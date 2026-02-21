import {
    Engine, Scene, ArcRotateCamera, Vector3, HemisphericLight, DirectionalLight,
    MeshBuilder, StandardMaterial, Color3, Color4, ParticleSystem,
    GlowLayer, ShadowGenerator, Mesh, TransformNode, KeyboardEventTypes, PointLight
} from '@babylonjs/core';

export interface SimState {
    phase: string; altitude: number; velocity: number; downrange: number;
    fuel: number; thrust: number; missionTime: number; throttle: number;
}

// Proportional scale: 1 unit ≈ 1 meter near ground
// Earth radius 6371km → we use 600 (compressed ~10,000x for rendering)
// Moon radius 1737km → 164 (Moon/Earth ratio 0.273 preserved)
// Earth-Moon distance 384,400km → 9000 (~15x Earth radius, compressed from real 60x)
const E_R = 600, M_R = 164, EM_DIST = 9000;
const lerp = (a: number, b: number, t: number) => a + (b - a) * Math.min(1, Math.max(0, t));

let engine: Engine | null = null, scene: Scene | null = null;
export function destroyScene() { scene?.dispose(); engine?.dispose(); engine = null; scene = null; }

export function initStarshipScene(canvas: HTMLCanvasElement, onTelemetry: (s: SimState) => void): () => void {
    engine = new Engine(canvas, true, { preserveDrawingBuffer: true, stencil: true });
    scene = new Scene(engine);
    scene.clearColor = new Color4(0.28, 0.52, 0.82, 1);

    // Camera starts looking at rocket on pad
    const cam = new ArcRotateCamera('cam', -Math.PI / 2, Math.PI / 3.2, 150, new Vector3(0, E_R + 50, 0), scene);
    cam.lowerRadiusLimit = 10; cam.upperRadiusLimit = 5000;
    cam.attachControl(canvas, true); cam.wheelPrecision = 5;

    const hemi = new HemisphericLight('hemi', new Vector3(0, 1, 0), scene); hemi.intensity = 0.3;
    const sun = new DirectionalLight('sun', new Vector3(-1, -2, -1), scene);
    sun.intensity = 1.5; sun.position = new Vector3(200, E_R + 400, 200);
    const glow = new GlowLayer('glow', scene); glow.intensity = 0.9;

    // ── EARTH (sphere centered at origin) ──
    const earth = buildEarth(scene);
    earth.clouds.rotation.y = 0; // Will animate in loop

    // ── Launch site ON Earth surface (top of sphere, y = E_R) ──
    const siteRoot = new TransformNode('siteRoot', scene);
    siteRoot.position.y = E_R; // Sits on north pole of Earth
    const launchSite = buildLaunchSite(scene, siteRoot);

    // Local ground patch (tangent to sphere at launch site)
    const ground = MeshBuilder.CreateGround('ground', { width: 2000, height: 2000, subdivisions: 2 }, scene);
    const gMat = new StandardMaterial('gm', scene);
    gMat.diffuseColor = new Color3(0.18, 0.15, 0.12); gMat.specularColor = Color3.Black();
    ground.material = gMat; ground.parent = siteRoot; ground.receiveShadows = true;

    // ── SHIP on pad ──
    const shipRoot = new TransformNode('shipRoot', scene);
    const ship = buildStarship(scene, shipRoot);
    const padTop = E_R + 28; // OLM top
    shipRoot.position.y = padTop + 35.5;

    const shadowGen = new ShadowGenerator(2048, sun);
    shadowGen.useBlurExponentialShadowMap = true;
    ship.meshes.forEach(m => shadowGen.addShadowCaster(m));

    // ── ENGINE FIRE: emissive nozzle glow + particles ──
    const engineLight = new PointLight('engLight', new Vector3(0, padTop - 5, 0), scene);
    engineLight.diffuse = new Color3(1, 0.6, 0.15);
    engineLight.intensity = 0; engineLight.range = 80;

    // Main exhaust plume
    const exEmit = MeshBuilder.CreateBox('exE', { size: 5 }, scene);
    exEmit.parent = shipRoot; exEmit.position.y = -37; exEmit.isVisible = false;
    const shipExhaust = makeExhaust(scene, exEmit, 6000, 3, 8, 25, 55, new Color4(1, 0.65, 0.1, 1), new Color4(1, 0.3, 0.05, 0.8));
    // Bright blue-white core (Mach diamonds)
    const shipCore = makeExhaust(scene, exEmit, 2000, 0.5, 2.5, 35, 65, new Color4(0.7, 0.8, 1, 1), new Color4(1, 1, 0.85, 0.95));

    // Launch smoke/steam (water deluge)
    const smokeEmit = MeshBuilder.CreateBox('smE', { size: 8 }, scene);
    smokeEmit.parent = siteRoot; smokeEmit.position.y = 28; smokeEmit.isVisible = false;
    const launchSmoke = makeSmokePS(scene, smokeEmit);

    // Booster exhaust (for return burns)
    const bExEmit = MeshBuilder.CreateBox('bExE', { size: 4 }, scene);
    bExEmit.parent = ship.booster; bExEmit.position.y = -35.5; bExEmit.isVisible = false;
    const boosterExhaust = makeExhaust(scene, bExEmit, 3000, 2, 5, 20, 45, new Color4(1, 0.65, 0.1, 1), new Color4(1, 0.3, 0.05, 0.8));

    createStarfield(scene);

    // ── MOON (sphere) ──
    const moonRoot = new TransformNode('moonRoot', scene);
    moonRoot.position.set(0, EM_DIST * 0.15, EM_DIST);
    const moonSphere = buildMoon(scene, moonRoot);

    // Moon base (on Moon surface)
    const moonBase = buildMoonBase(scene); moonBase.setEnabled(false);
    const astro = buildAstronaut(scene); astro.root.setEnabled(false);
    const rover = buildRover(scene); rover.root.setEnabled(false);

    // Keyboard
    const keys: Record<string, boolean> = {};
    scene.onKeyboardObservable.add(i => { keys[i.event.key.toLowerCase()] = i.type === KeyboardEventTypes.KEYDOWN; });

    const state: SimState = { phase: 'prelaunch', altitude: 0, velocity: 0, downrange: 0, fuel: 100, thrust: 0, missionTime: -10, throttle: 0 };
    let lastTime = performance.now(), launched = false;
    let boosterDetached = false, sepY = 0, boosterY = 0, boosterRotZ = 0;
    let moonLandingY = 0;

    const onLaunch = () => { launched = true; };
    window.addEventListener('starship-launch', onLaunch);

    engine.runRenderLoop(() => {
        const now = performance.now();
        const dt = Math.min((now - lastTime) / 1000, 0.1);
        lastTime = now;

        if (!launched) { state.missionTime = -10 + (performance.now() % 10000) / 1000; }
        else {
            state.missionTime += dt;
            const t = state.missionTime;

            // ── PHASES (separation in atmosphere ~25km, orbit at ~200km) ──
            if (t < 0) { state.phase = 'prelaunch'; state.throttle = 0; }
            else if (t < 3) { state.phase = 'ignition'; state.throttle = lerp(0, 100, t / 3); }
            else if (t < 18) { state.phase = 'liftoff'; state.throttle = 100; }
            else if (t < 28) { state.phase = 'maxq'; state.throttle = 80; }
            else if (t < 38) { state.phase = 'meco'; state.throttle = lerp(80, 0, (t - 28) / 10); }
            else if (t < 45) { state.phase = 'separation'; state.throttle = 0; } // ~25km alt
            else if (t < 55) { state.phase = 'ses'; state.throttle = 65; }
            else if (t < 90) { state.phase = 'orbit'; state.throttle = lerp(65, 5, (t - 55) / 35); }
            else if (t < 100) { state.phase = 'tli'; state.throttle = 75; } // Trans-Lunar Injection
            else if (t < 160) { state.phase = 'coast'; state.throttle = 2; }
            else if (t < 175) { state.phase = 'lunar-approach'; state.throttle = lerp(2, 15, (t - 160) / 15); }
            else if (t < 195) { state.phase = 'landing-burn'; state.throttle = lerp(0, 85, Math.min(1, (t - 175) / 5)); }
            else if (t < 205) { state.phase = 'touchdown'; state.throttle = lerp(30, 0, (t - 195) / 10); }
            else if (t < 215) { state.phase = 'landed'; state.throttle = 0; }
            else if (t < 235) { state.phase = 'eva'; state.throttle = 0; }
            else if (t < 275) { state.phase = 'exploration'; state.throttle = 0; }
            else { state.phase = 'complete'; state.throttle = 0; }

            // ── PHYSICS ──
            const accel = state.throttle / 100 * 3.5;
            const grav = state.altitude < 100 ? 0.0098 : 0.0005;
            const moving = !['touchdown', 'landed', 'eva', 'exploration', 'complete'].includes(state.phase);
            if (t >= 0 && moving) {
                state.velocity += (accel - grav) * dt;
                state.altitude += state.velocity * dt;
                state.downrange += state.velocity * dt * 0.7;
                state.fuel = Math.max(0, state.fuel - state.throttle * dt * 0.025);
                state.thrust = state.throttle;
            }
            if (!moving) { state.velocity = lerp(state.velocity, 0, dt * 3); state.thrust = state.throttle; }
            state.altitude = Math.max(0, state.altitude);
            state.velocity = Math.max(0, state.velocity);

            // ── SHIP POSITION: on Earth sphere surface, moving outward ──
            // Map altitude (km) to scene units above Earth surface
            const sceneAlt = state.altitude * 0.15; // Compressed: 100km → 15 units
            const targetY = E_R + 28 + 35.5 + sceneAlt;
            shipRoot.position.y = lerp(shipRoot.position.y, targetY, dt * 4);

            // Gravity turn
            if (['liftoff', 'maxq'].includes(state.phase)) {
                shipRoot.rotation.z = lerp(shipRoot.rotation.z, Math.min((t - 3) * 0.003, 0.2), dt * 2);
            } else if (['orbit', 'tli'].includes(state.phase)) {
                shipRoot.rotation.z = lerp(shipRoot.rotation.z, 0.6, dt * 0.5); // Nearly horizontal
            } else if (['landing-burn', 'touchdown', 'landed'].includes(state.phase)) {
                shipRoot.rotation.z = lerp(shipRoot.rotation.z, 0, dt * 2);
            }

            // ── ENGINE FIRE (visible from nozzles!) ──
            if (state.throttle > 3) {
                shipExhaust.start(); shipCore.start();
                shipExhaust.emitRate = state.throttle * 20;
                shipExhaust.minSize = 2 + state.throttle * 0.06;
                shipExhaust.maxSize = 6 + state.throttle * 0.1;
                shipCore.emitRate = state.throttle * 8;
                engineLight.intensity = state.throttle * 0.03;
                engineLight.position.y = shipRoot.position.y - 35;
                // Engine bell glow
                ship.engineGlow.emissiveColor = new Color3(
                    0.8 * state.throttle / 100, 0.3 * state.throttle / 100, 0.05
                );
            } else {
                shipExhaust.stop(); shipCore.stop();
                engineLight.intensity = 0;
                ship.engineGlow.emissiveColor = Color3.Black();
            }

            // Landing burn reignition flash
            if (state.phase === 'landing-burn' && t > 175 && t < 177) {
                shipExhaust.emitRate = 3000; shipExhaust.maxSize = 15;
                shipCore.emitRate = 1200;
                engineLight.intensity = 5;
            }

            // ── LAUNCH SMOKE (water deluge) ──
            if (t >= 0 && t < 8) {
                launchSmoke.start();
                launchSmoke.emitRate = lerp(3000, 500, t / 8);
            } else { launchSmoke.stop(); }

            // ══ BOOSTER SEPARATION + RETURN ══
            if (state.phase === 'separation' && !boosterDetached) {
                boosterDetached = true;
                sepY = shipRoot.position.y;
                ship.booster.setParent(null);
                ship.booster.position.set(0, sepY, 0);
                ship.booster.rotation.z = shipRoot.rotation.z;
                boosterY = sepY; boosterRotZ = shipRoot.rotation.z;
                exEmit.setParent(ship.shipNode);
                exEmit.position.y = -2;
            }
            if (boosterDetached) {
                if (t < 52) { // Flip
                    boosterRotZ = lerp(boosterRotZ, Math.PI, dt * 1.5);
                    boosterY = lerp(boosterY, sepY - 5, dt * 0.3);
                    boosterExhaust.stop();
                } else if (t < 68) { // Boostback
                    boosterExhaust.start(); boosterExhaust.emitRate = 900;
                    boosterY = lerp(boosterY, sepY * 0.5, dt * 0.7);
                } else if (t < 85) { // Entry
                    boosterExhaust.stop();
                    boosterY = lerp(boosterY, E_R + 60, dt * 0.5);
                } else if (t < 95) { // Landing burn
                    boosterExhaust.start(); boosterExhaust.emitRate = 1200;
                    boosterY = lerp(boosterY, E_R + 36, dt * 1.5);
                } else if (t < 100) { // Catch
                    boosterExhaust.emitRate = lerp(boosterExhaust.emitRate, 0, dt * 3);
                    boosterY = lerp(boosterY, E_R + 35, dt * 2);
                    launchSite.chopL.position.z = lerp(launchSite.chopL.position.z, -3.5, dt * 3);
                    launchSite.chopR.position.z = lerp(launchSite.chopR.position.z, 3.5, dt * 3);
                } else { boosterExhaust.stop(); boosterY = E_R + 35; }
                ship.booster.position.y = lerp(ship.booster.position.y, boosterY, dt * 5);
                ship.booster.rotation.z = lerp(ship.booster.rotation.z, boosterRotZ, dt * 3);
            }

            // ── CAMERA ──
            if (state.phase !== 'prelaunch') {
                const cTargetY = (boosterDetached && t > 42 && t < 100 && state.altitude < 200)
                    ? (shipRoot.position.y + boosterY) / 2
                    : shipRoot.position.y;
                cam.target.y = lerp(cam.target.y, cTargetY, dt * 2);
                // Zoom out as altitude increases (see Earth curvature)
                if (state.altitude > 30) {
                    const desiredR = 150 + state.altitude * 0.6;
                    cam.radius = lerp(cam.radius, Math.min(desiredR, 1500), dt * 0.8);
                }
            }
            // Shake
            const shake = (['liftoff', 'ignition'].includes(state.phase)) ? 0.3 : state.phase === 'landing-burn' ? 0.1 : 0;
            if (shake > 0) {
                cam.target.x += (Math.random() - 0.5) * shake;
                cam.target.y += (Math.random() - 0.5) * shake * 0.5;
            }

            // ── SKY: blue → dark → black ──
            const skyT = Math.min(1, state.altitude / 100);
            scene!.clearColor = new Color4(0.28 * (1 - skyT), 0.52 * (1 - skyT), 0.82 * (1 - skyT * 0.8), 1);

            // ── GROUND FADE (as Earth sphere takes over) ──
            if (state.altitude > 40) ground.visibility = lerp(ground.visibility, 0, dt * 2);
            if (state.altitude > 20) {
                // Show ground patch fading, Earth sphere taking over
                siteRoot.getChildMeshes().forEach(m => { if (m.name !== 'ground') m.visibility = lerp(m.visibility, Math.max(0, 1 - (state.altitude - 20) / 80), dt * 2); });
            }

            // ── MOON APPROACH ──
            if (['coast', 'lunar-approach', 'landing-burn', 'touchdown'].includes(state.phase)) {
                const aT = Math.min(1, (t - 100) / 95);
                moonRoot.position.z = lerp(EM_DIST, 200, aT);
                moonRoot.position.y = lerp(EM_DIST * 0.15, shipRoot.position.y - 50, aT);
                const ms = 1 + aT * 2;
                moonSphere.scaling.set(ms, ms, ms);
            }

            // Near Moon: land on sphere surface
            if (['landing-burn', 'touchdown', 'landed', 'eva', 'exploration', 'complete'].includes(state.phase)) {
                moonBase.setEnabled(true);
                moonLandingY = moonRoot.position.y + M_R + 2;
                moonBase.position.y = moonLandingY;
                if (['touchdown', 'landed', 'eva', 'exploration', 'complete'].includes(state.phase)) {
                    shipRoot.position.y = lerp(shipRoot.position.y, moonLandingY + 25, dt * 2);
                }
                ground.visibility = 0;
            }

            // ── EVA + ROVER ──
            if (['eva', 'exploration', 'complete'].includes(state.phase)) {
                astro.root.setEnabled(true);
                if (state.phase === 'eva') {
                    const ep = (t - 215) / 20;
                    astro.root.position.set(shipRoot.position.x + 15 + ep * 20, moonLandingY + 1.2 + Math.abs(Math.sin(t * 3)) * 0.3, 0);
                }
            }
            if (['exploration', 'complete'].includes(state.phase)) {
                rover.root.setEnabled(true);
                rover.root.position.y = moonLandingY + 0.8;
                if (state.phase === 'exploration') {
                    const sp = 15;
                    if (keys['w'] || keys['arrowup']) rover.root.position.z += sp * dt;
                    if (keys['s'] || keys['arrowdown']) rover.root.position.z -= sp * dt;
                    if (keys['a'] || keys['arrowleft']) { rover.root.rotation.y += 1.5 * dt; rover.root.position.x -= sp * dt * 0.5; }
                    if (keys['d'] || keys['arrowright']) { rover.root.rotation.y -= 1.5 * dt; rover.root.position.x += sp * dt * 0.5; }
                    cam.target.set(lerp(cam.target.x, rover.root.position.x, dt * 3), lerp(cam.target.y, rover.root.position.y + 2, dt * 3), lerp(cam.target.z, rover.root.position.z, dt * 3));
                    cam.radius = lerp(cam.radius, 40, dt * 2);
                    astro.root.position.set(rover.root.position.x, rover.root.position.y + 2, rover.root.position.z);
                }
            }
        }

        onTelemetry(state);
        scene!.render();
    });
    return () => { window.removeEventListener('starship-launch', onLaunch); };
}

// ═══ EARTH SPHERE ═══
function buildEarth(scene: Scene) {
    const earth = MeshBuilder.CreateSphere('earth', { diameter: E_R * 2, segments: 64 }, scene);
    const eM = new StandardMaterial('eM', scene);
    eM.diffuseColor = new Color3(0.12, 0.32, 0.62); eM.specularColor = new Color3(0.15, 0.15, 0.25);
    earth.material = eM;
    // Land masses
    const lM = new StandardMaterial('lnd', scene); lM.diffuseColor = new Color3(0.22, 0.42, 0.18); lM.specularColor = Color3.Black();
    for (let i = 0; i < 12; i++) {
        const l = MeshBuilder.CreateDisc('land' + i, { radius: 30 + Math.random() * 90, tessellation: 20 }, scene);
        l.material = lM;
        const th = Math.random() * Math.PI * 2, ph = Math.random() * Math.PI * 0.8 + 0.1, r = E_R + 0.3;
        l.position.set(r * Math.sin(ph) * Math.cos(th), r * Math.cos(ph), r * Math.sin(ph) * Math.sin(th));
        l.lookAt(Vector3.Zero());
    }
    // Atmosphere
    const atmo = MeshBuilder.CreateSphere('atmo', { diameter: E_R * 2 + 40, segments: 48 }, scene);
    const aM = new StandardMaterial('aM', scene); aM.diffuseColor = new Color3(0.3, 0.55, 0.95);
    aM.emissiveColor = new Color3(0.08, 0.15, 0.4); aM.alpha = 0.12; aM.backFaceCulling = false;
    atmo.material = aM;
    // Clouds
    const clouds = MeshBuilder.CreateSphere('clouds', { diameter: E_R * 2 + 12, segments: 48 }, scene);
    const cM = new StandardMaterial('cM', scene); cM.diffuseColor = Color3.White();
    cM.emissiveColor = new Color3(0.25, 0.25, 0.25); cM.alpha = 0.18; cM.backFaceCulling = false;
    clouds.material = cM;
    return { earth, atmo, clouds };
}

// ═══ MOON SPHERE ═══
function buildMoon(scene: Scene, parent: TransformNode): Mesh {
    const moon = MeshBuilder.CreateSphere('moon', { diameter: M_R * 2, segments: 48 }, scene);
    const mM = new StandardMaterial('mM', scene); mM.diffuseColor = new Color3(0.58, 0.55, 0.5);
    mM.specularColor = new Color3(0.04, 0.04, 0.04); moon.material = mM; moon.parent = parent;
    for (let i = 0; i < 8; i++) {
        const m = MeshBuilder.CreateDisc('mare' + i, { radius: 12 + Math.random() * 30, tessellation: 20 }, scene);
        m.material = mat(scene, 'mm' + i, 0.38, 0.36, 0.32, 0); m.parent = parent;
        const th = Math.random() * Math.PI * 2, ph = Math.random() * Math.PI, r = M_R + 0.2;
        m.position.set(r * Math.sin(ph) * Math.cos(th), r * Math.cos(ph), r * Math.sin(ph) * Math.sin(th));
        m.lookAt(new Vector3(0, 0, 0));
    }
    return moon;
}

// ═══ STARSHIP ═══
function buildStarship(scene: Scene, parent: TransformNode) {
    const meshes: Mesh[] = [];
    const steel = mat(scene, 'steel', 0.78, 0.78, 0.76, 0.5);
    const dark = mat(scene, 'dark', 0.3, 0.3, 0.3, 0.2);

    const black = mat(scene, 'black', 0.05, 0.05, 0.05, 0);
    // Engine glow material (shared, updated dynamically)
    const engineGlow = new StandardMaterial('engGlow', scene);
    engineGlow.diffuseColor = new Color3(0.1, 0.1, 0.1);
    engineGlow.emissiveColor = Color3.Black();

    // Booster (71m)
    const booster = new TransformNode('boosterRoot', scene); booster.parent = parent;
    meshes.push(cyl(scene, 'bb', 9, 9, 71, 48, steel, booster));
    meshes.push(cyl(scene, 'sk', 9, 9.4, 5, 48, dark, booster, 0, -33, 0));
    // 33 engines
    for (const { n, r } of [{ n: 3, r: 0.8 }, { n: 10, r: 2.3 }, { n: 20, r: 3.8 }]) {
        for (let i = 0; i < n; i++) {
            const a = (i * Math.PI * 2) / n;
            const e = cyl(scene, 'e' + n + i, 0.78, 1.3, 3, 12, engineGlow, booster, Math.cos(a) * r, -35.5, Math.sin(a) * r);
            meshes.push(e);
        }
    }
    // Grid fins (lattice)
    const finM = mat(scene, 'fin', 0.35, 0.33, 0.3, 0.15);
    for (let i = 0; i < 4; i++) {
        const a = (i * Math.PI) / 2 + Math.PI / 4;
        const fr = new TransformNode('fr' + i, scene); fr.parent = booster;
        fr.position.set(Math.cos(a) * 5.5, 33, Math.sin(a) * 5.5); fr.rotation.y = a;
        meshes.push(box(scene, 'ff' + i, 4.5, 3.5, 0.15, finM, fr));
        for (let h = 0; h < 5; h++) meshes.push(box(scene, 'fh' + i + h, 4.3, 0.06, 0.18, finM, fr, 0, -1.4 + h * 0.7, 0));
        for (let v = 0; v < 6; v++) meshes.push(box(scene, 'fv' + i + v, 0.06, 3.3, 0.18, finM, fr, -1.8 + v * 0.72, 0, 0));
    }
    meshes.push(cyl(scene, 'hs', 9.2, 9.2, 4, 48, dark, booster, 0, 37.5, 0));

    // Ship (52m)
    const shipNode = new TransformNode('shipNode', scene); shipNode.parent = parent;
    const bY = 39.5;
    meshes.push(cyl(scene, 'sb', 9, 9, 36, 48, steel, shipNode, 0, bY + 18, 0));
    meshes.push(cyl(scene, 'nl', 6, 9, 8, 48, steel, shipNode, 0, bY + 40, 0));
    meshes.push(cyl(scene, 'nu', 0, 6, 8, 48, steel, shipNode, 0, bY + 48, 0));
    const sh = MeshBuilder.CreateCylinder('shld', { diameterTop: 9.15, diameterBottom: 9.15, height: 36, tessellation: 48, arc: 0.5 }, scene);
    sh.material = black; sh.parent = shipNode; sh.position.y = bY + 18; meshes.push(sh);
    // Flaps (trapezoidal) 
    const flapM = mat(scene, 'flapM', 0.08, 0.08, 0.08, 0);
    for (let i = 0; i < 2; i++) {
        const fr = new TransformNode('fwR' + i, scene); fr.parent = shipNode;
        fr.position.set(i === 0 ? 4.8 : -4.8, bY + 35, 0); fr.rotation.z = i === 0 ? -0.12 : 0.12;
        const f = MeshBuilder.CreateCylinder('fwF' + i, { diameterTop: 3.5, diameterBottom: 5.5, height: 8, tessellation: 4 }, scene);
        f.material = flapM; f.parent = fr; f.rotation.y = Math.PI / 4; f.scaling.z = 0.04; meshes.push(f);
    }
    for (let i = 0; i < 2; i++) {
        const fr = new TransformNode('afR' + i, scene); fr.parent = shipNode;
        fr.position.set(0, bY + 3, i === 0 ? 4.8 : -4.8); fr.rotation.x = i === 0 ? -0.1 : 0.1;
        const f = MeshBuilder.CreateCylinder('afF' + i, { diameterTop: 2.8, diameterBottom: 4.5, height: 6, tessellation: 4 }, scene);
        f.material = flapM; f.parent = fr; f.rotation.y = Math.PI / 4; f.scaling.z = 0.04; meshes.push(f);
    }
    // Ship engines (glow material)
    for (let i = 0; i < 3; i++) {
        const a = (i * Math.PI * 2) / 3 + Math.PI / 6;
        meshes.push(cyl(scene, 'se' + i, 0.78, 1.3, 2.5, 12, engineGlow, shipNode, Math.cos(a) * 1.5, bY - 2, Math.sin(a) * 1.5));
    }
    for (let i = 0; i < 3; i++) {
        const a = (i * Math.PI * 2) / 3;
        const e = cyl(scene, 've' + i, 1.0, 2.4, 4, 16, engineGlow, shipNode, Math.cos(a) * 3.2, bY - 3, Math.sin(a) * 3.2);
        cyl(scene, 'vx' + i, 2.4, 2.8, 2.5, 16, mat(scene, 'vxm' + i, 0.25, 0.2, 0.15, 0.1), e, 0, -3.25, 0);
        meshes.push(e);
    }
    return { meshes, booster, shipNode, engineGlow };
}

// ═══ LAUNCH SITE (on Earth surface) ═══
function buildLaunchSite(scene: Scene, parent: TransformNode) {
    const tM = mat(scene, 'tw', 0.45, 0.4, 0.35, 0.15);
    const cM = mat(scene, 'cn', 0.35, 0.33, 0.3, 0);
    const sM = mat(scene, 'sp', 0.4, 0.38, 0.35, 0.2);
    // OLM: steel table with central opening, on 6 legs
    box(scene, 'olmL', 25, 3, 25, sM, parent, 0, 27.5, 0);
    // OLM hole (dark center representing flame opening)
    const holeMat = mat(scene, 'hole', 0.08, 0.08, 0.08, 0);
    box(scene, 'olmHole', 10, 3.1, 10, holeMat, parent, 0, 27.5, 0);
    // 6 support legs
    for (let i = 0; i < 6; i++) {
        const a = (i * Math.PI * 2) / 6;
        cyl(scene, 'olmLeg' + i, 2, 2.5, 26, 8, sM, parent, Math.cos(a) * 10, 13, Math.sin(a) * 10);
    }
    // Water-cooled steel plate (beneath OLM)
    box(scene, 'wPlate', 22, 1, 22, mat(scene, 'wcp', 0.3, 0.28, 0.25, 0.1), parent, 0, 25, 0);
    // Concrete foundation
    box(scene, 'found', 30, 2, 30, cM, parent, 0, 1, 0);
    // Flame trench
    box(scene, 'trench', 8, 4, 35, mat(scene, 'trM', 0.2, 0.18, 0.16, 0), parent, 0, 2, 18);

    // Tower (140m, 4 rails + cross-bracing)
    const tx = -18;
    for (const [dx, dz] of [[-2, -2], [2, -2], [-2, 2], [2, 2]]) {
        box(scene, 'tc' + dx + dz, 1.5, 140, 1.5, tM, parent, tx + dx, 70, dz);
    }
    for (let l = 0; l < 9; l++) {
        const y = 10 + l * 15;
        box(scene, 'hb' + l, 4, 0.4, 0.4, tM, parent, tx, y, 0);
        box(scene, 'hb2' + l, 0.4, 0.4, 4, tM, parent, tx, y, 0);
    }
    // Chopsticks
    const chM = mat(scene, 'ch', 0.5, 0.45, 0.4, 0.2);
    const chopL = box(scene, 'chopL', 36, 2, 1.8, chM, parent, tx + 20, 85, -5.5);
    const chopR = box(scene, 'chopR', 36, 2, 1.8, chM, parent, tx + 20, 85, 5.5);
    // QD arm
    box(scene, 'qd', 22, 1.5, 2, chM, parent, tx + 13, 105, 0);
    // Lightning rod
    cyl(scene, 'rod', 0.08, 0.3, 12, 8, tM, parent, tx, 146, 0);
    // Concrete pad
    box(scene, 'cpad', 80, 0.3, 80, cM, parent, 0, 0.15, 0);
    // Fuel tanks
    const wM = mat(scene, 'wh', 0.9, 0.9, 0.9, 0.3);
    for (let i = 0; i < 4; i++) cyl(scene, 'ft' + i, 5, 5, 20, 16, wM, parent, -40 + i * 8, 10, 40);
    return { chopL, chopR };
}

// ═══ MOON BASE ═══
function buildMoonBase(scene: Scene): TransformNode {
    const root = new TransformNode('mBase', scene);
    const hM = mat(scene, 'hab', 0.85, 0.85, 0.82, 0.3);
    const mM = mat(scene, 'met', 0.5, 0.5, 0.5, 0.2);
    const pM = new StandardMaterial('pan', scene); pM.diffuseColor = new Color3(0.1, 0.1, 0.35);
    pM.emissiveColor = new Color3(0.02, 0.02, 0.08);
    for (let i = 0; i < 3; i++) {
        const d = MeshBuilder.CreateSphere('dome' + i, { diameter: 12, slice: 0.5, segments: 16 }, scene);
        d.material = hM; d.parent = root; d.position.set(50 + i * 18, 0, 30);
    }
    for (let i = 0; i < 2; i++) { const t = cyl(scene, 'tun' + i, 3, 3, 16, 10, mM, root, 59 + i * 18, 1.5, 30); t.rotation.z = Math.PI / 2; }
    for (let i = 0; i < 4; i++) {
        cyl(scene, 'pole' + i, 0.3, 0.3, 6, 8, mM, root, 50 + i * 12, 3, 50);
        box(scene, 'spnl' + i, 8, 0.1, 4, pM, root, 50 + i * 12, 6.5, 50);
    }
    for (let i = 0; i < 8; i++) {
        const a = (i * Math.PI * 2) / 8;
        const lm = new StandardMaterial('lmm' + i, scene); lm.diffuseColor = new Color3(0.8, 0.4, 0.1); lm.emissiveColor = new Color3(0.3, 0.15, 0.05);
        const m = box(scene, 'lm' + i, 3, 0.05, 0.5, lm, root, Math.cos(a) * 15, 0.05, Math.sin(a) * 15);
        m.rotation.y = a;
    }
    return root;
}

function buildAstronaut(scene: Scene) {
    const root = new TransformNode('astro', scene);
    const sM = mat(scene, 'suit', 0.9, 0.9, 0.88, 0.2);
    MeshBuilder.CreateCapsule('torso', { height: 1.6, radius: 0.4 }, scene).material = sM;
    scene.getMeshByName('torso')!.parent = root;
    MeshBuilder.CreateSphere('helm', { diameter: 0.7, segments: 10 }, scene).material = sM;
    scene.getMeshByName('helm')!.parent = root; scene.getMeshByName('helm')!.position.y = 1.2;
    const vM = new StandardMaterial('vis', scene); vM.diffuseColor = new Color3(0.1, 0.15, 0.3); vM.emissiveColor = new Color3(0.05, 0.08, 0.15);
    const v = MeshBuilder.CreateSphere('visor', { diameter: 0.5, segments: 10 }, scene); v.material = vM; v.parent = root; v.position.set(0.15, 1.25, 0);
    box(scene, 'plss', 0.5, 0.7, 0.3, sM, root, -0.3, 0.5, 0);
    for (let i = 0; i < 2; i++) cyl(scene, 'leg' + i, 0.25, 0.25, 0.9, 8, sM, root, 0, -0.7, (i - 0.5) * 0.3);
    return { root };
}

function buildRover(scene: Scene) {
    const root = new TransformNode('rover', scene); root.position.set(70, 0, 30);
    const bM = mat(scene, 'rB', 0.7, 0.7, 0.68, 0.2);
    const wM = mat(scene, 'rW', 0.25, 0.25, 0.25, 0.1);
    const body = box(scene, 'rCh', 4, 1, 2.5, bM, root);
    for (const [x, z] of [[-1.8, -1.2], [-1.8, 1.2], [1.8, -1.2], [1.8, 1.2]]) {
        const w = cyl(scene, 'rw' + x + z, 0.8, 0.8, 0.3, 16, wM, root, x, -0.3, z); w.rotation.x = Math.PI / 2;
    }
    const spM = new StandardMaterial('rspM', scene); spM.diffuseColor = new Color3(0.1, 0.1, 0.35); spM.emissiveColor = new Color3(0.02, 0.02, 0.06);
    box(scene, 'rsp', 3, 0.05, 1.5, spM, root, 0, 1.2, 0);
    return { root, body };
}

// ═══ PARTICLES ═══
function makeExhaust(sc: Scene, em: Mesh, count: number, minS: number, maxS: number, minP: number, maxP: number, c1: Color4, c2: Color4): ParticleSystem {
    const ps = new ParticleSystem('ex', count, sc);
    ps.createPointEmitter(new Vector3(-2, -1, -2), new Vector3(2, -1, 2));
    ps.color1 = c1; ps.color2 = c2; ps.colorDead = new Color4(0.3, 0.2, 0.1, 0);
    ps.minSize = minS; ps.maxSize = maxS; ps.minLifeTime = 0.2; ps.maxLifeTime = 1.0;
    ps.emitRate = 0; ps.blendMode = ParticleSystem.BLENDMODE_ADD;
    ps.minEmitPower = minP; ps.maxEmitPower = maxP; ps.updateSpeed = 0.02;
    ps.gravity = new Vector3(0, -6, 0); ps.emitter = em; ps.start(); return ps;
}

function makeSmokePS(scene: Scene, emitter: Mesh): ParticleSystem {
    const ps = new ParticleSystem('smoke', 4000, scene);
    ps.createSphereEmitter(8);
    ps.color1 = new Color4(0.85, 0.85, 0.82, 0.7);
    ps.color2 = new Color4(0.7, 0.7, 0.68, 0.5);
    ps.colorDead = new Color4(0.5, 0.5, 0.5, 0);
    ps.minSize = 4; ps.maxSize = 15; ps.minLifeTime = 1.5; ps.maxLifeTime = 4;
    ps.emitRate = 0; ps.blendMode = ParticleSystem.BLENDMODE_STANDARD;
    ps.minEmitPower = 15; ps.maxEmitPower = 35; ps.updateSpeed = 0.02;
    ps.gravity = new Vector3(0, 3, 0); ps.emitter = emitter; ps.start(); return ps;
}

function createStarfield(scene: Scene) {
    const m = new StandardMaterial('sM', scene); m.emissiveColor = Color3.White(); m.disableLighting = true;
    for (let i = 0; i < 600; i++) {
        const s = MeshBuilder.CreateSphere('s' + i, { diameter: 0.3 + Math.random() * 0.6 }, scene);
        s.material = m; const r = 1200 + Math.random() * 1000;
        const th = Math.random() * Math.PI * 2, ph = Math.random() * Math.PI;
        s.position.set(r * Math.sin(ph) * Math.cos(th), r * Math.cos(ph), r * Math.sin(ph) * Math.sin(th));
    }
}

// ═══ HELPERS ═══
function mat(sc: Scene, n: string, r: number, g: number, b: number, sp: number): StandardMaterial {
    const m = new StandardMaterial(n, sc); m.diffuseColor = new Color3(r, g, b); m.specularColor = new Color3(sp, sp, sp); return m;
}
function cyl(sc: Scene, n: string, dT: number, dB: number, h: number, t: number, m: StandardMaterial, p?: TransformNode | Mesh, x = 0, y = 0, z = 0): Mesh {
    const c = MeshBuilder.CreateCylinder(n, { diameterTop: dT, diameterBottom: dB, height: h, tessellation: t }, sc);
    c.material = m; if (p) c.parent = p; if (x || y || z) c.position.set(x, y, z); return c;
}
function box(sc: Scene, n: string, w: number, h: number, d: number, m: StandardMaterial, p?: TransformNode | Mesh, x = 0, y = 0, z = 0): Mesh {
    const b = MeshBuilder.CreateBox(n, { width: w, height: h, depth: d }, sc);
    b.material = m; if (p) b.parent = p; if (x || y || z) b.position.set(x, y, z); return b;
}
