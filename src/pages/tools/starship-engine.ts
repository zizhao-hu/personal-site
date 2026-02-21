import {
    Engine, Scene, ArcRotateCamera, Vector3, HemisphericLight, DirectionalLight,
    MeshBuilder, StandardMaterial, Color3, Color4, ParticleSystem,
    GlowLayer, ShadowGenerator, Mesh, TransformNode, KeyboardEventTypes
} from '@babylonjs/core';

export interface SimState {
    phase: string; altitude: number; velocity: number; downrange: number;
    fuel: number; thrust: number; missionTime: number; throttle: number;
}

const EARTH_R = 500, MOON_R = 136, MOON_DIST = 6000;
const lerp = (a: number, b: number, t: number) => a + (b - a) * Math.min(1, Math.max(0, t));

let engine: Engine | null = null, scene: Scene | null = null;

export function destroyScene() { scene?.dispose(); engine?.dispose(); engine = null; scene = null; }

export function initStarshipScene(canvas: HTMLCanvasElement, onTelemetry: (s: SimState) => void): () => void {
    engine = new Engine(canvas, true, { preserveDrawingBuffer: true, stencil: true });
    scene = new Scene(engine);
    scene.clearColor = new Color4(0.25, 0.45, 0.75, 1);

    const cam = new ArcRotateCamera('cam', -Math.PI / 2, Math.PI / 3, 180, new Vector3(0, 60, 0), scene);
    cam.lowerRadiusLimit = 15; cam.upperRadiusLimit = 2000;
    cam.attachControl(canvas, true); cam.wheelPrecision = 5;

    const hemi = new HemisphericLight('hemi', new Vector3(0, 1, 0), scene);
    hemi.intensity = 0.35;
    const sun = new DirectionalLight('sun', new Vector3(-1, -2, -1), scene);
    sun.intensity = 1.4; sun.position = new Vector3(200, 400, 200);
    const glow = new GlowLayer('glow', scene); glow.intensity = 0.8;

    // ── Build scene ──
    const shipRoot = new TransformNode('shipRoot', scene);
    const ship = buildStarshipStack(scene, shipRoot);
    const padY = 30; // OLM top height
    shipRoot.position.y = padY + 35.5; // booster center

    const launchSite = buildLaunchSite(scene);

    const ground = MeshBuilder.CreateGround('ground', { width: 4000, height: 4000, subdivisions: 4 }, scene);
    const gMat = new StandardMaterial('gm', scene);
    gMat.diffuseColor = new Color3(0.15, 0.13, 0.1); gMat.specularColor = Color3.Black();
    ground.material = gMat; ground.receiveShadows = true;

    const shadowGen = new ShadowGenerator(2048, sun);
    shadowGen.useBlurExponentialShadowMap = true;
    ship.meshes.forEach(m => shadowGen.addShadowCaster(m));

    // Ship exhaust (follows ship after separation)
    const shipExhEmit = MeshBuilder.CreateSphere('sExE', { diameter: 4 }, scene);
    shipExhEmit.parent = shipRoot; shipExhEmit.position.y = -37; shipExhEmit.isVisible = false;
    const shipExhaust = createExhaustPS(scene, shipExhEmit, 5000);
    const shipCore = createCorePS(scene, shipExhEmit);

    // Booster exhaust (follows booster, used for return burns)
    const boosterExhEmit = MeshBuilder.CreateSphere('bExE', { diameter: 4 }, scene);
    boosterExhEmit.parent = ship.booster; boosterExhEmit.position.y = -35.5; boosterExhEmit.isVisible = false;
    const boosterExhaust = createExhaustPS(scene, boosterExhEmit, 3000);

    createStarfield(scene);

    // Earth
    const earthRoot = new TransformNode('earthRoot', scene);
    earthRoot.position.y = -EARTH_R;
    const earthParts = buildEarth(scene, earthRoot);
    earthParts.earthSphere.isVisible = false; earthParts.atmosphere.isVisible = false; earthParts.clouds.isVisible = false;

    // Moon
    const moonRoot = new TransformNode('moonRoot', scene);
    moonRoot.position.set(0, MOON_DIST * 0.3, MOON_DIST);
    const moonSphere = buildMoonSphere(scene, moonRoot);
    const moonSurface = buildMoonSurface(scene); moonSurface.setEnabled(false);
    const moonBase = buildMoonBase(scene); moonBase.setEnabled(false);
    const astro = buildAstronaut(scene); astro.root.setEnabled(false);
    const rover = buildRover(scene); rover.root.setEnabled(false);

    // Keyboard
    const keys: Record<string, boolean> = {};
    scene.onKeyboardObservable.add(info => {
        keys[info.event.key.toLowerCase()] = info.type === KeyboardEventTypes.KEYDOWN;
    });

    // State
    const state: SimState = { phase: 'prelaunch', altitude: 0, velocity: 0, downrange: 0, fuel: 100, thrust: 0, missionTime: -10, throttle: 0 };
    let lastTime = performance.now(), launched = false;

    // Booster return state
    let boosterDetached = false;
    let boosterWorldY = 0, boosterWorldX = 0;
    let boosterRotZ = 0;

    let sepY = 0, sepX = 0;

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

            // ── Phase transitions ──
            if (t < 0) { state.phase = 'prelaunch'; state.throttle = 0; }
            else if (t < 3) { state.phase = 'ignition'; state.throttle = lerp(0, 100, t / 3); }
            else if (t < 15) { state.phase = 'liftoff'; state.throttle = 100; }
            else if (t < 25) { state.phase = 'maxq'; state.throttle = 80; }
            else if (t < 40) { state.phase = 'meco'; state.throttle = lerp(80, 0, (t - 25) / 15); }
            else if (t < 48) { state.phase = 'separation'; state.throttle = 0; }
            else if (t < 55) { state.phase = 'ses'; state.throttle = 60; }
            else if (t < 120) { state.phase = 'coast'; state.throttle = 5; }
            else if (t < 160) { state.phase = 'lunar-approach'; state.throttle = lerp(5, 15, (t - 120) / 40); }
            else if (t < 180) { state.phase = 'landing-burn'; state.throttle = lerp(0, 90, (t - 160) / 8); }
            else if (t < 192) { state.phase = 'touchdown'; state.throttle = lerp(30, 0, (t - 180) / 12); }
            else if (t < 202) { state.phase = 'landed'; state.throttle = 0; }
            else if (t < 222) { state.phase = 'eva'; state.throttle = 0; }
            else if (t < 262) { state.phase = 'exploration'; state.throttle = 0; }
            else { state.phase = 'complete'; state.throttle = 0; }

            // ── Physics ──
            const accel = state.throttle / 100 * 3.5;
            const grav = state.altitude < 100 ? 0.0098 : 0.001;
            const moving = !['touchdown', 'landed', 'eva', 'exploration', 'complete'].includes(state.phase);
            if (t >= 0 && moving) {
                state.velocity += (accel - grav) * dt;
                state.altitude += state.velocity * dt;
                state.downrange += state.velocity * dt * 0.7;
                state.fuel = Math.max(0, state.fuel - state.throttle * dt * 0.03);
                state.thrust = state.throttle;
            }
            if (!moving) { state.velocity = lerp(state.velocity, 0, dt * 3); state.thrust = 0; }
            state.altitude = Math.max(0, state.altitude);
            state.velocity = Math.max(0, state.velocity);

            // ── Ship visual position (smooth) ──
            const targetY = padY + 35.5 + Math.min(state.altitude * 0.8, 400);
            shipRoot.position.y = lerp(shipRoot.position.y, targetY, dt * 4);

            // Gravity turn tilt (smooth)
            if (state.phase === 'liftoff' || state.phase === 'maxq') {
                shipRoot.rotation.z = lerp(shipRoot.rotation.z, Math.min((t - 3) * 0.002, 0.12), dt * 2);
            } else if (['landing-burn', 'touchdown', 'landed'].includes(state.phase)) {
                shipRoot.rotation.z = lerp(shipRoot.rotation.z, 0, dt * 3);
            }

            // ── Ship exhaust ──
            if (state.throttle > 5 && !['separation'].includes(state.phase)) {
                shipExhaust.start(); shipCore.start();
                shipExhaust.emitRate = state.throttle * 15;
                shipExhaust.minSize = 2 + state.throttle * 0.05;
                shipExhaust.maxSize = 6 + state.throttle * 0.08;
                shipCore.emitRate = state.throttle * 5;
            } else { shipExhaust.stop(); shipCore.stop(); }

            // ── Landing burn clear reignition == flash the exhaust bright ──
            if (state.phase === 'landing-burn' && t > 160 && t < 162) {
                shipExhaust.emitRate = 2000;
                shipExhaust.minSize = 5; shipExhaust.maxSize = 12;
                shipCore.emitRate = 800;
            }

            // ═══ BOOSTER SEPARATION + RETURN ═══
            if (state.phase === 'separation' && !boosterDetached) {
                boosterDetached = true;
                sepY = shipRoot.position.y;
                sepX = shipRoot.position.x;
                boosterWorldY = sepY;
                boosterWorldX = sepX;
                boosterRotZ = shipRoot.rotation.z;
                // Detach booster from ship
                ship.booster.setParent(null);
                ship.booster.position.set(sepX, sepY, 0);
                ship.booster.rotation.z = boosterRotZ;
                // Move ship exhaust to ship node (it was on root which included booster)
                shipExhEmit.setParent(ship.shipNode);
                shipExhEmit.position.y = -2;
            }

            if (boosterDetached) {
                // Phase-based booster trajectory (all smooth lerped)
                if (t < 50) {
                    // Flip: rotate 180 degrees smoothly
                    boosterRotZ = lerp(boosterRotZ, Math.PI, dt * 2);
                    boosterWorldY = lerp(boosterWorldY, sepY - 10, dt * 0.5);
                    boosterExhaust.stop();
                } else if (t < 65) {
                    // Boostback burn: engines fire, heading back
                    boosterExhaust.start();
                    boosterExhaust.emitRate = 800;
                    boosterWorldY = lerp(boosterWorldY, sepY * 0.6, dt * 0.8);
                    boosterWorldX = lerp(boosterWorldX, 0, dt * 0.5); // Return toward pad
                } else if (t < 85) {
                    // Entry coast: descending
                    boosterExhaust.stop();
                    boosterWorldY = lerp(boosterWorldY, padY + 60, dt * 0.6);
                    boosterWorldX = lerp(boosterWorldX, -2, dt * 0.8);
                } else if (t < 95) {
                    // Landing burn
                    boosterExhaust.start(); boosterExhaust.emitRate = 1200;
                    boosterWorldY = lerp(boosterWorldY, padY + 38, dt * 1.2);
                    boosterWorldX = lerp(boosterWorldX, 0, dt * 2);
                    boosterRotZ = lerp(boosterRotZ, Math.PI, dt * 3); // Keep vertical inverted... actually 0 is right-side-up
                } else if (t < 100) {
                    // Final approach + catch
                    boosterExhaust.emitRate = lerp(boosterExhaust.emitRate, 0, dt * 3);
                    boosterWorldY = lerp(boosterWorldY, padY + 36, dt * 2);
                    boosterWorldX = lerp(boosterWorldX, 0, dt * 5);
                    // Chopsticks close!

                    launchSite.chopLeft.position.z = lerp(launchSite.chopLeft.position.z, -3.5, dt * 3);
                    launchSite.chopRight.position.z = lerp(launchSite.chopRight.position.z, 3.5, dt * 3);
                } else {
                    boosterExhaust.stop();
                    boosterWorldY = padY + 36; boosterWorldX = 0;
                }

                // Apply booster position smoothly
                ship.booster.position.x = lerp(ship.booster.position.x, boosterWorldX, dt * 5);
                ship.booster.position.y = lerp(ship.booster.position.y, boosterWorldY, dt * 5);
                ship.booster.rotation.z = lerp(ship.booster.rotation.z, boosterRotZ, dt * 3);

                // Flip booster rotation: it launches pointing up, flips to engines-first
                if (t >= 42 && t < 50) {
                    const flipProg = (t - 42) / 8;
                    ship.booster.rotation.z = lerp(shipRoot.rotation.z, Math.PI, flipProg);
                }
            }

            // ── Camera follow (smooth) ──
            if (state.phase !== 'prelaunch') {
                const camTargetY = (boosterDetached && t < 100 && t > 42 && state.altitude < 200)
                    ? lerp(cam.target.y, (shipRoot.position.y + boosterWorldY) / 2, dt * 1.5)
                    : lerp(cam.target.y, shipRoot.position.y, dt * 2);
                cam.target.y = camTargetY;
                if (state.altitude > 50) {
                    cam.radius = lerp(cam.radius, 200 + state.altitude * 0.4, dt);
                }
            }

            // Camera shake (smooth)
            const shakeAmt = (state.phase === 'liftoff' || state.phase === 'ignition') ? 0.3
                : state.phase === 'landing-burn' ? 0.12 : 0;
            if (shakeAmt > 0) {
                cam.target.x += (Math.random() - 0.5) * shakeAmt;
                cam.target.y += (Math.random() - 0.5) * shakeAmt * 0.5;
            }

            // ── Sky color ──
            const skyT = Math.min(1, state.altitude / 120);
            scene!.clearColor = new Color4(0.25 * (1 - skyT), 0.45 * (1 - skyT), 0.75 * (1 - skyT * 0.7), 1);

            // ── Earth visibility ──
            if (state.altitude > 30) {
                earthParts.earthSphere.isVisible = true;
                earthParts.atmosphere.isVisible = true;
                earthParts.clouds.isVisible = true;
            }
            earthParts.clouds.rotation.y += dt * 0.01;

            // Ground fade
            if (state.altitude > 60) ground.visibility = Math.max(0, 1 - (state.altitude - 60) / 60);

            // ── Moon approach ──
            if (['coast', 'lunar-approach', 'landing-burn'].includes(state.phase)) {
                const aT = Math.min(1, (t - 55) / 125);
                moonRoot.position.z = lerp(MOON_DIST, MOON_DIST * 0.05, aT);
                moonRoot.position.y = lerp(MOON_DIST * 0.3, 0, aT);
                const ms = 1 + aT * 3;
                moonSphere.scaling.set(ms, ms, ms);
                earthRoot.position.y = lerp(-EARTH_R, -EARTH_R - 2000, aT);
            }

            // Moon surface
            if (['lunar-approach', 'landing-burn', 'touchdown', 'landed', 'eva', 'exploration', 'complete'].includes(state.phase)) {
                moonSurface.setEnabled(true); moonBase.setEnabled(true);
                if (['landing-burn', 'touchdown'].includes(state.phase)) {
                    const surfY = shipRoot.position.y - 70;
                    moonSurface.position.y = lerp(moonSurface.position.y, surfY, dt * 2);
                    moonBase.position.y = moonSurface.position.y;
                }
                ground.visibility = 0;
            }

            // ── EVA + Rover ──
            if (['eva', 'exploration', 'complete'].includes(state.phase)) {
                astro.root.setEnabled(true);
                const moonY = moonSurface.position.y;
                if (state.phase === 'eva') {
                    const ep = (t - 202) / 20;
                    astro.root.position.set(
                        shipRoot.position.x + 15 + ep * 20,
                        moonY + 1.2 + Math.abs(Math.sin(t * 3)) * 0.4,
                        shipRoot.position.z
                    );
                }
            }
            if (['exploration', 'complete'].includes(state.phase)) {
                rover.root.setEnabled(true);
                rover.root.position.y = moonSurface.position.y + 0.8;
                if (state.phase === 'exploration') {
                    const spd = 15;
                    if (keys['w'] || keys['arrowup']) rover.root.position.z += spd * dt;
                    if (keys['s'] || keys['arrowdown']) rover.root.position.z -= spd * dt;
                    if (keys['a'] || keys['arrowleft']) { rover.root.rotation.y += 1.5 * dt; rover.root.position.x -= spd * dt * 0.5; }
                    if (keys['d'] || keys['arrowright']) { rover.root.rotation.y -= 1.5 * dt; rover.root.position.x += spd * dt * 0.5; }
                    cam.target.x = lerp(cam.target.x, rover.root.position.x, dt * 3);
                    cam.target.z = lerp(cam.target.z, rover.root.position.z, dt * 3);
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

// ═══ STARSHIP STACK ═══
function buildStarshipStack(scene: Scene, parent: TransformNode) {
    const meshes: Mesh[] = [];
    const steel = mat(scene, 'steel', 0.78, 0.78, 0.76, 0.5);
    const dark = mat(scene, 'dark', 0.3, 0.3, 0.3, 0.2);
    const eng = mat(scene, 'eng', 0.15, 0.15, 0.15, 0.1);
    const black = mat(scene, 'black', 0.05, 0.05, 0.05, 0);

    // ── BOOSTER (71m) ──
    const booster = new TransformNode('boosterRoot', scene);
    booster.parent = parent;
    const bb = cyl(scene, 'bb', 9, 9, 71, 48); bb.material = steel; bb.parent = booster; meshes.push(bb);
    const sk = cyl(scene, 'sk', 9, 9.4, 5, 48); sk.material = dark; sk.parent = booster; sk.position.y = -33; meshes.push(sk);
    // 33 engines: 3+10+20
    for (const { n, r } of [{ n: 3, r: 0.8 }, { n: 10, r: 2.3 }, { n: 20, r: 3.8 }]) {
        for (let i = 0; i < n; i++) {
            const a = (i * Math.PI * 2) / n;
            const e = cyl(scene, 'e', 0.78, 1.3, 3, 12); e.material = eng; e.parent = booster;
            e.position.set(Math.cos(a) * r, -35.5, Math.sin(a) * r); meshes.push(e);
        }
    }

    // Grid fins — proper lattice structure
    const finMat = mat(scene, 'fin', 0.35, 0.33, 0.3, 0.15);
    for (let i = 0; i < 4; i++) {
        const a = (i * Math.PI) / 2 + Math.PI / 4;
        const finRoot = new TransformNode('finR' + i, scene);
        finRoot.parent = booster; finRoot.position.y = 33;
        finRoot.position.x = Math.cos(a) * 5.5; finRoot.position.z = Math.sin(a) * 5.5;
        finRoot.rotation.y = a;
        // Outer frame
        const frame = MeshBuilder.CreateBox('finFrame' + i, { width: 4.5, height: 3.5, depth: 0.15 }, scene);
        frame.material = finMat; frame.parent = finRoot; meshes.push(frame);
        // Lattice grid (horizontal bars)
        for (let h = 0; h < 5; h++) {
            const bar = MeshBuilder.CreateBox('fh' + i + h, { width: 4.3, height: 0.06, depth: 0.18 }, scene);
            bar.material = finMat; bar.parent = finRoot; bar.position.y = -1.4 + h * 0.7;
            meshes.push(bar);
        }
        // Lattice grid (vertical bars)
        for (let v = 0; v < 6; v++) {
            const bar = MeshBuilder.CreateBox('fv' + i + v, { width: 0.06, height: 3.3, depth: 0.18 }, scene);
            bar.material = finMat; bar.parent = finRoot; bar.position.x = -1.8 + v * 0.72;
            meshes.push(bar);
        }
    }

    // Hot-staging ring with vent slots
    const hs = cyl(scene, 'hs', 9.2, 9.2, 4, 48); hs.material = dark; hs.parent = booster; hs.position.y = 37.5; meshes.push(hs);

    // ── SHIP (52m) ──
    const shipNode = new TransformNode('shipNode', scene);
    shipNode.parent = parent;
    const bY = 39.5;
    const body = cyl(scene, 'sb', 9, 9, 36, 48); body.material = steel; body.parent = shipNode; body.position.y = bY + 18; meshes.push(body);
    // Ogive nose
    const nl = cyl(scene, 'nl', 6, 9, 8, 48); nl.material = steel; nl.parent = shipNode; nl.position.y = bY + 40; meshes.push(nl);
    const nu = cyl(scene, 'nu', 0, 6, 8, 48); nu.material = steel; nu.parent = shipNode; nu.position.y = bY + 48; meshes.push(nu);
    // Heat shield (half)
    const shield = MeshBuilder.CreateCylinder('sh', { diameterTop: 9.15, diameterBottom: 9.15, height: 36, tessellation: 48, arc: 0.5 }, scene);
    shield.material = black; shield.parent = shipNode; shield.position.y = bY + 18; meshes.push(shield);

    // ── Flaps — trapezoidal fins (wide base, narrow tip) ──
    const flapMat = mat(scene, 'flapM', 0.08, 0.08, 0.08, 0);
    // Forward flaps (2) — larger, near nose, rounded edges simulated with tapered box
    for (let i = 0; i < 2; i++) {
        const flapRoot = new TransformNode('fwdFlapR' + i, scene);
        flapRoot.parent = shipNode;
        flapRoot.position.y = bY + 35; flapRoot.position.x = i === 0 ? 4.8 : -4.8;
        flapRoot.rotation.z = i === 0 ? -0.12 : 0.12;
        // Main surface (tapers: wider at hinge, narrower at tip)
        const main = MeshBuilder.CreateCylinder('fwdF' + i, {
            diameterTop: 3.5, diameterBottom: 5.5, height: 8, tessellation: 4
        }, scene);
        main.material = flapMat; main.parent = flapRoot;
        main.rotation.y = Math.PI / 4; // Rotate so square faces become "flat"
        main.scaling.z = 0.04; // Flatten to fin thickness
        meshes.push(main);
        // Hinge cylinder
        const hinge = cyl(scene, 'fwdH' + i, 0.5, 0.5, 5.5, 10);
        hinge.material = dark; hinge.parent = flapRoot; hinge.position.y = 4; hinge.rotation.z = Math.PI / 2;
        meshes.push(hinge);
    }
    // Aft flaps (2) — smaller, near base
    for (let i = 0; i < 2; i++) {
        const flapRoot = new TransformNode('aftFlapR' + i, scene);
        flapRoot.parent = shipNode;
        flapRoot.position.y = bY + 3; flapRoot.position.z = i === 0 ? 4.8 : -4.8;
        flapRoot.rotation.x = i === 0 ? -0.1 : 0.1;
        const main = MeshBuilder.CreateCylinder('aftF' + i, {
            diameterTop: 2.8, diameterBottom: 4.5, height: 6, tessellation: 4
        }, scene);
        main.material = flapMat; main.parent = flapRoot;
        main.rotation.y = Math.PI / 4; main.scaling.z = 0.04;
        meshes.push(main);
        const hinge = cyl(scene, 'aftH' + i, 0.4, 0.4, 4.5, 10);
        hinge.material = dark; hinge.parent = flapRoot; hinge.position.y = 3; hinge.rotation.x = Math.PI / 2;
        meshes.push(hinge);
    }

    // Ship engines: 3 sea-level + 3 vacuum
    for (let i = 0; i < 3; i++) {
        const a = (i * Math.PI * 2) / 3 + Math.PI / 6;
        const e = cyl(scene, 'se' + i, 0.78, 1.3, 2.5, 12); e.material = eng; e.parent = shipNode;
        e.position.set(Math.cos(a) * 1.5, bY - 2, Math.sin(a) * 1.5); meshes.push(e);
    }
    for (let i = 0; i < 3; i++) {
        const a = (i * Math.PI * 2) / 3;
        const e = cyl(scene, 've' + i, 1.0, 2.4, 4, 16); e.material = eng; e.parent = shipNode;
        e.position.set(Math.cos(a) * 3.2, bY - 3, Math.sin(a) * 3.2);
        const ext = cyl(scene, 'vx' + i, 2.4, 2.8, 2.5, 16);
        ext.material = mat(scene, 'vxm' + i, 0.25, 0.2, 0.15, 0.1);
        ext.parent = e; ext.position.y = -3.25; meshes.push(e);
    }

    // Logo band
    const lb = cyl(scene, 'lb', 9.2, 9.2, 0.8, 48);
    const lm = new StandardMaterial('lm', scene); lm.diffuseColor = Color3.White(); lm.emissiveColor = new Color3(0.15, 0.15, 0.15);
    lb.material = lm; lb.parent = shipNode; lb.position.y = bY + 28; meshes.push(lb);

    return { meshes, booster, shipNode };
}

// ═══ LAUNCH SITE (returns chopstick refs for animation) ═══
function buildLaunchSite(scene: Scene) {
    const tM = mat(scene, 'tower', 0.45, 0.4, 0.35, 0.15);
    const cM = mat(scene, 'conc', 0.35, 0.33, 0.3, 0);
    const sM = mat(scene, 'sPlate', 0.4, 0.38, 0.35, 0.2);
    // OLM
    MeshBuilder.CreateBox('olmB', { width: 25, height: 4, depth: 25 }, scene).material = cM;
    scene.getMeshByName('olmB')!.position.y = 2;
    for (let i = 0; i < 4; i++) {
        const c = MeshBuilder.CreateBox('oc' + i, { width: 4, height: 28, depth: 4 }, scene);
        c.material = sM; c.position.set((i % 2 === 0 ? 1 : -1) * 8, 14, (i < 2 ? 1 : -1) * 8);
    }
    const top = MeshBuilder.CreateBox('ot', { width: 22, height: 3, depth: 22 }, scene);
    top.material = sM; top.position.y = 29;
    // Flame deflector
    const defl = MeshBuilder.CreateBox('defl', { width: 20, height: 1.5, depth: 20 }, scene);
    defl.material = mat(scene, 'dfl', 0.25, 0.22, 0.2, 0); defl.position.y = 27;

    // Tower
    const tx = -18;
    for (const [dx, dz] of [[-2.5, -2.5], [2.5, -2.5], [-2.5, 2.5], [2.5, 2.5]]) {
        const c = MeshBuilder.CreateBox('tc', { width: 1.5, height: 140, depth: 1.5 }, scene);
        c.material = tM; c.position.set(tx + dx, 70, dz);
    }
    for (let l = 0; l < 9; l++) {
        const y = 10 + l * 15;
        MeshBuilder.CreateBox('hb' + l, { width: 5, height: 0.5, depth: 0.5 }, scene).material = tM;
        scene.getMeshByName('hb' + l)!.position.set(tx, y, 0);
        MeshBuilder.CreateBox('hb2' + l, { width: 0.5, height: 0.5, depth: 5 }, scene).material = tM;
        scene.getMeshByName('hb2' + l)!.position.set(tx, y, 0);
    }
    // Chopstick arms — separate so we can animate closing
    const chM = mat(scene, 'chop', 0.5, 0.45, 0.4, 0.2);
    const chopLeft = MeshBuilder.CreateBox('chopL', { width: 36, height: 2, depth: 1.8 }, scene);
    chopLeft.material = chM; chopLeft.position.set(tx + 20.5, 85, -5); // Start open wide
    const chopRight = MeshBuilder.CreateBox('chopR', { width: 36, height: 2, depth: 1.8 }, scene);
    chopRight.material = chM; chopRight.position.set(tx + 20.5, 85, 5);
    // QD arm
    MeshBuilder.CreateBox('qd', { width: 22, height: 1.5, depth: 2 }, scene).material = chM;
    scene.getMeshByName('qd')!.position.set(tx + 13, 105, 0);
    // Lightning rod
    const rod = cyl(scene, 'rod', 0.08, 0.3, 12, 8); rod.material = tM; rod.position.set(tx, 146, 0);
    // Pad
    MeshBuilder.CreateBox('pad', { width: 60, height: 0.5, depth: 60 }, scene).material = cM;
    scene.getMeshByName('pad')!.position.y = 0.25;
    // Tanks
    const wM = mat(scene, 'wh', 0.9, 0.9, 0.9, 0.3);
    for (let i = 0; i < 4; i++) {
        cyl(scene, 'ft' + i, 5, 5, 20, 16).material = wM;
        scene.getMeshByName('ft' + i)!.position.set(-40 + i * 8, 10, 35);
    }
    return { chopLeft, chopRight };
}

// ═══ EARTH ═══
function buildEarth(scene: Scene, parent: TransformNode) {
    const earthSphere = MeshBuilder.CreateSphere('earth', { diameter: EARTH_R * 2, segments: 64 }, scene);
    const eM = new StandardMaterial('eM', scene); eM.diffuseColor = new Color3(0.15, 0.35, 0.65); eM.specularColor = new Color3(0.2, 0.2, 0.3);
    earthSphere.material = eM; earthSphere.parent = parent;
    const landMat = new StandardMaterial('lnd', scene); landMat.diffuseColor = new Color3(0.25, 0.45, 0.2); landMat.specularColor = Color3.Black();
    for (let i = 0; i < 8; i++) {
        const l = MeshBuilder.CreateDisc('land' + i, { radius: 40 + Math.random() * 80, tessellation: 20 }, scene);
        l.material = landMat; l.parent = parent;
        const th = Math.random() * Math.PI * 2, ph = Math.random() * Math.PI * 0.8 + 0.1, r = EARTH_R + 0.5;
        l.position.set(r * Math.sin(ph) * Math.cos(th), r * Math.cos(ph), r * Math.sin(ph) * Math.sin(th));
        l.lookAt(Vector3.Zero());
    }
    const atmosphere = MeshBuilder.CreateSphere('atmo', { diameter: EARTH_R * 2 + 30, segments: 48 }, scene);
    const aM = new StandardMaterial('aM', scene); aM.diffuseColor = new Color3(0.3, 0.5, 0.9); aM.emissiveColor = new Color3(0.1, 0.2, 0.5);
    aM.alpha = 0.15; aM.backFaceCulling = false; atmosphere.material = aM; atmosphere.parent = parent;
    const clouds = MeshBuilder.CreateSphere('clouds', { diameter: EARTH_R * 2 + 10, segments: 48 }, scene);
    const cM = new StandardMaterial('cM', scene); cM.diffuseColor = Color3.White(); cM.emissiveColor = new Color3(0.3, 0.3, 0.3);
    cM.alpha = 0.2; cM.backFaceCulling = false; clouds.material = cM; clouds.parent = parent;
    return { earthSphere, atmosphere, clouds };
}

// ═══ MOON SPHERE ═══
function buildMoonSphere(scene: Scene, parent: TransformNode): Mesh {
    const moon = MeshBuilder.CreateSphere('moon', { diameter: MOON_R * 2, segments: 48 }, scene);
    const mM = new StandardMaterial('mM', scene); mM.diffuseColor = new Color3(0.55, 0.53, 0.48); mM.specularColor = new Color3(0.05, 0.05, 0.05);
    moon.material = mM; moon.parent = parent;
    for (let i = 0; i < 6; i++) {
        const mare = MeshBuilder.CreateDisc('mare' + i, { radius: 15 + Math.random() * 25, tessellation: 20 }, scene);
        mare.material = mat(scene, 'mm' + i, 0.35, 0.33, 0.3, 0); mare.parent = parent;
        const th = Math.random() * Math.PI * 2, ph = Math.random() * Math.PI, r = MOON_R + 0.3;
        mare.position.set(r * Math.sin(ph) * Math.cos(th), r * Math.cos(ph), r * Math.sin(ph) * Math.sin(th));
        mare.lookAt(Vector3.Zero());
    }
    return moon;
}

// ═══ MOON SURFACE ═══
function buildMoonSurface(scene: Scene): TransformNode {
    const root = new TransformNode('moonSurf', scene);
    const g = MeshBuilder.CreateGround('mg', { width: 800, height: 800, subdivisions: 80 }, scene);
    g.material = mat(scene, 'mgs', 0.5, 0.48, 0.44, 0.05); g.parent = root;
    for (let i = 0; i < 30; i++) {
        const r = 4 + Math.random() * 15;
        const c = MeshBuilder.CreateDisc('cr' + i, { radius: r, tessellation: 24 }, scene);
        c.material = mat(scene, 'crm' + i, 0.38, 0.36, 0.32, 0); c.parent = root; c.rotation.x = Math.PI / 2;
        c.position.set((Math.random() - 0.5) * 500, 0.05, (Math.random() - 0.5) * 500);
        const rim = MeshBuilder.CreateTorus('rim' + i, { diameter: r * 2, thickness: r * 0.12, tessellation: 20 }, scene);
        rim.material = mat(scene, 'rmm' + i, 0.52, 0.5, 0.46, 0); rim.parent = root;
        rim.position.copyFrom(c.position); rim.position.y = 0.2; rim.rotation.x = Math.PI / 2;
    }
    return root;
}

// ═══ MOON BASE ═══
function buildMoonBase(scene: Scene): TransformNode {
    const root = new TransformNode('moonBase', scene);
    const hM = mat(scene, 'hab', 0.85, 0.85, 0.82, 0.3);
    const mM = mat(scene, 'met', 0.5, 0.5, 0.5, 0.2);
    const pM = new StandardMaterial('pan', scene); pM.diffuseColor = new Color3(0.1, 0.1, 0.35); pM.emissiveColor = new Color3(0.02, 0.02, 0.08);
    const bx = 50, bz = 30;
    for (let i = 0; i < 3; i++) {
        const d = MeshBuilder.CreateSphere('dome' + i, { diameter: 12, slice: 0.5, segments: 20 }, scene);
        d.material = hM; d.parent = root; d.position.set(bx + i * 18, 0, bz);
    }
    for (let i = 0; i < 2; i++) {
        const t = cyl(scene, 'tun' + i, 3, 3, 16, 12); t.material = mM; t.parent = root;
        t.position.set(bx + 9 + i * 18, 1.5, bz); t.rotation.z = Math.PI / 2;
    }
    for (let i = 0; i < 4; i++) {
        cyl(scene, 'pole' + i, 0.3, 0.3, 6, 8).material = mM;
        scene.getMeshByName('pole' + i)!.parent = root;
        scene.getMeshByName('pole' + i)!.position.set(bx + i * 12, 3, bz + 20);
        const sp = MeshBuilder.CreateBox('sp' + i, { width: 8, height: 0.1, depth: 4 }, scene);
        sp.material = pM; sp.parent = root; sp.position.set(bx + i * 12, 6.5, bz + 20); sp.rotation.x = -0.3;
    }
    for (let i = 0; i < 8; i++) {
        const a = (i * Math.PI * 2) / 8;
        const m = MeshBuilder.CreateBox('lm' + i, { width: 3, height: 0.05, depth: 0.5 }, scene);
        const lmM = new StandardMaterial('lmm' + i, scene); lmM.diffuseColor = new Color3(0.8, 0.4, 0.1); lmM.emissiveColor = new Color3(0.3, 0.15, 0.05);
        m.material = lmM; m.parent = root; m.position.set(Math.cos(a) * 15, 0.05, Math.sin(a) * 15); m.rotation.y = a;
    }
    return root;
}

// ═══ ASTRONAUT ═══
function buildAstronaut(scene: Scene) {
    const root = new TransformNode('astronaut', scene);
    const sM = mat(scene, 'suit', 0.9, 0.9, 0.88, 0.2);
    const vM = new StandardMaterial('visor', scene); vM.diffuseColor = new Color3(0.1, 0.15, 0.3); vM.emissiveColor = new Color3(0.05, 0.08, 0.15);
    MeshBuilder.CreateCapsule('torso', { height: 1.6, radius: 0.4 }, scene).material = sM;
    scene.getMeshByName('torso')!.parent = root;
    const h = MeshBuilder.CreateSphere('helm', { diameter: 0.7, segments: 12 }, scene); h.material = sM; h.parent = root; h.position.y = 1.2;
    const v = MeshBuilder.CreateSphere('vis', { diameter: 0.5, segments: 12 }, scene); v.material = vM; v.parent = root; v.position.set(0.15, 1.25, 0);
    MeshBuilder.CreateBox('plss', { width: 0.5, height: 0.7, depth: 0.3 }, scene).material = sM;
    scene.getMeshByName('plss')!.parent = root; scene.getMeshByName('plss')!.position.set(-0.3, 0.5, 0);
    for (let i = 0; i < 2; i++) {
        const l = cyl(scene, 'leg' + i, 0.25, 0.25, 0.9, 8); l.material = sM; l.parent = root; l.position.set(0, -0.7, (i - 0.5) * 0.3);
    }
    return { root };
}

// ═══ ROVER ═══
function buildRover(scene: Scene) {
    const root = new TransformNode('rover', scene); root.position.set(70, 0, 30);
    const bM = mat(scene, 'rB', 0.7, 0.7, 0.68, 0.2);
    const wM = mat(scene, 'rW', 0.25, 0.25, 0.25, 0.1);
    const body = MeshBuilder.CreateBox('rCh', { width: 4, height: 1, depth: 2.5 }, scene); body.material = bM; body.parent = root;
    for (const [x, z] of [[-1.8, -1.2], [-1.8, 1.2], [1.8, -1.2], [1.8, 1.2]]) {
        const w = cyl(scene, 'rw', 0.8, 0.8, 0.3, 16); w.material = wM; w.parent = root;
        w.position.set(x, -0.3, z); w.rotation.x = Math.PI / 2;
    }
    cyl(scene, 'rant', 0.02, 0.08, 2, 6).material = bM;
    scene.getMeshByName('rant')!.parent = root; scene.getMeshByName('rant')!.position.set(-1.5, 1.5, 0);
    const sp = MeshBuilder.CreateBox('rsp', { width: 3, height: 0.05, depth: 1.5 }, scene);
    const spM = new StandardMaterial('rspM', scene); spM.diffuseColor = new Color3(0.1, 0.1, 0.35); spM.emissiveColor = new Color3(0.02, 0.02, 0.06);
    sp.material = spM; sp.parent = root; sp.position.y = 1.2;
    return { root, body };
}

// ═══ PARTICLES ═══
function createExhaustPS(scene: Scene, emitter: Mesh, count: number): ParticleSystem {
    const ps = new ParticleSystem('ex', count, scene);
    ps.createPointEmitter(new Vector3(-2, -1, -2), new Vector3(2, -1, 2));
    ps.color1 = new Color4(1, 0.7, 0.15, 1); ps.color2 = new Color4(1, 0.4, 0.08, 0.9);
    ps.colorDead = new Color4(0.4, 0.3, 0.2, 0);
    ps.minSize = 2; ps.maxSize = 6; ps.minLifeTime = 0.3; ps.maxLifeTime = 1.2;
    ps.emitRate = 0; ps.blendMode = ParticleSystem.BLENDMODE_ADD;
    ps.minEmitPower = 25; ps.maxEmitPower = 50; ps.updateSpeed = 0.02;
    ps.gravity = new Vector3(0, -8, 0); ps.emitter = emitter; ps.start();
    return ps;
}
function createCorePS(scene: Scene, emitter: Mesh): ParticleSystem {
    const ps = new ParticleSystem('core', 2000, scene);
    ps.createPointEmitter(new Vector3(-0.5, -1, -0.5), new Vector3(0.5, -1, 0.5));
    ps.color1 = new Color4(0.8, 0.85, 1, 1); ps.color2 = new Color4(1, 1, 0.9, 0.95);
    ps.colorDead = new Color4(1, 0.6, 0.1, 0);
    ps.minSize = 0.5; ps.maxSize = 2; ps.minLifeTime = 0.1; ps.maxLifeTime = 0.4;
    ps.emitRate = 0; ps.blendMode = ParticleSystem.BLENDMODE_ADD;
    ps.minEmitPower = 30; ps.maxEmitPower = 60; ps.updateSpeed = 0.01;
    ps.gravity = new Vector3(0, -5, 0); ps.emitter = emitter; ps.start();
    return ps;
}

function createStarfield(scene: Scene) {
    const m = new StandardMaterial('sM', scene); m.emissiveColor = Color3.White(); m.disableLighting = true;
    for (let i = 0; i < 600; i++) {
        const s = MeshBuilder.CreateSphere('s' + i, { diameter: 0.3 + Math.random() * 0.6 }, scene);
        s.material = m; const r = 800 + Math.random() * 700;
        const th = Math.random() * Math.PI * 2, ph = Math.random() * Math.PI;
        s.position.set(r * Math.sin(ph) * Math.cos(th), r * Math.cos(ph), r * Math.sin(ph) * Math.sin(th));
    }
}

// ═══ HELPERS ═══
function mat(scene: Scene, name: string, r: number, g: number, b: number, spec: number): StandardMaterial {
    const m = new StandardMaterial(name, scene); m.diffuseColor = new Color3(r, g, b); m.specularColor = new Color3(spec, spec, spec); return m;
}
function cyl(scene: Scene, name: string, dTop: number, dBot: number, h: number, tess: number): Mesh {
    return MeshBuilder.CreateCylinder(name, { diameterTop: dTop, diameterBottom: dBot, height: h, tessellation: tess }, scene);
}
