import {
    Engine, Scene, ArcRotateCamera, Vector3, HemisphericLight, DirectionalLight,
    MeshBuilder, StandardMaterial, Color3, Color4, ParticleSystem,
    GlowLayer, ShadowGenerator, Mesh, TransformNode, KeyboardEventTypes
} from '@babylonjs/core';

export interface SimState {
    phase: string; altitude: number; velocity: number; downrange: number;
    fuel: number; thrust: number; missionTime: number; throttle: number;
}

// Scale: Earth radius=500, Moon radius=136, Moon distance=6000
const EARTH_R = 500, MOON_R = 136, MOON_DIST = 6000;

let engine: Engine | null = null, scene: Scene | null = null;

export function destroyScene() {
    scene?.dispose(); engine?.dispose(); engine = null; scene = null;
}

export function initStarshipScene(canvas: HTMLCanvasElement, onTelemetry: (s: SimState) => void): () => void {
    engine = new Engine(canvas, true, { preserveDrawingBuffer: true, stencil: true });
    scene = new Scene(engine);
    scene.clearColor = new Color4(0.25, 0.45, 0.75, 1); // Start with blue sky

    const cam = new ArcRotateCamera('cam', -Math.PI / 2, Math.PI / 3, 180, new Vector3(0, 60, 0), scene);
    cam.lowerRadiusLimit = 15; cam.upperRadiusLimit = 2000;
    cam.attachControl(canvas, true); cam.wheelPrecision = 5;

    const hemi = new HemisphericLight('hemi', new Vector3(0, 1, 0), scene);
    hemi.intensity = 0.35;
    const sun = new DirectionalLight('sun', new Vector3(-1, -2, -1), scene);
    sun.intensity = 1.4; sun.position = new Vector3(200, 400, 200);

    const glow = new GlowLayer('glow', scene); glow.intensity = 0.8;

    // ── Build everything ──
    const shipRoot = new TransformNode('shipRoot', scene);
    const ship = buildStarshipStack(scene, shipRoot);
    shipRoot.position.y = 30 + 35.5;

    buildLaunchSite(scene);

    const ground = MeshBuilder.CreateGround('ground', { width: 4000, height: 4000, subdivisions: 4 }, scene);
    const groundMat = new StandardMaterial('gm', scene);
    groundMat.diffuseColor = new Color3(0.15, 0.13, 0.1); groundMat.specularColor = Color3.Black();
    ground.material = groundMat; ground.receiveShadows = true;

    const shadowGen = new ShadowGenerator(2048, sun);
    shadowGen.useBlurExponentialShadowMap = true;
    ship.meshes.forEach(m => shadowGen.addShadowCaster(m));

    // Engine fire
    const exhaustEmitter = MeshBuilder.CreateSphere('exE', { diameter: 4 }, scene);
    exhaustEmitter.parent = shipRoot; exhaustEmitter.position.y = -37; exhaustEmitter.isVisible = false;
    const exhaust = createExhaustParticles(scene, exhaustEmitter);

    // Mach diamond core
    const coreEmitter = MeshBuilder.CreateSphere('coreE', { diameter: 1 }, scene);
    coreEmitter.parent = shipRoot; coreEmitter.position.y = -37; coreEmitter.isVisible = false;
    const exhaustCore = createExhaustCore(scene, coreEmitter);

    createStarfield(scene);

    // ── Earth sphere (below ground) ──
    const earthRoot = new TransformNode('earthRoot', scene);
    earthRoot.position.y = -EARTH_R;
    const { earthSphere, atmosphere, clouds } = buildEarth(scene, earthRoot);
    earthSphere.isVisible = false; atmosphere.isVisible = false; clouds.isVisible = false;

    // ── Moon sphere (far away) ──
    const moonRoot = new TransformNode('moonRoot', scene);
    moonRoot.position.set(0, MOON_DIST * 0.3, MOON_DIST);
    const moonSphere = buildMoonSphere(scene, moonRoot);

    // ── Moon surface detail (for landing, hidden initially) ──
    const moonSurface = buildMoonSurface(scene);
    moonSurface.setEnabled(false);

    // ── Moon base ──
    const moonBase = buildMoonBase(scene);
    moonBase.setEnabled(false);

    // ── Astronaut + Rover ──
    const astronaut = buildAstronaut(scene);
    astronaut.root.setEnabled(false);
    const rover = buildRover(scene);
    rover.root.setEnabled(false);

    // ── Rover keyboard controls ──
    const keys: Record<string, boolean> = {};
    scene.onKeyboardObservable.add((info) => {
        const key = info.event.key.toLowerCase();
        keys[key] = info.type === KeyboardEventTypes.KEYDOWN;
    });

    // ── Sim state ──
    const state: SimState = {
        phase: 'prelaunch', altitude: 0, velocity: 0, downrange: 0,
        fuel: 100, thrust: 0, missionTime: -10, throttle: 0,
    };

    let lastTime = performance.now(), launched = false, phaseTimer = 0;

    const onLaunch = () => { launched = true; };
    window.addEventListener('starship-launch', onLaunch);

    engine.runRenderLoop(() => {
        const now = performance.now();
        const dt = Math.min((now - lastTime) / 1000, 0.1);
        lastTime = now;

        if (launched) {
            state.missionTime += dt; phaseTimer += dt;
            updateSim(state, dt, shipRoot, exhaust, exhaustCore, cam, ground,
                earthSphere, atmosphere, clouds, earthRoot,
                moonRoot, moonSphere, moonSurface, moonBase,
                astronaut, rover, ship, keys);
        } else {
            state.missionTime = -10 + (performance.now() % 10000) / 1000;
        }

        // Camera shake
        const shake = (state.phase === 'liftoff' || state.phase === 'ignition') ? 0.3
            : state.phase === 'landing-burn' ? 0.12 : 0;
        if (shake > 0) {
            cam.target.x += (Math.random() - 0.5) * shake;
            cam.target.y += (Math.random() - 0.5) * shake * 0.5;
        }

        // Sky color transition (blue → black)
        const skyT = Math.min(1, state.altitude / 120);
        scene!.clearColor = new Color4(
            0.25 * (1 - skyT), 0.45 * (1 - skyT), 0.75 * (1 - skyT * 0.7), 1
        );

        onTelemetry(state);
        scene!.render();
    });

    return () => { window.removeEventListener('starship-launch', onLaunch); };
}

// ═══ SIMULATION UPDATE ═══
function updateSim(
    s: SimState, dt: number, shipRoot: TransformNode,
    exhaust: ParticleSystem, exhaustCore: ParticleSystem,
    cam: ArcRotateCamera, ground: Mesh,
    earthSphere: Mesh, atmosphere: Mesh, clouds: Mesh, earthRoot: TransformNode,
    moonRoot: TransformNode, moonSphere: Mesh, moonSurface: TransformNode,
    moonBase: TransformNode,
    astro: { root: TransformNode }, rover: { root: TransformNode, body: Mesh },
    ship: { meshes: Mesh[], booster: TransformNode },
    keys: Record<string, boolean>
) {
    const t = s.missionTime;

    // Phase transitions
    if (t < 0) { s.phase = 'prelaunch'; s.throttle = 0; }
    else if (t < 3) { s.phase = 'ignition'; s.throttle = Math.min(t / 3 * 100, 100); }
    else if (t < 15) { s.phase = 'liftoff'; s.throttle = 100; }
    else if (t < 25) { s.phase = 'maxq'; s.throttle = 80; }
    else if (t < 40) { s.phase = 'meco'; s.throttle = Math.max(0, 80 - (t - 25) / 15 * 80); }
    else if (t < 48) { s.phase = 'separation'; s.throttle = 0; }
    else if (t < 55) { s.phase = 'ses'; s.throttle = 60; }
    else if (t < 120) { s.phase = 'coast'; s.throttle = 5; }
    else if (t < 160) { s.phase = 'lunar-approach'; s.throttle = 10; }
    else if (t < 180) { s.phase = 'landing-burn'; s.throttle = Math.min((t - 160) / 5 * 90, 90); }
    else if (t < 190) { s.phase = 'touchdown'; s.throttle = Math.max(0, (190 - t) / 10 * 20); }
    else if (t < 200) { s.phase = 'landed'; s.throttle = 0; }
    else if (t < 220) { s.phase = 'eva'; s.throttle = 0; }
    else if (t < 260) { s.phase = 'exploration'; s.throttle = 0; }
    else { s.phase = 'complete'; s.throttle = 0; }

    // Physics
    const accel = s.throttle / 100 * 3.5;
    const gravity = s.altitude < 100 ? 0.0098 : 0.001;
    if (t >= 0 && !['touchdown', 'landed', 'eva', 'exploration', 'complete'].includes(s.phase)) {
        s.velocity += (accel - gravity) * dt;
        s.altitude += s.velocity * dt;
        s.downrange += s.velocity * dt * 0.7;
        s.fuel = Math.max(0, s.fuel - s.throttle * dt * 0.03);
        s.thrust = s.throttle;
    }
    if (['touchdown', 'landed', 'eva', 'exploration', 'complete'].includes(s.phase)) {
        s.velocity = Math.max(0, s.velocity - dt * 2); s.thrust = 0;
    }
    s.altitude = Math.max(0, s.altitude); s.velocity = Math.max(0, s.velocity);

    // ── Visual: Ship position ──
    const sceneY = Math.min(s.altitude * 0.8, 400);
    shipRoot.position.y = 30 + 35.5 + sceneY;

    // Tilt
    if (s.phase === 'liftoff' || s.phase === 'maxq') {
        shipRoot.rotation.z = Math.min((t - 3) * 0.002, 0.12);
    } else if (s.phase === 'landing-burn') {
        shipRoot.rotation.z = Math.max(shipRoot.rotation.z - dt * 0.05, 0);
    }

    // ── Exhaust fire ──
    if (s.throttle > 5) {
        exhaust.start(); exhaustCore.start();
        exhaust.emitRate = s.throttle * 15;
        exhaust.minSize = 2 + s.throttle * 0.05;
        exhaust.maxSize = 6 + s.throttle * 0.08;
        exhaustCore.emitRate = s.throttle * 5;
        exhaustCore.minSize = 0.5 + s.throttle * 0.02;
        exhaustCore.maxSize = 2 + s.throttle * 0.03;
    } else { exhaust.stop(); exhaustCore.stop(); }

    // ── Camera follow ──
    if (s.phase !== 'prelaunch') {
        cam.target.y += (shipRoot.position.y - cam.target.y) * dt * 2;
        if (s.altitude > 50) cam.radius += (200 + s.altitude * 0.4 - cam.radius) * dt;
    }

    // ── Stage separation ──
    if (s.phase === 'separation') {
        ship.booster.position.y -= dt * 8;
        ship.booster.rotation.z += dt * 0.05;
    }

    // ── Earth visibility (show sphere as altitude increases) ──
    if (s.altitude > 30) {
        earthSphere.isVisible = true; atmosphere.isVisible = true; clouds.isVisible = true;
    }
    // Rotate clouds slowly
    clouds.rotation.y += dt * 0.01;

    // ── Ground fade ──
    if (s.altitude > 60) {
        ground.visibility = Math.max(0, 1 - (s.altitude - 60) / 60);
    }

    // ── Moon approach (move moon closer during transit) ──
    if (s.phase === 'coast' || s.phase === 'lunar-approach' || s.phase === 'landing-burn') {
        const approachT = Math.min(1, (t - 55) / 125);
        moonRoot.position.z = MOON_DIST * (1 - approachT * 0.95);
        moonRoot.position.y = MOON_DIST * 0.3 * (1 - approachT);
        // Scale moon sphere up as it "approaches"
        const moonScale = 1 + approachT * 3;
        moonSphere.scaling.set(moonScale, moonScale, moonScale);
        // Move Earth away
        earthRoot.position.y = -EARTH_R - approachT * 2000;
    }

    // ── Moon surface for landing ──
    if (['lunar-approach', 'landing-burn', 'touchdown', 'landed', 'eva', 'exploration', 'complete'].includes(s.phase)) {
        moonSurface.setEnabled(true);
        moonBase.setEnabled(true);
        if (s.phase === 'landing-burn' || s.phase === 'touchdown') {
            moonSurface.position.y = shipRoot.position.y - 70;
            moonBase.position.y = moonSurface.position.y;
        }
        ground.visibility = 0;
    }

    // ── Post-landing: Astronaut EVA ──
    if (s.phase === 'eva' || s.phase === 'exploration' || s.phase === 'complete') {
        astro.root.setEnabled(true);
        astro.root.position.y = moonSurface.position.y + 1.2;
        astro.root.position.x = shipRoot.position.x + 15;
        astro.root.position.z = shipRoot.position.z;

        // Walk astronaut toward rover
        if (s.phase === 'eva') {
            const evaProgress = (t - 200) / 20;
            astro.root.position.x = shipRoot.position.x + 15 + evaProgress * 20;
            // Bobbing walk animation
            astro.root.position.y = moonSurface.position.y + 1.2 + Math.abs(Math.sin(t * 3)) * 0.4;
        }
    }

    // ── Rover exploration ──
    if (s.phase === 'exploration' || s.phase === 'complete') {
        rover.root.setEnabled(true);
        rover.root.position.y = moonSurface.position.y + 0.8;

        if (s.phase === 'exploration') {
            // Keyboard controls for rover
            const speed = 15;
            if (keys['w'] || keys['arrowup']) rover.root.position.z += speed * dt;
            if (keys['s'] || keys['arrowdown']) rover.root.position.z -= speed * dt;
            if (keys['a'] || keys['arrowleft']) {
                rover.root.rotation.y += 1.5 * dt;
                rover.root.position.x -= speed * dt * 0.5;
            }
            if (keys['d'] || keys['arrowright']) {
                rover.root.rotation.y -= 1.5 * dt;
                rover.root.position.x += speed * dt * 0.5;
            }
            // Camera follows rover
            cam.target.x += (rover.root.position.x - cam.target.x) * dt * 3;
            cam.target.z += (rover.root.position.z - cam.target.z) * dt * 3;
            cam.radius += (40 - cam.radius) * dt * 2;
            // Astronaut rides on rover
            astro.root.position.x = rover.root.position.x;
            astro.root.position.z = rover.root.position.z;
            astro.root.position.y = rover.root.position.y + 2;
        }
    }
}

// ═══ STARSHIP STACK ═══
function buildStarshipStack(scene: Scene, parent: TransformNode): { meshes: Mesh[], booster: TransformNode } {
    const meshes: Mesh[] = [];
    const steelMat = makeMat(scene, 'steel', 0.78, 0.78, 0.76, 0.5);
    const darkMat = makeMat(scene, 'dark', 0.3, 0.3, 0.3, 0.2);
    const engMat = makeMat(scene, 'eng', 0.15, 0.15, 0.15, 0.1);
    const hsMat = makeMat(scene, 'hs', 0.05, 0.05, 0.05, 0);
    const glowMat = new StandardMaterial('glow', scene);
    glowMat.diffuseColor = new Color3(0.1, 0.1, 0.1);
    glowMat.emissiveColor = new Color3(0.15, 0.05, 0);

    // Booster (71m)
    const boosterRoot = new TransformNode('boosterRoot', scene);
    boosterRoot.parent = parent;
    const bb = MeshBuilder.CreateCylinder('bb', { diameterTop: 9, diameterBottom: 9, height: 71, tessellation: 48 }, scene);
    bb.material = steelMat; bb.parent = boosterRoot; meshes.push(bb);
    // Skirt
    const sk = MeshBuilder.CreateCylinder('sk', { diameterTop: 9, diameterBottom: 9.4, height: 5, tessellation: 48 }, scene);
    sk.material = darkMat; sk.parent = boosterRoot; sk.position.y = -33; meshes.push(sk);
    // 33 engines: 3+10+20
    [{ count: 3, r: 0.8 }, { count: 10, r: 2.3 }, { count: 20, r: 3.8 }].forEach(ring => {
        for (let i = 0; i < ring.count; i++) {
            const a = (i * Math.PI * 2) / ring.count;
            const e = MeshBuilder.CreateCylinder('e', { diameterTop: 0.78, diameterBottom: 1.3, height: 3, tessellation: 12 }, scene);
            e.material = engMat; e.parent = boosterRoot;
            e.position.set(Math.cos(a) * ring.r, -35.5, Math.sin(a) * ring.r);
            meshes.push(e);
        }
    });
    // Grid fins
    for (let i = 0; i < 4; i++) {
        const a = (i * Math.PI) / 2 + Math.PI / 4;
        const f = MeshBuilder.CreateBox('gf' + i, { width: 4.5, height: 3.5, depth: 0.25 }, scene);
        f.material = darkMat; f.parent = boosterRoot; f.position.y = 33;
        f.position.x = Math.cos(a) * 5.5; f.position.z = Math.sin(a) * 5.5; f.rotation.y = a;
        meshes.push(f);
    }
    // Hot-staging ring
    const hs = MeshBuilder.CreateCylinder('hs', { diameterTop: 9.2, diameterBottom: 9.2, height: 4, tessellation: 48 }, scene);
    hs.material = darkMat; hs.parent = boosterRoot; hs.position.y = 37.5; meshes.push(hs);

    // Ship upper stage (52m)
    const shipNode = new TransformNode('shipNode', scene);
    shipNode.parent = parent;
    const baseY = 39.5;
    const body = MeshBuilder.CreateCylinder('sb', { diameterTop: 9, diameterBottom: 9, height: 36, tessellation: 48 }, scene);
    body.material = steelMat; body.parent = shipNode; body.position.y = baseY + 18; meshes.push(body);
    // Nose
    const nl = MeshBuilder.CreateCylinder('nl', { diameterTop: 6, diameterBottom: 9, height: 8, tessellation: 48 }, scene);
    nl.material = steelMat; nl.parent = shipNode; nl.position.y = baseY + 40; meshes.push(nl);
    const nu = MeshBuilder.CreateCylinder('nu', { diameterTop: 0, diameterBottom: 6, height: 8, tessellation: 48 }, scene);
    nu.material = steelMat; nu.parent = shipNode; nu.position.y = baseY + 48; meshes.push(nu);
    // Heat shield
    const shield = MeshBuilder.CreateCylinder('sh', { diameterTop: 9.15, diameterBottom: 9.15, height: 36, tessellation: 48, arc: 0.5 }, scene);
    shield.material = hsMat; shield.parent = shipNode; shield.position.y = baseY + 18; meshes.push(shield);
    // Flaps
    const flapMat = makeMat(scene, 'flap', 0.08, 0.08, 0.08, 0);
    for (let i = 0; i < 2; i++) {
        const ff = MeshBuilder.CreateBox('ff' + i, { width: 4.5, height: 7, depth: 0.3 }, scene);
        ff.material = flapMat; ff.parent = shipNode;
        ff.position.y = baseY + 34; ff.position.x = i === 0 ? 5.2 : -5.2;
        ff.rotation.z = i === 0 ? -0.15 : 0.15; meshes.push(ff);
        const af = MeshBuilder.CreateBox('af' + i, { width: 4, height: 5, depth: 0.3 }, scene);
        af.material = flapMat; af.parent = shipNode;
        af.position.y = baseY + 4; af.position.z = i === 0 ? 5.2 : -5.2;
        af.rotation.x = i === 0 ? -0.12 : 0.12; meshes.push(af);
    }
    // 6 ship engines: 3 sea-level + 3 vacuum
    for (let i = 0; i < 3; i++) {
        const a = (i * Math.PI * 2) / 3 + Math.PI / 6;
        const e = MeshBuilder.CreateCylinder('se' + i, { diameterTop: 0.78, diameterBottom: 1.3, height: 2.5, tessellation: 12 }, scene);
        e.material = engMat; e.parent = shipNode; e.position.set(Math.cos(a) * 1.5, baseY - 2, Math.sin(a) * 1.5);
        meshes.push(e);
    }
    for (let i = 0; i < 3; i++) {
        const a = (i * Math.PI * 2) / 3;
        const e = MeshBuilder.CreateCylinder('ve' + i, { diameterTop: 1.0, diameterBottom: 2.4, height: 4, tessellation: 16 }, scene);
        e.material = engMat; e.parent = shipNode; e.position.set(Math.cos(a) * 3.2, baseY - 3, Math.sin(a) * 3.2);
        // Vacuum extension
        const ext = MeshBuilder.CreateCylinder('vx' + i, { diameterTop: 2.4, diameterBottom: 2.8, height: 2.5, tessellation: 16 }, scene);
        const extMat = makeMat(scene, 'vxm' + i, 0.25, 0.2, 0.15, 0.1);
        ext.material = extMat; ext.parent = e; ext.position.y = -3.25;
        meshes.push(e);
    }
    return { meshes, booster: boosterRoot };
}

// ═══ LAUNCH SITE ═══
function buildLaunchSite(scene: Scene) {
    const tMat = makeMat(scene, 'tower', 0.45, 0.4, 0.35, 0.15);
    const cMat = makeMat(scene, 'concrete', 0.35, 0.33, 0.3, 0);
    const sMat = makeMat(scene, 'steelP', 0.4, 0.38, 0.35, 0.2);
    // OLM
    const base = MeshBuilder.CreateBox('olmB', { width: 25, height: 4, depth: 25 }, scene);
    base.material = cMat; base.position.y = 2;
    for (let i = 0; i < 4; i++) {
        const c = MeshBuilder.CreateBox('oc' + i, { width: 4, height: 28, depth: 4 }, scene);
        c.material = sMat; c.position.set((i % 2 === 0 ? 1 : -1) * 8, 14, (i < 2 ? 1 : -1) * 8);
    }
    const top = MeshBuilder.CreateBox('ot', { width: 22, height: 3, depth: 22 }, scene);
    top.material = sMat; top.position.y = 29;
    // Clamp ring
    const cr = MeshBuilder.CreateTorus('cr', { diameter: 10, thickness: 0.6, tessellation: 32 }, scene);
    cr.material = sMat; cr.position.y = 30.5;
    // Tower (140m)
    const tx = -18;
    for (const [dx, dz] of [[-2.5, -2.5], [2.5, -2.5], [-2.5, 2.5], [2.5, 2.5]]) {
        const c = MeshBuilder.CreateBox('tc', { width: 1.5, height: 140, depth: 1.5 }, scene);
        c.material = tMat; c.position.set(tx + dx, 70, dz);
    }
    // Cross bracing
    for (let l = 0; l < 9; l++) {
        const y = 10 + l * 15;
        const b1 = MeshBuilder.CreateBox('hb' + l, { width: 5, height: 0.5, depth: 0.5 }, scene);
        b1.material = tMat; b1.position.set(tx, y, 0);
        const b2 = MeshBuilder.CreateBox('hb2' + l, { width: 0.5, height: 0.5, depth: 5 }, scene);
        b2.material = tMat; b2.position.set(tx, y, 0);
    }
    // Chopstick arms
    const chMat = makeMat(scene, 'chop', 0.5, 0.45, 0.4, 0.2);
    for (let i = 0; i < 2; i++) {
        const arm = MeshBuilder.CreateBox('arm' + i, { width: 36, height: 2, depth: 1.8 }, scene);
        arm.material = chMat; arm.position.set(tx + 20.5, 85, (i - 0.5) * 7);
    }
    // QD arm
    const qd = MeshBuilder.CreateBox('qd', { width: 22, height: 1.5, depth: 2 }, scene);
    qd.material = chMat; qd.position.set(tx + 13, 105, 0);
    // Lightning rod
    const rod = MeshBuilder.CreateCylinder('rod', { diameterTop: 0.08, diameterBottom: 0.3, height: 12, tessellation: 8 }, scene);
    rod.material = tMat; rod.position.set(tx, 146, 0);
    // Concrete pad
    const pad = MeshBuilder.CreateBox('pad', { width: 60, height: 0.5, depth: 60 }, scene);
    pad.material = cMat; pad.position.y = 0.25;
    // Fuel tanks
    const wMat = makeMat(scene, 'white', 0.9, 0.9, 0.9, 0.3);
    for (let i = 0; i < 4; i++) {
        const tk = MeshBuilder.CreateCylinder('ft' + i, { diameterTop: 5, diameterBottom: 5, height: 20, tessellation: 16 }, scene);
        tk.material = wMat; tk.position.set(-40 + i * 8, 10, 35);
    }
}

// ═══ EARTH WITH ATMOSPHERE ═══
function buildEarth(scene: Scene, parent: TransformNode) {
    // Earth sphere
    const earthSphere = MeshBuilder.CreateSphere('earth', { diameter: EARTH_R * 2, segments: 64 }, scene);
    const earthMat = new StandardMaterial('earthMat', scene);
    earthMat.diffuseColor = new Color3(0.15, 0.35, 0.65);
    earthMat.specularColor = new Color3(0.2, 0.2, 0.3);
    earthSphere.material = earthMat;
    earthSphere.parent = parent;

    // Land masses (green/brown patches as slightly-offset spheres)
    const landMat = new StandardMaterial('landMat', scene);
    landMat.diffuseColor = new Color3(0.25, 0.45, 0.2);
    landMat.specularColor = Color3.Black();
    for (let i = 0; i < 8; i++) {
        const land = MeshBuilder.CreateDisc('land' + i, { radius: 40 + Math.random() * 80, tessellation: 20 }, scene);
        land.material = landMat; land.parent = parent;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.random() * Math.PI * 0.8 + 0.1;
        const r = EARTH_R + 0.5;
        land.position.set(r * Math.sin(phi) * Math.cos(theta), r * Math.cos(phi), r * Math.sin(phi) * Math.sin(theta));
        land.lookAt(Vector3.Zero());
    }

    // Atmosphere glow (slightly larger sphere with emissive blue)
    const atmosphere = MeshBuilder.CreateSphere('atmo', { diameter: EARTH_R * 2 + 30, segments: 48 }, scene);
    const atmoMat = new StandardMaterial('atmoMat', scene);
    atmoMat.diffuseColor = new Color3(0.3, 0.5, 0.9);
    atmoMat.emissiveColor = new Color3(0.1, 0.2, 0.5);
    atmoMat.alpha = 0.15;
    atmoMat.backFaceCulling = false;
    atmosphere.material = atmoMat;
    atmosphere.parent = parent;

    // Cloud layer
    const clouds = MeshBuilder.CreateSphere('clouds', { diameter: EARTH_R * 2 + 10, segments: 48 }, scene);
    const cloudMat = new StandardMaterial('cloudMat', scene);
    cloudMat.diffuseColor = Color3.White();
    cloudMat.emissiveColor = new Color3(0.3, 0.3, 0.3);
    cloudMat.alpha = 0.2;
    cloudMat.backFaceCulling = false;
    clouds.material = cloudMat;
    clouds.parent = parent;

    return { earthSphere, atmosphere, clouds };
}

// ═══ MOON SPHERE ═══
function buildMoonSphere(scene: Scene, parent: TransformNode): Mesh {
    const moon = MeshBuilder.CreateSphere('moon', { diameter: MOON_R * 2, segments: 48 }, scene);
    const moonMat = new StandardMaterial('moonMat', scene);
    moonMat.diffuseColor = new Color3(0.55, 0.53, 0.48);
    moonMat.specularColor = new Color3(0.05, 0.05, 0.05);
    moon.material = moonMat;
    moon.parent = parent;
    // Mare (dark patches)
    for (let i = 0; i < 6; i++) {
        const mare = MeshBuilder.CreateDisc('mare' + i, { radius: 15 + Math.random() * 25, tessellation: 20 }, scene);
        const mareMat = makeMat(scene, 'mm' + i, 0.35, 0.33, 0.3, 0);
        mare.material = mareMat; mare.parent = parent;
        const th = Math.random() * Math.PI * 2, ph = Math.random() * Math.PI;
        const r = MOON_R + 0.3;
        mare.position.set(r * Math.sin(ph) * Math.cos(th), r * Math.cos(ph), r * Math.sin(ph) * Math.sin(th));
        mare.lookAt(Vector3.Zero());
    }
    return moon;
}

// ═══ MOON SURFACE (for landing detail) ═══
function buildMoonSurface(scene: Scene): TransformNode {
    const root = new TransformNode('moonSurf', scene);
    const g = MeshBuilder.CreateGround('mg', { width: 800, height: 800, subdivisions: 80 }, scene);
    const gm = makeMat(scene, 'mgs', 0.5, 0.48, 0.44, 0.05);
    g.material = gm; g.parent = root;
    for (let i = 0; i < 30; i++) {
        const r = 4 + Math.random() * 15;
        const c = MeshBuilder.CreateDisc('cr' + i, { radius: r, tessellation: 24 }, scene);
        c.material = makeMat(scene, 'crm' + i, 0.38, 0.36, 0.32, 0);
        c.parent = root; c.rotation.x = Math.PI / 2;
        c.position.set((Math.random() - 0.5) * 500, 0.05, (Math.random() - 0.5) * 500);
        const rim = MeshBuilder.CreateTorus('rim' + i, { diameter: r * 2, thickness: r * 0.12, tessellation: 20 }, scene);
        rim.material = makeMat(scene, 'rmm' + i, 0.52, 0.5, 0.46, 0);
        rim.parent = root; rim.position.copyFrom(c.position); rim.position.y = 0.2; rim.rotation.x = Math.PI / 2;
    }
    return root;
}

// ═══ MOON BASE ═══
function buildMoonBase(scene: Scene): TransformNode {
    const root = new TransformNode('moonBase', scene);
    const habMat = makeMat(scene, 'hab', 0.85, 0.85, 0.82, 0.3);
    const metalMat = makeMat(scene, 'metal', 0.5, 0.5, 0.5, 0.2);
    const panelMat = new StandardMaterial('panel', scene);
    panelMat.diffuseColor = new Color3(0.1, 0.1, 0.35);
    panelMat.emissiveColor = new Color3(0.02, 0.02, 0.08);
    const bx = 50, bz = 30;
    // Habitat domes
    for (let i = 0; i < 3; i++) {
        const dome = MeshBuilder.CreateSphere('dome' + i, { diameter: 12, slice: 0.5, segments: 20 }, scene);
        dome.material = habMat; dome.parent = root; dome.position.set(bx + i * 18, 0, bz);
    }
    // Connecting tunnels
    for (let i = 0; i < 2; i++) {
        const tun = MeshBuilder.CreateCylinder('tun' + i, { diameter: 3, height: 16, tessellation: 12 }, scene);
        tun.material = metalMat; tun.parent = root;
        tun.position.set(bx + 9 + i * 18, 1.5, bz); tun.rotation.z = Math.PI / 2;
    }
    // Solar panel arrays
    for (let i = 0; i < 4; i++) {
        const pole = MeshBuilder.CreateCylinder('pole' + i, { diameter: 0.3, height: 6, tessellation: 8 }, scene);
        pole.material = metalMat; pole.parent = root; pole.position.set(bx + i * 12, 3, bz + 20);
        const panel = MeshBuilder.CreateBox('sp' + i, { width: 8, height: 0.1, depth: 4 }, scene);
        panel.material = panelMat; panel.parent = root; panel.position.set(bx + i * 12, 6.5, bz + 20);
        panel.rotation.x = -0.3;
    }
    // Comms dish
    const dish = MeshBuilder.CreateSphere('dish', { diameter: 5, slice: 0.3, segments: 16 }, scene);
    dish.material = habMat; dish.parent = root; dish.position.set(bx - 10, 8, bz);
    dish.rotation.x = -0.5;
    const dishPole = MeshBuilder.CreateCylinder('dp', { diameter: 0.4, height: 8, tessellation: 8 }, scene);
    dishPole.material = metalMat; dishPole.parent = root; dishPole.position.set(bx - 10, 4, bz);
    // Landing pad markers
    for (let i = 0; i < 8; i++) {
        const a = (i * Math.PI * 2) / 8;
        const marker = MeshBuilder.CreateBox('lm' + i, { width: 3, height: 0.05, depth: 0.5 }, scene);
        const mMat = new StandardMaterial('lmm' + i, scene);
        mMat.diffuseColor = new Color3(0.8, 0.4, 0.1); mMat.emissiveColor = new Color3(0.3, 0.15, 0.05);
        marker.material = mMat; marker.parent = root;
        marker.position.set(Math.cos(a) * 15, 0.05, Math.sin(a) * 15); marker.rotation.y = a;
    }
    return root;
}

// ═══ ASTRONAUT ═══
function buildAstronaut(scene: Scene): { root: TransformNode } {
    const root = new TransformNode('astronaut', scene);
    const suitMat = makeMat(scene, 'suit', 0.9, 0.9, 0.88, 0.2);
    const visorMat = new StandardMaterial('visor', scene);
    visorMat.diffuseColor = new Color3(0.1, 0.15, 0.3);
    visorMat.emissiveColor = new Color3(0.05, 0.08, 0.15);
    // Body
    MeshBuilder.CreateCapsule('torso', { height: 1.6, radius: 0.4 }, scene).material = suitMat;
    const torso = scene.getMeshByName('torso')!; torso.parent = root;
    // Helmet
    const head = MeshBuilder.CreateSphere('helmet', { diameter: 0.7, segments: 12 }, scene);
    head.material = suitMat; head.parent = root; head.position.y = 1.2;
    const visor = MeshBuilder.CreateSphere('visorM', { diameter: 0.5, segments: 12 }, scene);
    visor.material = visorMat; visor.parent = root; visor.position.set(0.15, 1.25, 0);
    // Backpack (PLSS)
    const bp = MeshBuilder.CreateBox('plss', { width: 0.5, height: 0.7, depth: 0.3 }, scene);
    bp.material = suitMat; bp.parent = root; bp.position.set(-0.3, 0.5, 0);
    // Legs
    for (let i = 0; i < 2; i++) {
        const leg = MeshBuilder.CreateCylinder('leg' + i, { diameter: 0.25, height: 0.9, tessellation: 8 }, scene);
        leg.material = suitMat; leg.parent = root;
        leg.position.set(0, -0.7, (i - 0.5) * 0.3);
    }
    return { root };
}

// ═══ ROVER ═══
function buildRover(scene: Scene): { root: TransformNode, body: Mesh } {
    const root = new TransformNode('rover', scene);
    root.position.set(70, 0, 30);
    const bodyMat = makeMat(scene, 'roverBody', 0.7, 0.7, 0.68, 0.2);
    const wheelMat = makeMat(scene, 'roverWheel', 0.25, 0.25, 0.25, 0.1);
    // Chassis
    const body = MeshBuilder.CreateBox('roverChassis', { width: 4, height: 1, depth: 2.5 }, scene);
    body.material = bodyMat; body.parent = root;
    // Wheels
    for (const [x, z] of [[-1.8, -1.2], [-1.8, 1.2], [1.8, -1.2], [1.8, 1.2]]) {
        const w = MeshBuilder.CreateCylinder('rw', { diameter: 0.8, height: 0.3, tessellation: 16 }, scene);
        w.material = wheelMat; w.parent = root; w.position.set(x, -0.3, z); w.rotation.x = Math.PI / 2;
    }
    // Antenna
    const ant = MeshBuilder.CreateCylinder('rant', { diameterTop: 0.02, diameterBottom: 0.08, height: 2, tessellation: 6 }, scene);
    ant.material = bodyMat; ant.parent = root; ant.position.set(-1.5, 1.5, 0);
    const dish = MeshBuilder.CreateDisc('rdish', { radius: 0.4, tessellation: 12 }, scene);
    dish.material = bodyMat; dish.parent = root; dish.position.set(-1.5, 2.5, 0); dish.rotation.z = 0.3;
    // Solar panel
    const sp = MeshBuilder.CreateBox('rsp', { width: 3, height: 0.05, depth: 1.5 }, scene);
    const spMat = new StandardMaterial('rspMat', scene);
    spMat.diffuseColor = new Color3(0.1, 0.1, 0.35);
    spMat.emissiveColor = new Color3(0.02, 0.02, 0.06);
    sp.material = spMat; sp.parent = root; sp.position.y = 1.2;
    return { root, body };
}

// ═══ PARTICLES ═══
function createExhaustParticles(scene: Scene, emitter: Mesh): ParticleSystem {
    const ps = new ParticleSystem('exhaust', 5000, scene);
    ps.createPointEmitter(new Vector3(-2, -1, -2), new Vector3(2, -1, 2));
    ps.color1 = new Color4(1, 0.7, 0.15, 1);
    ps.color2 = new Color4(1, 0.4, 0.08, 0.9);
    ps.colorDead = new Color4(0.4, 0.3, 0.2, 0);
    ps.minSize = 2; ps.maxSize = 6;
    ps.minLifeTime = 0.3; ps.maxLifeTime = 1.2;
    ps.emitRate = 0; ps.blendMode = ParticleSystem.BLENDMODE_ADD;
    ps.minEmitPower = 25; ps.maxEmitPower = 50;
    ps.updateSpeed = 0.02; ps.gravity = new Vector3(0, -8, 0);
    ps.emitter = emitter; ps.start();
    return ps;
}

function createExhaustCore(scene: Scene, emitter: Mesh): ParticleSystem {
    const ps = new ParticleSystem('core', 2000, scene);
    ps.createPointEmitter(new Vector3(-0.5, -1, -0.5), new Vector3(0.5, -1, 0.5));
    ps.color1 = new Color4(0.8, 0.85, 1, 1);
    ps.color2 = new Color4(1, 1, 0.9, 0.95);
    ps.colorDead = new Color4(1, 0.6, 0.1, 0);
    ps.minSize = 0.5; ps.maxSize = 2;
    ps.minLifeTime = 0.1; ps.maxLifeTime = 0.4;
    ps.emitRate = 0; ps.blendMode = ParticleSystem.BLENDMODE_ADD;
    ps.minEmitPower = 30; ps.maxEmitPower = 60;
    ps.updateSpeed = 0.01; ps.gravity = new Vector3(0, -5, 0);
    ps.emitter = emitter; ps.start();
    return ps;
}

function createStarfield(scene: Scene) {
    const m = new StandardMaterial('starM', scene);
    m.emissiveColor = Color3.White(); m.disableLighting = true;
    for (let i = 0; i < 600; i++) {
        const s = MeshBuilder.CreateSphere('s' + i, { diameter: 0.3 + Math.random() * 0.6 }, scene);
        s.material = m;
        const r = 800 + Math.random() * 700;
        const th = Math.random() * Math.PI * 2, ph = Math.random() * Math.PI;
        s.position.set(r * Math.sin(ph) * Math.cos(th), r * Math.cos(ph), r * Math.sin(ph) * Math.sin(th));
    }
}

// ═══ HELPER ═══
function makeMat(scene: Scene, name: string, r: number, g: number, b: number, spec: number): StandardMaterial {
    const m = new StandardMaterial(name, scene);
    m.diffuseColor = new Color3(r, g, b);
    m.specularColor = new Color3(spec, spec, spec);
    return m;
}
