import {
    Engine, Scene, ArcRotateCamera, Vector3, HemisphericLight, DirectionalLight,
    MeshBuilder, StandardMaterial, Color3, Color4, ParticleSystem,
    GlowLayer, ShadowGenerator,
    Mesh, TransformNode
} from '@babylonjs/core';

export interface SimState {
    phase: string;
    altitude: number;
    velocity: number;
    downrange: number;
    fuel: number;
    thrust: number;
    missionTime: number;
    throttle: number;
}

let engine: Engine | null = null;
let scene: Scene | null = null;
let animFrame = 0;

export function destroyScene() {
    if (animFrame) cancelAnimationFrame(animFrame);
    scene?.dispose();
    engine?.dispose();
    engine = null;
    scene = null;
}

export function initStarshipScene(canvas: HTMLCanvasElement, onTelemetry: (s: SimState) => void): () => void {
    engine = new Engine(canvas, true, { preserveDrawingBuffer: true, stencil: true });
    scene = new Scene(engine);
    scene.clearColor = new Color4(0.01, 0.01, 0.03, 1);

    // ── Camera ──
    const cam = new ArcRotateCamera('cam', -Math.PI / 2, Math.PI / 3, 80, Vector3.Zero(), scene);
    cam.lowerRadiusLimit = 20;
    cam.upperRadiusLimit = 300;
    cam.attachControl(canvas, true);
    cam.wheelPrecision = 10;

    // ── Lighting ──
    const hemi = new HemisphericLight('hemi', new Vector3(0, 1, 0), scene);
    hemi.intensity = 0.3;
    const sun = new DirectionalLight('sun', new Vector3(-1, -2, -1), scene);
    sun.intensity = 1.2;
    sun.position = new Vector3(100, 200, 100);

    // ── Glow layer ──
    const glow = new GlowLayer('glow', scene);
    glow.intensity = 0.6;

    // ── Build Starship ──
    const shipRoot = new TransformNode('shipRoot', scene);
    const ship = buildStarship(scene, shipRoot);
    shipRoot.position.y = 35;

    // ── Build Launch Pad ──
    buildLaunchPad(scene);

    // ── Ground ──
    const ground = MeshBuilder.CreateGround('ground', { width: 2000, height: 2000, subdivisions: 4 }, scene);
    const groundMat = new StandardMaterial('groundMat', scene);
    groundMat.diffuseColor = new Color3(0.12, 0.12, 0.1);
    groundMat.specularColor = Color3.Black();
    ground.material = groundMat;
    ground.receiveShadows = true;

    // ── Shadow ──
    const shadowGen = new ShadowGenerator(1024, sun);
    shadowGen.useBlurExponentialShadowMap = true;
    ship.meshes.forEach(m => shadowGen.addShadowCaster(m));

    // ── Exhaust particles ──
    const exhaustEmitter = MeshBuilder.CreateSphere('exEmit', { diameter: 0.5 }, scene);
    exhaustEmitter.parent = shipRoot;
    exhaustEmitter.position.y = -33;
    exhaustEmitter.isVisible = false;
    const exhaust = createExhaustParticles(scene, exhaustEmitter);

    // ── Stars ──
    createStarfield(scene);

    // ── Moon surface (hidden initially) ──
    const moonSurface = buildMoonSurface(scene);
    moonSurface.setEnabled(false);

    // ── Simulation state ──
    const state: SimState = {
        phase: 'prelaunch',
        altitude: 0,
        velocity: 0,
        downrange: 0,
        fuel: 100,
        thrust: 0,
        missionTime: -10,
        throttle: 0,
    };

    let lastTime = performance.now();
    let launched = false;
    let phaseTimer = 0;
    let camShake = 0;

    const onLaunch = () => { launched = true; };
    window.addEventListener('starship-launch', onLaunch);

    // ── Render loop ──
    engine.runRenderLoop(() => {
        const now = performance.now();
        const dt = Math.min((now - lastTime) / 1000, 0.1);
        lastTime = now;

        if (launched) {
            state.missionTime += dt;
            phaseTimer += dt;
            updateSimulation(state, dt, phaseTimer, shipRoot, exhaust, cam, ground, moonSurface, ship);
            camShake = state.phase === 'liftoff' || state.phase === 'ignition' ? 0.15 : state.phase === 'landing-burn' ? 0.08 : 0;
        } else {
            state.missionTime = -10 + (performance.now() % 10000) / 1000;
        }

        // Camera shake
        if (camShake > 0) {
            cam.target.x += (Math.random() - 0.5) * camShake;
            cam.target.y += (Math.random() - 0.5) * camShake * 0.5;
        }

        onTelemetry(state);
        scene!.render();
    });

    return () => {
        window.removeEventListener('starship-launch', onLaunch);
    };
}

function updateSimulation(
    s: SimState, dt: number, _pt: number,
    shipRoot: TransformNode, exhaust: ParticleSystem,
    cam: ArcRotateCamera, ground: Mesh, moonSurface: TransformNode,
    ship: { meshes: Mesh[], booster: Mesh }
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
    else if (t < 190) { s.phase = 'touchdown'; s.throttle = 0; }
    else { s.phase = 'complete'; s.throttle = 0; }

    // Physics (simplified orbital-ish)
    const accel = s.throttle / 100 * 3.5; // km/s²
    const gravity = s.altitude < 100 ? 0.0098 : 0.001;

    if (t >= 0 && s.phase !== 'touchdown' && s.phase !== 'complete') {
        s.velocity += (accel - gravity) * dt;
        s.altitude += s.velocity * dt;
        s.downrange += s.velocity * dt * 0.7;
        s.fuel = Math.max(0, s.fuel - s.throttle * dt * 0.03);
        s.thrust = s.throttle;
    }

    if (s.phase === 'touchdown' || s.phase === 'complete') {
        s.velocity = Math.max(0, s.velocity - dt * 2);
        s.thrust = 0;
    }

    // Clamp
    s.altitude = Math.max(0, s.altitude);
    s.velocity = Math.max(0, s.velocity);

    // ── Visual updates ──

    // Ship position — maps altitude to scene Y
    const sceneY = Math.min(s.altitude * 0.5, 200);
    shipRoot.position.y = 35 + sceneY;

    // Tilt during ascent
    if (s.phase === 'liftoff' || s.phase === 'maxq') {
        shipRoot.rotation.z = Math.min((t - 3) * 0.003, 0.15);
    } else if (s.phase === 'landing-burn') {
        shipRoot.rotation.z = Math.max(shipRoot.rotation.z - dt * 0.05, 0);
    }

    // Exhaust
    if (s.throttle > 5) {
        exhaust.start();
        exhaust.emitRate = s.throttle * 8;
        exhaust.minSize = 1 + s.throttle * 0.03;
        exhaust.maxSize = 3 + s.throttle * 0.05;
        exhaust.minLifeTime = 0.2;
        exhaust.maxLifeTime = 0.8 + s.throttle * 0.005;
    } else {
        exhaust.stop();
    }

    // Camera follow
    if (s.phase !== 'prelaunch') {
        cam.target.y += (shipRoot.position.y - cam.target.y) * dt * 2;
        if (s.altitude > 50) {
            cam.radius += (120 + s.altitude * 0.3 - cam.radius) * dt;
        }
    }

    // Stage separation visual
    if (s.phase === 'separation') {
        ship.booster.position.y -= dt * 5;
        ship.booster.rotation.z += dt * 0.1;
    }

    // Ground fade / Moon swap
    if (s.altitude > 80) {
        ground.visibility = Math.max(0, 1 - (s.altitude - 80) / 40);
    }
    if (s.phase === 'lunar-approach' || s.phase === 'landing-burn' || s.phase === 'touchdown' || s.phase === 'complete') {
        moonSurface.setEnabled(true);
        moonSurface.position.y = shipRoot.position.y - 40;
        ground.visibility = 0;
    }
}

// ── Starship model ──
function buildStarship(scene: Scene, parent: TransformNode): { meshes: Mesh[], booster: Mesh } {
    const meshes: Mesh[] = [];

    // ── Super Heavy Booster ──
    const booster = MeshBuilder.CreateCylinder('booster', {
        diameterTop: 9, diameterBottom: 9, height: 30, tessellation: 32
    }, scene);
    const boosterMat = new StandardMaterial('boosterMat', scene);
    boosterMat.diffuseColor = new Color3(0.75, 0.75, 0.75);
    boosterMat.specularColor = new Color3(0.3, 0.3, 0.3);
    booster.material = boosterMat;
    booster.parent = parent;
    booster.position.y = -15;
    meshes.push(booster);

    // Booster grid fins
    for (let i = 0; i < 4; i++) {
        const fin = MeshBuilder.CreateBox('gridFin' + i, { width: 4, height: 3, depth: 0.3 }, scene);
        fin.material = boosterMat;
        fin.parent = booster;
        fin.position.y = 13;
        const angle = (i * Math.PI) / 2;
        fin.position.x = Math.cos(angle) * 5;
        fin.position.z = Math.sin(angle) * 5;
        fin.rotation.y = angle;
        meshes.push(fin);
    }

    // Booster engines (33 raptor cluster — simplified as ring)
    const engineMat = new StandardMaterial('engMat', scene);
    engineMat.diffuseColor = new Color3(0.2, 0.2, 0.2);
    engineMat.emissiveColor = new Color3(0.05, 0.05, 0.05);
    for (let i = 0; i < 12; i++) {
        const eng = MeshBuilder.CreateCylinder('eng' + i, {
            diameterTop: 1.2, diameterBottom: 0.8, height: 2, tessellation: 12
        }, scene);
        eng.material = engineMat;
        eng.parent = booster;
        const angle = (i * Math.PI * 2) / 12;
        eng.position.x = Math.cos(angle) * 3;
        eng.position.z = Math.sin(angle) * 3;
        eng.position.y = -16;
        meshes.push(eng);
    }

    // ── Starship (upper stage) ──
    const shipBody = MeshBuilder.CreateCylinder('shipBody', {
        diameterTop: 9, diameterBottom: 9, height: 25, tessellation: 32
    }, scene);
    const shipMat = new StandardMaterial('shipMat', scene);
    shipMat.diffuseColor = new Color3(0.85, 0.85, 0.85);
    shipMat.specularColor = new Color3(0.4, 0.4, 0.4);
    shipBody.material = shipMat;
    shipBody.parent = parent;
    shipBody.position.y = 12.5;
    meshes.push(shipBody);

    // Nose cone
    const nose = MeshBuilder.CreateCylinder('nose', {
        diameterTop: 0, diameterBottom: 9, height: 12, tessellation: 32
    }, scene);
    nose.material = shipMat;
    nose.parent = parent;
    nose.position.y = 31;
    meshes.push(nose);

    // Forward flaps (2)
    for (let i = 0; i < 2; i++) {
        const flap = MeshBuilder.CreateBox('fwdFlap' + i, { width: 5, height: 8, depth: 0.2 }, scene);
        const flapMat = new StandardMaterial('flapMat' + i, scene);
        flapMat.diffuseColor = new Color3(0.15, 0.15, 0.15);
        flap.material = flapMat;
        flap.parent = parent;
        flap.position.y = 24;
        flap.position.x = i === 0 ? 5 : -5;
        flap.rotation.z = i === 0 ? -0.1 : 0.1;
        meshes.push(flap);
    }

    // Aft flaps (2)
    for (let i = 0; i < 2; i++) {
        const flap = MeshBuilder.CreateBox('aftFlap' + i, { width: 5, height: 6, depth: 0.2 }, scene);
        const flapMat = new StandardMaterial('aftFlapMat' + i, scene);
        flapMat.diffuseColor = new Color3(0.15, 0.15, 0.15);
        flap.material = flapMat;
        flap.parent = parent;
        flap.position.y = 2;
        flap.position.z = i === 0 ? 5 : -5;
        flap.rotation.x = i === 0 ? -0.1 : 0.1;
        meshes.push(flap);
    }

    // Heat shield tiles (visual band)
    const heatShield = MeshBuilder.CreateCylinder('heatShield', {
        diameterTop: 9.1, diameterBottom: 9.1, height: 15, tessellation: 32
    }, scene);
    const hsMat = new StandardMaterial('hsMat', scene);
    hsMat.diffuseColor = new Color3(0.05, 0.05, 0.05);
    hsMat.alpha = 0.4;
    heatShield.material = hsMat;
    heatShield.parent = parent;
    heatShield.position.y = 10;
    meshes.push(heatShield);

    // SpaceX logo band (white stripe)
    const logoBand = MeshBuilder.CreateCylinder('logoBand', {
        diameterTop: 9.15, diameterBottom: 9.15, height: 1, tessellation: 32
    }, scene);
    const logoMat = new StandardMaterial('logoMat', scene);
    logoMat.diffuseColor = Color3.White();
    logoMat.emissiveColor = new Color3(0.2, 0.2, 0.2);
    logoBand.material = logoMat;
    logoBand.parent = parent;
    logoBand.position.y = 20;
    meshes.push(logoBand);

    return { meshes, booster };
}

// ── Launch pad ──
function buildLaunchPad(scene: Scene) {
    const padMat = new StandardMaterial('padMat', scene);
    padMat.diffuseColor = new Color3(0.3, 0.3, 0.3);

    // Main pad
    const pad = MeshBuilder.CreateBox('pad', { width: 30, height: 2, depth: 30 }, scene);
    pad.material = padMat;
    pad.position.y = 1;

    // Tower
    const towerMat = new StandardMaterial('towerMat', scene);
    towerMat.diffuseColor = new Color3(0.4, 0.35, 0.3);
    const tower = MeshBuilder.CreateBox('tower', { width: 3, height: 80, depth: 3 }, scene);
    tower.material = towerMat;
    tower.position.set(-12, 40, 0);

    // Chopstick arms
    for (let i = 0; i < 2; i++) {
        const arm = MeshBuilder.CreateBox('arm' + i, { width: 12, height: 1.5, depth: 1.5 }, scene);
        arm.material = towerMat;
        arm.parent = tower;
        arm.position.set(7, 10 + i * 5, (i - 0.5) * 4);
    }

    // Lightning rods on top
    const rod = MeshBuilder.CreateCylinder('rod', { diameterTop: 0.1, diameterBottom: 0.3, height: 10, tessellation: 8 }, scene);
    rod.material = towerMat;
    rod.parent = tower;
    rod.position.y = 45;
}

// ── Exhaust particles ──
function createExhaustParticles(scene: Scene, emitter: Mesh): ParticleSystem {
    const ps = new ParticleSystem('exhaust', 3000, scene);
    ps.createPointEmitter(new Vector3(-1, -1, -1), new Vector3(1, -1, 1));

    ps.color1 = new Color4(1, 0.6, 0.1, 1);
    ps.color2 = new Color4(1, 0.3, 0.05, 0.8);
    ps.colorDead = new Color4(0.3, 0.3, 0.3, 0);

    ps.minSize = 1;
    ps.maxSize = 4;
    ps.minLifeTime = 0.3;
    ps.maxLifeTime = 1.0;
    ps.emitRate = 0;
    ps.blendMode = ParticleSystem.BLENDMODE_ADD;

    ps.minEmitPower = 15;
    ps.maxEmitPower = 30;
    ps.updateSpeed = 0.02;

    ps.gravity = new Vector3(0, -5, 0);
    ps.emitter = emitter;

    ps.start();
    return ps;
}

// ── Starfield ──
function createStarfield(scene: Scene) {
    const starMat = new StandardMaterial('starMat', scene);
    starMat.emissiveColor = Color3.White();
    starMat.disableLighting = true;

    for (let i = 0; i < 500; i++) {
        const star = MeshBuilder.CreateSphere('star' + i, { diameter: 0.3 + Math.random() * 0.5 }, scene);
        star.material = starMat;
        const r = 500 + Math.random() * 500;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.random() * Math.PI;
        star.position.set(
            r * Math.sin(phi) * Math.cos(theta),
            r * Math.cos(phi),
            r * Math.sin(phi) * Math.sin(theta)
        );
    }
}

// ── Moon surface ──
function buildMoonSurface(scene: Scene): TransformNode {
    const root = new TransformNode('moonRoot', scene);

    const moonGround = MeshBuilder.CreateGround('moonGround', { width: 500, height: 500, subdivisions: 64 }, scene);
    const moonMat = new StandardMaterial('moonMat', scene);
    moonMat.diffuseColor = new Color3(0.45, 0.43, 0.4);
    moonMat.specularColor = new Color3(0.1, 0.1, 0.1);
    moonMat.bumpTexture = null; // would use noise texture in production
    moonGround.material = moonMat;
    moonGround.parent = root;

    // Craters
    for (let i = 0; i < 20; i++) {
        const crater = MeshBuilder.CreateDisc('crater' + i, { radius: 3 + Math.random() * 10, tessellation: 24 }, scene);
        const cMat = new StandardMaterial('craterMat' + i, scene);
        cMat.diffuseColor = new Color3(0.35, 0.33, 0.3);
        cMat.specularColor = Color3.Black();
        crater.material = cMat;
        crater.parent = root;
        crater.rotation.x = Math.PI / 2;
        crater.position.set(
            (Math.random() - 0.5) * 300,
            0.05,
            (Math.random() - 0.5) * 300
        );
    }

    return root;
}
