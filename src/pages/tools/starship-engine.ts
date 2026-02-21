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

// ── Scale: 1 unit = 1 meter ──
// Real dimensions: Booster 71m tall, Ship 52m tall, 9m diameter
// Total stack ~123m, Mechazilla tower ~145m

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
    const cam = new ArcRotateCamera('cam', -Math.PI / 2, Math.PI / 3, 180, new Vector3(0, 60, 0), scene);
    cam.lowerRadiusLimit = 30;
    cam.upperRadiusLimit = 500;
    cam.attachControl(canvas, true);
    cam.wheelPrecision = 5;

    // ── Lighting ──
    const hemi = new HemisphericLight('hemi', new Vector3(0, 1, 0), scene);
    hemi.intensity = 0.35;
    const sun = new DirectionalLight('sun', new Vector3(-1, -2, -1), scene);
    sun.intensity = 1.4;
    sun.position = new Vector3(200, 400, 200);

    // ── Glow layer ──
    const glow = new GlowLayer('glow', scene);
    glow.intensity = 0.7;

    // ── Build Starship Stack ──
    const shipRoot = new TransformNode('shipRoot', scene);
    const ship = buildStarshipStack(scene, shipRoot);
    // Position: ship sits on the OLM, bottom of booster at ~30m (OLM height)
    shipRoot.position.y = 30 + 71 / 2; // Center of booster at OLM top + half booster height

    // ── Build Launch Site (Mechazilla + OLM) ──
    buildLaunchSite(scene);

    // ── Ground ──
    const ground = MeshBuilder.CreateGround('ground', { width: 4000, height: 4000, subdivisions: 4 }, scene);
    const groundMat = new StandardMaterial('groundMat', scene);
    groundMat.diffuseColor = new Color3(0.15, 0.13, 0.1);
    groundMat.specularColor = Color3.Black();
    ground.material = groundMat;
    ground.receiveShadows = true;

    // ── Shadow ──
    const shadowGen = new ShadowGenerator(2048, sun);
    shadowGen.useBlurExponentialShadowMap = true;
    ship.meshes.forEach(m => shadowGen.addShadowCaster(m));

    // ── Exhaust particles ──
    const exhaustEmitter = MeshBuilder.CreateSphere('exEmit', { diameter: 3 }, scene);
    exhaustEmitter.parent = shipRoot;
    exhaustEmitter.position.y = -71 / 2 - 2; // Bottom of booster
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
            camShake = state.phase === 'liftoff' || state.phase === 'ignition' ? 0.3 : state.phase === 'landing-burn' ? 0.12 : 0;
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
    ship: { meshes: Mesh[], booster: TransformNode }
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
    const accel = s.throttle / 100 * 3.5;
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
    const sceneY = Math.min(s.altitude * 0.8, 400);
    shipRoot.position.y = 30 + 71 / 2 + sceneY;

    // Tilt during ascent (gravity turn)
    if (s.phase === 'liftoff' || s.phase === 'maxq') {
        shipRoot.rotation.z = Math.min((t - 3) * 0.002, 0.12);
    } else if (s.phase === 'landing-burn') {
        shipRoot.rotation.z = Math.max(shipRoot.rotation.z - dt * 0.05, 0);
    }

    // Exhaust
    if (s.throttle > 5) {
        exhaust.start();
        exhaust.emitRate = s.throttle * 15;
        exhaust.minSize = 2 + s.throttle * 0.05;
        exhaust.maxSize = 6 + s.throttle * 0.08;
        exhaust.minLifeTime = 0.3;
        exhaust.maxLifeTime = 1.0 + s.throttle * 0.008;
    } else {
        exhaust.stop();
    }

    // Camera follow
    if (s.phase !== 'prelaunch') {
        cam.target.y += (shipRoot.position.y - cam.target.y) * dt * 2;
        if (s.altitude > 50) {
            cam.radius += (200 + s.altitude * 0.4 - cam.radius) * dt;
        }
    }

    // Stage separation visual
    if (s.phase === 'separation') {
        ship.booster.position.y -= dt * 8;
        ship.booster.rotation.z += dt * 0.05;
    }

    // Ground fade / Moon swap
    if (s.altitude > 80) {
        ground.visibility = Math.max(0, 1 - (s.altitude - 80) / 40);
    }
    if (s.phase === 'lunar-approach' || s.phase === 'landing-burn' || s.phase === 'touchdown' || s.phase === 'complete') {
        moonSurface.setEnabled(true);
        moonSurface.position.y = shipRoot.position.y - 60;
        ground.visibility = 0;
    }
}

// ═══════════════════════════════════════════════════════════════
// STARSHIP STACK — Accurate proportions
// Booster: 71m tall, 9m diameter
// Ship: 52m tall, 9m diameter (including nose)
// ═══════════════════════════════════════════════════════════════

function buildStarshipStack(scene: Scene, parent: TransformNode): { meshes: Mesh[], booster: TransformNode } {
    const meshes: Mesh[] = [];

    // ── Materials ──
    const steelMat = new StandardMaterial('steelMat', scene);
    steelMat.diffuseColor = new Color3(0.78, 0.78, 0.76);
    steelMat.specularColor = new Color3(0.5, 0.5, 0.5);
    steelMat.specularPower = 40;

    const darkSteelMat = new StandardMaterial('darkSteelMat', scene);
    darkSteelMat.diffuseColor = new Color3(0.3, 0.3, 0.3);
    darkSteelMat.specularColor = new Color3(0.2, 0.2, 0.2);

    const engineMat = new StandardMaterial('engMat', scene);
    engineMat.diffuseColor = new Color3(0.15, 0.15, 0.15);
    engineMat.specularColor = new Color3(0.1, 0.1, 0.1);

    const heatShieldMat = new StandardMaterial('hsMat', scene);
    heatShieldMat.diffuseColor = new Color3(0.05, 0.05, 0.05);
    heatShieldMat.specularColor = Color3.Black();

    const nozzleGlowMat = new StandardMaterial('nozzleGlow', scene);
    nozzleGlowMat.diffuseColor = new Color3(0.1, 0.1, 0.1);
    nozzleGlowMat.emissiveColor = new Color3(0.15, 0.05, 0);

    // ── SUPER HEAVY BOOSTER (71m tall, 9m diameter) ──
    const boosterRoot = new TransformNode('boosterRoot', scene);
    boosterRoot.parent = parent;
    boosterRoot.position.y = 0; // Center of stack at y=0

    // Main booster body
    const boosterBody = MeshBuilder.CreateCylinder('boosterBody', {
        diameterTop: 9, diameterBottom: 9, height: 71, tessellation: 48
    }, scene);
    boosterBody.material = steelMat;
    boosterBody.parent = boosterRoot;
    boosterBody.position.y = 0;
    meshes.push(boosterBody);

    // Booster bottom skirt (slightly wider at the base, engine section)
    const boosterSkirt = MeshBuilder.CreateCylinder('boosterSkirt', {
        diameterTop: 9, diameterBottom: 9.4, height: 5, tessellation: 48
    }, scene);
    boosterSkirt.material = darkSteelMat;
    boosterSkirt.parent = boosterRoot;
    boosterSkirt.position.y = -33;
    meshes.push(boosterSkirt);

    // ── 33 Raptor Engines: 3 center + 10 inner ring + 20 outer ring ──
    const engineBottom = -35.5; // Bottom of booster
    const engineLen = 3;

    // 3 center engines (gimbaled)
    for (let i = 0; i < 3; i++) {
        const angle = (i * Math.PI * 2) / 3;
        const r = 0.8;
        const eng = createRaptorEngine(scene, engineMat, nozzleGlowMat, 1.3, engineLen);
        eng.parent = boosterRoot;
        eng.position.set(Math.cos(angle) * r, engineBottom, Math.sin(angle) * r);
        meshes.push(eng);
    }

    // 10 inner ring engines (gimbaled)
    for (let i = 0; i < 10; i++) {
        const angle = (i * Math.PI * 2) / 10;
        const r = 2.3;
        const eng = createRaptorEngine(scene, engineMat, nozzleGlowMat, 1.3, engineLen);
        eng.parent = boosterRoot;
        eng.position.set(Math.cos(angle) * r, engineBottom, Math.sin(angle) * r);
        meshes.push(eng);
    }

    // 20 outer ring engines (fixed)
    for (let i = 0; i < 20; i++) {
        const angle = (i * Math.PI * 2) / 20;
        const r = 3.8;
        const eng = createRaptorEngine(scene, engineMat, nozzleGlowMat, 1.3, engineLen);
        eng.parent = boosterRoot;
        eng.position.set(Math.cos(angle) * r, engineBottom, Math.sin(angle) * r);
        meshes.push(eng);
    }

    // ── Grid fins (4, at the top of booster) ──
    const gridFinMat = new StandardMaterial('gridFinMat', scene);
    gridFinMat.diffuseColor = new Color3(0.2, 0.2, 0.2);
    for (let i = 0; i < 4; i++) {
        const angle = (i * Math.PI) / 2 + Math.PI / 4;
        const fin = MeshBuilder.CreateBox('gridFin' + i, { width: 4.5, height: 3.5, depth: 0.25 }, scene);
        fin.material = gridFinMat;
        fin.parent = boosterRoot;
        fin.position.y = 33;
        fin.position.x = Math.cos(angle) * 5.5;
        fin.position.z = Math.sin(angle) * 5.5;
        fin.rotation.y = angle;
        meshes.push(fin);

        // Grid pattern (cross members)
        for (let j = 0; j < 3; j++) {
            const bar = MeshBuilder.CreateBox('gridBar' + i + '_' + j, { width: 4.5, height: 0.08, depth: 0.28 }, scene);
            bar.material = gridFinMat;
            bar.parent = fin;
            bar.position.y = -1.2 + j * 1.2;
        }
    }

    // ── Hot-staging ring (interstage) ──
    const hotStageRing = MeshBuilder.CreateCylinder('hotStage', {
        diameterTop: 9.2, diameterBottom: 9.2, height: 4, tessellation: 48
    }, scene);
    const hotStageMat = new StandardMaterial('hotStageMat', scene);
    hotStageMat.diffuseColor = new Color3(0.25, 0.25, 0.22);
    hotStageRing.material = hotStageMat;
    hotStageRing.parent = boosterRoot;
    hotStageRing.position.y = 35.5 + 2; // Top of booster
    meshes.push(hotStageRing);

    // Hot-staging vent slots
    for (let i = 0; i < 8; i++) {
        const slot = MeshBuilder.CreateBox('ventSlot' + i, { width: 2.5, height: 3, depth: 0.1 }, scene);
        const slotMat = new StandardMaterial('slotMat' + i, scene);
        slotMat.diffuseColor = new Color3(0.08, 0.08, 0.08);
        slot.material = slotMat;
        const angle = (i * Math.PI * 2) / 8;
        slot.parent = hotStageRing;
        slot.position.set(Math.cos(angle) * 4.65, 0, Math.sin(angle) * 4.65);
        slot.rotation.y = angle;
        meshes.push(slot);
    }

    // ═══ STARSHIP UPPER STAGE (52m tall, 9m diameter) ═══
    const shipNode = new TransformNode('shipNode', scene);
    shipNode.parent = parent;
    const shipBaseY = 71 / 2 + 4; // Above the hot-staging ring

    // Main body (barrel section ~36m)
    const shipBody = MeshBuilder.CreateCylinder('shipBody', {
        diameterTop: 9, diameterBottom: 9, height: 36, tessellation: 48
    }, scene);
    shipBody.material = steelMat;
    shipBody.parent = shipNode;
    shipBody.position.y = shipBaseY + 18;
    meshes.push(shipBody);

    // Nose cone (ogive ~16m)
    const noseLower = MeshBuilder.CreateCylinder('noseLower', {
        diameterTop: 6, diameterBottom: 9, height: 8, tessellation: 48
    }, scene);
    noseLower.material = steelMat;
    noseLower.parent = shipNode;
    noseLower.position.y = shipBaseY + 36 + 4;
    meshes.push(noseLower);

    const noseUpper = MeshBuilder.CreateCylinder('noseUpper', {
        diameterTop: 0, diameterBottom: 6, height: 8, tessellation: 48
    }, scene);
    noseUpper.material = steelMat;
    noseUpper.parent = shipNode;
    noseUpper.position.y = shipBaseY + 36 + 12;
    meshes.push(noseUpper);

    // ── Heat shield (black tiles on one side — simplified as half-cylinder overlay) ──
    const heatShield = MeshBuilder.CreateCylinder('heatShield', {
        diameterTop: 9.15, diameterBottom: 9.15, height: 36, tessellation: 48,
        arc: 0.5
    }, scene);
    heatShield.material = heatShieldMat;
    heatShield.parent = shipNode;
    heatShield.position.y = shipBaseY + 18;
    heatShield.rotation.y = Math.PI / 2;
    meshes.push(heatShield);

    // ── Forward flaps (2, near nose — larger, angled) ──
    const flapMat = new StandardMaterial('flapMat', scene);
    flapMat.diffuseColor = new Color3(0.08, 0.08, 0.08);

    for (let i = 0; i < 2; i++) {
        const flap = MeshBuilder.CreateBox('fwdFlap' + i, { width: 4.5, height: 7, depth: 0.3 }, scene);
        flap.material = flapMat;
        flap.parent = shipNode;
        flap.position.y = shipBaseY + 34;
        flap.position.x = i === 0 ? 5.2 : -5.2;
        flap.rotation.z = i === 0 ? -0.15 : 0.15;

        // Hinge detail
        const hinge = MeshBuilder.CreateCylinder('fwdHinge' + i, {
            diameter: 0.6, height: 4.8, tessellation: 12
        }, scene);
        hinge.material = darkSteelMat;
        hinge.parent = flap;
        hinge.position.y = 3.5;
        hinge.rotation.z = Math.PI / 2;
        meshes.push(flap, hinge);
    }

    // ── Aft flaps (2, near bottom — smaller) ──
    for (let i = 0; i < 2; i++) {
        const flap = MeshBuilder.CreateBox('aftFlap' + i, { width: 4, height: 5, depth: 0.3 }, scene);
        flap.material = flapMat;
        flap.parent = shipNode;
        flap.position.y = shipBaseY + 4;
        flap.position.z = i === 0 ? 5.2 : -5.2;
        flap.rotation.x = i === 0 ? -0.12 : 0.12;

        const hinge = MeshBuilder.CreateCylinder('aftHinge' + i, {
            diameter: 0.5, height: 4.2, tessellation: 12
        }, scene);
        hinge.material = darkSteelMat;
        hinge.parent = flap;
        hinge.position.y = 2.5;
        hinge.rotation.x = Math.PI / 2;
        meshes.push(flap, hinge);
    }

    // ── 6 Raptor engines on Starship: 3 sea-level (center) + 3 vacuum (outer, big bell) ──
    const shipEngineY = shipBaseY - 2;

    // 3 sea-level Raptors (center, gimbaled)
    for (let i = 0; i < 3; i++) {
        const angle = (i * Math.PI * 2) / 3 + Math.PI / 6;
        const r = 1.5;
        const eng = createRaptorEngine(scene, engineMat, nozzleGlowMat, 1.3, 2.5);
        eng.parent = shipNode;
        eng.position.set(Math.cos(angle) * r, shipEngineY, Math.sin(angle) * r);
        meshes.push(eng);
    }

    // 3 vacuum Raptors (outer, larger extended nozzle)
    for (let i = 0; i < 3; i++) {
        const angle = (i * Math.PI * 2) / 3;
        const r = 3.2;
        const eng = createRaptorVacuum(scene, engineMat, nozzleGlowMat);
        eng.parent = shipNode;
        eng.position.set(Math.cos(angle) * r, shipEngineY - 1, Math.sin(angle) * r);
        meshes.push(eng);
    }

    // ── SpaceX logo band ──
    const logoBand = MeshBuilder.CreateCylinder('logoBand', {
        diameterTop: 9.2, diameterBottom: 9.2, height: 0.8, tessellation: 48
    }, scene);
    const logoMat = new StandardMaterial('logoMat', scene);
    logoMat.diffuseColor = Color3.White();
    logoMat.emissiveColor = new Color3(0.15, 0.15, 0.15);
    logoBand.material = logoMat;
    logoBand.parent = shipNode;
    logoBand.position.y = shipBaseY + 28;
    meshes.push(logoBand);

    // ── Payload door outline ──
    const doorFrame = MeshBuilder.CreateBox('doorFrame', { width: 5, height: 8, depth: 0.15 }, scene);
    const doorMat = new StandardMaterial('doorMat', scene);
    doorMat.diffuseColor = new Color3(0.6, 0.6, 0.6);
    doorMat.emissiveColor = new Color3(0.05, 0.05, 0.05);
    doorFrame.material = doorMat;
    doorFrame.parent = shipNode;
    doorFrame.position.set(4.6, shipBaseY + 24, 0);
    doorFrame.rotation.y = 0;
    meshes.push(doorFrame);

    return { meshes, booster: boosterRoot };
}

// Creates a single sea-level Raptor engine bell
function createRaptorEngine(scene: Scene, bodyMat: StandardMaterial, glowMat: StandardMaterial, diameter: number, height: number): Mesh {
    const eng = MeshBuilder.CreateCylinder('raptor', {
        diameterTop: diameter * 0.6, diameterBottom: diameter, height: height, tessellation: 16
    }, scene);
    eng.material = bodyMat;

    // Inner glow ring
    const innerRing = MeshBuilder.CreateTorus('raptorRing', {
        diameter: diameter * 0.7, thickness: 0.08, tessellation: 16
    }, scene);
    innerRing.material = glowMat;
    innerRing.parent = eng;
    innerRing.position.y = -height / 2;

    return eng;
}

// Creates a vacuum Raptor engine (much larger extended nozzle)
function createRaptorVacuum(scene: Scene, bodyMat: StandardMaterial, glowMat: StandardMaterial): Mesh {
    // Much larger bell for vacuum optimization
    const eng = MeshBuilder.CreateCylinder('raptorVac', {
        diameterTop: 1.0, diameterBottom: 2.4, height: 4, tessellation: 20
    }, scene);
    eng.material = bodyMat;

    // Extended nozzle (regeneratively cooled — slightly different color)
    const extension = MeshBuilder.CreateCylinder('vacExtension', {
        diameterTop: 2.4, diameterBottom: 2.8, height: 2.5, tessellation: 20
    }, scene);
    const extMat = new StandardMaterial('vacExtMat', scene);
    extMat.diffuseColor = new Color3(0.25, 0.2, 0.15);
    extMat.specularColor = new Color3(0.15, 0.1, 0.1);
    extension.material = extMat;
    extension.parent = eng;
    extension.position.y = -3.25;

    // Inner glow
    const innerRing = MeshBuilder.CreateTorus('vacRing', {
        diameter: 2.5, thickness: 0.1, tessellation: 20
    }, scene);
    innerRing.material = glowMat;
    innerRing.parent = eng;
    innerRing.position.y = -4.5;

    return eng;
}

// ═══════════════════════════════════════════════════════════════
// LAUNCH SITE — Mechazilla Tower + Orbital Launch Mount (OLM)
// Tower: ~145m tall (including lightning rod)
// OLM: ~30m tall steel-reinforced platform
// ═══════════════════════════════════════════════════════════════

function buildLaunchSite(scene: Scene) {
    const towerMat = new StandardMaterial('towerMat', scene);
    towerMat.diffuseColor = new Color3(0.45, 0.4, 0.35);
    towerMat.specularColor = new Color3(0.15, 0.15, 0.15);

    const concreteMat = new StandardMaterial('concreteMat', scene);
    concreteMat.diffuseColor = new Color3(0.35, 0.33, 0.3);
    concreteMat.specularColor = Color3.Black();

    const steelPlateMat = new StandardMaterial('steelPlate', scene);
    steelPlateMat.diffuseColor = new Color3(0.4, 0.38, 0.35);
    steelPlateMat.specularColor = new Color3(0.2, 0.2, 0.2);

    // ── Orbital Launch Mount (OLM) ──
    // Large reinforced steel/concrete platform ~30m tall
    const olmBase = MeshBuilder.CreateBox('olmBase', { width: 25, height: 4, depth: 25 }, scene);
    olmBase.material = concreteMat;
    olmBase.position.y = 2;

    // OLM support columns (4 massive legs)
    for (let i = 0; i < 4; i++) {
        const col = MeshBuilder.CreateBox('olmCol' + i, { width: 4, height: 28, depth: 4 }, scene);
        col.material = steelPlateMat;
        const xOff = (i % 2 === 0 ? 1 : -1) * 8;
        const zOff = (i < 2 ? 1 : -1) * 8;
        col.position.set(xOff, 14, zOff);
    }

    // OLM top plate (launch table) where the rocket sits
    const olmTop = MeshBuilder.CreateBox('olmTop', { width: 22, height: 3, depth: 22 }, scene);
    olmTop.material = steelPlateMat;
    olmTop.position.y = 29;

    // Flame deflector / water-cooled steel plate
    const deflector = MeshBuilder.CreateBox('deflector', { width: 20, height: 1.5, depth: 20 }, scene);
    const deflectorMat = new StandardMaterial('deflMat', scene);
    deflectorMat.diffuseColor = new Color3(0.25, 0.22, 0.2);
    deflector.material = deflectorMat;
    deflector.position.y = 27;

    // Hold-down clamp ring (simplified)
    const clampRing = MeshBuilder.CreateTorus('clampRing', {
        diameter: 10, thickness: 0.6, tessellation: 32
    }, scene);
    clampRing.material = steelPlateMat;
    clampRing.position.y = 30.5;

    // Individual hold-down clamps (20 clamps around circumference)
    for (let i = 0; i < 20; i++) {
        const angle = (i * Math.PI * 2) / 20;
        const clamp = MeshBuilder.CreateBox('clamp' + i, { width: 0.8, height: 1.5, depth: 0.5 }, scene);
        clamp.material = steelPlateMat;
        clamp.position.set(Math.cos(angle) * 5, 31, Math.sin(angle) * 5);
        clamp.rotation.y = angle;
    }

    // ── Mechazilla Tower (145m tall steel truss tower) ──
    // Main tower structure — 4 vertical rails with cross bracing
    const towerX = -18;
    const towerHeight = 140;

    // 4 vertical columns
    const colPositions = [
        [towerX - 2.5, 0, -2.5],
        [towerX + 2.5, 0, -2.5],
        [towerX - 2.5, 0, 2.5],
        [towerX + 2.5, 0, 2.5]
    ];

    for (let i = 0; i < 4; i++) {
        const col = MeshBuilder.CreateBox('towerCol' + i, { width: 1.5, height: towerHeight, depth: 1.5 }, scene);
        col.material = towerMat;
        col.position.set(colPositions[i][0], towerHeight / 2, colPositions[i][2]);
    }

    // Cross bracing (every 15m)
    for (let level = 0; level < 9; level++) {
        const y = 10 + level * 15;

        // Horizontal beams
        const beam1 = MeshBuilder.CreateBox('hBeam' + level + 'a', { width: 5, height: 0.5, depth: 0.5 }, scene);
        beam1.material = towerMat;
        beam1.position.set(towerX, y, 0);

        const beam2 = MeshBuilder.CreateBox('hBeam' + level + 'b', { width: 0.5, height: 0.5, depth: 5 }, scene);
        beam2.material = towerMat;
        beam2.position.set(towerX, y, 0);

        // Diagonal brace (X-pattern, simplified)
        if (level % 2 === 0) {
            const diag = MeshBuilder.CreateBox('diag' + level, { width: 0.3, height: 18, depth: 0.3 }, scene);
            diag.material = towerMat;
            diag.position.set(towerX, y + 7.5, 0);
            diag.rotation.z = 0.35;
        }
    }

    // ── Chopstick Arms (2, 36m long tubular steel truss) ──
    const chopstickMat = new StandardMaterial('chopMat', scene);
    chopstickMat.diffuseColor = new Color3(0.5, 0.45, 0.4);
    chopstickMat.specularColor = new Color3(0.2, 0.2, 0.2);

    const chopstickY = 85; // Arms positioned about halfway up tower

    for (let i = 0; i < 2; i++) {
        const armRoot = new TransformNode('chopstick' + i, scene);
        armRoot.position.set(towerX, chopstickY, (i - 0.5) * 7);

        // Main arm beam (36m long)
        const arm = MeshBuilder.CreateBox('arm' + i, { width: 36, height: 2, depth: 1.8 }, scene);
        arm.material = chopstickMat;
        arm.parent = armRoot;
        arm.position.x = 18 + 2.5; // Extends from tower toward rocket

        // Arm top rail
        const rail = MeshBuilder.CreateBox('armRail' + i, { width: 20, height: 0.3, depth: 0.5 }, scene);
        const railMat = new StandardMaterial('railMat' + i, scene);
        railMat.diffuseColor = new Color3(0.6, 0.55, 0.5);
        rail.material = railMat;
        rail.parent = armRoot;
        rail.position.set(28, 1.2, 0);

        // Arm support strut (triangular brace back to tower)
        const strut = MeshBuilder.CreateBox('armStrut' + i, { width: 0.6, height: 25, depth: 0.6 }, scene);
        strut.material = towerMat;
        strut.parent = armRoot;
        strut.position.set(8, -10, 0);
        strut.rotation.z = 0.5;
    }

    // ── QD (Quick Disconnect) Arm ──
    const qdY = 105; // Higher up, connects to the ship
    const qdArm = MeshBuilder.CreateBox('qdArm', { width: 22, height: 1.5, depth: 2 }, scene);
    const qdMat = new StandardMaterial('qdMat', scene);
    qdMat.diffuseColor = new Color3(0.55, 0.5, 0.45);
    qdArm.material = qdMat;
    qdArm.position.set(towerX + 13, qdY, 0);

    // QD connector plate at end
    const qdPlate = MeshBuilder.CreateBox('qdPlate', { width: 2, height: 3, depth: 2.5 }, scene);
    qdPlate.material = steelPlateMat;
    qdPlate.position.set(towerX + 24, qdY, 0);

    // ── Lightning rod (10m tall, top of tower) ──
    const rod = MeshBuilder.CreateCylinder('lightningRod', {
        diameterTop: 0.08, diameterBottom: 0.3, height: 12, tessellation: 8
    }, scene);
    rod.material = towerMat;
    rod.position.set(towerX, towerHeight + 6, 0);

    // ── Platform/carriage for chopsticks (on tower rails) ──
    const carriage = MeshBuilder.CreateBox('carriage', { width: 7, height: 4, depth: 7 }, scene);
    carriage.material = steelPlateMat;
    carriage.position.set(towerX, chopstickY - 3, 0);

    // ── Ground-level infrastructure ──
    // Concrete pad
    const padGround = MeshBuilder.CreateBox('padGround', { width: 60, height: 0.5, depth: 60 }, scene);
    padGround.material = concreteMat;
    padGround.position.y = 0.25;

    // Propellant tank farm (simplified cylindrical tanks)
    const tankMat = new StandardMaterial('tankMat', scene);
    tankMat.diffuseColor = Color3.White();
    tankMat.specularColor = new Color3(0.3, 0.3, 0.3);

    for (let i = 0; i < 4; i++) {
        const tank = MeshBuilder.CreateCylinder('fuelTank' + i, {
            diameterTop: 5, diameterBottom: 5, height: 20, tessellation: 20
        }, scene);
        tank.material = tankMat;
        tank.position.set(-40 + i * 8, 10, 35);

        // Tank dome
        const dome = MeshBuilder.CreateSphere('tankDome' + i, {
            diameter: 5, slice: 0.5, segments: 16
        }, scene);
        dome.material = tankMat;
        dome.position.set(-40 + i * 8, 20, 35);
    }
}

// ── Exhaust particles ──
function createExhaustParticles(scene: Scene, emitter: Mesh): ParticleSystem {
    const ps = new ParticleSystem('exhaust', 5000, scene);
    ps.createPointEmitter(new Vector3(-2, -1, -2), new Vector3(2, -1, 2));

    ps.color1 = new Color4(1, 0.7, 0.15, 1);
    ps.color2 = new Color4(1, 0.4, 0.08, 0.9);
    ps.colorDead = new Color4(0.4, 0.3, 0.2, 0);

    ps.minSize = 2;
    ps.maxSize = 6;
    ps.minLifeTime = 0.3;
    ps.maxLifeTime = 1.2;
    ps.emitRate = 0;
    ps.blendMode = ParticleSystem.BLENDMODE_ADD;

    ps.minEmitPower = 25;
    ps.maxEmitPower = 50;
    ps.updateSpeed = 0.02;

    ps.gravity = new Vector3(0, -8, 0);
    ps.emitter = emitter;

    ps.start();
    return ps;
}

// ── Starfield ──
function createStarfield(scene: Scene) {
    const starMat = new StandardMaterial('starMat', scene);
    starMat.emissiveColor = Color3.White();
    starMat.disableLighting = true;

    for (let i = 0; i < 600; i++) {
        const star = MeshBuilder.CreateSphere('star' + i, { diameter: 0.3 + Math.random() * 0.6 }, scene);
        star.material = starMat;
        const r = 800 + Math.random() * 700;
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

    const moonGround = MeshBuilder.CreateGround('moonGround', { width: 800, height: 800, subdivisions: 80 }, scene);
    const moonMat = new StandardMaterial('moonMat', scene);
    moonMat.diffuseColor = new Color3(0.5, 0.48, 0.44);
    moonMat.specularColor = new Color3(0.05, 0.05, 0.05);
    moonGround.material = moonMat;
    moonGround.parent = root;

    // Craters (more realistic spread)
    for (let i = 0; i < 30; i++) {
        const radius = 4 + Math.random() * 15;
        const crater = MeshBuilder.CreateDisc('crater' + i, { radius: radius, tessellation: 28 }, scene);
        const cMat = new StandardMaterial('craterMat' + i, scene);
        cMat.diffuseColor = new Color3(0.38, 0.36, 0.32);
        cMat.specularColor = Color3.Black();
        crater.material = cMat;
        crater.parent = root;
        crater.rotation.x = Math.PI / 2;
        crater.position.set(
            (Math.random() - 0.5) * 500,
            0.05,
            (Math.random() - 0.5) * 500
        );

        // Crater rim
        const rim = MeshBuilder.CreateTorus('rim' + i, {
            diameter: radius * 2, thickness: radius * 0.15, tessellation: 24
        }, scene);
        const rimMat = new StandardMaterial('rimMat' + i, scene);
        rimMat.diffuseColor = new Color3(0.52, 0.5, 0.46);
        rim.material = rimMat;
        rim.parent = root;
        rim.position.set(crater.position.x, 0.2, crater.position.z);
        rim.rotation.x = Math.PI / 2;
    }

    return root;
}
