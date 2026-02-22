# Starship Simulation — Design Requirements

All user requirements for the Starship HLS mission simulation.

## Physics Engine
- **Real physics** — no scripted trajectories. Orbits emerge from Newtonian gravity + thrust.
- **Gravity must be exactly Earth's**: g = 9.81 m/s² at surface.
- **Thrust must be realistic**: match real Starship TWR (~1.5 at launch).
- **Center of mass** for the rocket — each stage has its own CoM.
- **Thrust comes from the bottom** (engine position), creating torque relative to CoM.
- **Centrifugal force / orbital mechanics**: ship should rotate in orbit naturally.
- **Feel like Kerbal Space Program**: real planet, realistic sizes, real physics.
- **Even if it takes long, use time warp** — don't compress physics, compress time.
- **Fast-forward everything except launch and moon landing** — only ignition→separation and landing-burn→exploration play in real time; everything else (orbit, TLI, coast, LOI) is time-warped.

## Scale & Telemetry
- All displayed values (altitude, velocity, downrange) must be **scaled to real-life** using the game-to-Earth ratio.
- Planet sizes should maintain **realistic ratios** (Moon/Earth radius = 0.273, mass ratio = 1/81.3).

## Visual
- Cloud layer the rocket punches through.
- Atmosphere and cloud spheres proportional to Earth diameter.
- Engine fire, exhaust plume, launch smoke.
- Sky transitions from blue → black in space.
- Booster separation and return to pad.

## Mission Profile
- Full Starship HLS sequence: ignition → liftoff → max-Q → MECO → separation → SES → orbit → TLI → coast → LOI → powered descent → touchdown → EVA → exploration.
- LOI burn is a separate step from powered descent.

## Technical
- Fix WebGL texture errors (no mipmaps on 1×1 textures).
- Keep all requirements in this prompt file for reference.
