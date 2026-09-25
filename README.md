# Particle Accelerator from Scratch

### A small accelerator-physics simulation written from scratch in Python + NumPy.

> I'm using this project to learn **accelerator physics, beam optics, and beam dynamics** by implementing the underlying physics myself.

## Current

The current simulation tracks a **10 MeV proton** through a simple magnetic beamline:

```text
Dipole → Quadrupole
```

- 0.5 m hard-edge dipole, `By = 0.5 T`
- 0.4 m hard-edge quadrupole, `G = 2 T/m`
- Relativistic **Boris integrator**
- No electric field, so the magnetic field only changes the particle's direction
- Checks that the particle's speed remains constant up to numerical roundoff

### Files

```text
elements.py       # Dipole, quadrupole, magnetic fields
boris.py          # Constants, relativistic velocity, Boris solver
track.py          # Beamline tracking
accelerator.py    # Example simulation
```

## Run

Requirements:

```bash
pip install numpy
```

Then:

```bash
python accelerator.py
```

The program prints the particle position after each magnet and the relative speed change.

## Planned

The project will gradually move from single-particle tracking toward beam optics and beam dynamics:

- FODO lattice
- Phase-space tracking
- Transfer matrices
- Twiss parameters and emittance
- RF cavities and longitudinal dynamics
- Energy-recovery linac

The goal is to learn the physics by building the accelerator model from the ground up.