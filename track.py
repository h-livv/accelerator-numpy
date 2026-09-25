"""Leapfrog tracking through a beamline."""

import numpy as np

from boris import _per_particle, boris_push
from elements import _as_rows, beamline_length, magnetic_field


def track(position, velocity, charge, mass, elements, dt, max_steps=100000):
    """Track particles through a beamline with a leapfrog Boris step.

    The stored positions are at times 0, dt, 2 dt, ... The velocity used
    inside the loop sits halfway between those times. The returned velocity
    is brought back onto the time of the last position.

    position and velocity: shape (3,) or (N, 3). charge is in units of e.
    mass is kilograms. dt is seconds and must be positive.

    Returns positions of shape (n + 1, N, 3), or (n + 1, 3) for one particle,
    and the velocity at the last position. Tracking stops when every particle
    has z beyond the end of the last magnet, or after max_steps.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if not elements:
        raise ValueError("beamline needs at least one element")

    positions, single = _as_rows(position, "position")
    velocities, velocity_single = _as_rows(velocity, "velocity")
    if positions.shape != velocities.shape or single != velocity_single:
        raise ValueError("position and velocity must have the same shape")

    positions = positions.copy()
    velocities = velocities.copy()
    n_particles = positions.shape[0]
    _per_particle(charge, n_particles, "charge")
    _per_particle(mass, n_particles, "mass")

    end = beamline_length(elements)
    # Place velocity at t = -dt/2 so the first full step lands on t = +dt/2.
    velocities = boris_push(
        velocities,
        magnetic_field(positions, elements),
        charge,
        mass,
        -0.5 * dt,
    )

    history = [positions.copy()]
    for _ in range(max_steps):
        if np.all(positions[:, 2] >= end):
            break
        velocities = boris_push(
            velocities,
            magnetic_field(positions, elements),
            charge,
            mass,
            dt,
        )
        positions = positions + velocities * dt
        history.append(positions.copy())

    # Bring velocity from the last half-step onto the time of the last position.
    velocities = boris_push(
        velocities,
        magnetic_field(positions, elements),
        charge,
        mass,
        0.5 * dt,
    )

    history = np.stack(history)
    if single:
        return history[:, 0, :], velocities[0]
    return history, velocities
