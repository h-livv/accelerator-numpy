"""Relativistic Boris magnetic rotation.

Positions are metres, velocities are metres per second, fields are tesla.
Charge is in units of the elementary charge (a proton is +1). Mass is kilograms.

The proper velocity is rotated by the local B field so that the speed stays
constant.
"""

import numpy as np

from elements import _as_rows

C_LIGHT = 299792458.0
ELEMENTARY_CHARGE = 1.602176634e-19
PROTON_MASS_KG = 1.67262192369e-27
PROTON_REST_ENERGY_MEV = 938.2720813


def speed_from_kinetic_energy(kinetic_energy_mev, rest_energy_mev=PROTON_REST_ENERGY_MEV):
    """Speed in metres per second for a kinetic energy in MeV."""
    if kinetic_energy_mev <= 0.0:
        raise ValueError("kinetic energy must be positive")
    gamma = 1.0 + float(kinetic_energy_mev) / float(rest_energy_mev)
    return C_LIGHT * np.sqrt(1.0 - 1.0 / gamma ** 2)


def boris_push(velocity, field, charge, mass, dt):
    """Advance velocity by one relativistic Boris magnetic rotation.

    velocity and field have shape (3,) or (N, 3). charge is in units of e
    and mass is in kilograms; each is a scalar or one value per particle.
    dt is in seconds. A positive dt rotates forward. A negative dt rotates
    backward, which is how the leapfrog stagger is started.

    The electric field is zero, so the speed of each particle is unchanged.
    """
    velocity, single = _as_rows(velocity, "velocity")
    field, _ = _as_rows(field, "field")
    if velocity.shape != field.shape:
        raise ValueError("velocity and field must have the same shape")

    charges = _per_particle(charge, velocity.shape[0], "charge") * ELEMENTARY_CHARGE
    masses = _per_particle(mass, velocity.shape[0], "mass")
    c2 = C_LIGHT ** 2

    speed2 = np.sum(velocity ** 2, axis=1, keepdims=True)
    if np.any(speed2 >= c2):
        raise ValueError("speed must be below the speed of light")

    gamma = 1.0 / np.sqrt(1.0 - speed2 / c2)
    proper_velocity = gamma * velocity

    # t = tan(φ/2), where φ is the rotation about B over this step.
    t_vector = charges * field * dt / (2.0 * gamma * masses)
    t_squared = np.sum(t_vector ** 2, axis=1, keepdims=True)
    s_vector = 2.0 * t_vector / (1.0 + t_squared)

    primed = proper_velocity + np.cross(proper_velocity, t_vector)
    rotated = proper_velocity + np.cross(primed, s_vector)

    gamma_new = np.sqrt(1.0 + np.sum(rotated ** 2, axis=1, keepdims=True) / c2)
    updated = rotated / gamma_new
    if single:
        return updated[0]
    return updated


def _per_particle(values, n_particles, name):
    """Broadcast a scalar or per-particle array to shape (n_particles, 1)."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(n_particles, float(array))
    array = np.reshape(array, -1)
    if array.shape != (n_particles,):
        raise ValueError(f"{name} must be a scalar or length {n_particles}")
    return array.reshape(n_particles, 1)
