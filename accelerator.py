"""A relativistic beamline in NumPy: Boris pusher, one dipole, one quadrupole.

Positions are metres, velocities are metres per second, fields are tesla.
Charge is in units of the elementary charge (a proton is +1). Mass is kilograms.

The magnetic update is the relativistic Boris rotation: the proper velocity
is rotated by the local B field so that the speed stays constant. The dipole
is a hard-edged uniform vertical field. The quadrupole is the hard-edged
linear field Bx = G y, By = G x.
"""

import numpy as np

from boris import ELEMENTARY_CHARGE, PROTON_MASS_KG, PROTON_REST_ENERGY_MEV, speed_from_kinetic_energy
from elements import beamline_length, dipole, quadrupole
from track import track


def main():
    """Track one 10 MeV proton through a dipole and then a quadrupole."""
    kinetic_energy_mev = 10.0
    speed = speed_from_kinetic_energy(kinetic_energy_mev)
    gamma = 1.0 + kinetic_energy_mev / PROTON_REST_ENERGY_MEV
    by = 0.5
    elements = [
        dipole(length=0.5, by=by),
        quadrupole(length=0.4, gradient=2.0),
    ]

    # A fraction of a degree of cyclotron rotation per step.
    omega = (ELEMENTARY_CHARGE * by) / (gamma * PROTON_MASS_KG)
    dt = 0.002 / omega

    position = np.array([0.0, 0.002, 0.0])
    velocity = np.array([0.0, 0.0, speed])
    history, final_velocity = track(
        position,
        velocity,
        charge=+1.0,
        mass=PROTON_MASS_KG,
        elements=elements,
        dt=dt,
    )

    print(f"proton  {kinetic_energy_mev:.1f} MeV   gamma {gamma:.6f}   speed {speed:.6e} m/s")
    print("beamline")
    for element in elements:
        if element["kind"] == "dipole":
            print(f"  dipole       {element['length']:.3f} m   By = {element['by']:.3f} T")
        else:
            print(f"  quadrupole   {element['length']:.3f} m   G = {element['gradient']:.3f} T/m")
    print(f"steps {len(history) - 1}   dt {dt:.6e} s")
    print()
    print(f"{'':<18} {'z (m)':>10} {'x (m)':>12} {'y (mm)':>12} {'angle (deg)':>12}")

    labels = ("start", "after dipole", "after quadrupole")
    z_marks = [0.0, elements[0]["length"], beamline_length(elements)]
    for label, z_mark in zip(labels, z_marks):
        index = int(np.argmax(history[:, 2] >= z_mark))
        x, y, z = history[index]
        if index == 0:
            step = velocity
        else:
            step = history[index] - history[index - 1]
        angle = np.degrees(np.arctan2(step[0], step[2]))
        print(f"{label:<18} {z:10.4f} {x:12.6f} {y * 1e3:12.4f} {angle:12.3f}")

    speed_in = np.linalg.norm(velocity)
    speed_out = np.linalg.norm(final_velocity)
    print()
    print(f"speed change {abs(speed_out - speed_in) / speed_in:.3e}")


if __name__ == "__main__":
    main()
