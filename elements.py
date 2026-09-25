"""Hard-edged beamline elements and their magnetic field."""

import numpy as np


def dipole(length, by):
    """Uniform vertical dipole.

    length: metres along z. by: By in tesla. Positive By bends a positive
    charge toward negative x.
    """
    length = float(length)
    if length <= 0.0:
        raise ValueError("dipole length must be positive")
    return {"kind": "dipole", "length": length, "by": float(by)}


def quadrupole(length, gradient):
    """Linear quadrupole.

    length: metres along z. gradient: G = dBy/dx in tesla per metre.
    Inside the magnet, Bx = G y and By = G x. For a positive charge and
    G > 0 this focuses in x and defocuses in y.
    """
    length = float(length)
    if length <= 0.0:
        raise ValueError("quadrupole length must be positive")
    return {"kind": "quadrupole", "length": length, "gradient": float(gradient)}


def beamline_length(elements):
    """Total length of the beamline in metres."""
    return float(sum(element["length"] for element in elements))


def magnetic_field(position, elements):
    """Magnetic field of a beamline at each position.

    Elements sit end to end along z, starting at z = 0. The edge at the
    exit of one magnet belongs to the next magnet. Outside the beamline
    the field is zero.

    position: shape (3,) or (N, 3). Returns B with the same shape.
    """
    coordinates, single = _as_rows(position, "position")
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    z = coordinates[:, 2]
    bx = np.zeros(len(coordinates))
    by = np.zeros(len(coordinates))
    bz = np.zeros(len(coordinates))

    z0 = 0.0
    for element in elements:
        z1 = z0 + element["length"]
        inside = (z >= z0) & (z < z1)
        kind = element["kind"]
        if kind == "dipole":
            by = np.where(inside, element["by"], by)
        elif kind == "quadrupole":
            gradient = element["gradient"]
            bx = np.where(inside, gradient * y, bx)
            by = np.where(inside, gradient * x, by)
        else:
            raise ValueError(f"unknown element kind {kind!r}")
        z0 = z1

    field = np.column_stack((bx, by, bz))
    if single:
        return field[0]
    return field


def _as_rows(values, name):
    """Return values as an (N, 3) array and whether the input was a single vector."""
    array = np.asarray(values, dtype=float)
    if array.shape == (3,):
        return array.reshape(1, 3), True
    if array.ndim == 2 and array.shape[1] == 3:
        return array, False
    raise ValueError(f"{name} must have shape (3,) or (N, 3)")
