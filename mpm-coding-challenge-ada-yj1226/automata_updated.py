"""Cellular automata: Lorenz 96 and Conway's Game of Life.

AI-updated version (Part 2). See README.md and SPEC.md for details.
"""
import numpy as np
from scipy.ndimage import convolve


def lorenz96(initial_state, nsteps, constants=(1 / 101, 100, 8)):
    """Perform iterations of the Lorenz 96 update.

    The state is updated synchronously with periodic boundaries.
    """
    alpha, beta, gamma = constants
    state = np.array(initial_state, dtype=np.float64, copy=True)

    for _ in range(nsteps):
        previous = np.roll(state, 1)
        previous_two = np.roll(state, 2)
        following = np.roll(state, -1)
        state = alpha * (
            beta * state + (previous_two - following) * previous + gamma
        )

    return state


def _neighbour_kernel(ndim):
    """Return a kernel that counts every neighbour except the centre."""
    kernel = np.ones((3,) * ndim, dtype=np.int8)
    kernel[(1,) * ndim] = 0
    return kernel


def _count_neighbours(cells, kernel, periodic):
    """Count live neighbours, treating nonzero cells as alive."""
    mode = "wrap" if periodic else "constant"
    return convolve(cells.astype(np.int16, copy=False), kernel, mode=mode)


def life(initial_state, nsteps, rules="basic", periodic=False):
    """Perform iterations of Conway's Game of Life.

    Supports basic 2D Life, two-colour 2D Life, and the specified 3D
    Life rules. Neighbour counts are computed by SciPy's compiled
    convolution routine rather than Python loops over individual cells.
    """
    if rules not in ("basic", "2colour", "3d"):
        raise ValueError("rules must be 'basic', '2colour' or '3d'")

    state = np.array(initial_state, copy=True)
    expected_ndim = 3 if rules == "3d" else 2
    if state.ndim != expected_ndim:
        raise ValueError(f"rules={rules!r} requires a {expected_ndim}D grid")

    if rules == "2colour":
        state = state.astype(np.int8, copy=False)
    elif state.dtype == np.bool_:
        state = state.astype(bool, copy=False)
    else:
        state = state.astype(np.int8, copy=False)

    kernel = _neighbour_kernel(expected_ndim)

    for _ in range(nsteps):
        alive = state != 0
        neighbours = _count_neighbours(alive, kernel, periodic)

        if rules == "3d":
            next_state = (alive & ((neighbours == 5) | (neighbours == 6)))
            next_state |= (~alive & (neighbours == 4))
            state = next_state.astype(state.dtype, copy=False)
            continue

        survives = alive & ((neighbours == 2) | (neighbours == 3))
        births = ~alive & (neighbours == 3)

        if rules == "2colour":
            positive = _count_neighbours(state == 1, kernel, periodic)
            negative = _count_neighbours(state == -1, kernel, periodic)
            next_state = np.zeros(state.shape, dtype=np.int8)
            next_state[survives] = state[survives]
            next_state[births & (positive > negative)] = 1
            next_state[births & (negative > positive)] = -1
            state = next_state
        else:
            next_state = np.zeros(state.shape, dtype=state.dtype)
            next_state[survives | births] = 1
            state = next_state

    return state
