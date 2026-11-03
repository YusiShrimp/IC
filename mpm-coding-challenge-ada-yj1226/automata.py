"""Cellular automata: Lorenz 96 and Conway's Game of Life.

Base version (Part 1): write this yourself, without AI tools. See
README.md for how the day runs and SPEC.md for what to implement.
"""
import numpy as np


def lorenz96(initial_state, nsteps, constants=(1/101, 100, 8)):

    alpha, beta, gamma = constants
    x = np.array(initial_state, dtype=float)
    for step in range(nsteps):
        y = np.zeros(len(x))
        for i in range(len(x)):
            x_i = x[i]
            x_im1 = x[(i - 1) % len(x)]
            x_im2 = x[(i - 2) % len(x)]
            x_ip1 = x[(i + 1) % len(x)]

            y[i] = alpha * (beta * x_i + (x_im2 - x_ip1) * x_im1 + gamma)

        x = y

    return x


def life(initial_state, nsteps, rules="basic", periodic=False):

    if rules != "basic":
        raise ValueError("Unknown rules")

    state = np.array(initial_state, copy=True)
    rows, cols = state.shape

    for step in range(nsteps):

        new_state = np.zeros_like(state)

        for i in range(rows):
            for j in range(cols):

                live_neighbours = 0

                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):

                        if di == 0 and dj == 0:
                            continue

                        ni = i + di
                        nj = j + dj

                        if (ni < 0 or ni >= rows
                                or nj < 0 or nj >= cols):
                            continue

                        if state[ni, nj] != 0:
                            live_neighbours += 1

                if state[i, j] != 0:
                    if live_neighbours == 2 or live_neighbours == 3:
                        new_state[i, j] = 1
                else:
                    if live_neighbours == 3:
                        new_state[i, j] = 1

        state = new_state

    return state
