"""Tests for the automata module.

Run ``pytest tests/`` for automata.py or ``pytest -U tests/`` for the
AI-updated version.
"""
import numpy as np
import pytest

import automata


class TestLorenz96:
    def test_steady_state(self):
        """Every cell 8 is a steady state."""
        x = np.full(10, 8.0)
        np.testing.assert_allclose(automata.lorenz96(x, 10), x)

    def test_all_sevens(self):
        """Every cell 7 becomes 708/101 after one step."""
        x = np.full(10, 7.0)
        expected = np.full(10, 708 / 101)
        np.testing.assert_allclose(automata.lorenz96(x, 1), expected)

    @pytest.mark.parametrize("i", [4, 0, 9])
    def test_one_raised_cell(self, i):
        """One raised cell checks ordinary and wrapped indices."""
        x = np.full(10, 8.0)
        x[i] = 9.0
        expected = np.full(10, 8.0)
        expected[(i - 1) % 10] = 800 / 101
        expected[i] = 908 / 101
        expected[(i + 2) % 10] = 816 / 101
        np.testing.assert_allclose(automata.lorenz96(x, 1), expected)

    def test_input_not_modified(self):
        """lorenz96 doesn't change its input."""
        x = np.linspace(7.0, 9.0, 10)
        original = x.copy()
        automata.lorenz96(x, 5)
        np.testing.assert_array_equal(x, original)

    def test_zero_steps_returns_float_copy(self):
        """Zero steps preserve values and return a new float array."""
        x = np.array([1, 2, 3])
        result = automata.lorenz96(x, 0)
        np.testing.assert_array_equal(result, x)
        assert result.dtype == np.float64
        assert not np.shares_memory(result, x)


class TestLife:
    def test_blinker(self):
        """A horizontal blinker turns vertical after one step."""
        grid = np.zeros((5, 5), dtype=int)
        grid[2, 1:4] = 1
        expected = np.zeros((5, 5), dtype=int)
        expected[1:4, 2] = 1
        np.testing.assert_array_equal(automata.life(grid, 1), expected)

    def test_block_is_still(self):
        """A 2x2 block never changes."""
        grid = np.zeros((4, 4), dtype=int)
        grid[1:3, 1:3] = 1
        np.testing.assert_array_equal(automata.life(grid, 3), grid)

    def test_glider_moves(self):
        """A glider moves diagonally after four steps."""
        grid = np.zeros((8, 8), dtype=int)
        grid[3, 2:5] = 1
        grid[4, 4] = 1
        grid[5, 3] = 1
        expected = np.roll(grid, (-1, 1), axis=(0, 1))
        np.testing.assert_array_equal(automata.life(grid, 4), expected)

    def test_zero_steps(self):
        """With nsteps=0, the starting grid comes back unchanged."""
        grid = np.zeros((4, 6), dtype=int)
        grid[1, 1:4] = 1
        np.testing.assert_array_equal(automata.life(grid, 0), grid)

    def test_unknown_rules(self):
        """An unknown rules value raises ValueError."""
        with pytest.raises(ValueError):
            automata.life(np.zeros((4, 4), dtype=int), 1, rules="hexagonal")

    def test_non_square_grid_updates(self):
        """A blinker works on a non-square grid."""
        grid = np.zeros((5, 8), dtype=int)
        grid[2, 2:5] = 1
        expected = np.zeros_like(grid)
        expected[1:4, 3] = 1
        np.testing.assert_array_equal(automata.life(grid, 1), expected)


class TestLifePeriodic:
    def test_blinker_across_edge(self):
        """A blinker split across edges turns vertical."""
        grid = np.zeros((5, 5), dtype=int)
        grid[2, [4, 0, 1]] = 1
        expected = np.zeros((5, 5), dtype=int)
        expected[1:4, 0] = 1
        np.testing.assert_array_equal(
            automata.life(grid, 1, periodic=True), expected
        )

    def test_corner_birth_wraps_both_axes(self):
        """A corner sees neighbours across both periodic edges."""
        grid = np.zeros((5, 6), dtype=int)
        grid[4, 5] = 1
        grid[4, 0] = 1
        grid[0, 5] = 1
        assert automata.life(grid, 1, periodic=True)[0, 0] == 1


class TestLife2Colour:
    def test_birth_takes_majority_colour(self):
        """A new cell takes the majority colour of its neighbours."""
        grid = np.zeros((5, 5), dtype=int)
        grid[1, 1:4] = [1, 1, -1]
        assert automata.life(grid, 1, rules="2colour")[2, 2] == 1

    def test_birth_takes_negative_majority(self):
        """A birth uses the negative colour when it is the majority."""
        grid = np.zeros((5, 5), dtype=int)
        grid[1, 1:4] = [-1, -1, 1]
        assert automata.life(grid, 1, rules="2colour")[2, 2] == -1


class TestLife3D:
    def test_lone_cell_dies(self):
        """A live cell with no neighbours dies."""
        grid = np.zeros((5, 5, 5), dtype=int)
        grid[2, 2, 2] = 1
        result = automata.life(grid, 1, rules="3d")
        np.testing.assert_array_equal(result, np.zeros((5, 5, 5)))

    def test_dead_cell_with_four_neighbours_is_born(self):
        """A dead 3D cell is born with exactly four neighbours."""
        grid = np.zeros((5, 5, 5), dtype=int)
        grid[1, 2, 2] = 1
        grid[3, 2, 2] = 1
        grid[2, 1, 2] = 1
        grid[2, 2, 3] = 1
        assert automata.life(grid, 1, rules="3d")[2, 2, 2] == 1
