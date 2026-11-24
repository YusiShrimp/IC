import pytest

import pandas as pd
import numpy as np

from deepimpact import Planet


class TestCalculateEnergy:
    """Tests for the function calculate_energy in solver.py"""

    @pytest.mark.parametrize("result", [
            (pd.DataFrame({
                "velocity": [10, 5],
                "mass": [10, 8],
                "angle": [1, 1],
                "altitude": [100, 90],
                "horizontal distance": [1, 1],
                "radius": [1, 1],
                "spreading rate": [1, 1],
                }))
            ])
    def test_output_format(self, result):
        test_planet = Planet()
        output = test_planet.calculate_energy(result)
        size_input = result.shape
        size_output = output.shape
        is_dataFrame = isinstance(output, pd.DataFrame)
        correct_shape = (size_output == (size_input[0], size_input[1] + 1))
        assert is_dataFrame and correct_shape

    @pytest.mark.parametrize("result", [
            (pd.DataFrame({
                "velocity": [10, 5],
                "mass": [10, 8],
                "angle": [1, 1],
                "altitude": [100, 90],
                "horizontal distance": [1, 1],
                "radius": [1, 1],
                "spreading rate": [1, 1],
                }))
            ])
    def test_unchanged_result(self, result):
        original = result.copy(deep=True)
        output = Planet().calculate_energy(result)
        assert (
            result.equals(original)
            and output[original.columns].equals(original)
            and "dedz" in output.columns
        )

    @pytest.mark.parametrize("result, expected", [
            (pd.DataFrame({
                "velocity": [10, 10, 10],
                "mass": [10, 10, 10],
                "angle": [1, 1, 1],
                "altitude": [100, 90, 80],
                "horizontal distance": [1, 1, 1],
                "radius": [1, 1, 1],
                "spreading rate": [1, 1, 1],
                }),
                [0, 0, 0])
            ])
    def test_constant_velocity_mass(self, result, expected):
        test_planet = Planet()
        output = test_planet.calculate_energy(result)
        assert output['dedz'].eq(expected).all()

    @pytest.mark.parametrize("result", [
            (pd.DataFrame({
                "velocity": [10, 8, 6],
                "mass": [3, 2, 1],
                "angle": [1, 1, 1],
                "altitude": [100, 100, 100],
                "horizontal distance": [1, 1, 1],
                "radius": [1, 1, 1],
                "spreading rate": [1, 1, 1],
                }))
            ])
    def test_constant_altitude(self, result):
        test_planet = Planet()
        output = test_planet.calculate_energy(result)
        assert output["dedz"].isna().all()

    @pytest.mark.parametrize("result, expected", [
            (pd.DataFrame({
                "velocity": [20000, 15000, 1000],
                "mass": [10000, 8000, 6000],
                "angle": [1, 1, 1],
                "altitude": [50000, 40000, 30000],
                "horizontal distance": [1, 1, 1],
                "radius": [1, 1, 1],
                "spreading rate": [1, 1, 1],
                }),
                [0.026290627, 0.02386471915, 0.02143881129])
            ])
    def test_airburst_case(self, result, expected):
        test_planet = Planet()
        output = test_planet.calculate_energy(result)
        assert np.allclose(output['dedz'], expected)

    @pytest.mark.parametrize("result, expected", [
            (pd.DataFrame({
                "velocity": [25000, 24000, 23000],
                "mass": [10000, 10000, 10000],
                "angle": [1, 1, 1],
                "altitude": [20000, 10000, 0],
                "horizontal distance": [1, 1, 1],
                "radius": [1, 1, 1],
                "spreading rate": [1, 1, 1],
                }),
                [0.00585563965, 0.0057361368, 0.00561663395])
            ])
    def test_cratering_case(self, result, expected):
        test_planet = Planet()
        output = test_planet.calculate_energy(result)
        assert np.allclose(output['dedz'], expected)

    @pytest.mark.parametrize("result", [
            (pd.DataFrame({
                "mass": [10000, 10000, 10000],
                "angle": [1, 1, 1],
                "horizontal distance": [1, 1, 1],
                "radius": [1, 1, 1],
                "spreading rate": [1, 1, 1],
                }))
            ])
    def test_invalid_input_parameter(self, result):
        test_planet = Planet()
        with pytest.raises(ValueError):
            test_planet.calculate_energy(result)


class TestAnalyseOutcome:
    """Tests for the function analyse_outcome in solver.py"""

    @pytest.mark.parametrize("result", [
        (pd.DataFrame({
            "velocity": [25000, 24000, 23000],
            "mass": [10000, 10000, 10000],
            "angle": [1, 1, 1],
            "altitude": [20000, 10000, 0],
            "distance": [1, 1, 1],
            "radius": [1, 1, 1],
            "spreading rate": [1, 1, 1],
            "dedz": [0.0058556405, 0.0057361377, 0.0056166348],
            }))
        ])
    def test_output_format(self, result):
        test_planet = Planet()
        output = test_planet.analyse_outcome(result)
        expected_keys = {
            "outcome",
            "burst_peak_dedz",
            "burst_altitude",
            "burst_distance",
            "burst_energy",
            "burst_angle",
        }
        is_dict = isinstance(output, dict)
        correct_keys = set(output) == expected_keys
        assert is_dict and correct_keys

    @pytest.mark.parametrize("result", [
        (pd.DataFrame({
            "velocity": [25000, 24000, 23000],
            "mass": [10000, 10000, 10000],
            "angle": [1, 5, 1],
            "altitude": [20000, 10000, 0],
            "distance": [1, 10, 1],
            "radius": [1, 1, 1],
            "spreading rate": [1, 1, 1],
            "dedz": [5, 20, 3],
            }))
        ])
    def test_airburst_case(self, result):
        test_planet = Planet()
        output = test_planet.analyse_outcome(result)
        correct_dict = {
            "outcome": "Airburst",
            "burst_peak_dedz": 20,
            "burst_altitude": 10000,
            "burst_distance": 10,
            "burst_energy": 2.45e11/4.184e12,
            "burst_angle": 5,
        }
        assert correct_dict == output

    @pytest.mark.parametrize("result", [
        (pd.DataFrame({
            "velocity": [50000, 24000, 23000],
            "mass": [10000, 10000, 10000],
            "angle": [1, 2, 3],
            "altitude": [20000, 10000, 0],
            "distance": [1, 10, 100],
            "radius": [1, 1, 1],
            "spreading rate": [1, 1, 1],
            "dedz": [5, 7, 20],
            }))
        ])
    def test_cratering_case_total_kinetic(self, result):
        test_planet = Planet()
        output = test_planet.analyse_outcome(result)
        correct_dict = {
            "outcome": "Cratering",
            "burst_peak_dedz": 20,
            "burst_altitude": 0,
            "burst_distance": 100,
            "burst_energy": 9.855e12/4.184e12,
            "burst_angle": 3,
        }
        assert correct_dict == output

    @pytest.mark.parametrize("result", [
        (pd.DataFrame({
            "velocity": [25000, 24000, 23000],
            "mass": [10000, 10000, 10000],
            "angle": [1, 2, 3],
            "altitude": [20000, 10000, 0],
            "distance": [1, 10, 100],
            "radius": [1, 1, 1],
            "spreading rate": [1, 1, 1],
            "dedz": [5, 7, 20],
            }))
        ])
    def test_cratering_case_residual_kinetic(self, result):
        test_planet = Planet()
        output = test_planet.analyse_outcome(result)
        correct_dict = {
            "outcome": "Cratering",
            "burst_peak_dedz": 20,
            "burst_altitude": 0,
            "burst_distance": 100,
            "burst_energy": 2.645e12/4.184e12,
            "burst_angle": 3,
        }
        assert correct_dict == output

    @pytest.mark.parametrize("result", [
        (pd.DataFrame({
            "velocity": [25000, 24000, 23000],
            "mass": [10000, 10000, 10000],
            "angle": [1, 2, 3],
            "altitude": [20000, 10000, 0],
            "distance": [1, 10, 100],
            "radius": [1, 1, 1],
            "spreading rate": [1, 1, 1],
            "dedz": [5, 7, np.nan],
            }))
        ])
    def test_handles_peak_selection(self, result):
        test_planet = Planet()
        output = test_planet.analyse_outcome(result)
        correct_dict = {
            "outcome": "Airburst",
            "burst_peak_dedz": 7,
            "burst_altitude": 10000,
            "burst_distance": 10,
            "burst_energy": 2.45e11/4.184e12,
            "burst_angle": 2,
        }
        assert correct_dict == output

    @pytest.mark.parametrize("result", [
        (pd.DataFrame({
            "velocity": [25000, 24000, 23000],
            "mass": [10000, 10000, 10000],
            "angle": [1, 2, 3],
            "altitude": [20000, 10000, 0],
            "distance": [1, 10, 100],
            "radius": [1, 1, 1],
            "spreading rate": [1, 1, 1],
            "dedz": [np.nan, np.nan, np.nan],
            }))
        ])
    def test_invalid_dedz(self, result):
        test_planet = Planet()
        output = test_planet.analyse_outcome(result)
        correct_dict = {
            "outcome": "Unknown",
            "burst_peak_dedz": 0,
            "burst_altitude": 0,
            "burst_distance": 0,
            "burst_energy": 0,
            "burst_angle": 0,
        }
        assert correct_dict == output

    @pytest.mark.parametrize("result", [
        (pd.DataFrame({
            "velocity": [25000, 24000, 23000],
            "mass": [10000, 10000, 10000],
            "angle": [1, 2, 3],
            "altitude": [20000, 0.5, 0],
            "distance": [1, 10, 100],
            "radius": [1, 1, 1],
            "spreading rate": [1, 1, 1],
            "dedz": [5, 20, 3],
            }))
        ])
    def test_cratering_case_radius_contact(self, result):
        test_planet = Planet()
        output = test_planet.analyse_outcome(result)
        correct_dict = {
            "outcome": "Cratering",
            "burst_peak_dedz": 3,
            "burst_altitude": 0,
            "burst_distance": 100,
            "burst_energy": 2.645e12/4.184e12,
            "burst_angle": 3,
        }
        assert correct_dict == output

    @pytest.mark.parametrize("result", [
        (pd.DataFrame({
            "velocity": [25000, 24000, 23000],
            "mass": [10000, 10000, 10000],
            "angle": [1, 2, 3],
            "altitude": [20000, 0.5, 0],
            "distance": [1, 10, 100],
            "radius": [1, 1, 1],
            "spreading rate": [1, 1, 1],
            "dedz": [5, 10, 20],
            }))
        ])
    def test_cratering_case_peak_at_ground(self, result):
        test_planet = Planet()
        output = test_planet.analyse_outcome(result)
        correct_dict = {
            "outcome": "Cratering",
            "burst_peak_dedz": 20,
            "burst_altitude": 0,
            "burst_distance": 100,
            "burst_energy": 2.645e12 / 4.184e12,
            "burst_angle": 3,
        }
        assert correct_dict == output

    @pytest.mark.parametrize("result", [
        (pd.DataFrame({
            "velocity": [25000, 24000, 23000],
            "mass": [10000, 10000, 10000],
            "angle": [1, 2, 3],
            "altitude": [20000, 10000, 0],
            "distance": [1, 10, 100],
            "radius": [1, 1, 1],
            "spreading rate": [1, 1, 1],
            }))
        ])
    def test_invalid_input_parameter(self, result):
        test_planet = Planet()
        with pytest.raises(ValueError):
            test_planet.analyse_outcome(result)


class TestAnalyticalSolution:
    """Tests comparing the entry solver with its analytical reference."""

    def test_solver_matches_analytical_solution(self):
        """Check the no-gravity, no-ablation, no-breakup case."""

        planet = Planet(g=0.0, Cl=0.0, Ch=0.0, Rp=np.inf)
        radius = 1.0
        velocity = 20_000.0
        density = 3000.0
        angle = 30.0
        init_altitude = 100_000.0

        result = planet.solve_atmospheric_entry(
            radius=radius,
            velocity=velocity,
            density=density,
            strength=1e30,
            angle=angle,
            init_altitude=init_altitude,
            dt=0.05,
        )

        initial_mass = (4.0 / 3.0) * np.pi * radius**3 * density
        expected_velocity = planet.analytical_solution(
            z=result['altitude'].to_numpy(),
            v0=velocity,
            C_D=planet.Cd,
            R=radius,
            m0=initial_mass,
            theta=angle,
            rho_0=planet.rho0,
            H=planet.H,
            z0=init_altitude,
        )

        assert len(result) > 2
        np.testing.assert_allclose(
            result['velocity'].to_numpy(),
            expected_velocity,
            rtol=1e-3,
            atol=1e-6,
        )
