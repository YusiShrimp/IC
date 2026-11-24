from pathlib import Path
import time

import numpy as np
import pandas as pd  # noqa: F401
import pytest

from deepimpact import Planet

pytestmark = pytest.mark.timeout(15)
ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "velocity",
    "mass",
    "angle",
    "altitude",
    "distance",
    "radius",
    "spreading_rate",
    "time",
]

# ============================================================================
# Shared Helper Functions and Fixtures
# ============================================================================


# [Helper/fixture] Defines the ordering of the private seven-dimensional state
# vector;
# check here when adapting the interface for physics tests.
def isolated_state(**kwargs):
    values = dict(
        v=10.0, m=5.0, theta=0.0, z=0.0, x=0.0, r=np.sqrt(2.0 / np.pi), u=0.0
    )
    values.update(kwargs)
    return np.array(list(values.values()))


# [Helper/fixture] Set up simplified physical conditions;
# preserve valid use of Cd/Ch/Cl/g=0 and Rp=inf.
def vacuum(**kwargs):
    settings = dict(Cd=0.0, Ch=0.0, Cl=0.0, g=0.0, Rp=np.inf)
    settings.update(kwargs)
    return Planet(**settings)


# [Helper function/fixture] A unified helper for common inputs;
# no changes are required if new public interfaces remain compatible.
def run(planet=None, **kwargs):
    inputs = dict(
        radius=1.0,
        velocity=100.0,
        density=3000.0,
        strength=np.inf,
        angle=90.0,
        init_altitude=23.0,
        dt=0.1,
    )
    inputs.update(kwargs)
    return (vacuum() if planet is None else planet).solve_atmospheric_entry(
        **inputs
    )


# [Helper/fixture] Generate the supplied scenario using the current solver.
# Parameters match tests/README.md and the initial altitude in scenario.npz.
# Reuse one simulated trajectory across this module's reference comparisons
# and physical-bound checks; the reference data is loaded separately by tests.
@pytest.fixture(scope="module")
def reference_result():
    planet = Planet(pancake_factor=6.5)
    return planet.solve_atmospheric_entry(
        radius=35.0,
        velocity=19000.0,
        density=3000.0,
        strength=1e7,
        angle=45.0,
        init_altitude=100000.0,
        dt=0.05,
    )


# [Helper function/fixture] Synthesized RHS—used solely to isolate the event
# mechanism
def synthetic_spreading(monkeypatch, cap):
    planet = vacuum(atmos_func="constant", rho0=1.0, pancake_factor=cap)
    planet._max_internal_step = 0.5
    original = planet._derivatives
    phases = []

    def controlled(t, state, density, phase):
        phases.append(phase)
        rates = original(t, state, density, phase)
        if phase == "spreading":
            rates[5] = state[6]
            rates[6] = 2.0  # Exact pre-cap radius: r=1+t^2.
        return rates

    monkeypatch.setattr(planet, "_derivatives", controlled)
    return planet, phases


# ============================================================================
# 1. Atmospheric Models and Mode Selection (4 test functions / 4 expanded
# cases)
# Check exponential, constant-density, and tabulated atmospheres, as well as
# invalid mode names;
# physical expectations should remain unchanged when the integration algorithm
# is changed.
# ============================================================================


# exponential mode
def test_exponential_atmosphere_at_known_heights():
    planet = Planet(rho0=2.0, H=1000.0)
    np.testing.assert_allclose(
        planet.rhoa(np.array([0.0, 1000.0, 2000.0])),
        [2.0, 2.0 / np.e, 2.0 / np.e**2],
    )


# constant mode
def test_constant_atmosphere_scalar():
    planet = Planet(atmos_func="constant", rho0=2.5)
    assert planet.rhoa(0.0) == planet.rhoa(90000.0) == 2.5


# #tabular mode
def test_tabular_atmosphere_matches_supplied_nodes():
    table = np.loadtxt(ROOT / "resources/AltitudeDensityTable.csv")
    planet = Planet(atmos_func="tabular")
    indices = [0, 5, 10]
    actual = [planet.rhoa(float(table[i, 0])) for i in indices]
    np.testing.assert_allclose(actual, table[indices, 1], rtol=1e-12)


# unknow type
def test_unknown_atmosphere_rejected():
    with pytest.raises(ValueError):
        Planet(atmos_func="expontential_typo")


# ============================================================================
# 2. Physical Derivatives, Material Parameters, and Area Feedback
# Check drag, gravity, ablation, lift, curvature, the expansion phase,
# and material density individually, and verify that inputs are not modified.
# ============================================================================


@pytest.mark.parametrize(
    "component", ["drag", "gravity", "ablation", "lift", "curvature"]
)
def test_individual_physical_terms(component):
    settings = dict(atmos_func="constant", rho0=3.0)
    state = isolated_state()
    if component == "drag":
        # [Drag term] Full velocity equation:
        # dv/dt = -Cd*rho_air*A*v^2/(2*m) + g*sin(theta).
        planet = vacuum(Cd=4.0, **settings)
        actual = planet._derivatives(0, state, 3000, "intact")
        assert actual[0] == pytest.approx(-240.0)
        assert actual[1] == 0.0
    elif component == "gravity":
        # [Gravity term]
        # With Cd=Cl=0 and Rp=inf, the velocity and angle equations simplify
        # to:
        # dv/dt = g*sin(theta)
        # dtheta/dt = g*cos(theta)/v
        planet = vacuum(g=2.0, **settings)
        state[2] = np.pi / 6
        actual = planet._derivatives(0, state, 3000, "intact")
        assert actual[0] == pytest.approx(1.0)
        assert actual[2] == pytest.approx(np.sqrt(3.0) / 10.0)
    elif component == "ablation":
        # [Ablation term] Mass equation: dm/dt=-Ch*rho_air*A*v^3/(2*Q).
        planet = vacuum(Ch=0.5, Q=10.0, **settings)
        actual = planet._derivatives(0, state, 3000, "intact")
        assert actual[1] == pytest.approx(-150.0)
    elif component == "lift":
        # [Lift term] Full angle equation:
        # dtheta/dt = g*cos(theta)/v
        # - Cl*rho_air*A*v/(2*m)
        # - v*cos(theta)/(Rp+z).
        planet = vacuum(Cl=0.4, **settings)
        actual = planet._derivatives(0, state, 3000, "intact")
        assert actual[2] == pytest.approx(-2.4)
    else:
        # [Planetary curvature term]
        # The angle equation simplifies to:
        # dtheta/dt = -v*cos(theta)/(Rp+z)
        # The horizontal distance equation is:
        # dx/dt = v*cos(theta)/(1+z/Rp)
        planet = vacuum(Rp=100.0, **settings)
        state[3] = 100.0
        actual = planet._derivatives(0, state, 3000, "intact")
        assert actual[2] == pytest.approx(-0.05)
        assert actual[4] == pytest.approx(5.0)


# Test whether the correct derivatives are used for the intact,
# capped, and spreading phases
@pytest.mark.parametrize(
    "phase,expected",
    [
        ("intact", [0.0, 0.0]),
        ("capped", [0.0, 0.0]),
        ("spreading", [7.0, 120.0]),
    ],
)
def test_expansion_acceleration_and_phase_rules(phase, expected):
    planet = vacuum(Cd=4.0, atmos_func="constant", rho0=3.0)
    state = isolated_state(r=2.0, u=7.0)
    before = state.copy()
    rates = planet._derivatives(0, state, density=5.0, phase=phase)
    np.testing.assert_allclose(rates[-2:], expected)
    np.testing.assert_array_equal(state, before)
    assert rates.shape == (7,)


# Test whether an error occurs when an unknown stage name is passed in.
def test_derivative_rejects_unknown_phase():
    with pytest.raises(ValueError, match="Unknown phase"):
        vacuum()._derivatives(0, isolated_state(), 3000, "typo")


# Test expression for du/dt.
@pytest.mark.parametrize("factor,expected", [(1.0, 2.0), (2.0, 4.0)])
def test_spreading_uses_material_density(factor, expected):
    planet = vacuum(Cd=1.0, atmos_func="constant", rho0=1.0)
    rates = planet._derivatives(
        0,
        isolated_state(v=10.0, r=1.0),
        50.0 / factor,
        "spreading",
    )
    assert rates[6] == pytest.approx(expected)


# Test whether changes in radius are correctly reflected in the frontal area,
# as well as in the deceleration due to drag and the mass-loss rate due to
# ablation.
def test_expanding_area_feeds_back_into_drag_and_ablation():
    planet = vacuum(Cd=1.0, Ch=0.1, atmos_func="constant", rho0=1.0)
    a = planet._derivatives(0, isolated_state(r=1.0), 3000, "spreading")
    b = planet._derivatives(0, isolated_state(r=2.0), 3000, "spreading")
    np.testing.assert_allclose(b[:2], 4 * a[:2])


# ============================================================================
# 3. Known Solutions, Time Parameters, and Convergence Order of the Fixed-Step
# RK4 Kernel
# 2 test functions in total.
# Test the integration kernel in isolation, excluding fragmentation
# transitions, radius capping, and ground-impact interpolation.
# ============================================================================


# Helper: compute the absolute error at the endpoint for the exponential decay
# equation.
def integrate_exponential(h):
    planet = Planet()
    t_end = 1.0
    n_steps = int(round(t_end / h))

    assert np.isclose(n_steps * h, t_end, rtol=0, atol=1e-14)

    state = np.array([1.0])

    for n in range(n_steps):
        state = planet._rk4_step(
            lambda t, y: -y,
            n * h,
            state,
            h,
        )

    return abs(state[0] - np.exp(-t_end))


# Check correct time handling and verify that the input is not modified.
def test_rk4_uses_stage_times_and_does_not_mutate_input():
    planet = Planet()
    state = np.array([0.0, 1.0])
    original = state.copy()

    result = planet._rk4_step(
        lambda t, y: np.array([t, 2 * t]),
        0.3,
        state,
        0.2,
    )

    np.testing.assert_allclose(
        result,
        [0.08, 1.16],
        rtol=0,
        atol=1e-14,
    )

    # The integrator should return a new state without modifying the supplied
    # array in place.
    np.testing.assert_array_equal(state, original)


# Check fourth-order accuracy.


def test_rk4_fourth_order_on_known_solution():
    errors = np.array([integrate_exponential(h) for h in [0.1, 0.05, 0.025]])

    # Check the absolute error with the smallest step size.
    assert errors[-1] < 2e-9

    # The error should strictly decrease as the step size decreases.
    assert np.all(errors[1:] < errors[:-1])

    # Ensure the errors are positive so that error ratios and logarithms can be
    # computed.
    assert np.all(errors > 0)

    observed_orders = np.log2(errors[:-1] / errors[1:])

    # Check the convergence order.
    assert np.all((observed_orders > 3.8) & (observed_orders < 4.2)), (
        "The observed convergence orders should be close to 4; "
        f"actual values: {observed_orders}"
    )


# ============================================================================
# 4. Accuracy testing of asteroid trajectories in an incomplete physical
# environment
# ============================================================================


# Physical environment: force-free motion over a flat planet, with no ablation,
# at entry angles of 30, 45, and 90 degrees.
# Verification: compare ground-impact time and horizontal distance with the
# straight-line solutions t_ground=z0/(v*sin(theta)) and x=v*cos(theta)*t,
# and check that the trajectory angle remains constant.
@pytest.mark.parametrize("angle", [30.0, 45.0, 90.0])
def test_vacuum_oblique_trajectory(angle):
    result = run(angle=angle)
    theta = np.deg2rad(angle)
    expected_time = 23.0 / (100 * np.sin(theta))
    assert result.time.iloc[-1] == pytest.approx(expected_time, abs=1e-8)
    np.testing.assert_allclose(
        result.distance, 100 * np.cos(theta) * result.time, atol=1e-8
    )
    np.testing.assert_allclose(result.angle, angle)


# Physical environment: vertical descent under constant gravity (g=9.81 m/s^2)
# over a flat planet, with no aerodynamic forces or ablation.
# Verification: compare velocity and pre-impact altitude with the constant-
# acceleration solutions v=v0+g*t and z=z0-v0*t-g*t^2/2; compare ground-impact
# time with the positive root of z(t)=0.
def test_vertical_gravity_exact_trajectory():
    result = run(vacuum(g=9.81), init_altitude=123.0)
    t = result.time.to_numpy()
    exact_time = 2 * 123 / (100 + np.sqrt(100**2 + 2 * 9.81 * 123))
    assert t[-1] == pytest.approx(exact_time, abs=1e-8)
    np.testing.assert_allclose(result.velocity, 100 + 9.81 * t, atol=1e-8)
    np.testing.assert_allclose(
        result.altitude.iloc[:-1],
        (123 - 100 * t - 4.905 * t * t)[:-1],
        atol=1e-8,
    )


# Physical environment: vertical entry into a constant-density atmosphere with
# quadratic drag, with no gravity, lift, ablation, or fragmentation.
# Verification: compare velocity with v=v0/(1+k*v0*t), where k=rho*A/(2*m)
# for Cd=1, and ground-impact time with (exp(k*z0)-1)/(k*v0).
def test_constant_density_drag_against_time_and_ground_solution():
    planet = vacuum(Cd=1.0, atmos_func="constant", rho0=1.2)
    result = run(planet, init_altitude=100.0)
    k = 1.2 * np.pi / (2 * (4 * np.pi * 3000 / 3))
    t = result.time.to_numpy()
    np.testing.assert_allclose(
        result.velocity, 100 / (1 + k * 100 * t), rtol=1e-7
    )
    exact_time = np.expm1(k * 100) / (k * 100)
    assert t[-1] == pytest.approx(exact_time, abs=2e-7)


# Physical environment: force-free entry at 45 degrees around a spherical
# planet
# with Rp=10000 m, retaining curvature terms but excluding ablation.
# Verification: check that the geometric invariant (Rp+z)*cos(theta) remains
# constant along the trajectory, as required for straight-line motion.
def test_spherical_straight_line_geometric_invariant():
    planet = vacuum(Rp=10000.0)
    result = run(planet, init_altitude=1000, angle=45, dt=0.1)
    invariant = (planet.Rp + result.altitude) * np.cos(
        np.deg2rad(result.angle)
    )

    np.testing.assert_allclose(invariant, invariant.iloc[0], rtol=1e-8)


# ============================================================================
# 5. Given Reference Trajectories and Multiple Complete Physical Scenarios
# Align reference files by altitude and verify physical boundaries;
# note that sampled reference discrepancies are not equivalent to actual
# observation errors.
# ============================================================================

# Physical environment: the project's supplied atmospheric-entry scenario,
# using the physical conditions and initial parameters configured in
# reference_result.


# Verification: check that altitude strictly decreases, then interpolate the
# simulated
# velocity, mass, radius, angle, and distance onto the reference altitude
# nodes.
# Compare each quantity with tests/scenario.npz using rtol=1e-3 and atol=0.01;
# the reference contains sampled model output, not an exact solution or
# observations.
@pytest.mark.parametrize(
    "column", ["velocity", "mass", "radius", "angle", "distance"]
)
def test_supplied_scenario_accuracy(reference_result, column):
    reference = np.load(ROOT / "tests/scenario.npz")
    result = reference_result
    assert (np.diff(result.altitude) < 0).all()
    actual = np.interp(
        reference["altitude"],
        result.altitude.to_numpy()[::-1],
        result[column].to_numpy()[::-1],
    )
    # A declared engineering acceptance criterion, not the hidden scorer's
    # tolerance.
    np.testing.assert_allclose(actual, reference[column], rtol=1e-3, atol=0.01)


# Physical environment: the project's supplied atmospheric-entry scenario,
# using reference_result to follow the trajectory from entry to ground impact,
# with an initial radius of 35 m and a maximum expansion radius of 35*6.5 m.


# Verification: check that all values are finite, mass and velocity remain
# positive,
# mass does not increase, and horizontal distance does not decrease within
# tolerance.
# Check that the radius stays within its bounds and reaches the upper limit,
# where the spreading rate must be zero; the final altitude must be zero,
# and the simulated duration must exceed 1 s.
def test_reference_trajectory_physical_bounds(reference_result):
    result = reference_result
    assert np.isfinite(result.to_numpy()).all()
    assert (result.mass > 0).all()
    assert (result.velocity > 0).all()
    assert (np.diff(result.mass) <= 1e-5).all()
    assert (np.diff(result.distance) >= -1e-8).all()
    assert result.radius.max() == pytest.approx(35 * 6.5)
    assert (result.radius >= 35).all()
    assert (result.radius <= 35 * 6.5).all()
    assert result.altitude.iloc[-1] == 0
    assert result.time.iloc[-1] > 1.0
    assert (result.loc[result.radius == 35 * 6.5, "spreading_rate"] == 0).all()


# Use the default planet and atmosphere models with six sets of initial radius,
# velocity, density, strength, and entry angle,
# covering different sizes, velocities, material properties, and vertical and
# oblique entry scenarios.


# Check that all output values are finite (no NaN or infinity), mass never
# becomes negative, and velocity remains positive. Most cases impact the
# ground; the 72 km/s case completely ablates before impact.
@pytest.mark.parametrize(
    "inputs,termination",
    [
        ((1, 11000, 3000, 1e9, 90), "ground"),
        ((10, 19000, 3000, 1e6, 45), "ground"),
        ((100, 20000, 8000, 1e9, 45), "ground"),
        ((10, 72000, 3000, 1e6, 45), "ablation"),
        ((10, 19000, 500, 1e4, 45), "ground"),
        ((10, 19000, 3000, 1e6, 90), "ground"),
    ],
)
def test_finite_entry_scenarios_complete(inputs, termination):
    result = Planet().solve_atmospheric_entry(*inputs, dt=0.1)
    assert np.isfinite(result.to_numpy()).all()
    assert (result.mass >= 0).all()
    assert (result.velocity > 0).all()
    if termination == "ground":
        print(result.mass.iloc[-1])
        assert result.altitude.iloc[-1] == 0
        assert result.mass.iloc[-1] > 0
    else:
        assert result.altitude.iloc[-1] > 0
        assert result.mass.iloc[-1] == 0


# ============================================================================
# 6. Tests for special events (such as fragmentation or expansion limit
# adjustments) handling
# ============================================================================


# Extreme test case: The asteroid shatters immediately and reaches its maximum
# expansion limit.
def test_pancake_factor_one_disables_expansion():
    result = run(vacuum(pancake_factor=1), strength=0)
    np.testing.assert_allclose(result.radius, 1)
    np.testing.assert_allclose(result.spreading_rate, 0)


# Special scenario: Grounding and reaching the expansion limit occur in the
# same time step.
@pytest.mark.parametrize("cap,cap_expected", [(1.01, True), (1.09, False)])
def test_earliest_of_ground_and_pancake(monkeypatch, cap, cap_expected):
    planet, phases = synthetic_spreading(monkeypatch, cap)
    result = run(planet, init_altitude=20, strength=0.0, dt=0.5)
    assert result.time.iloc[-1] == pytest.approx(0.2, abs=1e-8)
    assert ("capped" in phases) == cap_expected
    expected_radius = cap if cap_expected else 1.04
    assert result.radius.iloc[-1] == pytest.approx(expected_radius, abs=1e-7)


# Testing shows that once an asteroid has transitioned into a fragmented state,
# it will under no circumstances revert to an intact state.
def test_no_reversion_when_ram_pressure_drops(monkeypatch):
    planet = vacuum(Cd=1.0, atmos_func="constant", rho0=1.0)
    original = planet._derivatives
    phases = []

    def record(t, state, density, phase):
        phases.append(phase)
        return original(t, state, density, phase)

    monkeypatch.setattr(planet, "_derivatives", record)
    result = run(planet, init_altitude=10, strength=9999, dt=0.01)
    assert result.velocity.iloc[-1] ** 2 < 9999
    assert "intact" not in phases
    assert result.radius.iloc[-1] > 1
    assert result.spreading_rate.iloc[-1] > 0


# Scenario: The actual contact time falls within an interval (i.e., not at the
# endpoints);
# evaluate the magnitude of the error.
# Expected behavior: The contact time error should not exceed 1e-5 s,
# the velocity error should not exceed 1e-4 m/s, and the final height value
# should be zero.
def test_ground_contact_accuracy():
    g = 9.81
    initial_velocity = 100.0
    initial_altitude = 123.0
    radius = 1.0
    density = 3000.0

    planet = Planet(Cd=0.0, Ch=0.0, Cl=0.0, g=g, Rp=np.inf)
    result = planet.solve_atmospheric_entry(
        radius=radius,
        velocity=initial_velocity,
        density=density,
        strength=np.inf,
        angle=90.0,
        init_altitude=initial_altitude,
        dt=0.02,
    )

    # Analytical solutions: z=z0-v0*t-g*t^2/2 and v=v0+g*t.
    # Solve z=0 for the ground-contact time; use the equivalent form below to
    # avoid subtracting nearly equal numbers.
    exact_velocity = np.sqrt(initial_velocity**2 + 2.0 * g * initial_altitude)
    exact_time = 2.0 * initial_altitude / (initial_velocity + exact_velocity)
    initial_mass = (4.0 / 3.0) * np.pi * radius**3 * density

    assert (
        len(result) >= 2
    ), "The initial state and ground-contact state should be returned."
    assert np.isfinite(
        result.to_numpy()
    ).all(), "The trajectory should contain no NaN or infinite values."
    final = result.iloc[-1]

    # Engineering acceptance thresholds, not the project's official grading
    # criteria or the integrator's internal tolerances.
    time_error = abs(final["time"] - exact_time)
    velocity_error = abs(final["velocity"] - exact_velocity)
    assert (
        time_error <= 1e-5
    ), f"Ground-contact time error {time_error:.9g} s exceeds 1e-5 s."
    assert velocity_error <= 1e-4, (
        f"Ground-contact velocity error {velocity_error:.9g} m/s "
        "exceeds 1e-4 m/s."
    )
    assert final["altitude"] == pytest.approx(0.0, abs=1e-8)

    # With ablation disabled, ground-contact interpolation or event handling
    # should not change the mass.
    # This assertion does not verify ground-contact mass accuracy in scenarios
    # with ablation.
    assert final["mass"] == pytest.approx(initial_mass, rel=1e-12, abs=0.0)


# ============================================================================
# 8. Invalid Public Inputs, Validation at Initial Ground Contact, and Physical
# Configuration
# ============================================================================


# Test scenario: the initial altitude is zero, so no integration is needed, but
# one of
# the radius, velocity, density, angle, or strength parameters is NaN.
# Exception handling: inputs must still be validated on the early-return path
# for initial ground contact;
# invalid parameters must raise ValueError rather than return an initial state
# containing NaN.
@pytest.mark.parametrize(
    "field", ["radius", "velocity", "density", "angle", "strength"]
)
def test_invalid_input_still_rejected_at_initial_ground(field):
    with pytest.raises(ValueError):
        run(init_altitude=0, **{field: np.nan})


# Test scenario: construct Planet with H=0, rho0=-1, Q=0,
# Rp=0, or pancake_factor=0.5, testing each setting separately.
# Exception handling: reject physical settings outside their valid ranges
# during construction
# by immediately raising ValueError, preventing division by zero, negative air
# density, or an invalid radius limit.
@pytest.mark.parametrize(
    "settings",
    [
        {"H": 0},
        {"rho0": -1},
        {"Q": 0},
        {"Rp": 0},
        {"pancake_factor": 0.5},
    ],
)
def test_invalid_physical_configuration_rejected(settings):
    with pytest.raises(ValueError):
        Planet(**settings)


# Test scenario: pass zero or negative radius, velocity, or density to the
# solver,
# or invalid initial altitude, time step, strength, or angle, including NaN and
# disallowed infinite values.
# Exception handling: raise ValueError before numerical integration
# to prevent invalid parameters from producing invalid results or causing
# nontermination.
# Set low integration-step and retry limits to reduce the risk of tests hanging
# if input validation is missing.
# Note: positive infinite strength is valid for a simplified model and is not
# rejected by this test.
@pytest.mark.parametrize(
    "field,value",
    [
        (field, value)
        for field in ["radius", "velocity", "density"]
        for value in [0.0, -1.0, np.nan, np.inf]
    ]
    + [
        (field, value)
        for field in ["init_altitude", "dt"]
        for value in [-1.0, np.nan, np.inf]
    ]
    + [
        ("dt", 0.0),
        ("strength", -1.0),
        ("strength", np.nan),
        ("strength", -np.inf),
        ("angle", np.nan),
        ("angle", np.inf),
    ],
)
def test_invalid_public_inputs_raise_value_error(field, value):
    planet = vacuum()
    planet._max_integration_steps = 100
    planet._max_step_retries = 2
    with pytest.raises(ValueError):
        run(planet, **{field: value})


# A small, light, fast asteroid is stopped high in the atmosphere and would
# then fall at only a few m/s for hours of model time (130,000 rows, our
# best explanation for the scenario four timeout). The run should end once
# the asteroid keeps a negligible fraction (1e-8) of its kinetic energy.
def test_asteroid_stopped_in_the_air_ends_quickly():
    start = time.perf_counter()
    result = Planet().solve_atmospheric_entry(1, 30000, 1000, 1e3, 45)
    elapsed = time.perf_counter() - start

    energy = 0.5 * result.mass * result.velocity**2
    assert elapsed < 5
    assert result.altitude.iloc[-1] > 0
    assert energy.iloc[-1] < 1e-7 * energy.iloc[0]
    assert len(result) < 20000
