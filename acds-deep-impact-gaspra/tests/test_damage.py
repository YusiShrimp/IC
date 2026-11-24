"""
Tests for the surface zero location and airblast damage radii
calculated by deepimpact.damage_zones.

The reference formulae (overpressure, haversine distance and initial
bearing) are re-implemented here, independently of the package, so that
the tests do not simply compare the code with itself.
"""
import numpy as np
import pytest

import deepimpact

RP = 6371e3
PRESSURES = [1e3, 3.5e3, 27e3, 43e3]


def overpressure(r, zb, ek):
    """Empirical airblast overpressure (Pa) from the project brief."""
    s = (r**2 + zb**2) / ek**(2. / 3.)
    return 3.14e11 * s**(-1.3) + 1.8e7 * s**(-0.565)


def haversine(lat1, lon1, lat2, lon2):
    """Great circle distance (m) between two points given in degrees."""
    p1, l1, p2, l2 = np.radians([lat1, lon1, lat2, lon2])
    a = (np.sin((p2 - p1) / 2)**2
         + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2)**2)
    return 2 * RP * np.arcsin(np.sqrt(a))


def initial_bearing(lat1, lon1, lat2, lon2):
    """Bearing (degrees, 0 to 360) to travel from point 1 to point 2."""
    p1, l1, p2, l2 = np.radians([lat1, lon1, lat2, lon2])
    num = np.cos(p2) * np.sin(l2 - l1)
    den = (np.cos(p1) * np.sin(p2)
           - np.sin(p1) * np.cos(p2) * np.cos(l2 - l1))
    return np.degrees(np.arctan2(num, den)) % 360.


def make_outcome(energy=7e3, altitude=8e3, distance=90e3):
    """Build an outcome dictionary like the one from analyse_outcome."""
    return {'outcome': 'Airburst',
            'burst_peak_dedz': 1e3,
            'burst_altitude': altitude,
            'burst_distance': distance,
            'burst_energy': energy,
            'burst_angle': 45.}


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------

def test_list_of_pressures_returns_list_of_floats():
    blat, blon, damrad = deepimpact.damage_zones(
        make_outcome(), 52.79, -2.95, 135., PRESSURES)

    assert isinstance(blat, float)
    assert isinstance(blon, float)
    assert type(damrad) is list
    assert len(damrad) == len(PRESSURES)
    assert all(isinstance(r, float) for r in damrad)


def test_numpy_array_of_pressures_returns_list():
    _, _, damrad = deepimpact.damage_zones(
        make_outcome(), 52.79, -2.95, 135., np.array(PRESSURES))

    assert type(damrad) is list
    assert len(damrad) == len(PRESSURES)


def test_single_pressure_returns_float():
    blat, blon, damrad = deepimpact.damage_zones(
        make_outcome(), 52.79, -2.95, 135., 27e3)

    assert isinstance(blat, float)
    assert isinstance(blon, float)
    assert isinstance(damrad, float)


def test_single_pressure_matches_list_entry():
    _, _, as_list = deepimpact.damage_zones(
        make_outcome(), 52.79, -2.95, 135., PRESSURES)
    _, _, as_float = deepimpact.damage_zones(
        make_outcome(), 52.79, -2.95, 135., PRESSURES[2])

    assert as_float == pytest.approx(as_list[2], rel=1e-12)


# ---------------------------------------------------------------------------
# Surface zero location
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('bearing', [0., 77., 135., 270.])
def test_zero_distance_returns_entry_point(bearing):
    blat, blon, _ = deepimpact.damage_zones(
        make_outcome(distance=0.), 52.79, -2.95, bearing, PRESSURES)

    assert blat == pytest.approx(52.79, abs=1e-10)
    assert blon == pytest.approx(-2.95, abs=1e-10)


def test_due_north_changes_latitude_only():
    distance = 100e3
    blat, blon, _ = deepimpact.damage_zones(
        make_outcome(distance=distance), 50., -1., 0., PRESSURES)

    assert blat == pytest.approx(50. + np.degrees(distance / RP), abs=1e-9)
    assert blon == pytest.approx(-1., abs=1e-9)


def test_due_south_changes_latitude_only():
    distance = 100e3
    blat, blon, _ = deepimpact.damage_zones(
        make_outcome(distance=distance), 50., -1., 180., PRESSURES)

    assert blat == pytest.approx(50. - np.degrees(distance / RP), abs=1e-9)
    assert blon == pytest.approx(-1., abs=1e-9)


@pytest.mark.parametrize('bearing, sign', [(90., 1.), (270., -1.)])
def test_along_equator_changes_longitude_only(bearing, sign):
    distance = 100e3
    blat, blon, _ = deepimpact.damage_zones(
        make_outcome(distance=distance), 0., 0., bearing, PRESSURES)

    assert blat == pytest.approx(0., abs=1e-9)
    assert blon == pytest.approx(sign * np.degrees(distance / RP), abs=1e-9)


@pytest.mark.parametrize('bearing', [0., 45., 115., 135., 217., 300.])
@pytest.mark.parametrize('lat, lon', [(52.79, -2.95), (55.2, -2.5),
                                      (51.5, 0.1), (57.1, -6.2)])
def test_surface_zero_is_right_distance_and_direction(lat, lon, bearing):
    """Travel back-check: the point returned must be burst_distance away
    from the entry point, in the direction of the bearing."""
    distance = 90e3
    blat, blon, _ = deepimpact.damage_zones(
        make_outcome(distance=distance), lat, lon, bearing, PRESSURES)

    assert haversine(lat, lon, blat, blon) == pytest.approx(distance,
                                                            rel=1e-9)
    found = initial_bearing(lat, lon, blat, blon)
    # compare angles on the circle, so that 359.999 and 0 count as close
    assert abs((found - bearing + 180.) % 360. - 180.) < 1e-6


def test_longitude_stays_in_range_across_date_line():
    _, blon, _ = deepimpact.damage_zones(
        make_outcome(distance=200e3), 10., 179.5, 90., PRESSURES)

    assert -180. <= blon <= 180.
    assert blon < 0.


@pytest.mark.parametrize('distance', [1e3, 90e3, 400e3])
def test_surface_zero_distance_for_short_and_long_tracks(distance):
    blat, blon, _ = deepimpact.damage_zones(
        make_outcome(distance=distance), 52.79, -2.95, 135., PRESSURES)

    assert haversine(52.79, -2.95, blat, blon) == pytest.approx(distance,
                                                                rel=1e-6)


def test_bearing_360_same_as_bearing_0():
    north = deepimpact.damage_zones(
        make_outcome(), 52.79, -2.95, 0., PRESSURES)
    full_turn = deepimpact.damage_zones(
        make_outcome(), 52.79, -2.95, 360., PRESSURES)

    assert full_turn[0] == pytest.approx(north[0], abs=1e-9)
    assert full_turn[1] == pytest.approx(north[1], abs=1e-9)


def test_surface_zero_does_not_depend_on_pressures():
    one = deepimpact.damage_zones(
        make_outcome(), 52.79, -2.95, 135., [1e3])
    four = deepimpact.damage_zones(
        make_outcome(), 52.79, -2.95, 135., PRESSURES)

    assert one[0] == pytest.approx(four[0], abs=1e-12)
    assert one[1] == pytest.approx(four[1], abs=1e-12)


# ---------------------------------------------------------------------------
# Damage radii
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('energy, altitude', [(7e3, 8e3), (500., 30e3),
                                              (1e4, 0.), (5., 2e3)])
@pytest.mark.parametrize('r_true', [500., 5e3, 30e3, 150e3])
def test_radius_recovers_known_value(energy, altitude, r_true):
    """Choose the radius first, then ask for the pressure found there."""
    pressure = overpressure(r_true, altitude, energy)

    _, _, damrad = deepimpact.damage_zones(
        make_outcome(energy=energy, altitude=altitude),
        52.79, -2.95, 135., [pressure])

    assert damrad[0] == pytest.approx(r_true, rel=1e-6)


@pytest.mark.parametrize('energy, altitude', [(7e3, 8e3), (6e3, 9e3),
                                              (500., 30e3), (2e4, 0.),
                                              (30., 1e3), (1e6, 5e3)])
def test_pressure_at_each_radius_equals_threshold(energy, altitude):
    """Round trip: putting each radius back into p(r) must give back the
    threshold pressure, unless the level never reaches the ground."""
    _, _, damrad = deepimpact.damage_zones(
        make_outcome(energy=energy, altitude=altitude),
        52.79, -2.95, 135., PRESSURES)

    # pressure directly below the burst (unbounded for a surface burst)
    if altitude > 0.:
        p_below = overpressure(0., altitude, energy)
    else:
        p_below = np.inf

    for radius, pressure in zip(damrad, PRESSURES):
        if p_below > pressure:
            assert radius > 0.
            assert overpressure(radius, altitude, energy) == pytest.approx(
                pressure, rel=1e-8)
        else:
            assert radius == 0.


def test_radii_decrease_as_pressure_increases():
    _, _, damrad = deepimpact.damage_zones(
        make_outcome(), 52.79, -2.95, 135., PRESSURES)

    assert all(a > b for a, b in zip(damrad[:-1], damrad[1:]))


def test_more_energy_gives_larger_radii():
    # 1 kPa reaches the ground for every energy in this list
    radii = []
    for energy in (1e2, 1e3, 1e4, 1e5):
        _, _, radius = deepimpact.damage_zones(
            make_outcome(energy=energy), 52.79, -2.95, 135., PRESSURES[0])
        radii.append(radius)

    assert all(a < b for a, b in zip(radii[:-1], radii[1:]))


def test_radii_keep_the_order_of_the_input_pressures():
    _, _, forward = deepimpact.damage_zones(
        make_outcome(), 52.79, -2.95, 135., PRESSURES)
    _, _, backward = deepimpact.damage_zones(
        make_outcome(), 52.79, -2.95, 135., PRESSURES[::-1])

    assert backward == pytest.approx(forward[::-1], rel=1e-12)


def test_radii_do_not_depend_on_entry_point_or_bearing():
    _, _, here = deepimpact.damage_zones(
        make_outcome(), 52.79, -2.95, 135., PRESSURES)
    _, _, there = deepimpact.damage_zones(
        make_outcome(), 10., 100., 20., PRESSURES)

    assert there == pytest.approx(here, rel=1e-12)


def test_level_not_reached_returns_zero():
    """A 500 kt burst at 30 km gives about 1.8 kPa directly below, so
    only the 1 kPa level reaches the ground."""
    _, _, damrad = deepimpact.damage_zones(
        make_outcome(energy=500., altitude=30e3),
        52.79, -2.95, 135., PRESSURES)

    assert damrad[0] > 0.
    assert damrad[1:] == [0., 0., 0.]


def test_no_level_reached_returns_all_zeros():
    """A 1 kt burst at 40 km gives about 0.1 kPa directly below."""
    assert overpressure(0., 40e3, 1.) < min(PRESSURES)

    _, _, damrad = deepimpact.damage_zones(
        make_outcome(energy=1., altitude=40e3),
        52.79, -2.95, 135., PRESSURES)

    assert damrad == [0., 0., 0., 0.]


def test_threshold_equal_to_pressure_below_burst_gives_zero_radius():
    energy, altitude = 7e3, 8e3
    p0 = overpressure(0., altitude, energy)
    _, _, radius = deepimpact.damage_zones(
        make_outcome(energy=energy, altitude=altitude),
        52.79, -2.95, 135., p0)

    assert radius == pytest.approx(0., abs=1.)


def test_surface_burst_gives_finite_radii():
    """burst_altitude = 0 (cratering event) must not divide by zero."""
    _, _, damrad = deepimpact.damage_zones(
        make_outcome(energy=2e4, altitude=0.),
        52.79, -2.95, 135., PRESSURES)

    assert all(np.isfinite(damrad))
    assert all(r > 0. for r in damrad)


def test_zero_energy_gives_zero_radii():
    _, _, damrad = deepimpact.damage_zones(
        make_outcome(energy=0.), 52.79, -2.95, 135., PRESSURES)

    assert damrad == [0., 0., 0., 0.]


@pytest.mark.parametrize('bad', [0., -1e3, [1e3, -5.]])
def test_non_positive_pressure_raises(bad):
    with pytest.raises(ValueError):
        deepimpact.damage_zones(make_outcome(), 52.79, -2.95, 135., bad)


def test_docstring_example_values():
    """Reference values found separately with a simple bisection."""
    blat, blon, damrad = deepimpact.damage_zones(
        make_outcome(energy=7e3, altitude=8e3, distance=90e3),
        52.79, -2.95, 135., PRESSURES)

    assert blat == pytest.approx(52.21397, abs=1e-5)
    assert blon == pytest.approx(-2.01591, abs=1e-5)
    assert damrad == pytest.approx([115971.3, 42628.4, 9575.2, 5836.0],
                                   rel=1e-5)


# ---------------------------------------------------------------------------
# Damage ellipses (extension)
# ---------------------------------------------------------------------------

def outcome_at_angle(angle):
    """Outcome dictionary with the given burst angle (degrees)."""
    outcome = make_outcome()
    outcome['burst_angle'] = angle
    return outcome


def expected_axes(radius, angle_deg, kf=0.4, kb=0.2):
    """Semi-axes from the formulas in DamageMapper.ipynb."""
    cos2 = np.cos(np.radians(angle_deg))**2
    q_f = 1. - kf * cos2
    q_b = 1. - kb * cos2
    a_perp = radius * np.sqrt(2. / (q_f + q_b))
    return a_perp, q_f * a_perp, q_b * a_perp


def test_ellipses_list_input_returns_three_lists_of_floats():
    axes = deepimpact.damage_ellipses(outcome_at_angle(30.), [5e3, 2e3])

    assert len(axes) == 3
    for axis in axes:
        assert type(axis) is list
        assert len(axis) == 2
        assert all(isinstance(a, float) for a in axis)


def test_ellipses_scalar_input_returns_floats():
    axes = deepimpact.damage_ellipses(outcome_at_angle(30.), 5e3)

    assert len(axes) == 3
    for axis in axes:
        assert isinstance(axis, float)


@pytest.mark.parametrize('angle', [0., 10., 30., 45., 60., 80.])
@pytest.mark.parametrize('kf, kb', [(0.4, 0.2), (0.5, 0.1), (0.3, 0.3)])
def test_ellipses_match_formula(angle, kf, kb):
    radii = [9575.2, 1000.]

    a_perp, a_front, a_behind = deepimpact.damage_ellipses(
        outcome_at_angle(angle), radii, kf=kf, kb=kb)

    for i, radius in enumerate(radii):
        perp, front, behind = expected_axes(radius, angle, kf, kb)
        assert a_perp[i] == pytest.approx(perp, rel=1e-12)
        assert a_front[i] == pytest.approx(front, rel=1e-12)
        assert a_behind[i] == pytest.approx(behind, rel=1e-12)


def test_vertical_entry_gives_circle():
    a_perp, a_front, a_behind = deepimpact.damage_ellipses(
        outcome_at_angle(90.), 5e3)

    assert a_perp == pytest.approx(5e3, rel=1e-12)
    assert a_front == pytest.approx(5e3, rel=1e-12)
    assert a_behind == pytest.approx(5e3, rel=1e-12)


def test_zero_shape_factors_give_circle():
    a_perp, a_front, a_behind = deepimpact.damage_ellipses(
        outcome_at_angle(20.), 5e3, kf=0., kb=0.)

    assert a_perp == pytest.approx(5e3, rel=1e-12)
    assert a_front == pytest.approx(5e3, rel=1e-12)
    assert a_behind == pytest.approx(5e3, rel=1e-12)


@pytest.mark.parametrize('angle', [0., 18., 45., 70., 90.])
def test_ellipse_area_equals_circle_area(angle):
    """Two half-ellipses: area = (pi / 2) * a_perp * (a_front + a_behind)."""
    radius = 5e3
    a_perp, a_front, a_behind = deepimpact.damage_ellipses(
        outcome_at_angle(angle), radius)

    area = 0.5 * np.pi * a_perp * (a_front + a_behind)
    assert area == pytest.approx(np.pi * radius**2, rel=1e-12)


def test_shallow_entry_squashes_front_more_than_back():
    a_perp, a_front, a_behind = deepimpact.damage_ellipses(
        outcome_at_angle(20.), 5e3)

    assert a_front < a_behind < a_perp


def test_horizontal_entry_values():
    """At 0 degrees q_f = 0.6 and q_b = 0.8."""
    radius = 5e3
    a_perp, a_front, a_behind = deepimpact.damage_ellipses(
        outcome_at_angle(0.), radius)

    expected_perp = radius * np.sqrt(2. / 1.4)
    assert a_perp == pytest.approx(expected_perp, rel=1e-12)
    assert a_front == pytest.approx(0.6 * expected_perp, rel=1e-12)
    assert a_behind == pytest.approx(0.8 * expected_perp, rel=1e-12)


def test_radians_flag_matches_degrees():
    in_degrees = deepimpact.damage_ellipses(outcome_at_angle(30.), 5e3)

    outcome = outcome_at_angle(np.radians(30.))
    in_radians = deepimpact.damage_ellipses(outcome, 5e3, radians=True)

    assert in_radians == pytest.approx(in_degrees, rel=1e-12)


def test_zero_radius_gives_zero_axes():
    axes = deepimpact.damage_ellipses(outcome_at_angle(30.), [0., 5e3])

    for axis in axes:
        assert axis[0] == 0.
        assert axis[1] > 0.


def test_axes_scale_with_radius():
    small = deepimpact.damage_ellipses(outcome_at_angle(30.), 1e3)
    large = deepimpact.damage_ellipses(outcome_at_angle(30.), 3e3)

    for s, big in zip(small, large):
        assert big == pytest.approx(3. * s, rel=1e-12)
