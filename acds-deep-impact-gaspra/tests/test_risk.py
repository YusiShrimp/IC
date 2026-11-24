"""
Tests for deepimpact.impact_risk.

impact_risk runs the whole tool (solver, damage zones, locator) for every
scenario of an impact file. To test its own logic (looping over the
scenarios, counting the postcodes and computing the statistics) without
depending on the other modules, the solver and the locator are replaced
here by small fake versions whose answers are known in advance:

* FakePlanet returns an outcome that depends only on the scenario.
* FakeLocator knows a tiny made-up "country" of four postcodes placed
  1, 5, 10 and 50 km north of a centre point.

In the "by hand" tests damage_zones is replaced as well, so that the
'radius' column of the impact file is used directly as the blast radius
in km. The expected probabilities and populations can then be worked out
with pen and paper.
"""
import os

import numpy as np
import pandas as pd
import pytest

import deepimpact
import deepimpact.damage as damage

RP = 6371e3
COLUMNS = ['radius', 'angle', 'strength', 'density', 'velocity',
           'entry latitude', 'entry longitude', 'bearing']
DEFAULT_FILE = os.sep.join((os.path.dirname(__file__), '..', 'resources',
                            'impact_parameter_list.csv'))

# The made-up country: centre point, then four postcodes to the north
CENTRE = (52.0, -1.0)
POSTCODES = ['AA1 1AA', 'BB2 2BB', 'CC3 3CC', 'DD4 4DD']
DISTANCES = np.array([1e3, 5e3, 10e3, 50e3])
POPULATIONS = np.array([100, 20, 3, 1000])
POSTCODE_LATS = CENTRE[0] + np.degrees(DISTANCES / RP)
POSTCODE_LONS = np.full(4, CENTRE[1])


def haversine(lat1, lon1, lat2, lon2):
    """Great circle distance (m) between points given in degrees."""
    p1, l1, p2, l2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    a = (np.sin((p2 - p1) / 2)**2
         + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2)**2)
    return 2 * RP * np.arcsin(np.sqrt(a))


class FakePlanet:
    """
    Stand-in for deepimpact.Planet.

    The outcome is an airburst directly above the entry point
    (burst_distance = 0), with an energy proportional to the asteroid
    radius. Every call is recorded so that the tests can check what
    impact_risk asked for.
    """

    def __init__(self):
        self.solve_calls = []
        self.energy_inputs = []
        self.outcome_inputs = []

    def solve_atmospheric_entry(self, radius, velocity, density, strength,
                                angle, init_altitude=100e3, dt=0.05,
                                radians=False):
        self.solve_calls.append({'radius': radius, 'velocity': velocity,
                                 'density': density, 'strength': strength,
                                 'angle': angle,
                                 'init_altitude': init_altitude,
                                 'radians': radians})
        return {'stage': 'solved', 'radius': radius, 'angle': angle}

    def calculate_energy(self, result):
        self.energy_inputs.append(result)
        return dict(result, stage='energy')

    def analyse_outcome(self, result):
        self.outcome_inputs.append(result)
        return {'outcome': 'Airburst',
                'burst_peak_dedz': 1e3,
                'burst_altitude': 9e3,
                'burst_distance': 0.,
                'burst_energy': 200. * result['radius'],
                'burst_angle': result['angle'],
                # only used by the fake damage_zones below
                'zone_radius': 1e3 * result['radius']}


class NoDamagePlanet(FakePlanet):
    """Every scenario releases no energy at all."""

    def analyse_outcome(self, result):
        outcome = super().analyse_outcome(result)
        outcome['burst_energy'] = 0.
        outcome['zone_radius'] = 0.
        return outcome


class NanPlanet(FakePlanet):
    """Every scenario gives a burst energy that is not a number."""

    def analyse_outcome(self, result):
        outcome = super().analyse_outcome(result)
        outcome['burst_energy'] = np.nan
        outcome['zone_radius'] = np.nan
        return outcome


class FakeLocator:
    """
    Stand-in for deepimpact.GeospatialLocator, knowing only the four
    postcodes of the made-up country. Calls are recorded on the class.
    """

    created = 0
    calls = []

    def __init__(self, *args, **kwargs):
        FakeLocator.created += 1

    @staticmethod
    def _inside(X, radius):
        return haversine(X[0], X[1], POSTCODE_LATS, POSTCODE_LONS) < radius

    def get_postcodes_by_radius(self, X, radii):
        FakeLocator.calls.append(('postcodes_radius', tuple(X), list(radii)))
        return [[p for p, inside in zip(POSTCODES, self._inside(X, r))
                 if inside] for r in radii]

    def get_population_by_radius(self, X, radii, method='naive'):
        FakeLocator.calls.append(('population_radius', tuple(X),
                                  list(radii)))
        return [int(POPULATIONS[self._inside(X, r)].sum()) for r in radii]

    def get_postcodes_by_ellipse(self, X, a_perp, a_front, a_behind,
                                 bearing):
        FakeLocator.calls.append(('postcodes_ellipse', tuple(X),
                                  list(a_perp), list(a_front),
                                  list(a_behind), bearing))
        return [[p for p, inside in zip(POSTCODES, self._inside(X, r))
                 if inside] for r in a_perp]

    def get_population_by_ellipse(self, X, a_perp, a_front, a_behind,
                                  bearing, method='naive'):
        FakeLocator.calls.append(('population_ellipse', tuple(X),
                                  list(a_perp), list(a_front),
                                  list(a_behind), bearing))
        return [int(POPULATIONS[self._inside(X, r)].sum()) for r in a_perp]


def fake_damage_zones(outcome, lat, lon, bearing, pressures):
    """Surface zero is the entry point and the radius is read directly
    from the outcome, so that expected results can be found by hand."""
    return float(lat), float(lon), float(outcome['zone_radius'])


def write_scenarios(folder, radii, name='scenarios.csv', **changes):
    """
    Write an impact file with one scenario per value in radii, all
    entering at CENTRE, and return its filename. Other columns can be
    changed with keyword arguments, using '_' for the spaces in the
    column names (e.g. entry_latitude=[...]).
    """
    n = len(radii)
    table = pd.DataFrame({'radius': np.asarray(radii, dtype=float),
                          'angle': np.full(n, 45.),
                          'strength': np.full(n, 1e7),
                          'density': np.full(n, 3000.),
                          'velocity': np.full(n, 19e3),
                          'entry latitude': np.full(n, CENTRE[0]),
                          'entry longitude': np.full(n, CENTRE[1]),
                          'bearing': np.full(n, 115.)})
    for key, values in changes.items():
        table[key.replace('_', ' ')] = values
    filename = os.path.join(str(folder), name)
    table[COLUMNS].to_csv(filename, index=False)
    return filename


@pytest.fixture
def fake_locator(monkeypatch):
    """Make impact_risk use FakeLocator instead of the real locator."""
    FakeLocator.created = 0
    FakeLocator.calls = []
    monkeypatch.setattr(damage, 'GeospatialLocator', FakeLocator)
    # start every test without a locator kept from an earlier call
    monkeypatch.setattr(damage, '_locators', {})
    return FakeLocator


@pytest.fixture
def by_hand(monkeypatch, fake_locator):
    """Fake locator and fake damage_zones: the 'radius' column of the
    impact file is the blast radius in km around CENTRE."""
    monkeypatch.setattr(damage, 'damage_zones', fake_damage_zones)
    return fake_locator


# Blast radii (km) used in most of the "by hand" tests.
#   radius  postcodes inside          population inside
#   2 km    AA1 1AA                   100
#   6 km    AA1 1AA, BB2 2BB          120
#   11 km   AA1 1AA, BB2 2BB, CC3 3CC 123
#   0.5 km  none                      0
#   6 km    AA1 1AA, BB2 2BB          120
RADII = [2., 6., 11., 0.5, 6.]
EXPECTED_PROBABILITY = {'AA1 1AA': 4 / 5, 'BB2 2BB': 3 / 5, 'CC3 3CC': 1 / 5}
EXPECTED_POPULATIONS = [100., 120., 123., 0., 120.]


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------

def test_probability_is_dataframe_indexed_by_postcode(by_hand, tmp_path):
    probability, _ = deepimpact.impact_risk(
        FakePlanet(), impact_file=write_scenarios(tmp_path, RADII))

    assert type(probability) is pd.DataFrame
    assert probability.index.name == 'postcode'
    assert list(probability.columns) == ['probability']
    assert probability['probability'].dtype == float


def test_population_is_dict_of_python_floats(by_hand, tmp_path):
    _, population = deepimpact.impact_risk(
        FakePlanet(), impact_file=write_scenarios(tmp_path, RADII))

    assert type(population) is dict
    assert set(population.keys()) == {'mean', 'stdev'}
    assert type(population['mean']) is float
    assert type(population['stdev']) is float


def test_format_is_kept_when_nothing_is_hit(by_hand, tmp_path):
    probability, population = deepimpact.impact_risk(
        NoDamagePlanet(), impact_file=write_scenarios(tmp_path, RADII))

    assert type(probability) is pd.DataFrame
    assert probability.index.name == 'postcode'
    assert list(probability.columns) == ['probability']
    assert type(population['mean']) is float
    assert type(population['stdev']) is float


# ---------------------------------------------------------------------------
# Probabilities and population statistics, worked out by hand
# ---------------------------------------------------------------------------

def test_probabilities_by_hand(by_hand, tmp_path):
    probability, _ = deepimpact.impact_risk(
        FakePlanet(), impact_file=write_scenarios(tmp_path, RADII))

    assert set(probability.index) == set(EXPECTED_PROBABILITY)
    for postcode, expected in EXPECTED_PROBABILITY.items():
        assert probability.loc[postcode, 'probability'] == pytest.approx(
            expected)


def test_population_statistics_by_hand(by_hand, tmp_path):
    _, population = deepimpact.impact_risk(
        FakePlanet(), impact_file=write_scenarios(tmp_path, RADII))

    assert population['mean'] == pytest.approx(463. / 5.)
    # standard deviation over all the scenarios (divided by N, not N - 1)
    assert population['stdev'] == pytest.approx(
        np.std(EXPECTED_POPULATIONS))


def test_postcode_never_hit_is_not_listed(by_hand, tmp_path):
    probability, _ = deepimpact.impact_risk(
        FakePlanet(), impact_file=write_scenarios(tmp_path, RADII))

    assert 'DD4 4DD' not in probability.index


def test_each_postcode_is_listed_once(by_hand, tmp_path):
    probability, _ = deepimpact.impact_risk(
        FakePlanet(), impact_file=write_scenarios(tmp_path, RADII))

    assert probability.index.is_unique


def test_postcode_strings_are_not_changed(by_hand, tmp_path):
    """Postcodes come back exactly as the locator gave them, including
    the space."""
    probability, _ = deepimpact.impact_risk(
        FakePlanet(), impact_file=write_scenarios(tmp_path, [60.]))

    assert sorted(probability.index) == sorted(POSTCODES)
    assert all(type(postcode) is str for postcode in probability.index)


def test_probabilities_are_between_zero_and_one(by_hand, tmp_path):
    probability, _ = deepimpact.impact_risk(
        FakePlanet(), impact_file=write_scenarios(tmp_path, RADII))

    assert (probability['probability'] > 0.).all()
    assert (probability['probability'] <= 1.).all()


def test_probabilities_are_multiples_of_one_over_n(by_hand, tmp_path):
    probability, _ = deepimpact.impact_risk(
        FakePlanet(), impact_file=write_scenarios(tmp_path, RADII))

    counts = probability['probability'].to_numpy() * len(RADII)
    assert np.allclose(counts, np.round(counts))


def test_single_scenario(by_hand, tmp_path):
    probability, population = deepimpact.impact_risk(
        FakePlanet(), impact_file=write_scenarios(tmp_path, [6.]))

    assert set(probability.index) == {'AA1 1AA', 'BB2 2BB'}
    assert (probability['probability'] == 1.).all()
    assert population['mean'] == pytest.approx(120.)
    assert population['stdev'] == 0.


def test_identical_scenarios(by_hand, tmp_path):
    probability, population = deepimpact.impact_risk(
        FakePlanet(), impact_file=write_scenarios(tmp_path, [11.] * 7))

    assert set(probability.index) == {'AA1 1AA', 'BB2 2BB', 'CC3 3CC'}
    assert (probability['probability'] == 1.).all()
    assert population['mean'] == pytest.approx(123.)
    assert population['stdev'] == pytest.approx(0.)


def test_everything_hit_in_every_scenario(by_hand, tmp_path):
    probability, population = deepimpact.impact_risk(
        FakePlanet(), impact_file=write_scenarios(tmp_path, [60., 70.]))

    assert sorted(probability.index) == sorted(POSTCODES)
    assert (probability['probability'] == 1.).all()
    assert population['mean'] == pytest.approx(POPULATIONS.sum())


def test_scenarios_at_different_locations(by_hand, tmp_path):
    """The blast zone is centred on each scenario's own entry point: the
    second scenario is 49 km further north, next to DD4 4DD only."""
    far_north = CENTRE[0] + np.degrees(49e3 / RP)
    filename = write_scenarios(tmp_path, [2., 2.],
                               entry_latitude=[CENTRE[0], far_north])

    probability, population = deepimpact.impact_risk(FakePlanet(),
                                                     impact_file=filename)

    assert set(probability.index) == {'AA1 1AA', 'DD4 4DD'}
    assert (probability['probability'] == 0.5).all()
    assert population['mean'] == pytest.approx((100. + 1000.) / 2.)
    assert population['stdev'] == pytest.approx(np.std([100., 1000.]))


def test_results_are_repeatable(by_hand, tmp_path):
    filename = write_scenarios(tmp_path, RADII)

    first = deepimpact.impact_risk(FakePlanet(), impact_file=filename)
    second = deepimpact.impact_risk(FakePlanet(), impact_file=filename)

    pd.testing.assert_frame_equal(first[0].sort_index(),
                                  second[0].sort_index())
    assert first[1] == second[1]


# ---------------------------------------------------------------------------
# Scenarios that cause no damage
# ---------------------------------------------------------------------------

def test_scenario_with_zero_radius_counts_in_the_total(by_hand, tmp_path):
    """Two of the four scenarios have a blast radius of zero. They hit
    nothing, but still count as scenarios and as a population of 0."""
    probability, population = deepimpact.impact_risk(
        FakePlanet(), impact_file=write_scenarios(tmp_path,
                                                  [6., 0., 6., 0.]))

    assert set(probability.index) == {'AA1 1AA', 'BB2 2BB'}
    assert (probability['probability'] == 0.5).all()
    assert population['mean'] == pytest.approx(60.)
    assert population['stdev'] == pytest.approx(
        np.std([120., 0., 120., 0.]))


def test_no_scenario_causes_damage(by_hand, tmp_path):
    probability, population = deepimpact.impact_risk(
        NoDamagePlanet(), impact_file=write_scenarios(tmp_path, RADII))

    assert len(probability) == 0
    assert population == {'mean': 0., 'stdev': 0.}


def test_blast_zone_smaller_than_distance_to_any_postcode(by_hand, tmp_path):
    probability, population = deepimpact.impact_risk(
        FakePlanet(), impact_file=write_scenarios(tmp_path, [0.5, 0.9]))

    assert len(probability) == 0
    assert population == {'mean': 0., 'stdev': 0.}


def test_outcome_that_is_not_a_number_does_not_crash(by_hand, tmp_path):
    probability, population = deepimpact.impact_risk(
        NanPlanet(), impact_file=write_scenarios(tmp_path, RADII))

    assert len(probability) == 0
    assert population == {'mean': 0., 'stdev': 0.}


def test_pressure_never_reached_with_real_damage_zones(fake_locator,
                                                       tmp_path):
    """No asteroid in the file can produce 1 GPa on the ground."""
    probability, population = deepimpact.impact_risk(
        FakePlanet(), impact_file=write_scenarios(tmp_path, [30., 35.]),
        pressure=1e9)

    assert len(probability) == 0
    assert population == {'mean': 0., 'stdev': 0.}


# ---------------------------------------------------------------------------
# nsamples
# ---------------------------------------------------------------------------

def test_nsamples_none_uses_every_scenario(by_hand, tmp_path):
    planet = FakePlanet()
    deepimpact.impact_risk(planet, nsamples=None,
                           impact_file=write_scenarios(tmp_path, RADII))

    assert len(planet.solve_calls) == len(RADII)


def test_nsamples_uses_the_first_rows(by_hand, tmp_path):
    """With the first two scenarios only (2 km and 6 km): AA1 1AA is hit
    twice and BB2 2BB once."""
    planet = FakePlanet()
    probability, population = deepimpact.impact_risk(
        planet, nsamples=2, impact_file=write_scenarios(tmp_path, RADII))

    assert [call['radius'] for call in planet.solve_calls] == [2., 6.]
    assert set(probability.index) == {'AA1 1AA', 'BB2 2BB'}
    assert probability.loc['AA1 1AA', 'probability'] == pytest.approx(1.)
    assert probability.loc['BB2 2BB', 'probability'] == pytest.approx(0.5)
    assert population['mean'] == pytest.approx(110.)
    assert population['stdev'] == pytest.approx(np.std([100., 120.]))


def test_nsamples_divides_by_the_number_of_scenarios_used(by_hand,
                                                          tmp_path):
    """The probability is out of nsamples, not out of the file length."""
    probability, _ = deepimpact.impact_risk(
        FakePlanet(), nsamples=4,
        impact_file=write_scenarios(tmp_path, RADII))

    assert probability.loc['AA1 1AA', 'probability'] == pytest.approx(3 / 4)
    assert probability.loc['BB2 2BB', 'probability'] == pytest.approx(2 / 4)
    assert probability.loc['CC3 3CC', 'probability'] == pytest.approx(1 / 4)


def test_nsamples_equal_to_file_length_matches_none(by_hand, tmp_path):
    filename = write_scenarios(tmp_path, RADII)

    everything = deepimpact.impact_risk(FakePlanet(), impact_file=filename)
    counted = deepimpact.impact_risk(FakePlanet(), impact_file=filename,
                                     nsamples=len(RADII))

    pd.testing.assert_frame_equal(everything[0].sort_index(),
                                  counted[0].sort_index())
    assert everything[1] == counted[1]


def test_nsamples_larger_than_file_uses_every_scenario(by_hand, tmp_path):
    planet = FakePlanet()
    probability, population = deepimpact.impact_risk(
        planet, nsamples=1000, impact_file=write_scenarios(tmp_path, RADII))

    assert len(planet.solve_calls) == len(RADII)
    for postcode, expected in EXPECTED_PROBABILITY.items():
        assert probability.loc[postcode, 'probability'] == pytest.approx(
            expected)
    assert population['mean'] == pytest.approx(463. / 5.)


def test_nsamples_one(by_hand, tmp_path):
    probability, population = deepimpact.impact_risk(
        FakePlanet(), nsamples=1,
        impact_file=write_scenarios(tmp_path, RADII))

    assert list(probability.index) == ['AA1 1AA']
    assert probability.loc['AA1 1AA', 'probability'] == 1.
    assert population == {'mean': 100., 'stdev': 0.}


def test_nsamples_zero_does_not_crash(by_hand, tmp_path):
    planet = FakePlanet()
    probability, population = deepimpact.impact_risk(
        planet, nsamples=0, impact_file=write_scenarios(tmp_path, RADII))

    assert planet.solve_calls == []
    assert len(probability) == 0
    assert population == {'mean': 0., 'stdev': 0.}


# ---------------------------------------------------------------------------
# Reading the impact file
# ---------------------------------------------------------------------------

def test_default_file_is_the_one_in_resources(by_hand):
    planet = FakePlanet()
    deepimpact.impact_risk(planet)

    expected = pd.read_csv(DEFAULT_FILE)
    assert len(planet.solve_calls) == len(expected)
    assert [call['radius'] for call in planet.solve_calls] == pytest.approx(
        list(expected['radius']))


def test_columns_are_read_by_name_not_by_position(by_hand, tmp_path):
    filename = write_scenarios(tmp_path, RADII)
    shuffled = os.path.join(str(tmp_path), 'shuffled.csv')
    pd.read_csv(filename)[COLUMNS[::-1]].to_csv(shuffled, index=False)

    original = deepimpact.impact_risk(FakePlanet(), impact_file=filename)
    reordered = deepimpact.impact_risk(FakePlanet(), impact_file=shuffled)

    pd.testing.assert_frame_equal(original[0].sort_index(),
                                  reordered[0].sort_index())
    assert original[1] == reordered[1]


def test_extra_columns_are_ignored(by_hand, tmp_path):
    filename = write_scenarios(tmp_path, RADII)
    extra = os.path.join(str(tmp_path), 'extra.csv')
    table = pd.read_csv(filename)
    table['comment'] = 'ignore me'
    table.to_csv(extra, index=False)

    probability, population = deepimpact.impact_risk(FakePlanet(),
                                                     impact_file=extra)

    for postcode, expected in EXPECTED_PROBABILITY.items():
        assert probability.loc[postcode, 'probability'] == pytest.approx(
            expected)
    assert population['mean'] == pytest.approx(463. / 5.)


@pytest.mark.parametrize('missing', COLUMNS)
def test_missing_column_raises(by_hand, tmp_path, missing):
    filename = write_scenarios(tmp_path, RADII)
    broken = os.path.join(str(tmp_path), 'broken.csv')
    pd.read_csv(filename).drop(columns=missing).to_csv(broken, index=False)

    with pytest.raises(KeyError):
        deepimpact.impact_risk(FakePlanet(), impact_file=broken)


def test_missing_file_raises(by_hand, tmp_path):
    with pytest.raises(FileNotFoundError):
        deepimpact.impact_risk(
            FakePlanet(),
            impact_file=os.path.join(str(tmp_path), 'not_there.csv'))


def test_file_with_no_scenarios(by_hand, tmp_path):
    planet = FakePlanet()
    probability, population = deepimpact.impact_risk(
        planet, impact_file=write_scenarios(tmp_path, []))

    assert planet.solve_calls == []
    assert len(probability) == 0
    assert probability.index.name == 'postcode'
    assert population == {'mean': 0., 'stdev': 0.}


def test_impact_file_is_not_modified(by_hand, tmp_path):
    filename = write_scenarios(tmp_path, RADII)
    with open(filename) as handle:
        before = handle.read()

    deepimpact.impact_risk(FakePlanet(), impact_file=filename)

    with open(filename) as handle:
        assert handle.read() == before


# ---------------------------------------------------------------------------
# What impact_risk passes to the solver
# ---------------------------------------------------------------------------

def test_each_scenario_is_passed_to_the_solver(by_hand, tmp_path):
    filename = write_scenarios(tmp_path, [2., 6., 11.],
                               angle=[20., 45., 70.],
                               strength=[1e6, 5e6, 1e7],
                               density=[2500., 3000., 3500.],
                               velocity=[15e3, 19e3, 25e3])
    planet = FakePlanet()

    deepimpact.impact_risk(planet, impact_file=filename)

    assert len(planet.solve_calls) == 3
    expected = pd.read_csv(filename)
    for call, (_, row) in zip(planet.solve_calls, expected.iterrows()):
        for name in ['radius', 'angle', 'strength', 'density', 'velocity']:
            assert call[name] == pytest.approx(row[name])


def test_solver_is_called_with_angles_in_degrees(by_hand, tmp_path):
    planet = FakePlanet()
    deepimpact.impact_risk(planet,
                           impact_file=write_scenarios(tmp_path, RADII))

    assert all(call['radians'] is False for call in planet.solve_calls)


def test_solver_starts_from_100_km(by_hand, tmp_path):
    planet = FakePlanet()
    deepimpact.impact_risk(planet,
                           impact_file=write_scenarios(tmp_path, RADII))

    assert all(call['init_altitude'] == 100e3
               for call in planet.solve_calls)


def test_solver_functions_are_chained(by_hand, tmp_path):
    """calculate_energy gets the solver result, and analyse_outcome gets
    the result of calculate_energy."""
    planet = FakePlanet()
    deepimpact.impact_risk(planet,
                           impact_file=write_scenarios(tmp_path, RADII))

    assert len(planet.energy_inputs) == len(RADII)
    assert len(planet.outcome_inputs) == len(RADII)
    assert all(r['stage'] == 'solved' for r in planet.energy_inputs)
    assert all(r['stage'] == 'energy' for r in planet.outcome_inputs)


# ---------------------------------------------------------------------------
# What impact_risk passes to damage_zones and to the locator
# ---------------------------------------------------------------------------

def test_entry_point_bearing_and_pressure_reach_damage_zones(
        monkeypatch, fake_locator, tmp_path):
    seen = []

    def spy(outcome, lat, lon, bearing, pressures):
        seen.append((lat, lon, bearing, pressures))
        return fake_damage_zones(outcome, lat, lon, bearing, pressures)

    monkeypatch.setattr(damage, 'damage_zones', spy)
    filename = write_scenarios(tmp_path, [2., 6.],
                               entry_latitude=[52.1, 52.2],
                               entry_longitude=[-1.1, -1.2],
                               bearing=[10., 200.])

    deepimpact.impact_risk(FakePlanet(), impact_file=filename,
                           pressure=3.5e3)

    assert seen == [(52.1, -1.1, 10., 3.5e3), (52.2, -1.2, 200., 3.5e3)]


def test_locator_is_asked_about_surface_zero(fake_locator, tmp_path):
    """With the real damage_zones, the locator must be queried at the
    surface zero point and with the radius that damage_zones returns."""
    planet = FakePlanet()
    deepimpact.impact_risk(planet, pressure=27e3,
                           impact_file=write_scenarios(tmp_path, [35.]))

    outcome = FakePlanet().analyse_outcome({'radius': 35., 'angle': 45.})
    blat, blon, radius = deepimpact.damage_zones(outcome, CENTRE[0],
                                                 CENTRE[1], 115., 27e3)
    assert radius > 0.
    for call in fake_locator.calls:
        assert call[1] == pytest.approx((blat, blon))
        assert call[2] == pytest.approx([radius])


def test_locator_is_created_once(by_hand, tmp_path):
    """Loading the postcode data is slow, so it must not be repeated for
    every scenario."""
    deepimpact.impact_risk(FakePlanet(),
                           impact_file=write_scenarios(tmp_path, RADII))

    assert by_hand.created == 1


def test_locator_is_reused_between_calls(by_hand, tmp_path):
    """The data files are read when the locator is created, so repeated
    calls (circles, ellipses, other pressures) must not create it again."""
    filename = write_scenarios(tmp_path, RADII)

    deepimpact.impact_risk(FakePlanet(), impact_file=filename)
    deepimpact.impact_risk(FakePlanet(), impact_file=filename,
                           ellipse=True)
    deepimpact.impact_risk(FakePlanet(), impact_file=filename,
                           pressure=43e3)

    assert by_hand.created == 1


def test_circular_zones_are_used_by_default(by_hand, tmp_path):
    deepimpact.impact_risk(FakePlanet(),
                           impact_file=write_scenarios(tmp_path, RADII))

    kinds = {call[0] for call in by_hand.calls}
    assert kinds == {'postcodes_radius', 'population_radius'}


# ---------------------------------------------------------------------------
# Angle-dependent (elliptical) blast zones
# ---------------------------------------------------------------------------

def test_ellipse_uses_the_ellipse_locator_functions(by_hand, tmp_path):
    deepimpact.impact_risk(FakePlanet(), ellipse=True,
                           impact_file=write_scenarios(tmp_path, RADII))

    kinds = {call[0] for call in by_hand.calls}
    assert kinds == {'postcodes_ellipse', 'population_ellipse'}


def test_ellipse_passes_shape_coefficients_and_bearing(monkeypatch, by_hand,
                                                       tmp_path):
    seen = []

    def spy(outcome, damrad, kf=0.4, kb=0.2, radians=False):
        seen.append((damrad, kf, kb))
        return 3e3, 2e3, 1e3

    monkeypatch.setattr(damage, 'damage_ellipses', spy)
    filename = write_scenarios(tmp_path, [6., 11.], bearing=[30., 250.])

    deepimpact.impact_risk(FakePlanet(), impact_file=filename,
                           ellipse=True, kf=0.3, kb=0.1)

    # damage_ellipses gets the circular radius and the coefficients
    assert seen == [(6e3, 0.3, 0.1), (11e3, 0.3, 0.1)]
    # the locator gets the three semi-axes and the scenario's bearing
    for call, bearing in zip(by_hand.calls[::2], [30., 250.]):
        assert call[2:] == ([3e3], [2e3], [1e3], bearing)


def test_ellipse_returns_the_same_format(by_hand, tmp_path):
    probability, population = deepimpact.impact_risk(
        FakePlanet(), ellipse=True,
        impact_file=write_scenarios(tmp_path, RADII))

    assert type(probability) is pd.DataFrame
    assert probability.index.name == 'postcode'
    assert list(probability.columns) == ['probability']
    assert type(population['mean']) is float
    assert type(population['stdev']) is float


def test_ellipse_results_come_from_the_ellipse_functions(monkeypatch,
                                                         by_hand, tmp_path):
    """The fake locator uses the crossrange semi-axis as its radius. With
    semi-axes of 6 km in every scenario, AA1 1AA and BB2 2BB are always
    hit, whatever the circular radius was."""
    monkeypatch.setattr(damage, 'damage_ellipses',
                        lambda outcome, damrad, kf=0.4, kb=0.2,
                        radians=False: (6e3, 6e3, 6e3))

    probability, population = deepimpact.impact_risk(
        FakePlanet(), ellipse=True,
        impact_file=write_scenarios(tmp_path, [2., 11., 60.]))

    assert set(probability.index) == {'AA1 1AA', 'BB2 2BB'}
    assert (probability['probability'] == 1.).all()
    assert population == {'mean': 120., 'stdev': 0.}


# ---------------------------------------------------------------------------
# With the real damage_zones
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('pressure', [1e3, 3.5e3, 27e3, 43e3])
def test_matches_separate_calculation_with_real_damage_zones(
        fake_locator, tmp_path, pressure):
    """Repeat the calculation scenario by scenario in the test, using
    damage_zones directly, and compare with impact_risk."""
    radii = [5., 20., 35., 50., 80.]
    filename = write_scenarios(tmp_path, radii)

    probability, population = deepimpact.impact_risk(
        FakePlanet(), impact_file=filename, pressure=pressure)

    hits = dict.fromkeys(POSTCODES, 0)
    people = []
    for radius in radii:
        outcome = FakePlanet().analyse_outcome({'radius': radius,
                                                'angle': 45.})
        blat, blon, damrad = deepimpact.damage_zones(
            outcome, CENTRE[0], CENTRE[1], 115., pressure)
        inside = haversine(blat, blon, POSTCODE_LATS,
                           POSTCODE_LONS) < damrad
        for postcode in np.array(POSTCODES)[inside]:
            hits[postcode] += 1
        people.append(POPULATIONS[inside].sum())

    expected = {p: n / len(radii) for p, n in hits.items() if n > 0}
    assert set(probability.index) == set(expected)
    for postcode, value in expected.items():
        assert probability.loc[postcode, 'probability'] == pytest.approx(
            value)
    assert population['mean'] == pytest.approx(np.mean(people))
    assert population['stdev'] == pytest.approx(np.std(people))


def test_higher_pressure_never_increases_the_risk(fake_locator, tmp_path):
    """A higher pressure gives a smaller blast zone, so no postcode can
    become more likely to be hit, and no more people can be affected."""
    filename = write_scenarios(tmp_path, [5., 20., 35., 50., 80.])

    previous, previous_population = deepimpact.impact_risk(
        FakePlanet(), impact_file=filename, pressure=1e3)

    for pressure in [3.5e3, 27e3, 43e3]:
        current, current_population = deepimpact.impact_risk(
            FakePlanet(), impact_file=filename, pressure=pressure)

        assert set(current.index) <= set(previous.index)
        for postcode in current.index:
            assert (current.loc[postcode, 'probability']
                    <= previous.loc[postcode, 'probability'] + 1e-12)
        assert current_population['mean'] <= previous_population['mean']
        previous, previous_population = current, current_population


# ---------------------------------------------------------------------------
# With the real Planet and the real locator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('ellipse', [False, True])
def test_runs_with_the_real_planet_and_locator(ellipse):
    """Only the format is checked here: the values depend on the solver
    and on the postcode data."""
    probability, population = deepimpact.impact_risk(
        deepimpact.Planet(), pressure=27e3, nsamples=2, ellipse=ellipse)

    assert type(probability) is pd.DataFrame
    assert probability.index.name == 'postcode'
    assert list(probability.columns) == ['probability']
    assert (probability['probability'] >= 0.).all()
    assert (probability['probability'] <= 1.).all()
    assert type(population) is dict
    assert type(population['mean']) is float
    assert type(population['stdev']) is float
    assert population['mean'] >= 0.
    assert population['stdev'] >= 0.


# ---------------------------------------------------------------------------
# Scenario order, population method and speed
# ---------------------------------------------------------------------------

def test_scenario_order_does_not_matter(by_hand, tmp_path):
    forward = write_scenarios(tmp_path, RADII, name='forward.csv')
    backward = write_scenarios(tmp_path, RADII[::-1], name='backward.csv')

    first = deepimpact.impact_risk(FakePlanet(), impact_file=forward)
    second = deepimpact.impact_risk(FakePlanet(), impact_file=backward)

    pd.testing.assert_frame_equal(first[0].sort_index(),
                                  second[0].sort_index())
    assert first[1]['mean'] == pytest.approx(second[1]['mean'])
    assert first[1]['stdev'] == pytest.approx(second[1]['stdev'])


@pytest.mark.parametrize('ellipse', [False, True])
def test_population_uses_the_naive_method(monkeypatch, by_hand, tmp_path,
                                          ellipse):
    """The scorer only marks the naive method, so impact_risk must keep
    using it even if a better method is added to the locator."""
    methods = []

    class MethodSpy(FakeLocator):
        def get_population_by_radius(self, X, radii, method='naive'):
            methods.append(method)
            return super().get_population_by_radius(X, radii, method)

        def get_population_by_ellipse(self, X, a_perp, a_front, a_behind,
                                      bearing, method='naive'):
            methods.append(method)
            return super().get_population_by_ellipse(
                X, a_perp, a_front, a_behind, bearing, method)

    monkeypatch.setattr(damage, 'GeospatialLocator', MethodSpy)

    deepimpact.impact_risk(FakePlanet(), ellipse=ellipse,
                           impact_file=write_scenarios(tmp_path, RADII))

    assert methods == ['naive'] * len(RADII)


@pytest.mark.timeout(20)
def test_many_scenarios(by_hand, tmp_path):
    """2000 scenarios repeating RADII give the same answers as RADII
    alone, and the loop itself must not be slow."""
    probability, population = deepimpact.impact_risk(
        FakePlanet(), impact_file=write_scenarios(tmp_path, RADII * 400))

    assert set(probability.index) == set(EXPECTED_PROBABILITY)
    for postcode, expected in EXPECTED_PROBABILITY.items():
        assert probability.loc[postcode, 'probability'] == pytest.approx(
            expected)
    assert population['mean'] == pytest.approx(463. / 5.)
    assert population['stdev'] == pytest.approx(
        np.std(EXPECTED_POPULATIONS))


# ---------------------------------------------------------------------------
# With the real damage_ellipses
# ---------------------------------------------------------------------------

def test_locator_gets_the_axes_from_damage_ellipses(fake_locator, tmp_path):
    """With the real damage_zones and damage_ellipses, the locator must
    receive the semi-axes given by the formulas in DamageMapper.ipynb."""
    deepimpact.impact_risk(FakePlanet(), ellipse=True, pressure=27e3,
                           impact_file=write_scenarios(tmp_path, [35.]))

    outcome = FakePlanet().analyse_outcome({'radius': 35., 'angle': 45.})
    _, _, radius = deepimpact.damage_zones(outcome, CENTRE[0], CENTRE[1],
                                           115., 27e3)
    # at 45 degrees cos^2 = 1/2, so q_f = 0.8 and q_b = 0.9
    a_perp = radius * np.sqrt(2. / 1.7)
    expected = [a_perp, 0.8 * a_perp, 0.9 * a_perp]

    assert len(fake_locator.calls) == 2
    for call in fake_locator.calls:
        assert call[0] in ('postcodes_ellipse', 'population_ellipse')
        axes = [call[2][0], call[3][0], call[4][0]]
        assert axes == pytest.approx(expected)
        assert call[5] == 115.


# ---------------------------------------------------------------------------
# With the real locator and postcode data
# ---------------------------------------------------------------------------

# Entry point in central London. FakePlanet bursts directly above the
# entry point, so this is also surface zero.
LONDON = (51.5074, -0.1278)


@pytest.fixture(scope='module')
def real_locator():
    return deepimpact.GeospatialLocator()


def london_scenarios(folder, radii):
    n = len(radii)
    return write_scenarios(folder, radii,
                           entry_latitude=[LONDON[0]] * n,
                           entry_longitude=[LONDON[1]] * n)


def test_single_scenario_matches_real_locator(tmp_path, real_locator):
    probability, population = deepimpact.impact_risk(
        FakePlanet(), pressure=27e3,
        impact_file=london_scenarios(tmp_path, [35.]))

    outcome = FakePlanet().analyse_outcome({'radius': 35., 'angle': 45.})
    blat, blon, radius = deepimpact.damage_zones(outcome, LONDON[0],
                                                 LONDON[1], 115., 27e3)
    postcodes = real_locator.get_postcodes_by_radius((blat, blon),
                                                     [radius])[0]
    people = real_locator.get_population_by_radius((blat, blon),
                                                   [radius])[0]

    assert len(postcodes) > 0
    assert set(probability.index) == set(postcodes)
    assert (probability['probability'] == 1.).all()
    assert population['mean'] == pytest.approx(float(people))
    assert population['stdev'] == pytest.approx(0.)


def test_real_ellipse_matches_real_locator(tmp_path, real_locator):
    probability, population = deepimpact.impact_risk(
        FakePlanet(), pressure=27e3, ellipse=True,
        impact_file=london_scenarios(tmp_path, [35.]))

    outcome = FakePlanet().analyse_outcome({'radius': 35., 'angle': 45.})
    blat, blon, radius = deepimpact.damage_zones(outcome, LONDON[0],
                                                 LONDON[1], 115., 27e3)
    axes = deepimpact.damage_ellipses(outcome, radius)
    postcodes = real_locator.get_postcodes_by_ellipse(
        (blat, blon), [axes[0]], [axes[1]], [axes[2]], 115.)[0]
    people = real_locator.get_population_by_ellipse(
        (blat, blon), [axes[0]], [axes[1]], [axes[2]], 115.)[0]

    assert len(postcodes) > 0
    assert set(probability.index) == set(postcodes)
    assert population['mean'] == pytest.approx(float(people))


def test_ellipse_with_zero_shape_factors_matches_circle(tmp_path):
    """With kf = kb = 0 the two half-ellipses are the circle."""
    filename = london_scenarios(tmp_path, [35., 20.])

    circle = deepimpact.impact_risk(FakePlanet(), impact_file=filename)
    ellipse = deepimpact.impact_risk(FakePlanet(), impact_file=filename,
                                     ellipse=True, kf=0., kb=0.)

    assert set(ellipse[0].index) == set(circle[0].index)
    assert ellipse[1]['mean'] == pytest.approx(circle[1]['mean'])
    assert ellipse[1]['stdev'] == pytest.approx(circle[1]['stdev'])


# ---------------------------------------------------------------------------
# With the real Planet: the whole chain
# ---------------------------------------------------------------------------

def test_real_planet_matches_manual_chain(tmp_path, real_locator):
    """The first row of the default file, run through every step here."""
    planet = deepimpact.Planet()
    first = pd.read_csv(DEFAULT_FILE).iloc[0]

    result = planet.solve_atmospheric_entry(
        radius=first['radius'], velocity=first['velocity'],
        density=first['density'], strength=first['strength'],
        angle=first['angle'])
    outcome = planet.analyse_outcome(planet.calculate_energy(result))
    blat, blon, radius = deepimpact.damage_zones(
        outcome, first['entry latitude'], first['entry longitude'],
        first['bearing'], 27e3)
    postcodes = real_locator.get_postcodes_by_radius((blat, blon),
                                                     [radius])[0]
    people = real_locator.get_population_by_radius((blat, blon),
                                                   [radius])[0]

    filename = os.path.join(str(tmp_path), 'first.csv')
    first.to_frame().T[COLUMNS].to_csv(filename, index=False)
    probability, population = deepimpact.impact_risk(
        planet, impact_file=filename, pressure=27e3)

    assert set(probability.index) == set(postcodes)
    assert population['mean'] == pytest.approx(float(people))


@pytest.mark.timeout(60)
@pytest.mark.parametrize('ellipse', [False, True])
def test_default_file_runs_quickly_with_real_planet(ellipse):
    """The scorer times impact_risk on the full default file."""
    n = len(pd.read_csv(DEFAULT_FILE))

    probability, population = deepimpact.impact_risk(
        deepimpact.Planet(), pressure=27e3, ellipse=ellipse)

    counts = probability['probability'].to_numpy() * n
    assert np.allclose(counts, np.round(counts))
    assert (probability['probability'] > 0.).all()
    assert (probability['probability'] <= 1.).all()
    assert population['mean'] >= 0.


# ---------------------------------------------------------------------------
# Default arguments (the scorer calls impact_risk with pressure=27e3)
# ---------------------------------------------------------------------------

def test_default_pressure_is_27_kpa(monkeypatch, fake_locator, tmp_path):
    seen = []

    def spy(outcome, lat, lon, bearing, pressures):
        seen.append(pressures)
        return fake_damage_zones(outcome, lat, lon, bearing, pressures)

    monkeypatch.setattr(damage, 'damage_zones', spy)

    deepimpact.impact_risk(FakePlanet(),
                           impact_file=write_scenarios(tmp_path, [2., 6.]))

    assert seen == [27e3, 27e3]


def test_default_shape_coefficients(monkeypatch, by_hand, tmp_path):
    seen = []

    def spy(outcome, damrad, kf=None, kb=None, radians=False):
        seen.append((kf, kb))
        return damrad, damrad, damrad

    monkeypatch.setattr(damage, 'damage_ellipses', spy)

    deepimpact.impact_risk(FakePlanet(), ellipse=True,
                           impact_file=write_scenarios(tmp_path, [2., 6.]))

    assert seen == [(0.4, 0.2), (0.4, 0.2)]


# ---------------------------------------------------------------------------
# Postcodes returned are real UK postcodes
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def uk_postcodes():
    """Every postcode in the data file, with spaces removed."""
    table = pd.read_csv(os.sep.join((os.path.dirname(__file__), '..',
                                     'resources', 'full_postcodes.csv')),
                        usecols=['Postcode'])
    return set(table['Postcode'].str.replace(' ', '', regex=False))


@pytest.mark.parametrize('ellipse', [False, True])
def test_every_postcode_is_a_real_postcode(tmp_path, uk_postcodes,
                                           ellipse):
    probability, _ = deepimpact.impact_risk(
        FakePlanet(), ellipse=ellipse,
        impact_file=london_scenarios(tmp_path, [35., 20.]))

    assert len(probability) > 0
    found = {str(p).replace(' ', '') for p in probability.index}
    assert found <= uk_postcodes


# ---------------------------------------------------------------------------
# Integration: solver -> damage_zones -> locator -> impact_risk
# ---------------------------------------------------------------------------

def test_solver_outcome_works_in_damage_zones():
    """An outcome from the real solver has the keys and values that
    damage_zones needs, and a large asteroid gives a real blast zone."""
    planet = deepimpact.Planet()
    result = planet.solve_atmospheric_entry(
        radius=35., velocity=19e3, density=3000., strength=1e7, angle=45.)
    outcome = planet.analyse_outcome(planet.calculate_energy(result))

    blat, blon, damrad = deepimpact.damage_zones(
        outcome, 52.79, -2.95, 135., [1e3, 3.5e3, 27e3, 43e3])

    assert outcome['burst_energy'] > 0.
    assert np.isfinite([blat, blon]).all()
    assert all(np.isfinite(damrad))
    assert damrad[0] > 0.


def test_whole_default_file_matches_manual_chain(real_locator):
    """Every row of the default file, run through each step here, gives
    the same probabilities and population as impact_risk."""
    planet = deepimpact.Planet()
    table = pd.read_csv(DEFAULT_FILE)

    hits = {}
    people = []
    for _, row in table.iterrows():
        result = planet.solve_atmospheric_entry(
            radius=row['radius'], velocity=row['velocity'],
            density=row['density'], strength=row['strength'],
            angle=row['angle'])
        outcome = planet.analyse_outcome(planet.calculate_energy(result))
        blat, blon, radius = deepimpact.damage_zones(
            outcome, row['entry latitude'], row['entry longitude'],
            row['bearing'], 27e3)
        for postcode in real_locator.get_postcodes_by_radius(
                (blat, blon), [radius])[0]:
            hits[postcode] = hits.get(postcode, 0) + 1
        people.append(float(real_locator.get_population_by_radius(
            (blat, blon), [radius])[0]))

    probability, population = deepimpact.impact_risk(planet, pressure=27e3)

    assert set(probability.index) == set(hits)
    for postcode, count in hits.items():
        assert probability.loc[postcode, 'probability'] == pytest.approx(
            count / len(table))
    assert population['mean'] == pytest.approx(np.mean(people))
    assert population['stdev'] == pytest.approx(np.std(people))
