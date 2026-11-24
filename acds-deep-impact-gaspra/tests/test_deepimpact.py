import pandas as pd
import numpy as np
import os

from pytest import fixture


# Use pytest fixtures to generate objects we know we'll reuse.
# This makes sure tests run quickly

@fixture(scope='module')
def deepimpact():
    import deepimpact
    return deepimpact


@fixture(scope='module')
def planet(deepimpact):
    return deepimpact.Planet()


@fixture(scope='module')
def loc(deepimpact):
    return deepimpact.GeospatialLocator()


@fixture(scope='module')
def result(planet):
    input = {'radius': 1.,
             'velocity': 2.0e4,
             'density': 3000.,
             'strength': 1e5,
             'angle': 30.0,
             'init_altitude': 0.0,
             }

    result = planet.solve_atmospheric_entry(**input)

    return result


@fixture(scope='module')
def energy(planet, result):
    energy = planet.calculate_energy(result=result)
    return energy


@fixture(scope='module')
def outcome(planet, energy):
    outcome = planet.analyse_outcome(result=energy)
    return outcome


def test_import(deepimpact):
    assert deepimpact


def test_planet_signature(deepimpact):
    inputs = {
        'atmos_func': 'exponential',
        'atmos_filename': None,
        'Cd': 1.0, 'Ch': 0.1, 'Q': 1e7, 'Cl': 1e-3,
        'Rp': 6371e3, 'g': 9.81, 'H': 8000.0, 'rho0': 1.2,
        'pancake_factor': 6.5
    }

    # initialise using keyword arguments
    planet_kw = deepimpact.Planet(**inputs)
    # initialise using positional arguments
    planet_pos = deepimpact.Planet(*inputs.values())

    # Check all attributes are set correctly
    for key in inputs.keys():
        if key[:5] == 'atmos':
            # you may want to modify this to check that
            # atmos_filename is set correctly
            continue
        else:
            assert planet_kw.__dict__[key] == inputs[key]
            assert planet_pos.__dict__[key] == inputs[key]


def test_attributes(planet):
    for key in ('Cd', 'Ch', 'Q', 'Cl',
                'Rp', 'g', 'H', 'rho0', 'pancake_factor'):
        assert hasattr(planet, key)


def test_atmos_filename(planet):

    assert os.path.isfile(planet.atmos_filename)


def test_solve_atmospheric_entry(result):

    assert type(result) is pd.DataFrame

    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time'):
        assert key in result.columns


def test_calculate_energy(energy):

    assert type(energy) is pd.DataFrame

    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time', 'dedz'):
        assert key in energy.columns


def test_analyse_outcome(outcome):

    assert type(outcome) is dict

    for key in ('outcome', 'burst_peak_dedz', 'burst_altitude',
                'burst_distance', 'burst_energy', 'burst_angle'):
        assert key in outcome.keys()


def test_scenario(planet):

    inputs = {'radius': 35.,
              'angle': 45.,
              'strength': 1e7,
              'density': 3000.,
              'velocity': 19e3}

    result = planet.solve_atmospheric_entry(**inputs)

    # For now, we just check the result is a DataFrame
    # and the columns are as expected.

    # You should add more tests here to check the values
    # are correct and match the expected output
    # given in the tests/scenario.npz file

    assert type(result) is pd.DataFrame

    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time'):
        assert key in result.columns


def test_damage_zones(deepimpact):

    outcome = {'burst_peak_dedz': 1000.,
               'burst_altitude': 9000.,
               'burst_distance': 90000.,
               'burst_energy': 6000.,
               'burst_angle': 45.,
               'outcome': 'Airburst'}

    blat, blon, damrad = deepimpact.damage_zones(outcome, 55.0, 0.,
                                                 135., [27e3, 43e3])

    # for now, we just check the output format (if pressures is list-like,
    # damrad should be a list, otherwise a float). You should add more
    # tests of the damage_zones functionality as you write that function

    assert isinstance(blat, float)
    assert isinstance(blon, float)
    assert type(damrad) is list
    assert len(damrad) == 2

    blat, blon, damrad = deepimpact.damage_zones(outcome, 55.0, 0.,
                                                 135., 27e3)

    assert isinstance(blat, float)
    assert isinstance(blon, float)
    assert isinstance(damrad, float)


def test_damage_ellipses(deepimpact):

    outcome = {'burst_peak_dedz': 1000.,
               'burst_altitude': 9000.,
               'burst_distance': 90000.,
               'burst_energy': 6000.,
               'burst_angle': 45.,
               'outcome': 'Airburst'}

    # for now, we just check the output format (if damrad is list-like,
    # each item in the returned tuple should be a list, otherwise a float).
    # You should add more tests of the damage_ellipses functionality as
    # you write that function

    axes = deepimpact.damage_ellipses(outcome, [5000., 2000.])

    assert len(axes) == 3
    for axis in axes:
        assert type(axis) is list
        assert len(axis) == 2

    axes = deepimpact.damage_ellipses(outcome, 5000.)

    assert len(axes) == 3
    for axis in axes:
        assert isinstance(axis, float)


def test_great_circle_distance(deepimpact):

    pnts1 = np.array([[54.0, 0.0], [55.0, 1.0], [54.2, -3.0]])
    pnts2 = np.array([[55.0, 1.0], [56.0, -2.1], [54.001, -0.003]])

    data = np.array([[1.28580537e+05, 2.59579735e+05, 2.25409117e+02],
                    [0.00000000e+00, 2.24656571e+05, 1.28581437e+05],
                    [2.72529953e+05, 2.08175028e+05, 1.96640630e+05]])

    dist = deepimpact.great_circle_distance(pnts1, pnts2)

    assert np.allclose(data, dist, rtol=1.0e-4)


def test_locator_postcodes(loc):

    latlon = (52.2074, 0.1170)

    result = loc.get_postcodes_by_radius(latlon, [0.2e3, 0.1e3])

    assert type(result) is list
    if len(result) > 0:
        for element in result:
            assert type(element) is list


def test_population_by_radius(loc):

    latlon = (52.2074, 0.1170)

    result = loc.get_population_by_radius(latlon, [5e2, 1e3], method='naive')

    assert type(result) is list
    if len(result) > 0:
        for element in result:
            assert isinstance(element, (int, np.integer))


def test_locator_postcodes_ellipse(loc):

    latlon = (52.2074, 0.1170)

    result = loc.get_postcodes_by_ellipse(latlon, [0.2e3, 0.1e3],
                                          [0.15e3, 0.08e3], [0.18e3, 0.09e3],
                                          45.)

    assert type(result) is list
    if len(result) > 0:
        for element in result:
            assert type(element) is list


def test_population_by_ellipse(loc):

    latlon = (52.2074, 0.1170)

    result = loc.get_population_by_ellipse(latlon, [5e2, 1e3],
                                           [4e2, 8e2], [4.5e2, 9e2],
                                           45., method='naive')

    assert type(result) is list
    if len(result) > 0:
        for element in result:
            assert isinstance(element, (int, np.integer))


def test_impact_risk(deepimpact, planet):

    probability, population = deepimpact.impact_risk(planet)

    assert type(probability) is pd.DataFrame
    assert 'probability' in probability.columns
    assert type(population) is dict
    assert 'mean' in population.keys()
    assert 'stdev' in population.keys()
    assert isinstance(population['mean'], float)
    assert isinstance(population['stdev'], float)


def test_impact_risk_ellipse(deepimpact, planet):

    probability, population = deepimpact.impact_risk(planet, nsamples=5,
                                                     ellipse=True)

    assert type(probability) is pd.DataFrame
    assert 'probability' in probability.columns
    assert type(population) is dict
    assert 'mean' in population.keys()
    assert 'stdev' in population.keys()
    assert isinstance(population['mean'], float)
    assert isinstance(population['stdev'], float)
