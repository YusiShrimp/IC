"""Module to calculate the damage and impact risk for given scenarios"""
import numpy as np
import pandas as pd
import os
from .locator import GeospatialLocator

__all__ = ['damage_zones', 'damage_ellipses', 'impact_risk']

R_EARTH = 6371e3

# One GeospatialLocator per class, kept between calls of impact_risk.
# Loading the postcode and population files takes about a second, so they
# are only read once per Python session.
_locators = {}


def _shared_locator():
    """
    Return a GeospatialLocator, creating it on the first call only.

    The locator only reads its data files when it is created, and its
    look-up methods do not change it, so one instance can safely be reused
    by every call of impact_risk. It is stored per class, so a replacement
    class (for example a fake locator in the tests) gets its own instance.

    Returns
    -------
    GeospatialLocator
        the locator shared by all calls of impact_risk
    """
    if GeospatialLocator not in _locators:
        _locators[GeospatialLocator] = GeospatialLocator()
    return _locators[GeospatialLocator]


def surface_zero(lat, lon, bearing, distance):
    """
    Calculate the latitude and longitude of the surface zero location
    for a given impact scenario.

    Parameters
    ----------

    lat: float
        latitude of the meteoroid entry point (degrees)
    lon: float
        longitude of the meteoroid entry point (degrees)
    bearing: float
        Bearing (azimuth) relative to north of meteoroid trajectory (degrees)
    distance: float
        Distance from the meteoroid entry point to the surface zero point (m)

    Returns
    -------

    blat: float
        latitude of the surface zero point (degrees)
    blon: float
        longitude of the surface zero point (degrees)

    """
    phi1 = np.radians(lat)
    lambda1 = np.radians(lon)
    beta = np.radians(bearing)
    delta = distance / R_EARTH
    sin_phi2 = (np.sin(phi1) * np.cos(delta)
                + np.cos(phi1) * np.sin(delta) * np.cos(beta))

    sin_phi2 = np.clip(sin_phi2, -1., 1.)
    phi2 = np.arcsin(sin_phi2)
    lambda2 = lambda1 + np.arctan2(np.sin(beta) * np.sin(delta) * np.cos(phi1),
                                   np.cos(delta)-np.sin(phi1) * np.sin(phi2))
    blat = np.degrees(phi2)
    blon = (np.degrees(lambda2)+180.) % 360. - 180.
    return float(blat), float(blon)


def blast_radii(energy, altitude, pressures, tol=1e-12, max_iter=50):
    """
    Find the distance from surface zero at which the airblast overpressure
    falls to each of the given pressures.

    The overpressure (Pa) at range r (m) is

        p = 3.14e11 * s**(-1.3) + 1.8e7 * s**(-0.565),

    with s = (r**2 + zb**2) / Ek**(2/3). The pressure depends on r only
    through s, so the equation p(s) = pressure is solved for s, and then

        r = sqrt(s * Ek**(2/3) - zb**2).

    If this is negative, the pressure directly below the burst is already
    lower than the threshold: that damage level does not reach the ground
    and the radius returned is 0.

    The equation for s is solved with Newton's method applied to
    f(s) = p(s) - pressure. f is decreasing and convex, so starting from
    a point where f > 0 the iterations approach the root from one side
    and cannot diverge.

    Parameters
    ----------

    energy: float
        burst energy Ek (kt TNT)
    altitude: float
        burst altitude zb (m)
    pressures: arraylike
        threshold pressures (Pa), all greater than zero
    tol: float, optional
        the iteration stops when no value of s changes by more than
        tol times its size
    max_iter: int, optional
        maximum number of Newton iterations

    Returns
    -------

    radii: numpy.ndarray
        blast radius (m) for each pressure, 0 where the level is not
        reached
    """
    pressures = np.asarray(pressures, dtype=float)

    if energy <= 0.:
        return np.zeros_like(pressures)

    # Starting guess: solve with each term of the formula on its own and
    # keep the larger s. Dropping a (positive) term underestimates the
    # pressure, so both guesses lie below the root, where f > 0.
    s = np.maximum((3.14e11 / pressures)**(1. / 1.3),
                   (1.8e7 / pressures)**(1. / 0.565))

    for _ in range(max_iter):
        # f(s) = p(s) - pressure, and its derivative with respect to s
        f = 3.14e11 * s**(-1.3) + 1.8e7 * s**(-0.565) - pressures
        dfds = (-1.3 * 3.14e11 * s**(-2.3)
                - 0.565 * 1.8e7 * s**(-1.565))
        step = f / dfds
        s = s - step
        if np.all(np.abs(step) < tol * s):
            break

    r_squared = s * energy**(2. / 3.) - altitude**2

    return np.sqrt(np.maximum(r_squared, 0.))


def damage_zones(outcome, lat, lon, bearing, pressures):
    """
    Calculate the latitude and longitude of the surface zero location and the
    list of airblast damage radii (m) for a given impact scenario.

    Parameters
    ----------

    outcome: Dict
        the outcome dictionary from an impact scenario
    lat: float
        latitude of the meteoroid entry point (degrees)
    lon: float
        longitude of the meteoroid entry point (degrees)
    bearing: float
        Bearing (azimuth) relative to north of meteoroid trajectory (degrees)
    pressures: float, arraylike
        List of threshold pressures to define airblast damage levels

    Returns
    -------

    blat: float
        latitude of the surface zero point (degrees)
    blon: float
        longitude of the surface zero point (degrees)
    damagerad: arraylike, float
        List of distances specifying the blast radii
        for the input damage levels

    Examples
    --------

    >>> import deepimpact
    >>> outcome = {'burst_altitude': 8e3, 'burst_energy': 7e3,
    ...            'burst_distance': 90e3, 'burst_peak_dedz': 1e3,
    ...            'burst_angle': 45, 'outcome': 'Airburst'}
    >>> deepimpact.damage_zones(outcome, 52.79, -2.95, 135,
    ...                         pressures=[1e3, 3.5e3, 27e3, 43e3])
    """
    if np.any(np.asarray(pressures, dtype=float) <= 0.):
        raise ValueError("pressures must all be greater than zero")

    blat, blon = surface_zero(lat, lon, bearing, outcome['burst_distance'])

    radii = blast_radii(outcome['burst_energy'], outcome['burst_altitude'],
                        pressures)

    if np.ndim(pressures) == 0:
        damagerad = float(radii)
    else:
        damagerad = [float(r) for r in radii]

    return blat, blon, damagerad


def damage_ellipses(outcome, damrad, kf=0.4, kb=0.2, radians=False):
    """
    Calculate the shape of damage ellipses when taking angle to the horizontal
    into account.

    NOTE: This is an extension task.

    Parameters
    ----------

    outcome: Dict
        the outcome dictionary from an impact scenario
    damrad: arraylike, float
        the radii (or radius) of the circular blast for a range (or single)
        damage level.
    kf: float
        Scaling factor to determine the ellipticity of the half-ellipse
        downrange of surface zero.
    kb: float
        Scaling factor to determine the ellipticity of the half-ellipse
        uprange of surface zero.
    radians: bool
        Whether the burst angle is stored in the outcome dict in radians
        (True) or degrees (False)

    Returns
    -------

    a_perp: arraylike, float
        Crossrange semi-axis of blast ellipse for each damage
        radius in damrad, in m
    a_front: arraylike, float
        Downrange semi-axis of blast ellipse for each damage
        radius in damrad, in m
    a_behind: arraylike, float
        Uprange semi-axis of blast ellipse for each damage
        radius in damrad, in m

    Examples
    --------

    >>> import deepimpact
    >>> outcome = {'burst_altitude': 8e3, 'burst_energy': 7e3,
    ...            'burst_distance': 90e3, 'burst_peak_dedz': 1e3,
    ...            'burst_angle': 45, 'outcome': 'Airburst'}
    >>> deepimpact.damage_ellipses(outcome, [5000., 4500.])
    """
    theta_b = outcome['burst_angle']
    if not radians:
        theta_b = np.radians(theta_b)

    # how much each half-ellipse is squashed along the direction of travel
    q_front = 1. - kf * np.cos(theta_b)**2
    q_behind = 1. - kb * np.cos(theta_b)**2

    if q_front <= 0. or q_behind <= 0.:
        raise ValueError("kf and kb must give positive semi-axes")

    radii = np.asarray(damrad, dtype=float)

    # widening the zone across the trajectory keeps its area equal to
    # the area of the circle
    a_perp = radii * np.sqrt(2. / (q_front + q_behind))
    a_front = q_front * a_perp
    a_behind = q_behind * a_perp

    if np.ndim(damrad) == 0:
        return float(a_perp), float(a_front), float(a_behind)

    return ([float(a) for a in a_perp], [float(a) for a in a_front],
            [float(a) for a in a_behind])


def impact_risk(planet,
                impact_file=os.sep.join((os.path.dirname(__file__),
                                         '..', 'resources',
                                         'impact_parameter_list.csv')),
                pressure=27.e3, nsamples=None,
                ellipse=False, kf=0.4, kb=0.2):
    """
    Perform an uncertainty analysis to calculate the probability for
    each affected UK postcode and the total population affected.

    NOTE: This is an extension task.

    Parameters
    ----------
    planet: deepimpact.Planet instance
        The Planet instance from which to solve the atmospheric entry

    impact_file: str
        Filename of a .csv file containing the impact parameter list
        with columns for 'radius', 'angle', 'velocity', 'strength',
        'density', 'entry latitude', 'entry longitude', 'bearing'

    pressure: float
        A single pressure at which to calculate the damage zone for
        each impact

    nsamples: int or None
        The number of iterations to perform in the uncertainty analysis.
        If None, the full set of impact parameters provided in impact_file
        is used.

    ellipse: bool
        If ellipse is True, use the angle-dependent blast zones calculated
        by damage_ellipses. Otherwise, use circular blast zones.

    kf, kb: float
        shape coefficients passed to damage_ellipses; only used
        if ellipse=True

    Returns
    -------
    probability: DataFrame
        A pandas DataFrame with the index named "postcode", and the
        probability the postcode was inside the blast radius as a column
        named "probability".
    population: dict
        A dictionary containing the mean and standard deviation of the
        population affected by the impact, with keys 'mean' and 'stdev'.
        Values are floats.
    """
    scenarios = pd.read_csv(impact_file)
    if nsamples is not None:
        scenarios = scenarios.iloc[:nsamples]
    n_scenarios = len(scenarios)

    locator = _shared_locator()
    # list of every postcode hit
    hit_postcodes = []
    # population inside the blast zone for each scenario
    populations = np.zeros(n_scenarios)
    for i, scenario in enumerate(scenarios.to_dict('records')):
        result = planet.solve_atmospheric_entry(
            radius=scenario['radius'], velocity=scenario['velocity'],
            density=scenario['density'], strength=scenario['strength'],
            angle=scenario['angle'])
        result = planet.calculate_energy(result)
        outcome = planet.analyse_outcome(result)

        bearing = scenario['bearing']
        blat, blon, radius = damage_zones(
            outcome, scenario['entry latitude'],
            scenario['entry longitude'], bearing, pressure)
        if not radius > 0.:
            continue

        if ellipse:
            a_perp, a_front, a_behind = damage_ellipses(outcome, radius,
                                                        kf=kf, kb=kb)
            postcodes = locator.get_postcodes_by_ellipse(
                (blat, blon), [a_perp], [a_front], [a_behind], bearing)
            population = locator.get_population_by_ellipse(
                (blat, blon), [a_perp], [a_front], [a_behind], bearing)
        else:
            postcodes = locator.get_postcodes_by_radius((blat, blon),
                                                        [radius])
            population = locator.get_population_by_radius((blat, blon),
                                                          [radius])

        # only one zone was asked for, so each result holds one entry
        for zone in postcodes:
            hit_postcodes.extend(zone)
        populations[i] = np.sum(population)

    # number of scenarios that hit each postcode
    counts = pd.Series(hit_postcodes, dtype=object).value_counts()
    counts = counts.sort_index()

    probability = pd.DataFrame(
        {'probability': counts.to_numpy(dtype=float) / max(n_scenarios, 1)},
        index=pd.Index(counts.index, name='postcode'))

    if n_scenarios > 0:
        population = {'mean': float(np.mean(populations)),
                      'stdev': float(np.std(populations))}
    else:
        population = {'mean': 0., 'stdev': 0.}

    return probability, population
