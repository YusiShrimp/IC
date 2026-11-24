import numpy as np
import pandas as pd
import time
from pytest import fixture, mark, raises
from deepimpact.locator import great_circle_distance, GeospatialLocator
from deepimpact.locator import _initial_bearing, _in_bounding_box


R = 6371e3  # radius of the Earth in meters

# ---------------------------------------------------------------------------
# great_circle_distance
# ---------------------------------------------------------------------------


# Values Test - great_circle_distance
def test_one_degree_on_equator():
    """1° on the equator is pi * R / 180.0 meters."""
    pnts1 = np.array([[0.0, 0.0]])
    pnts2 = np.array([[0.0, 1.0]])
    dist = great_circle_distance(pnts1, pnts2)
    assert np.allclose(dist, np.pi * R / 180.0, rtol=1.0e-4)


def test_half_circumference():
    """Half the circumference is pi * R meters."""
    pnts1 = np.array([[0.0, 0.0]])
    pnts2 = np.array([[0.0, 180.0]])
    dist = great_circle_distance(pnts1, pnts2)
    assert np.allclose(dist, np.pi * R, rtol=1.0e-4)


def test_teacher_data():
    """Test with 3x3 data in test_deepimpact.py."""
    pnts1 = np.array([[54.0, 0.0], [55.0, 1.0], [54.2, -3.0]])
    pnts2 = np.array([[55.0, 1.0], [56.0, -2.1], [54.001, -0.003]])
    data = np.array([[1.28580537e+05, 2.59579735e+05, 2.25409117e+02],
                    [0.00000000e+00, 2.24656571e+05, 1.28581437e+05],
                    [2.72529953e+05, 2.08175028e+05, 1.96640630e+05]])
    dist = great_circle_distance(pnts1, pnts2)
    assert np.allclose(data, dist, rtol=1.0e-4)


# Symmetry Test - great_circle_distance
def test_symmetry():
    """Test with symmetric points."""
    pnts1 = np.array([[54.0, 0.0], [55.0, 1.0], [54.2, -3.0]])
    pnts2 = np.array([[55.0, 1.0], [56.0, -2.1], [54.001, -0.003]])
    dist12 = great_circle_distance(pnts1, pnts2)
    dist21 = great_circle_distance(pnts2, pnts1)
    assert np.allclose(dist12, dist21.T, rtol=1.0e-4)


def test_zero_distance():
    """Test that the distance from a point to itself is zero."""
    pnts1 = np.array([[54.0, 0.0], [55.0, 1.0], [54.2, -3.0]])
    dist = great_circle_distance(pnts1, pnts1)
    assert np.allclose(np.diag(dist), 0.0, rtol=1.0e-4)


# Range Test - great_circle_distance
def test_range():
    """Test every distance lies in [0, pi * R]."""
    rng = np.random.default_rng(11)
    pnts1 = rng.uniform([-90, -180], [90, 180], (100, 2))
    pnts2 = rng.uniform([-90, -180], [90, 180], (80, 2))
    dist = great_circle_distance(pnts1, pnts2)
    assert np.all(dist >= 0)
    assert np.all(dist <= np.pi * R)


# Domain Test - great_circle_distance
def test_domain_latitude_edge_cases():
    """Test the boundaries of the latitude domain."""
    pnts1 = np.array([[90.0, 0.0]])
    pnts2 = np.array([[-90.0, 0.0]])
    dist = great_circle_distance(pnts1, pnts2)
    assert np.allclose(dist, np.pi * R, rtol=1.0e-4)


def test_domain_longitude_edge_cases():
    """Test the boundaries of the longitude domain."""
    pnts1 = np.array([[0.0, 180.0]])
    pnts2 = np.array([[0.0, -180.0]])
    dist = great_circle_distance(pnts1, pnts2)
    assert np.allclose(dist, 0.0, atol=1.0e-6)


@mark.parametrize('point', [
    [[90.1, 0.0]],     # latitude too big
    [[-91.0, 0.0]],    # latitude too small
    [[0.0, 180.5]],    # longitude too big
    [[0.0, -181.0]],   # longitude too small
])
def test_distance_invalid_input_raises(point):
    with raises(ValueError):
        great_circle_distance(point, [[0.0, 0.0]])
    with raises(ValueError):
        great_circle_distance([[0.0, 0.0]], point)


# Format Test - great_circle_distance
def test_output_format():
    """Test that n points against m points gives an (n, m) numpy array."""
    pnts1 = np.array([[54.0, 0.0], [55.0, 1.0], [54.2, -3.0]])
    pnts2 = np.array([[55.0, 1.0], [56.0, -2.1]])
    dist = great_circle_distance(pnts1, pnts2)
    assert isinstance(dist, np.ndarray)
    assert dist.shape == (len(pnts1), len(pnts2))


def test_distance_accepts_lists():
    """Arraylike input: plain Python lists work as well as arrays."""
    dist = great_circle_distance([[54.0, 0.0]], [[55.0, 1.0]])
    assert dist.shape == (1, 1)
    assert np.isclose(dist[0, 0], 1.28580537e+05, rtol=1e-4)


def test_distance_single_point_values():
    """Test that a single 1-D point gives the correct values."""
    dist = great_circle_distance([[54.0, 0.0], [55, 0.0]], [55, 1.0])
    assert np.allclose(np.ravel(dist), [1.286e5, 6.378e4], rtol=1e-3)


def test_distance_single_point_shape():
    """Test that a single 1-D point gives the correct shape."""
    dist = great_circle_distance([[54.0, 0.0], [55, 0.0]], [55, 1.0])
    assert dist.shape == (2,)


# Performance Test - great_circle_distance
def test_distance_performance():
    """1000 points against 1000 points runs in under 1 second.

    That is a million distances. A vectorised numpy version takes a few
    hundredths of a second; a Python loop would take several seconds.
    Best of 3 runs, so one slow moment on the CI machine does not fail it.
    """
    rng = np.random.default_rng(11)
    pnts1 = rng.uniform([50.0, -10.0], [60.0, 1.0], size=(1000, 2))
    pnts2 = rng.uniform([50.0, -10.0], [60.0, 1.0], size=(1000, 2))

    times = []
    for _ in range(3):
        start = time.perf_counter()
        dist = great_circle_distance(pnts1, pnts2)
        times.append(time.perf_counter() - start)

    assert dist.shape == (1000, 1000)
    assert min(times) < 1.0

# ---------------------------------------------------------------------------
# GeospatialLocator
# ---------------------------------------------------------------------------


@fixture(scope='module')
def loc():
    """One locator for the whole file: loading the data is slow."""
    return GeospatialLocator()


PC_COL, LAT_COL, LON_COL = 'Postcode', 'Latitude', 'Longitude'


# A generous box around the UK, Channel Islands and Shetland included
UK_LAT = (49.0, 61.5)
UK_LON = (-9.0, 2.5)


# full_postcodes.csv
def test_postcodes_loaded(loc):
    """Test the function output over million postcodes UK."""
    assert len(loc.postcodes) > 1_000_000


def test_postcodes_are_7_character_strings(loc):
    """Fixed 7-character postcode format: 'SW7 2AZ', 'E1  6AN', 'SW195AE'."""
    col = loc.postcodes[PC_COL]
    assert col.map(lambda p: isinstance(p, str)).all()
    assert (col.str.len() == 7).all()


def test_postcodes_unique(loc):
    assert not loc.postcodes[PC_COL].duplicated().any()


def test_postcodes_no_missing_coordinates(loc):
    """A NaN coordinate would silently drop out of every distance check."""
    assert not loc.postcodes[LAT_COL].isna().any()
    assert not loc.postcodes[LON_COL].isna().any()


def test_postcodes_inside_uk(loc):
    lat = loc.postcodes[LAT_COL]
    lon = loc.postcodes[LON_COL]
    assert lat.between(UK_LAT[0], UK_LAT[1]).all()
    assert lon.between(UK_LON[0], UK_LON[1]).all()


# UK_residential_population_2011_latlon.arc
def test_population_arrays_same_length(loc):
    """Each cell has a latitude, a longitude and a population."""
    for arr in (loc.pop_lat, loc.pop_lon, loc.pop):
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 1
    assert len(loc.pop_lat) == len(loc.pop_lon) == len(loc.pop) > 0


def test_population_no_nodata_left(loc):
    """Sea cells hold -9999 in the file; __init__ must drop them.

    If they were kept, any zone touching the sea would lose 9999 people
    per sea cell and could even come out negative.
    """
    assert not np.any(loc.pop == -9999)
    assert np.all(loc.pop >= 0)


def test_population_cells_inside_uk(loc):
    assert np.all((loc.pop_lat >= UK_LAT[0]) & (loc.pop_lat <= UK_LAT[1]))
    assert np.all((loc.pop_lon >= UK_LON[0]) & (loc.pop_lon <= UK_LON[1]))


def test_population_total(loc):
    """The 2011 census counted about 63 million people in the UK.

    The range is wide on purpose: the grid may not cover every island.
    A total far outside it means cells were lost or -9999 was added in.
    """
    total = loc.pop.sum()
    assert 4e7 < total < 8e7


def test_default_norm(loc):
    assert loc.norm is great_circle_distance

# ---------------------------------------------------------------------------
# get_postcodes_by_radius
# ---------------------------------------------------------------------------


IMPERIAL = (51.498317, -0.176923)   # look up 'SW7 2AZ' in the postcode file
NORTH_SEA = (56.0, 3.0)         # open sea, no postcodes nearby


def brute_force_postcodes(centre, radius):
    """Slow but obviously right: distance from centre to every postcode."""
    data = pd.read_csv('resources/full_postcodes.csv',
                       usecols=[PC_COL, LAT_COL, LON_COL])
    data = data[data[LAT_COL] <= 90]   # drop postcodes with no location
    points = data[[LAT_COL, LON_COL]].to_numpy()
    dist = great_circle_distance([centre], points)[0]
    return list(data[PC_COL][dist < radius])   # 'closer than' the radius


# Domain Test
def test_postcodes_radius_zero_is_empty(loc):
    """Nothing is closer than 0 m, not even SW7 2AZ at the centre."""
    assert loc.get_postcodes_by_radius(IMPERIAL, [0.0]) == [[]]


def test_postcodes_at_sea_is_empty(loc):
    assert loc.get_postcodes_by_radius(NORTH_SEA, [1e4]) == [[]]


def test_postcodes_negative_radius_raises(loc):
    with raises(ValueError):
        loc.get_postcodes_by_radius(IMPERIAL, [-1e3])


# Format Test
def test_postcodes_format(loc):
    radii = [1e3, 2e3, 5e3]
    result = loc.get_postcodes_by_radius(IMPERIAL, radii)
    assert type(result) is list
    assert len(result) == len(radii)
    for zone in result:
        assert type(zone) is list
        for postcode in zone:
            assert isinstance(postcode, str)
            assert len(postcode) == 7


# Range Test
def test_postcodes_no_duplicates(loc):
    (zone,) = loc.get_postcodes_by_radius(IMPERIAL, [2e3])
    assert len(zone) == len(set(zone))


# Nesting Test
def test_postcodes_nested(loc):
    small, big = loc.get_postcodes_by_radius(IMPERIAL, [5e2, 2e3])
    assert set(small) <= set(big)
    assert len(big) > len(small)


# Order Test
def test_postcodes_follow_radii_order(loc):
    """Radii given largest first: result[0] must be the bigger zone."""
    big, small = loc.get_postcodes_by_radius(IMPERIAL, [2e3, 5e2])
    assert len(big) > len(small)
    assert set(small) <= set(big)


# Values Test
def test_postcodes_imperial(loc):
    (zone,) = loc.get_postcodes_by_radius(IMPERIAL, [1e3])
    assert 'SW7 2AZ' in zone


# Brute force Test
def test_postcodes_match_brute_force(loc):
    fast = loc.get_postcodes_by_radius(IMPERIAL, [2e3])[0]
    slow = brute_force_postcodes(IMPERIAL, 2e3)
    assert set(fast) == set(slow)


# Performance Test
def test_postcodes_performance(loc):
    """Best of 3 runs, as in the distance performance test."""
    times = []
    for _ in range(3):
        start = time.perf_counter()
        loc.get_postcodes_by_radius(IMPERIAL, [1e3, 1e4, 1e5])
        times.append(time.perf_counter() - start)
    assert min(times) < 1.0

# ---------------------------------------------------------------------------
# get_population_by_radius
# ---------------------------------------------------------------------------


# look up 'BN2 1TW' in the postcode file
BRIGHTON = (50.819106, -0.136477)
# look up 'WC2N5DU' in the postcode file
LONDON = (51.507209, -0.12829400000000002)
# roughly the middle of the UK
GB_MIDDLE = (54.0, -2.5)


# Format Test
def test_population_format(loc):
    radii = [1e2, 5e2, 1e3]
    result = loc.get_population_by_radius(IMPERIAL, radii)
    assert type(result) is list
    assert len(result) == len(radii)
    for value in result:
        assert isinstance(value, (int, np.integer))


# Test Range
def test_population_in_range(loc):
    total = loc.pop.sum()
    result = loc.get_population_by_radius(IMPERIAL, [1e3, 1e4, 1e5])
    for value in result:
        assert 0 <= value <= total


# Monotonic Test
def test_population_monotonic(loc):
    radii = [1e3, 5e3, 1e4, 5e4]
    result = loc.get_population_by_radius(IMPERIAL, radii)
    for smaller, larger in zip(result, result[1:]):
        assert smaller <= larger


# Order Test
def test_population_follows_radii_order(loc):
    """Radii given largest first: result[0] belongs to the larger radius."""
    big, small = loc.get_population_by_radius(IMPERIAL, [1e4, 1e3])
    assert big > small


# Domain Test
def test_population_radius_zero(loc):
    assert loc.get_population_by_radius(IMPERIAL, [0.0]) == [0]


def test_population_at_sea(loc):
    assert loc.get_population_by_radius(NORTH_SEA, [5e3]) == [0]


def test_population_unknown_method_raises(loc):
    with raises(NotImplementedError):
        loc.get_population_by_radius(IMPERIAL, [1e3], method='banana')


def test_population_negative_radius_raises(loc):
    with raises(ValueError):
        loc.get_population_by_radius(IMPERIAL, [-1e3])


# Values Test
def test_population_single_cell(loc):
    """Centre exactly on a cell centre, radius 1 m: only that cell counts.

    Cells are 1 km apart, so no other cell centre is within 1 m.
    """
    i = int(np.argmax(loc.pop))  # the most populated cell
    centre = (loc.pop_lat[i], loc.pop_lon[i])
    assert loc.get_population_by_radius(centre, [1.0]) == [int(loc.pop[i])]


def test_population_whole_uk(loc):
    """2000 km from the middle of the UK covers every cell."""
    (pop,) = loc.get_population_by_radius(GB_MIDDLE, [2e6])
    assert pop == int(loc.pop.sum())


def test_population_coast(loc):
    """Near the sea the land cells must still count."""
    result = loc.get_population_by_radius(BRIGHTON, [2e3, 2e4])
    assert result[0] > 0
    assert result[1] > result[0]


def test_population_london(loc):
    """Within 10 km of central London live a few million people."""
    (pop,) = loc.get_population_by_radius(LONDON, [1e4])
    assert 1e6 < pop < 6e6


# Brute force
def brute_force_population(loc, centre, radius):
    """Slow but obviously right: a Python loop over the nearby cells.

    Only cells within 1 degree are looped over (any radius used below is
    far smaller than that), and each distance is the haversine formula
    written out again here, independent of locator.py.
    """
    lat0, lon0 = np.radians(centre)
    near = ((np.abs(loc.pop_lat - centre[0]) < 1)
            & (np.abs(loc.pop_lon - centre[1]) < 2))
    total = 0
    for lat, lon, pop in zip(loc.pop_lat[near], loc.pop_lon[near],
                             loc.pop[near]):
        lat, lon = np.radians(lat), np.radians(lon)
        a = (np.sin((lat - lat0) / 2) ** 2
             + np.cos(lat0) * np.cos(lat) * np.sin((lon - lon0) / 2) ** 2)
        if 2 * R * np.arcsin(np.sqrt(a)) < radius:
            total += pop
    return int(total)


@mark.parametrize('centre, radius', [
    (IMPERIAL, 5e3),
    ((52.2074, 0.1170), 1e4),      # Cambridge, as in test_deepimpact.py
    ((55.9533, -3.1883), 2e4),     # Edinburgh
    (BRIGHTON, 1.5e4),             # coast
])
def test_population_match_brute_force(loc, centre, radius):
    (fast,) = loc.get_population_by_radius(centre, [radius])
    assert fast == brute_force_population(loc, centre, radius)


# Performance Test
def test_population_performance(loc):
    """Best of 3 runs, as in the distance performance test."""
    times = []
    for _ in range(3):
        start = time.perf_counter()
        loc.get_population_by_radius(IMPERIAL, [1e3, 1e4, 1e5])
        times.append(time.perf_counter() - start)
    assert min(times) < 1.0

# ---------------------------------------------------------------------------
# Extension: ellipse
# ---------------------------------------------------------------------------


def destination(start, bearing, distance):
    """Point reached from start after distance (m) along bearing (deg).

    Standard great-circle destination formula, written out here so the
    tests do not depend on locator.py.
    """
    lat1, lon1 = np.radians(start)
    theta = np.radians(bearing)
    delta = distance / R
    lat2 = np.arcsin(np.sin(lat1) * np.cos(delta)
                     + np.cos(lat1) * np.sin(delta) * np.cos(theta))
    lon2 = lon1 + np.arctan2(np.sin(theta) * np.sin(delta) * np.cos(lat1),
                             np.cos(delta) - np.sin(lat1) * np.sin(lat2))
    return np.degrees(lat2), np.degrees(lon2)


# _initial_bearing Test

@mark.parametrize('point, expected', [
    ((1.0, 0.0), 0.0),      # north
    ((0.0, 1.0), 90.0),     # east
    ((-1.0, 0.0), 180.0),   # south
    ((0.0, -1.0), 270.0),   # west
])
def test_initial_bearing_compass(point, expected):
    """From (0, 0) on the equator the compass directions are exact.

    np.mod(..., 360) accepts both -90 and 270 for west.
    """
    bearing = np.ravel(_initial_bearing((0.0, 0.0), np.array([point])))[0]
    assert np.isclose(np.mod(bearing, 360), expected)


@mark.parametrize('theta', [10., 45., 120., 200., 315.])
def test_initial_bearing_matches_destination(theta):
    """Go 5 km from Imperial along theta, then ask for the bearing back."""
    target = destination(IMPERIAL, theta, 5e3)
    bearing = np.ravel(_initial_bearing(IMPERIAL, np.array([target])))[0]
    assert np.isclose(np.mod(bearing, 360), theta, atol=1e-6)


# _in_ellipse_zones -----------------------------------------------------------
#
# Zone used below: a_front = 2 km, a_behind = 0.5 km, a_perp = 1 km.
# Each point is (angle from the trajectory in degrees, distance in m,
# expected inside?). Worked out by hand with (x/a)^2 + (y/a_perp)^2 < 1,
# where x = d cos(angle), y = d sin(angle), a = a_front ahead, a_behind
# behind:
#   ahead  1.5 km : (1.5/2)^2           = 0.56  inside
#   ahead  2.5 km : (2.5/2)^2           = 1.56  outside
#   behind 0.4 km : (0.4/0.5)^2         = 0.64  inside
#   behind 0.6 km : (0.6/0.5)^2         = 1.44  outside
#   side   0.9 km : (0.9/1)^2           = 0.81  inside (left and right)
#   side   1.1 km : (1.1/1)^2           = 1.21  outside
#   45 deg 1.2 km : x = y = 0.849 km -> 0.18 + 0.72 = 0.90  inside
#   45 deg 1.4 km : x = y = 0.990 km -> 0.25 + 0.98 = 1.23  outside
#   135 deg 0.55 km: x = y = 0.389 km -> 0.61 + 0.15 = 0.76  inside
#   135 deg 0.70 km: x = y = 0.495 km -> 0.98 + 0.25 = 1.23  outside

A_PERP, A_FRONT, A_BEHIND = 1e3, 2e3, 5e2
ELLIPSE_POINTS = [
    (0., 1.5e3, True), (0., 2.5e3, False),
    (180., 4e2, True), (180., 6e2, False),
    (90., 9e2, True), (270., 9e2, True),
    (90., 1.1e3, False), (270., 1.1e3, False),
    (45., 1.2e3, True), (45., 1.4e3, False),
    (135., 5.5e2, True), (135., 7e2, False),
]


@mark.parametrize('bearing', [0., 70., 225.])
def test_in_ellipse_zones_hand_checked(loc, bearing):
    """Same answers whatever the bearing, because the points turn with it."""
    centre = (0.0, 0.0)
    points = np.array([destination(centre, bearing + angle, d)
                       for angle, d, _ in ELLIPSE_POINTS])
    expected = np.array([inside for _, _, inside in ELLIPSE_POINTS])

    (mask,) = loc._in_ellipse_zones(centre, points, [A_PERP], [A_FRONT],
                                    [A_BEHIND], bearing)

    assert np.array_equal(mask, expected)


def test_in_ellipse_zones_format(loc):
    """One boolean array of shape (N,) per zone."""
    points = np.array([destination(IMPERIAL, 30., d)
                       for d in (1e2, 1e3, 1e4)])
    masks = loc._in_ellipse_zones(IMPERIAL, points, [1e3, 2e3],
                                  [2e3, 4e3], [5e2, 1e3], 30.)
    assert len(masks) == 2
    for mask in masks:
        assert mask.dtype == bool
        assert mask.shape == (3,)


# ---------------------------------------------------------------------------
# get_postcodes_by_ellipse
# ---------------------------------------------------------------------------


def ellipse_zones(loc, centre, a_perp, a_front, a_behind, bearing):
    """Shorthand used by the tests below."""
    return loc.get_postcodes_by_ellipse(centre, a_perp, a_front,
                                        a_behind, bearing)


# Format Test
def test_ellipse_postcodes_format(loc):
    result = ellipse_zones(loc, IMPERIAL, [2e3, 1e3], [3e3, 1.5e3],
                           [1e3, 5e2], 45.)
    assert type(result) is list
    assert len(result) == 2
    for zone in result:
        assert type(zone) is list
        for postcode in zone:
            assert isinstance(postcode, str)
            assert len(postcode) == 7


def test_ellipse_postcodes_float_input(loc):
    """Floats instead of lists: same zone as one-element lists."""
    from_floats = ellipse_zones(loc, IMPERIAL, 1e3, 3e3, 5e2, 45.)
    from_lists = ellipse_zones(loc, IMPERIAL, [1e3], [3e3], [5e2], 45.)
    assert type(from_floats) is list
    assert len(from_floats) == 1
    assert set(from_floats[0]) == set(from_lists[0])


# Range Test
def test_ellipse_postcodes_no_duplicates(loc):
    (zone,) = ellipse_zones(loc, IMPERIAL, [2e3], [3e3], [1e3], 45.)
    assert len(zone) == len(set(zone))


# Nesting Test
def test_ellipse_postcodes_nested(loc):
    """All three semi-axes smaller -> the zone is a subset."""
    small, big = ellipse_zones(loc, IMPERIAL, [5e2, 2e3], [8e2, 3e3],
                               [3e2, 1e3], 45.)
    assert set(small) <= set(big)
    assert len(big) > len(small)


# Order Test
def test_ellipse_postcodes_follow_axes_order(loc):
    """Bigger zone given first: result[0] must be the bigger zone."""
    big, small = ellipse_zones(loc, IMPERIAL, [2e3, 5e2], [3e3, 8e2],
                               [1e3, 3e2], 45.)
    assert len(big) > len(small)
    assert set(small) <= set(big)


# Domain Test
def test_ellipse_postcodes_zero_axes_is_empty(loc):
    """Nothing is strictly inside a zone of size 0."""
    assert ellipse_zones(loc, IMPERIAL, [0.], [0.], [0.], 45.) == [[]]


def test_ellipse_postcodes_at_sea_is_empty(loc):
    assert ellipse_zones(loc, NORTH_SEA, [1e4], [2e4], [5e3], 45.) == [[]]


def test_ellipse_postcodes_negative_axis_raises(loc):
    with raises(ValueError):
        ellipse_zones(loc, IMPERIAL, [-1e3], [1e3], [1e3], 45.)
    with raises(ValueError):
        ellipse_zones(loc, IMPERIAL, [1e3], [-1e3], [1e3], 45.)
    with raises(ValueError):
        ellipse_zones(loc, IMPERIAL, [1e3], [1e3], [-1e3], 45.)


def test_ellipse_postcodes_different_lengths_raises(loc):
    """One value per damage level: the three lists must match."""
    with raises(ValueError):
        ellipse_zones(loc, IMPERIAL, [1e3, 2e3], [1e3], [1e3], 45.)


# Values Test
def test_ellipse_postcodes_imperial(loc):
    """The centre is SW7 2AZ itself, so it is inside for any bearing."""
    for bearing in (0., 90., 200.):
        (zone,) = ellipse_zones(loc, IMPERIAL, [5e2], [8e2], [3e2], bearing)
        assert 'SW7 2AZ' in zone


def test_ellipse_postcodes_direction(loc):
    """SW7 2AZ is 3 km due north of the centre used here.

    bearing 0:   north is in front, a_front = 5 km  -> inside
    bearing 180: north is behind,   a_behind = 1 km -> outside
    bearing 90:  north is to the side, so a_perp decides
    """
    centre = (IMPERIAL[0] - np.degrees(3e3 / R), IMPERIAL[1])

    (zone,) = ellipse_zones(loc, centre, [1e3], [5e3], [1e3], 0.)
    assert 'SW7 2AZ' in zone

    (zone,) = ellipse_zones(loc, centre, [1e3], [5e3], [1e3], 180.)
    assert 'SW7 2AZ' not in zone

    (zone,) = ellipse_zones(loc, centre, [2e3], [5e3], [1e3], 90.)
    assert 'SW7 2AZ' not in zone

    (zone,) = ellipse_zones(loc, centre, [4e3], [5e3], [1e3], 90.)
    assert 'SW7 2AZ' in zone


# Ellipse Test
def test_ellipse_postcodes_equal_axes_is_circle(loc):
    """With all three semi-axes equal to r the zone is the circle."""
    radii = [5e2, 2e3]
    ellipse = ellipse_zones(loc, IMPERIAL, radii, radii, radii, 45.)
    circle = loc.get_postcodes_by_radius(IMPERIAL, radii)
    for e_zone, c_zone in zip(ellipse, circle):
        assert set(e_zone) == set(c_zone)


def test_ellipse_postcodes_between_circles(loc):
    """circle(smallest semi-axis) <= ellipse <= circle(largest semi-axis)."""
    a_perp, a_front, a_behind = 1e3, 3e3, 5e2
    (zone,) = ellipse_zones(loc, IMPERIAL, [a_perp], [a_front], [a_behind],
                            45.)
    inner, outer = loc.get_postcodes_by_radius(
        IMPERIAL, [min(a_perp, a_front, a_behind),
                   max(a_perp, a_front, a_behind)])
    assert set(inner) <= set(zone) <= set(outer)


def test_ellipse_postcodes_reverse_bearing(loc):
    """Flying the other way with front and behind swapped: same zone."""
    (one_way,) = ellipse_zones(loc, IMPERIAL, [1e3], [3e3], [5e2], 30.)
    (other_way,) = ellipse_zones(loc, IMPERIAL, [1e3], [5e2], [3e3], 210.)
    assert set(one_way) == set(other_way)


def test_ellipse_postcodes_bearing_360(loc):
    (zero,) = ellipse_zones(loc, IMPERIAL, [1e3], [3e3], [5e2], 0.)
    (full_turn,) = ellipse_zones(loc, IMPERIAL, [1e3], [3e3], [5e2], 360.)
    assert set(zero) == set(full_turn)


# Performance Test
def test_ellipse_postcodes_performance(loc):
    """Best of 3 runs, as in the circle performance test."""
    times = []
    for _ in range(3):
        start = time.perf_counter()
        ellipse_zones(loc, IMPERIAL, [1e3, 1e4, 1e5], [2e3, 2e4, 2e5],
                      [5e2, 5e3, 5e4], 45.)
        times.append(time.perf_counter() - start)
    assert min(times) < 1.0

# ---------------------------------------------------------------------------
# get_population_by_ellipse
# ---------------------------------------------------------------------------


def ellipse_population(loc, centre, a_perp, a_front, a_behind, bearing,
                       **kwargs):
    """Shorthand used by the tests below."""
    return loc.get_population_by_ellipse(centre, a_perp, a_front,
                                         a_behind, bearing, **kwargs)


# Format Test
def test_ellipse_population_format(loc):
    result = ellipse_population(loc, IMPERIAL, [2e3, 1e3], [3e3, 1.5e3],
                                [1e3, 5e2], 45.)
    assert type(result) is list
    assert len(result) == 2
    for value in result:
        assert isinstance(value, (int, np.integer))


def test_ellipse_population_float_input(loc):
    """Floats instead of lists: same answer as one-element lists."""
    from_floats = ellipse_population(loc, IMPERIAL, 5e3, 1.5e4, 2e3, 45.)
    from_lists = ellipse_population(loc, IMPERIAL, [5e3], [1.5e4], [2e3], 45.)
    assert type(from_floats) is list
    assert from_floats == from_lists


# Range Test
def test_ellipse_population_in_range(loc):
    total = loc.pop.sum()
    result = ellipse_population(loc, IMPERIAL, [1e3, 1e4], [2e3, 2e4],
                                [5e2, 5e3], 45.)
    for value in result:
        assert 0 <= value <= total


# Monotonic Test
def test_ellipse_population_nested(loc):
    """All three semi-axes larger -> never fewer people."""
    small, big = ellipse_population(loc, IMPERIAL, [2e3, 8e3], [4e3, 1.6e4],
                                    [1e3, 4e3], 45.)
    assert small <= big


# Order Test
def test_ellipse_population_follow_axes_order(loc):
    """Bigger zone given first: result[0] belongs to the bigger zone."""
    big, small = ellipse_population(loc, IMPERIAL, [8e3, 2e3], [1.6e4, 4e3],
                                    [4e3, 1e3], 45.)
    assert big > small


# Domain Test
def test_ellipse_population_zero_axes(loc):
    assert ellipse_population(loc, IMPERIAL, [0.], [0.], [0.], 45.) == [0]


def test_ellipse_population_at_sea(loc):
    assert ellipse_population(loc, NORTH_SEA, [5e3], [1e4], [2e3],
                              45.) == [0]


def test_ellipse_population_unknown_method_raises(loc):
    with raises(NotImplementedError):
        ellipse_population(loc, IMPERIAL, [1e3], [2e3], [5e2], 45.,
                           method='banana')


def test_ellipse_population_negative_axis_raises(loc):
    with raises(ValueError):
        ellipse_population(loc, IMPERIAL, [-1e3], [1e3], [1e3], 45.)
    with raises(ValueError):
        ellipse_population(loc, IMPERIAL, [1e3], [-1e3], [1e3], 45.)
    with raises(ValueError):
        ellipse_population(loc, IMPERIAL, [1e3], [1e3], [-1e3], 45.)


def test_ellipse_population_different_lengths_raises(loc):
    with raises(ValueError):
        ellipse_population(loc, IMPERIAL, [1e3, 2e3], [1e3], [1e3], 45.)


# Values Test
def test_ellipse_population_whole_uk(loc):
    """2000 km in every direction from the middle of the UK: every cell."""
    (pop,) = ellipse_population(loc, GB_MIDDLE, [2e6], [2e6], [2e6], 45.)
    assert pop == int(loc.pop.sum())


def test_ellipse_population_single_cell(loc):
    """Centre on a cell centre, all semi-axes 1 m: only that cell counts."""
    i = int(np.argmax(loc.pop))
    centre = (loc.pop_lat[i], loc.pop_lon[i])
    (pop,) = ellipse_population(loc, centre, [1.], [1.], [1.], 45.)
    assert pop == int(loc.pop[i])


def test_ellipse_population_coast(loc):
    """Near the sea the land cells must still count."""
    small, big = ellipse_population(loc, BRIGHTON, [2e3, 2e4], [4e3, 4e4],
                                    [1e3, 1e4], 0.)
    assert small > 0
    assert big > small


# Ellipse Test
def test_ellipse_population_equal_axes_is_circle(loc):
    """With all three semi-axes equal to r the zone is the circle."""
    radii = [1e3, 5e3, 2e4]
    ellipse = ellipse_population(loc, IMPERIAL, radii, radii, radii, 45.)
    circle = loc.get_population_by_radius(IMPERIAL, radii)
    assert ellipse == circle


def test_ellipse_population_between_circles(loc):
    """circle(smallest semi-axis) <= ellipse <= circle(largest semi-axis)."""
    a_perp, a_front, a_behind = 5e3, 1.5e4, 2e3
    (pop,) = ellipse_population(loc, IMPERIAL, [a_perp], [a_front],
                                [a_behind], 45.)
    inner, outer = loc.get_population_by_radius(
        IMPERIAL, [min(a_perp, a_front, a_behind),
                   max(a_perp, a_front, a_behind)])
    assert inner <= pop <= outer


def test_ellipse_population_reverse_bearing(loc):
    """Flying the other way with front and behind swapped: same people."""
    one_way = ellipse_population(loc, IMPERIAL, [5e3], [1.5e4], [2e3], 30.)
    other_way = ellipse_population(loc, IMPERIAL, [5e3], [2e3], [1.5e4],
                                   210.)
    assert one_way == other_way


def test_ellipse_population_bearing_360(loc):
    zero = ellipse_population(loc, IMPERIAL, [5e3], [1.5e4], [2e3], 0.)
    full_turn = ellipse_population(loc, IMPERIAL, [5e3], [1.5e4], [2e3], 360.)
    assert zero == full_turn


# Performance Test
def test_ellipse_population_performance(loc):
    """Best of 3 runs, as in the circle performance test."""
    times = []
    for _ in range(3):
        start = time.perf_counter()
        ellipse_population(loc, IMPERIAL, [1e3, 1e4, 1e5], [2e3, 2e4, 2e5],
                           [5e2, 5e3, 5e4], 45.)
        times.append(time.perf_counter() - start)
    assert min(times) < 1.0

# ---------------------------------------------------------------------------
# Speedup and Bounding boxes
# ---------------------------------------------------------------------------


SHETLAND = (60.15, -1.15)    # highest latitude: narrowest longitude degrees
CORNWALL = (50.12, -5.54)    # far south-west corner


# Format Test
def test_box_format():
    points = np.array([[51.5, -0.1], [55.9, -3.2], [0.0, 0.0]])
    keep = _in_bounding_box(IMPERIAL, points, 1e4)
    assert keep.dtype == bool
    assert keep.shape == (3,)


# Values Test
@mark.parametrize('centre', [IMPERIAL, SHETLAND, CORNWALL,
                             (0.0, 0.0), (75.0, 20.0), (-60.0, 100.0)])
@mark.parametrize('r', [1e2, 1e4, 1e5, 1e6])
def test_box_keeps_the_whole_circle_edge(centre, r):
    """Points just inside the circle, every 1 degree round it, are kept.

    The widest point of the circle in longitude is somewhere on this
    ring, so a box that is too narrow loses at least one of them.
    """
    ring = np.array([destination(centre, b, r * (1 - 1e-6))
                     for b in np.arange(0., 360., 1.)])
    assert np.all(_in_bounding_box(centre, ring, r))


def test_box_keeps_every_close_point():
    """Random centres and points all over the globe: no close point lost."""
    rng = np.random.default_rng(7)
    points = rng.uniform([-90, -180], [90, 180], size=(20000, 2))
    for _ in range(30):
        centre = rng.uniform([-90, -180], [90, 180])
        r = rng.choice([1e4, 1e5, 1e6, 5e6])
        close = great_circle_distance([centre], points)[0] < r
        assert np.all(_in_bounding_box(centre, points, r)[close])


def test_box_drops_far_points():
    """The box must actually filter, or there is no speed-up."""
    far = np.array([destination(IMPERIAL, b, 3e4) for b in (0., 90., 180.,
                                                            270.)])
    assert not np.any(_in_bounding_box(IMPERIAL, far, 1e4))


# Domain Test
def test_box_across_180th_meridian():
    """179.95 E and 179.96 W are only 0.09 degrees (about 10 km) apart."""
    centre = (0.0, 179.95)
    point = np.array([[0.0, -179.96]])
    assert great_circle_distance([centre], point)[0, 0] < 2e4
    assert np.all(_in_bounding_box(centre, point, 2e4))


def test_box_circle_over_the_pole():
    """A circle round the North Pole keeps every longitude near the pole."""
    centre = (89.5, 0.0)
    ring = np.array([[89.9, lon] for lon in range(-180, 180, 10)])
    assert np.all(_in_bounding_box(centre, ring, 2e5))


# Brute force Test: the lookups where a box goes wrong first
@mark.parametrize('centre, radius', [
    (SHETLAND, 2e4),
    (CORNWALL, 3e4),
    (IMPERIAL, 1e5),
])
def test_postcodes_box_match_brute_force(loc, centre, radius):
    fast = loc.get_postcodes_by_radius(centre, [radius])[0]
    slow = brute_force_postcodes(centre, radius)
    assert set(fast) == set(slow)


@mark.parametrize('centre, radius', [
    (SHETLAND, 2e4),
    (CORNWALL, 3e4),
    (IMPERIAL, 5e4),
])
def test_population_box_match_brute_force(loc, centre, radius):
    (fast,) = loc.get_population_by_radius(centre, [radius])
    assert fast == brute_force_population(loc, centre, radius)


def test_postcodes_whole_uk(loc):
    """2000 km from the middle of the UK: every postcode with a location."""
    (zone,) = loc.get_postcodes_by_radius(GB_MIDDLE, [2e6])
    assert len(zone) == len(loc._pc_codes)


def test_postcodes_far_north_centre(loc):
    """A centre near the pole with a big radius must not crash."""
    assert loc.get_postcodes_by_radius((89.0, 0.0), [1e6]) == [[]]
    assert loc.get_population_by_radius((89.0, 0.0), [1e6]) == [0]
