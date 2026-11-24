"""Tests for deepimpact/mapping.py."""

import folium
from folium.plugins import PolyLineTextPath

import numpy as np
import pandas as pd
from pytest import raises, fixture, mark

from deepimpact import mapping, damage_zones
from deepimpact.locator import (GeospatialLocator, _initial_bearing,
                                great_circle_distance)
from deepimpact.mapping import _LEVEL_COLOURS

IMPERIAL = (51.498317, -0.176923)        # 'SW7 2AZ' in the postcode file
DEFAULT_PRESSURES = [1e3, 3.5e3, 27e3, 43e3]


def find_all(element, cls):
    """Every child of a folium element (at any depth) of type cls.

    Searching all levels means the tests still work when circles are
    moved into a FeatureGroup or another layer.
    """
    found = []
    for child in element._children.values():
        if isinstance(child, cls):
            found.append(child)
        found.extend(find_all(child, cls))
    return found


def radii_on(fmap):
    """Radii of the circles on a map, in the order they were drawn."""
    return [c.options['radius'] for c in find_all(fmap, folium.Circle)]


# =============================================================================
# plot_circle Tests
# The starter function from the repository. Draws one circle and returns
# the map; passing fmap draws on an existing map.
# =============================================================================


# Format Test
def test_plot_circle_returns_map():
    fmap = mapping.plot_circle(*IMPERIAL, 5000)
    assert isinstance(fmap, folium.Map)
    assert len(find_all(fmap, folium.Circle)) == 1


# Values Test
def test_plot_circle_centre_and_radius():
    fmap = mapping.plot_circle(*IMPERIAL, 5000)
    (circle,) = find_all(fmap, folium.Circle)
    assert circle.location == list(IMPERIAL)
    assert circle.options['radius'] == 5000


def test_plot_circle_style():
    """Extra keyword arguments such as color reach the circle."""
    fmap = mapping.plot_circle(*IMPERIAL, 5000, color='red')
    (circle,) = find_all(fmap, folium.Circle)
    assert circle.options['color'] == 'red'


# Layers Test
def test_plot_circle_add_to_existing_map():
    """Passing fmap draws on the same map instead of making a new one."""
    fmap = mapping.plot_circle(*IMPERIAL, 5000)
    same = mapping.plot_circle(*IMPERIAL, 2000, fmap=fmap)
    assert same is fmap
    assert radii_on(fmap) == [5000, 2000]


# =============================================================================
# plot_damage_zones Tests
# One circle per damage level around surface zero, drawn from the largest
# to the smallest, coloured by the pressure of its level. The expected
# colour of each test case is picked from _LEVEL_COLOURS by a level worked
# out by hand from the thresholds 1, 3.5, 27 and 43 kPa.
# =============================================================================


POINT = (51.4981, -0.1773)            # the docstring example point
PRESSURES = [1e3, 3.5e3, 27e3, 43e3]
RADII = [9e3, 4e3, 1.5e3, 1e3]
R = 6371e3


@fixture(scope='module')
def loc():
    """Used only for _in_ellipse_zones, the tested ellipse definition."""
    return GeospatialLocator()


def html_of(fmap):
    """The whole map as the HTML text a browser would get."""
    return fmap.get_root().render()


def markers_on(fmap):
    """Markers that are not circles (folium.Circle is a kind of Marker)."""
    return [m for m in find_all(fmap, folium.Marker)
            if not isinstance(m, folium.Circle)]


def layer_names(fmap):
    return [g.layer_name for g in find_all(fmap, folium.FeatureGroup)]


# Format Test
def test_damage_zones_returns_map():
    fmap = mapping.plot_damage_zones(*IMPERIAL, [9e3, 4e3, 1.5e3, 1e3])
    assert isinstance(fmap, folium.Map)
    assert len(find_all(fmap, folium.Circle)) == 4


def test_damage_zones_float_input():
    """One radius and one pressure as floats: one circle."""
    fmap = mapping.plot_damage_zones(*IMPERIAL, 5e3, pressures=27e3)
    assert radii_on(fmap) == [5e3]


# Values Test
def test_damage_zones_centred_on_surface_zero():
    fmap = mapping.plot_damage_zones(*IMPERIAL, [9e3, 4e3, 1.5e3, 1e3])
    for circle in find_all(fmap, folium.Circle):
        assert circle.location == list(IMPERIAL)
    assert list(fmap.location) == list(IMPERIAL)   # new map centred there


def test_damage_zones_radii():
    radii = [9e3, 4e3, 1.5e3, 1e3]
    fmap = mapping.plot_damage_zones(*IMPERIAL, radii)
    assert sorted(radii_on(fmap)) == sorted(radii)


# Order Test
def test_damage_zones_largest_first():
    """Radii given in a mixed order are still drawn largest first."""
    fmap = mapping.plot_damage_zones(*IMPERIAL, [1.5e3, 9e3, 1e3, 4e3],
                                     pressures=[27e3, 1e3, 43e3, 3.5e3])
    assert radii_on(fmap) == [9e3, 4e3, 1.5e3, 1e3]


def test_damage_zones_colour_stays_with_its_radius():
    """Sorting by radius must not mix up which pressure goes with which.

    Mixed input: 9 km goes with 1 kPa (level 0), 1 km with 43 kPa
    (level 3), and so on. After sorting, each circle keeps its own colour.
    """
    fmap = mapping.plot_damage_zones(*IMPERIAL, [1.5e3, 9e3, 1e3, 4e3],
                                     pressures=[27e3, 1e3, 43e3, 3.5e3])
    colours = [c.options['color'] for c in find_all(fmap, folium.Circle)]
    assert colours == [_LEVEL_COLOURS[0], _LEVEL_COLOURS[1],
                       _LEVEL_COLOURS[2], _LEVEL_COLOURS[3]]


# Colour Test
def colour_of(pressure):
    """Colour of a single circle drawn with the given pressure."""
    fmap = mapping.plot_damage_zones(*IMPERIAL, [5e3], pressures=[pressure])
    (circle,) = find_all(fmap, folium.Circle)
    return circle.options['color']


def test_damage_zones_colour_at_thresholds():
    """Each threshold pressure has its own level."""
    for level, pressure in enumerate(DEFAULT_PRESSURES):
        assert colour_of(pressure) == _LEVEL_COLOURS[level]


def test_damage_zones_colour_between_thresholds():
    """Between two thresholds the lower level is used."""
    assert colour_of(2e3) == _LEVEL_COLOURS[0]      # 1 <= 2 < 3.5 kPa
    assert colour_of(30e3) == _LEVEL_COLOURS[2]     # 27 <= 30 < 43 kPa


def test_damage_zones_colour_outside_thresholds():
    """Below 1 kPa is the lightest level, above 43 kPa the darkest."""
    assert colour_of(5e2) == _LEVEL_COLOURS[0]
    assert colour_of(1e5) == _LEVEL_COLOURS[3]


def test_damage_zones_same_pressure_and_colour():
    """The colour depends only on the pressure, not on the radius."""
    big = mapping.plot_damage_zones(*IMPERIAL, [9e3], pressures=[27e3])
    small = mapping.plot_damage_zones(*IMPERIAL, [1e3], pressures=[27e3])
    (c_big,) = find_all(big, folium.Circle)
    (c_small,) = find_all(small, folium.Circle)
    assert c_big.options['color'] == c_small.options['color']


# Marker Test
def test_damage_zones_one_marker_at_surface_zero():
    fmap = mapping.plot_damage_zones(*POINT, RADII)
    (marker,) = markers_on(fmap)
    assert marker.location == list(POINT)


def test_damage_zones_marker_text():
    fmap = mapping.plot_damage_zones(*POINT, RADII)
    (tooltip,) = find_all(markers_on(fmap)[0], folium.Tooltip)
    assert tooltip.text == 'Surface zero'
    assert '51.4981, -0.1773' in html_of(fmap)     # popup, 4 decimals


def test_damage_zones_two_calls_two_markers():
    fmap = mapping.plot_damage_zones(*POINT, RADII)
    mapping.plot_damage_zones(52.2074, 0.1170, RADII, fmap=fmap)
    assert len(markers_on(fmap)) == 2


# Popup Test
def test_damage_zones_circle_text():
    """Each circle says its level when hovered and its radius when clicked."""
    fmap = mapping.plot_damage_zones(*POINT, RADII, pressures=PRESSURES)
    tooltips = [t.text for c in find_all(fmap, folium.Circle)
                for t in find_all(c, folium.Tooltip)]
    assert 'Level 3: 27 kPa' in tooltips
    assert 'Radius: 1.5 km' in html_of(fmap)


def test_damage_zones_population_in_popup():
    fmap = mapping.plot_damage_zones(*POINT, RADII, pressures=PRESSURES,
                                     populations=[1, 2, 78598, 4])
    assert 'Population inside: 78,598' in html_of(fmap)


# Domain Test
def test_damage_zones_different_lengths_raises():
    with raises(ValueError):
        mapping.plot_damage_zones(*POINT, [9e3, 4e3],
                                  pressures=[1e3, 3.5e3, 27e3])


def test_damage_zones_negative_radius_raises():
    with raises(ValueError):
        mapping.plot_damage_zones(*POINT, [9e3, -4e3],
                                  pressures=[1e3, 3.5e3])


def test_damage_zones_population_wrong_length_raises():
    with raises(ValueError):
        mapping.plot_damage_zones(*POINT, RADII, pressures=PRESSURES,
                                  populations=[1, 2])


# Layers Test
def test_damage_zones_adds_to_existing_map():
    fmap = mapping.plot_circle(*POINT, 2e4)
    same = mapping.plot_damage_zones(*POINT, [9e3, 4e3, 1.5e3, 1e3],
                                     fmap=fmap)
    assert same is fmap
    assert len(find_all(fmap, folium.Circle)) == 5


def test_damage_zones_one_layer_per_level():
    fmap = mapping.plot_damage_zones(*POINT, RADII, pressures=PRESSURES)
    names = layer_names(fmap)
    assert len(names) == 4
    assert 'Level 3: 27 kPa (circle)' in names


def test_damage_zones_one_layer_control_after_layers():
    """Two calls: still one control, and it comes after every layer."""
    fmap = mapping.plot_damage_zones(*POINT, RADII)
    mapping.plot_damage_zones(52.2074, 0.1170, RADII, fmap=fmap)
    children = list(fmap._children.values())
    (control,) = [c for c in children if isinstance(c, folium.LayerControl)]
    groups = [c for c in children if isinstance(c, folium.FeatureGroup)]
    assert all(children.index(g) < children.index(control) for g in groups)


def test_damage_zones_one_legend():
    fmap = mapping.plot_damage_zones(*POINT, RADII)
    mapping.plot_damage_zones(52.2074, 0.1170, RADII, fmap=fmap)
    assert html_of(fmap).count('Airblast damage level') == 1


def test_damage_zones_fit_bounds():
    """The map zooms to the zones when there is something to show."""
    fmap = mapping.plot_damage_zones(*POINT, RADII)
    assert len(find_all(fmap, folium.FitBounds)) >= 1


# =============================================================================
# _ellipse_outline and plot_damage_ellipses Tests
# The outline is checked against _in_ellipse_zones, the zone definition
# already tested in test_locator.py: a point moved 1% towards the centre
# must be inside the zone, 1% away from it outside.
# =============================================================================


ELLIPSE_ARGS = ([2e3, 1e3], [5e3, 2.5e3], [1e3, 5e2], 30.)


def nudge(centre, points, factor):
    """Move each point towards (factor < 1) or away from the centre."""
    centre = np.asarray(centre)
    return centre + factor * (np.asarray(points) - centre)


def inside(loc, centre, points, a_perp, a_front, a_behind, bearing):
    (mask,) = loc._in_ellipse_zones(centre, points, [a_perp], [a_front],
                                    [a_behind], bearing)
    return mask


# Format Test
def test_outline_has_180_points():
    outline = np.asarray(mapping._ellipse_outline(*IMPERIAL, 2e3, 5e3, 1e3,
                                                  30.))
    assert outline.shape == (180, 2)


@mark.parametrize('n', [36, 360])
def test_outline_number_of_points(n):
    outline = mapping._ellipse_outline(*IMPERIAL, 2e3, 5e3, 1e3, 30., n=n)
    assert len(outline) == n


def test_ellipses_one_polygon_per_level():
    fmap = mapping.plot_damage_ellipses(*IMPERIAL, *ELLIPSE_ARGS,
                                        pressures=[27e3, 43e3])
    assert isinstance(fmap, folium.Map)
    assert len(find_all(fmap, folium.Polygon)) == 2
    assert all(name.endswith('(ellipse)') for name in layer_names(fmap))


# Values Test
@mark.parametrize('bearing', [0., 70., 225.])
def test_outline_on_the_zone_edge(loc, bearing):
    axes = (2e3, 5e3, 1e3)                 # a_perp, a_front, a_behind
    outline = mapping._ellipse_outline(*IMPERIAL, *axes, bearing)
    assert np.all(inside(loc, IMPERIAL, nudge(IMPERIAL, outline, 0.99),
                         *axes, bearing))
    assert not np.any(inside(loc, IMPERIAL, nudge(IMPERIAL, outline, 1.01),
                             *axes, bearing))


def test_outline_equal_axes_is_circle():
    outline = mapping._ellipse_outline(*IMPERIAL, 3e3, 3e3, 3e3, 30.)
    dist = great_circle_distance([IMPERIAL], outline)[0]
    assert np.allclose(dist, 3e3, rtol=1e-3)


def test_outline_points_the_right_way():
    """Bearing 0: the northmost point is a_front away,
    the southmost a_behind."""
    outline = np.asarray(mapping._ellipse_outline(*IMPERIAL, 2e3, 5e3, 1e3,
                                                  0.))
    north = outline[np.argmax(outline[:, 0])]
    south = outline[np.argmin(outline[:, 0])]
    assert np.isclose(great_circle_distance([IMPERIAL], [north])[0, 0], 5e3,
                      rtol=1e-2)
    assert np.isclose(great_circle_distance([IMPERIAL], [south])[0, 0], 1e3,
                      rtol=1e-2)


def test_outline_reverse_bearing(loc):
    """Reverse the bearing and swap front/behind: same zone, so the first
    outline lies on the edge of the second zone."""
    outline = mapping._ellipse_outline(*IMPERIAL, 2e3, 5e3, 1e3, 30.)
    other = (2e3, 1e3, 5e3, 210.)
    assert np.all(inside(loc, IMPERIAL, nudge(IMPERIAL, outline, 0.99),
                         *other))
    assert not np.any(inside(loc, IMPERIAL, nudge(IMPERIAL, outline, 1.01),
                             *other))


def test_ellipses_polygons_are_the_outlines():
    fmap = mapping.plot_damage_ellipses(*IMPERIAL, *ELLIPSE_ARGS,
                                        pressures=[27e3, 43e3])
    drawn = [np.asarray(p.locations) for p in find_all(fmap, folium.Polygon)]
    a_perp, a_front, a_behind, bearing = ELLIPSE_ARGS
    for level in range(2):
        expected = np.asarray(mapping._ellipse_outline(
            *IMPERIAL, a_perp[level], a_front[level], a_behind[level],
            bearing))
        assert any(np.allclose(d, expected) for d in drawn)


# Color Test
def test_ellipses_colour_by_pressure():
    fmap = mapping.plot_damage_ellipses(*IMPERIAL, [2e3], [5e3], [1e3], 30.,
                                        pressures=[30e3])
    (polygon,) = find_all(fmap, folium.Polygon)
    assert polygon.options['color'] == mapping._LEVEL_COLOURS[2]


# Marker Test
def test_ellipses_marker_at_surface_zero():
    fmap = mapping.plot_damage_ellipses(*POINT, *ELLIPSE_ARGS,
                                        pressures=[27e3, 43e3])
    (marker,) = markers_on(fmap)
    assert marker.location == list(POINT)


#  Popup Test
def test_ellipses_popup_text():
    """Hovering shows the level; clicking shows the three semi-axes."""
    fmap = mapping.plot_damage_ellipses(*IMPERIAL, [2e3], [5e3], [1e3], 30.,
                                        pressures=[27e3], populations=[78598])
    html = html_of(fmap)
    assert 'Level 3: 27 kPa' in html
    assert 'Semi-axes: 2.0 km across, 5.0 km ahead, 1.0 km behind' in html
    assert 'Population inside: 78,598' in html


# Domain Test
def test_ellipses_different_lengths_raises():
    with raises(ValueError):
        mapping.plot_damage_ellipses(*IMPERIAL, [2e3, 1e3], [5e3], [1e3],
                                     30., pressures=[27e3])


def test_ellipses_negative_axis_raises():
    with raises(ValueError):
        mapping.plot_damage_ellipses(*IMPERIAL, [2e3], [-5e3], [1e3], 30.,
                                     pressures=[27e3])


def test_ellipses_population_wrong_length_raises():
    with raises(ValueError):
        mapping.plot_damage_ellipses(*IMPERIAL, *ELLIPSE_ARGS,
                                     pressures=[27e3, 43e3], populations=[1])


# Layers Test
def test_ellipses_adds_to_existing_map():
    fmap = mapping.plot_damage_zones(*IMPERIAL, RADII)
    same = mapping.plot_damage_ellipses(*IMPERIAL, *ELLIPSE_ARGS,
                                        pressures=[27e3, 43e3], fmap=fmap)
    assert same is fmap
    assert len(find_all(fmap, folium.Polygon)) == 2
    assert len(find_all(fmap, folium.Circle)) == 4


# =============================================================================
# _postcode_sector and plot_risk Tests
# The risk map draws one dot per postcode sector (e.g. 'SW7 2'), at the
# mean location of the sector's postcodes and coloured by the highest
# probability in it. A small hand-made postcode table is passed in, so
# every expected value can be worked out by hand:
#
#   postcode   sector  lat    lon    probability
#   SW7 2AZ    SW7 2   51.50  -0.18  0.5
#   SW7 2BU    SW7 2   51.52  -0.16  1.0
#   SW7 2AA    SW7 2   51.51  -0.17  (not hit)
#   E1  6AN    E1 6    51.52  -0.07  0.2
#   N1  1AA    N1 1    51.54  -0.10  (not hit)
#
#   SW7 2: mean location (51.51, -0.17), highest 100%, mean 50%, 2 of 3
#   E1 6:  highest 20%, mean 20%, 1 of 1
#   N1 1:  nothing hit, so no dot
# =============================================================================


POSTCODES = pd.DataFrame({
    'Postcode': ['SW7 2AZ', 'SW7 2BU', 'SW7 2AA', 'E1  6AN', 'N1  1AA'],
    'Latitude': [51.50, 51.52, 51.51, 51.52, 51.54],
    'Longitude': [-0.18, -0.16, -0.17, -0.07, -0.10],
})
PROBABILITY = pd.DataFrame({'probability': [0.5, 1.0, 0.2]},
                           index=['SW7 2AZ', 'SW7 2BU', 'E1  6AN'])


def risk_map(probability=PROBABILITY, **kwargs):
    """Risk map of the hand-made postcode table."""
    return mapping.plot_risk(probability, postcodes=POSTCODES, **kwargs)


def dots_on(fmap):
    """The sector dots, in the order they were drawn."""
    return find_all(fmap, folium.CircleMarker)


def tooltip_of(dot):
    (tooltip,) = find_all(dot, folium.Tooltip)
    return tooltip.text


# Values Test: _postcode_sector
def test_sector_of_each_format():
    """One space, two spaces and no space all give the right sector."""
    codes = pd.Series(['SW7 2AZ', 'E1  6AN', 'SW195AE'])
    assert list(mapping._postcode_sector(codes)) == ['SW7 2', 'E1 6',
                                                     'SW19 5']


def test_sector_same_length():
    sectors = mapping._postcode_sector(POSTCODES['Postcode'])
    assert len(sectors) == len(POSTCODES)


def test_sector_of_every_real_postcode(loc):
    """Sector without spaces + the last two letters gives the postcode back,
    for all 1.7 million postcodes in the data."""
    codes = loc.postcodes['Postcode']
    sectors = mapping._postcode_sector(codes)
    rebuilt = sectors.str.replace(' ', '') + codes.str[-2:]
    assert (rebuilt == codes.str.replace(' ', '')).all()


# Format Test
def test_risk_returns_map():
    fmap = risk_map()
    assert isinstance(fmap, folium.Map)
    assert len(dots_on(fmap)) == 2            # SW7 2 and E1 6


# Values Test:plot_risk
def test_risk_no_dot_for_safe_sector():
    tooltips = [tooltip_of(d) for d in dots_on(risk_map())]
    assert not any(t.startswith('N1 1') for t in tooltips)


def test_risk_dot_at_mean_of_all_sector_postcodes():
    """The unaffected SW7 2AA counts towards the location too."""
    (dot,) = [d for d in dots_on(risk_map())
              if tooltip_of(d).startswith('SW7 2')]
    assert np.allclose(dot.location, [51.51, -0.17])


def test_risk_tooltip_shows_highest_probability():
    tooltips = {tooltip_of(d) for d in dots_on(risk_map())}
    assert tooltips == {'SW7 2: 100%', 'E1 6: 20%'}


def test_risk_popup_text():
    """Mean over all three SW7 2 postcodes: (0.5 + 1.0 + 0) / 3 = 50%."""
    html = html_of(risk_map())
    assert 'Sector SW7 2' in html
    assert 'Mean probability: 50%' in html
    assert 'Postcodes affected: 2 of 3' in html
    assert 'Postcodes affected: 1 of 1' in html


# Order Test
def test_risk_riskiest_drawn_last():
    """Lowest probability first, so the riskiest dot is on top."""
    tooltips = [tooltip_of(d) for d in dots_on(risk_map())]
    assert tooltips == ['E1 6: 20%', 'SW7 2: 100%']


# Colour Test
def test_risk_same_probability_same_colour():
    """The scale is fixed from 0 to 1, so the colour of a probability does
    not depend on the other sectors on the map."""
    alone = PROBABILITY.loc[['E1  6AN']]          # only E1 6 at 20%
    colour_alone = dots_on(risk_map(alone))[0].options['color']
    colour_with_others = dots_on(risk_map())[0].options['color']   # E1 6
    assert colour_alone == colour_with_others


def test_risk_higher_probability_other_colour():
    low, high = dots_on(risk_map())
    assert low.options['color'] != high.options['color']


# Domain Test
def test_risk_missing_column_raises():
    with raises(ValueError):
        mapping.plot_risk(pd.DataFrame({'p': [0.5]}, index=['SW7 2AZ']),
                          postcodes=POSTCODES)


def test_risk_nothing_hit():
    """No affected postcode: an empty map, not an error."""
    empty = pd.DataFrame({'probability': []}, index=pd.Index([], dtype=str))
    assert dots_on(risk_map(empty)) == []


def test_risk_unknown_postcode_ignored():
    """A postcode missing from the table is skipped, not an error."""
    extra = pd.concat([PROBABILITY,
                       pd.DataFrame({'probability': [0.9]},
                                    index=['ZZ9 9ZZ'])])
    assert len(dots_on(risk_map(extra))) == 2


# Layers Test
def test_risk_layer_name():
    assert 'Risk by postcode sector' in layer_names(risk_map())


def test_risk_adds_to_existing_map():
    fmap = mapping.plot_damage_zones(*POINT, RADII)
    same = risk_map(fmap=fmap)
    assert same is fmap
    assert len(dots_on(fmap)) == 2
    assert len(find_all(fmap, folium.LayerControl)) == 1


# =============================================================================
# plot_trajectory Tests
# The ground track from the entry point to surface zero, drawn as a line
# of points along the bearing. The points are checked with functions
# already tested in test_locator.py: great_circle_distance for how far
# each point is from the entry point, _initial_bearing for its direction.
# The track must end exactly at the surface zero of damage_zones, so the
# track and the damage zones meet on the same map.
# =============================================================================

ENTRY = (55.2, -2.5)                    # docstring example
BEARING = 217.
DISTANCE = 90e3
OUTCOME = {'burst_peak_dedz': 1000., 'burst_altitude': 9000.,
           'burst_distance': DISTANCE, 'burst_energy': 6000.,
           'burst_angle': 45., 'outcome': 'Airburst'}   # test_deepimpact.py


def track_of(fmap):
    """The points of the one track line on a map, as an (n, 2) array."""
    (line,) = find_all(fmap, folium.PolyLine)
    return np.asarray(line.locations)


def trajectory(**kwargs):
    return mapping.plot_trajectory(*ENTRY, BEARING, DISTANCE, **kwargs)


# Format Test
def test_trajectory_returns_map():
    fmap = trajectory()
    assert isinstance(fmap, folium.Map)
    assert len(find_all(fmap, folium.PolyLine)) == 1
    assert len(markers_on(fmap)) == 1


def test_trajectory_number_of_points():
    assert len(track_of(trajectory())) == 50
    assert len(track_of(trajectory(n=10))) == 10


# Values Test
def test_trajectory_starts_at_entry():
    assert np.allclose(track_of(trajectory())[0], ENTRY)


def test_trajectory_ends_at_surface_zero():
    """The last point is the centre of the damage zones."""
    blat, blon, _ = damage_zones(OUTCOME, *ENTRY, BEARING, 27e3)
    assert np.allclose(track_of(trajectory())[-1], [blat, blon])


def test_trajectory_evenly_spaced():
    """Distances from the entry point go 0, d/49, 2d/49, ... up to d."""
    track = track_of(trajectory())
    dist = great_circle_distance([ENTRY], track)[0]
    assert np.allclose(dist, np.linspace(0., DISTANCE, 50), atol=1e-3)


def test_trajectory_follows_bearing():
    """Seen from the entry point, every later point lies along the bearing."""
    track = track_of(trajectory())
    bearings = np.ravel(_initial_bearing(ENTRY, track[1:]))
    assert np.allclose(np.mod(bearings, 360), BEARING, atol=1e-6)


def test_trajectory_due_north():
    """Bearing 0: same longitude all the way, latitude going up."""
    fmap = mapping.plot_trajectory(*ENTRY, 0., DISTANCE)
    track = track_of(fmap)
    assert np.allclose(track[:, 1], ENTRY[1])
    assert np.all(np.diff(track[:, 0]) > 0)


# Text Test
def test_trajectory_line_tooltip():
    (line,) = find_all(trajectory(), folium.PolyLine)
    (tooltip,) = find_all(line, folium.Tooltip)
    assert tooltip.text == 'Ground track (bearing 217°, 90.0 km)'


def test_trajectory_entry_marker():
    fmap = trajectory()
    (marker,) = markers_on(fmap)
    assert marker.location == list(ENTRY)
    (tooltip,) = find_all(marker, folium.Tooltip)
    assert tooltip.text == 'Atmospheric entry'
    html = html_of(fmap)
    assert '55.2000, -2.5000' in html
    assert 'Bearing 217°' in html


# Domain Test
def test_trajectory_negative_distance_raises():
    with raises(ValueError):
        mapping.plot_trajectory(*ENTRY, BEARING, -1e3)


def test_trajectory_zero_distance():
    """No distance: every point at the entry point, and no error."""
    fmap = mapping.plot_trajectory(*ENTRY, BEARING, 0.)
    assert np.allclose(track_of(fmap), ENTRY)


# Layers Test
def test_trajectory_layer_and_arrows():
    fmap = trajectory()
    assert 'Trajectory' in layer_names(fmap)
    assert len(find_all(fmap, PolyLineTextPath)) == 1


def test_trajectory_with_damage_zones():
    """Track and zones on one map: two markers, one layer control."""
    blat, blon, _ = damage_zones(OUTCOME, *ENTRY, BEARING, 27e3)
    fmap = mapping.plot_damage_zones(blat, blon, RADII)
    same = trajectory(fmap=fmap)
    assert same is fmap
    assert len(markers_on(fmap)) == 2           # entry and surface zero
    assert len(find_all(fmap, folium.LayerControl)) == 1


# =============================================================================
# plot_damage_zones: a whole scenario Test
# The same function drawing ellipses instead of circles when semi_axes is
# given, and the ground track when entry is given, on one map with one
# layer control and one legend. Also levels with no zone (radius or area
# zero) and the new errors.
# =============================================================================

# a_perp, a_front, a_behind
SEMI_AXES = ([2e3, 1e3], [5e3, 2.5e3], [1e3, 5e2])
TWO_RADII = [5e3, 2.5e3]
TWO_PRESSURES = [27e3, 43e3]


def ellipse_scenario(**kwargs):
    return mapping.plot_damage_zones(*IMPERIAL, TWO_RADII,
                                     pressures=TWO_PRESSURES,
                                     semi_axes=SEMI_AXES, bearing=30.,
                                     **kwargs)


def entry_scenario(**kwargs):
    """Surface zero from damage_zones, so it lies on the track."""
    blat, blon, _ = damage_zones(OUTCOME, *ENTRY, BEARING, 27e3)
    fmap = mapping.plot_damage_zones(blat, blon, RADII, entry=ENTRY,
                                     bearing=BEARING, **kwargs)
    return fmap, blat, blon


# Domain Test
def test_damage_zones_zero_radius_left_out():
    """A level not reached on the ground (radius 0) is not drawn."""
    fmap = mapping.plot_damage_zones(*POINT, [9e3, 0., 1e3],
                                     pressures=[1e3, 3.5e3, 27e3])
    assert sorted(radii_on(fmap)) == [1e3, 9e3]
    assert len(layer_names(fmap)) == 2


def test_damage_zones_all_zero():
    """No zone at all: no circles and nothing to zoom to, but no error."""
    fmap = mapping.plot_damage_zones(*POINT, [0., 0., 0., 0.])
    assert find_all(fmap, folium.Circle) == []
    assert len(markers_on(fmap)) == 1          # surface zero is still shown
    assert find_all(fmap, folium.FitBounds) == []


def test_damage_ellipses_zero_area_left_out():
    fmap = mapping.plot_damage_ellipses(*IMPERIAL, [2e3, 0.], [5e3, 2e3],
                                        [1e3, 5e2], 30.,
                                        pressures=TWO_PRESSURES)
    assert len(find_all(fmap, folium.Polygon)) == 1


def test_damage_zones_semi_axes_without_bearing_raises():
    with raises(ValueError):
        mapping.plot_damage_zones(*IMPERIAL, TWO_RADII,
                                  pressures=TWO_PRESSURES,
                                  semi_axes=SEMI_AXES)


def test_damage_zones_entry_without_bearing_raises():
    with raises(ValueError):
        mapping.plot_damage_zones(*IMPERIAL, RADII, entry=ENTRY)


def test_damage_zones_two_semi_axes_raises():
    """semi_axes must hold three arrays: a_perp, a_front and a_behind."""
    with raises(ValueError):
        mapping.plot_damage_zones(*IMPERIAL, TWO_RADII,
                                  pressures=TWO_PRESSURES,
                                  semi_axes=SEMI_AXES[:2], bearing=30.)


# Ellipse Test
def test_scenario_circles_without_semi_axes():
    fmap = mapping.plot_damage_zones(*IMPERIAL, RADII)
    assert len(find_all(fmap, folium.Circle)) == 4
    assert find_all(fmap, folium.Polygon) == []


def test_scenario_ellipses_with_semi_axes():
    fmap = ellipse_scenario()
    assert len(find_all(fmap, folium.Polygon)) == 2
    assert find_all(fmap, folium.Circle) == []
    assert all(name.endswith('(ellipse)') for name in layer_names(fmap))


def test_scenario_semi_axes_in_the_right_order():
    """a_perp, a_front and a_behind reach the ellipses unswapped: each
    polygon is the outline worked out directly from the three axes."""
    fmap = ellipse_scenario()
    drawn = [np.asarray(p.locations) for p in find_all(fmap, folium.Polygon)]
    for a_perp, a_front, a_behind in zip(*SEMI_AXES):
        expected = np.asarray(mapping._ellipse_outline(
            *IMPERIAL, a_perp, a_front, a_behind, 30.))
        assert any(np.allclose(d, expected) for d in drawn)


def test_scenario_population_reaches_ellipses():
    html = html_of(ellipse_scenario(populations=[78598, 1234]))
    assert 'Population inside: 78,598' in html
    assert 'Semi-axes:' in html


# Trajectory Test
def test_scenario_track_from_entry_to_surface_zero():
    fmap, blat, blon = entry_scenario()
    track = track_of(fmap)
    assert np.allclose(track[0], ENTRY)
    assert np.allclose(track[-1], [blat, blon])


def test_scenario_entry_and_surface_zero_markers():
    fmap, _, _ = entry_scenario()
    assert len(markers_on(fmap)) == 2


def test_scenario_track_drawn_under_the_zones():
    """The track layer comes first, so the zones are drawn on top."""
    fmap, _, _ = entry_scenario()
    assert layer_names(fmap)[0] == 'Trajectory'


def test_scenario_zoom_includes_entry():
    """The last zoom set on the map shows the entry point too."""
    fmap, _, _ = entry_scenario()
    (south, west), (north, east) = find_all(fmap, folium.FitBounds)[-1].bounds
    assert south <= ENTRY[0] <= north
    assert west <= ENTRY[1] <= east


# Layers Test
def test_scenario_everything_on_one_map():
    """Ellipses, track and circles together: one control, one legend."""
    blat, blon, _ = damage_zones(OUTCOME, *ENTRY, BEARING, 27e3)
    fmap = mapping.plot_damage_zones(blat, blon, TWO_RADII,
                                     pressures=TWO_PRESSURES,
                                     semi_axes=SEMI_AXES, bearing=BEARING,
                                     entry=ENTRY)
    mapping.plot_damage_zones(blat, blon, RADII, fmap=fmap)
    assert len(find_all(fmap, folium.LayerControl)) == 1
    assert html_of(fmap).count('Airblast damage level') == 1
