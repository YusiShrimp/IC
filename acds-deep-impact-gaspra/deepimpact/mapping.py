"""
This module contains some useful mapping functions. Note that mapping
the output of your solver and locator is an extension task. plot_circle
is provided as a starting point.
"""
import folium
import numpy as np
from folium.plugins import PolyLineTextPath
from branca.colormap import LinearColormap
from branca.element import MacroElement, Template
from .damage import surface_zero
from .locator import great_circle_distance

__all__ = ['plot_circle', 'plot_damage_zones', 'plot_damage_ellipses',
           'plot_risk', 'plot_trajectory']

# The four damage levels of DamageMapper.ipynb: lowest pressure (Pa) of
# each level and its colour, from least to most severe (ColorBrewer
# YlOrRd, readable with colour blindness)
_LEVEL_PRESSURES = [1e3, 3.5e3, 27e3, 43e3]
_LEVEL_COLOURS = ['#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']
_LEVEL_DESCRIPTIONS = ['About 10% of glass windows shatter',
                       'About 90% of glass windows shatter',
                       'Trees and wood buildings flattened',
                       'Multistorey brick buildings collapse']


def plot_circle(lat, lon, radius, fmap=None, **kwargs):
    """
    Plot a circle on a map (creating a new folium map instance if necessary).

    Parameters
    ----------

    lat: float
        latitude of circle to plot (degrees)
    lon: float
        longitude of circle to plot (degrees)
    radius: float
        radius of circle to plot (m)
    fmap: folium.Map
        existing map object

    Returns
    -------

    Folium map object

    Examples
    --------

    >>> import folium
    >>> deepimpact.plot_circle(52.79, -2.95, 1e3, fmap=None)
    """

    if not fmap:
        fmap = folium.Map(location=[lat, lon], control_scale=True)

    folium.Circle([lat, lon], radius, fill=True,
                  fillOpacity=0.6, **kwargs).add_to(fmap)

    return fmap


def _zone_popup(level, pressure, size, population=None):
    """
    Popup text for one damage zone: level, what happens there, pressure,
    size and (if given) the population inside.

    Parameters
    ----------
    level : int
        Damage level, 0 (1 kPa) to 3 (43 kPa)
    pressure : float
        Threshold pressure of the zone (Pa)
    size : str
        Size of the zone, already formatted (e.g. 'Radius: 5.2 km')
    population : int, optional
        Number of people inside the zone

    Returns
    -------
    folium.Popup
    """
    lines = [f'<b>Damage level {level + 1}: {pressure / 1e3:g} kPa</b>',
             _LEVEL_DESCRIPTIONS[level], size]
    if population is not None:
        lines.append(f'Population inside: {int(population):,}')
    return folium.Popup('<br>'.join(lines), max_width=300)


def _damage_legend():
    """
    Legend box (bottom left of the map) with the colour, pressure and
    description of each damage level.

    Returns
    -------
    branca.element.MacroElement
    """
    rows = ''.join(
        f'<div><span style="background:{colour}; width:14px; height:14px; '
        f'display:inline-block; margin-right:6px;"></span>'
        f'{i + 1}: {p / 1e3:g} kPa, {text}</div>'
        for i, (p, colour, text) in enumerate(zip(
            _LEVEL_PRESSURES, _LEVEL_COLOURS, _LEVEL_DESCRIPTIONS)))
    html = ('<div style="position: fixed; bottom: 30px; left: 30px; '
            'z-index: 9999; background: white; padding: 8px 10px; '
            'border: 1px solid grey; border-radius: 4px; font-size: 13px;">'
            f'<b>Airblast damage level</b>{rows}</div>')
    legend = MacroElement()
    legend._template = Template(
        '{% macro html(this, kwargs) %}' + html + '{% endmacro %}')
    return legend


def _finish_map(fmap):
    """
    Add the damage legend (once) and a layer control to the map.
    """
    if not getattr(fmap, '_has_damage_legend', False):
        fmap.get_root().add_child(_damage_legend())
        fmap._has_damage_legend = True
    _add_layer_control(fmap)


def _add_layer_control(fmap):
    """
    Give the map exactly one layer control, added last, so that it lists
    every layer on the map (it can only list the layers added before it).
    """
    for key, child in list(fmap._children.items()):
        if isinstance(child, folium.LayerControl):
            del fmap._children[key]
    folium.LayerControl(collapsed=False).add_to(fmap)


def _fit_to_circle(fmap, lat, lon, r, extra=()):
    """
    Zoom the map so that a circle of radius r (m) around (lat, lon) fits,
    together with any extra [lat, lon] points (e.g. the entry point).

    Uses the same bounding box as the locator: r / R either side in
    latitude, and r / (R cos(lat)) in longitude.
    """
    if r <= 0 and not extra:
        return
    dlat = np.degrees(max(r, 0.) / 6371e3)
    dlon = dlat / np.cos(np.radians(lat))
    lats = [lat - dlat, lat + dlat] + [p[0] for p in extra]
    lons = [lon - dlon, lon + dlon] + [p[1] for p in extra]
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])


def _plot_circles(blat, blon, radii, pressures, populations, fmap):
    """
    Draw the circular damage zones (see plot_damage_zones): one circle per
    damage level, largest first, coloured by level, with popups, a
    surface-zero marker, legend, layer control and zoom. The inputs are
    already checked.
    """

    if fmap is None:
        fmap = folium.Map(location=[blat, blon], control_scale=True)

    # Damage level of each pressure: 0 (1 kPa) to 3 (43 kPa); pressures
    # between two thresholds take the lower level
    level = np.searchsorted(_LEVEL_PRESSURES, pressures, side='right') - 1
    level = np.clip(level, 0, len(_LEVEL_PRESSURES) - 1)

    # Largest circle first, so the smaller ones are drawn on top of it;
    # each circle in its own layer, which can be switched on and off
    for i in np.argsort(radii)[::-1]:
        if radii[i] <= 0:     # level not reached on the ground: no zone
            continue
        colour = _LEVEL_COLOURS[level[i]]
        name = f'Level {level[i] + 1}: {pressures[i] / 1e3:g} kPa'
        population = None if populations is None else populations[i]
        popup = _zone_popup(level[i], pressures[i],
                            f'Radius: {radii[i] / 1e3:.1f} km', population)
        layer = folium.FeatureGroup(name=f'{name} (circle)').add_to(fmap)
        folium.Circle([blat, blon], radius=float(radii[i]), color=colour,
                      weight=2, fill=True, fill_color=colour,
                      fill_opacity=0.25, tooltip=name,
                      popup=popup).add_to(layer)

    # Surface zero: the point on the ground directly below the airburst
    folium.Marker([blat, blon], tooltip='Surface zero',
                  popup=f'Surface zero<br>{blat:.4f}, {blon:.4f}',
                  icon=folium.Icon(color='black', icon='crosshairs',
                                   prefix='fa')).add_to(fmap)

    # Legend, layer control, and zoom to fit the largest circle
    _finish_map(fmap)
    _fit_to_circle(fmap, blat, blon, radii.max(initial=0.))

    return fmap


def plot_damage_zones(blat, blon, damage_rad,
                      pressures=(1e3, 3.5e3, 27e3, 43e3), populations=None,
                      fmap=None, semi_axes=None, bearing=None, entry=None):
    """
    Plot the airblast damage zones of an impact on a map: circles, or the
    angle-dependent zones if semi_axes is given, and optionally the
    ground track from the entry point (creating a new folium map instance
    if necessary).

    Parameters
    ----------
    blat : float
        Latitude of the surface zero point (degrees)
    blon : float
        Longitude of the surface zero point (degrees)
    damage_rad : arraylike or float
        Damage radii (m), one per damage level, as returned by
        damage_zones
    pressures : arraylike or float, optional
        Threshold pressure (Pa) of each damage level, in the same order
        as damage_rad. Defaults to the four levels of DamageMapper.ipynb.
    populations : arraylike, optional
        Population inside each zone (e.g. from
        GeospatialLocator.get_population_by_radius or
        get_population_by_ellipse), shown in the popups
    fmap : folium.Map, optional
        Existing map to draw on
    semi_axes : tuple of arraylike, optional
        (a_perp, a_front, a_behind) as returned by damage_ellipses, one
        value each per damage level. If given, the angle-dependent zones
        are drawn instead of circles (see plot_damage_ellipses).
    bearing : float, optional
        Bearing of the trajectory (degrees clockwise from north). Needed
        with semi_axes or entry.
    entry : arraylike, optional
        Latitude-longitude pair of the meteoroid entry point (degrees). If
        given, the ground track from there to surface zero is drawn (see
        plot_trajectory).

    Returns
    -------
    folium.Map
        The map with one zone per damage level and a marker at surface
        zero. Hovering over a zone shows its level; clicking it shows
        what happens there, its size and (if given) its population.
        Each level is a separate layer that can be switched on and off,
        and a legend explains the colours. The map is zoomed to fit the
        largest zone (and the entry point, if given). Levels with a
        radius of zero (not reached on the ground) are left out.

    Raises
    ------
    ValueError
        If damage_rad, pressures, populations and semi-axes (if given)
        have different lengths, any radius is negative, or semi_axes or
        entry is given without a bearing.

    Notes
    -----
    This is the one function to call for a whole scenario; it uses
    plot_damage_ellipses and plot_trajectory, which can also be used on
    their own. Circles are drawn from the largest to the smallest, so the
    smaller, more severe zones stay on top. Each zone is coloured by its
    damage level in DamageMapper.ipynb, from light yellow (1 kPa) to
    dark red (43 kPa), so the same pressure always has the same colour.
    Implementation aided by Claude (Anthropic).

    Examples
    --------
    >>> fmap = plot_damage_zones(51.4981, -0.1773, [9e3, 4e3, 1.5e3, 1e3])
    """

    # One radius and one pressure per damage level
    radii = np.atleast_1d(np.asarray(damage_rad, dtype=float))
    pressures = np.atleast_1d(np.asarray(pressures, dtype=float))
    if radii.shape != pressures.shape:
        raise ValueError('damage_rad and pressures must have the same length')
    if np.any(radii < 0):
        raise ValueError('damage radii must be non-negative')
    if populations is not None and len(populations) != len(radii):
        raise ValueError('populations must have one value per damage level')
    if (semi_axes is not None or entry is not None) and bearing is None:
        raise ValueError('bearing is needed with semi_axes or entry')

    # Ground track first, so that the zones are drawn on top of it
    if entry is not None:
        distance = float(great_circle_distance(entry, [blat, blon]))
        fmap = plot_trajectory(entry[0], entry[1], bearing, distance,
                               fmap=fmap)

    # The zones: angle-dependent if semi-axes are given, otherwise circles
    if semi_axes is not None:
        if len(semi_axes) != 3:
            raise ValueError('semi_axes must hold a_perp, a_front and '
                             'a_behind')
        a_perp, a_front, a_behind = semi_axes
        fmap = plot_damage_ellipses(blat, blon, a_perp, a_front, a_behind,
                                    bearing, pressures=pressures,
                                    populations=populations, fmap=fmap)
        reach = max(np.max(np.atleast_1d(a), initial=0.) for a in semi_axes)
    else:
        fmap = _plot_circles(blat, blon, radii, pressures, populations,
                             fmap)
        reach = radii.max(initial=0.)

    # Zoom to show both the zones and the entry point
    if entry is not None:
        _fit_to_circle(fmap, blat, blon, reach, extra=[list(entry)])

    return fmap


def _ellipse_outline(blat, blon, a_perp, a_front, a_behind, bearing,
                     n=180):
    """
    Points on the edge of one angle-dependent blast zone, as
    [lat, lon] pairs for a folium polygon.

    Each point is first placed in along-track (x, ahead of surface zero)
    and crossrange (y) coordinates on the two half-ellipses, then turned
    into a distance and a bearing from surface zero and moved there with
    surface_zero. This is the same geometry the locator uses to decide
    whether a point is inside the zone, in reverse.

    Parameters
    ----------
    blat, blon : float
        Surface zero (degrees)
    a_perp, a_front, a_behind : float
        Crossrange, downrange and uprange semi-axes (m)
    bearing : float
        Bearing of the trajectory (degrees clockwise from north)
    n : int, optional
        Number of points on the edge

    Returns
    -------
    list of [float, float]
    """
    t = np.linspace(0., 2 * np.pi, n, endpoint=False)
    # Front half-ellipse ahead of surface zero (cos t >= 0), back half
    # behind it; both share the crossrange semi-axis
    x = np.where(np.cos(t) >= 0, a_front, a_behind) * np.cos(t)
    y = a_perp * np.sin(t)

    # Distance and bearing of each edge point as seen from surface zero
    dist = np.hypot(x, y)
    angle = bearing + np.degrees(np.arctan2(y, x))
    return [list(surface_zero(blat, blon, a, d))
            for a, d in zip(angle, dist)]


def plot_damage_ellipses(blat, blon, a_perp, a_front, a_behind, bearing,
                         pressures=(1e3, 3.5e3, 27e3, 43e3),
                         populations=None, fmap=None):
    """
    Plot the angle-dependent airblast damage zones (two half-ellipses
    each) around surface zero on a map (creating a new folium map
    instance if necessary).

    Parameters
    ----------
    blat : float
        Latitude of the surface zero point (degrees)
    blon : float
        Longitude of the surface zero point (degrees)
    a_perp, a_front, a_behind : arraylike or float
        Crossrange, downrange (ahead) and uprange (behind) semi-axes (m),
        one per damage level, as returned by damage_ellipses
    bearing : float
        Bearing of the trajectory (degrees clockwise from north)
    pressures : arraylike or float, optional
        Threshold pressure (Pa) of each damage level, in the same order
        as the semi-axes. Defaults to the four levels of
        DamageMapper.ipynb.
    populations : arraylike, optional
        Population inside each zone (e.g. from
        GeospatialLocator.get_population_by_ellipse), shown in the popups
    fmap : folium.Map, optional
        Existing map to draw on

    Returns
    -------
    folium.Map
        The map with one zone per damage level, a marker at surface zero,
        a legend and a layer control, zoomed to fit the largest zone.
        Hovering and clicking work as in plot_damage_zones.

    Raises
    ------
    ValueError
        If the semi-axes, pressures and populations (if given) have
        different lengths, or any semi-axis is negative.

    Notes
    -----
    Colours, popups and layers are the same as in plot_damage_zones, so
    circular and elliptical zones can be compared on the same map. The
    edges are drawn with the same geometry the locator uses
    (get_postcodes_by_ellipse). Implementation aided by Claude
    (Anthropic).

    Examples
    --------
    >>> fmap = plot_damage_ellipses(51.4981, -0.1773, [9e3, 4e3],
    ...                             [7e3, 3e3], [8e3, 3.5e3], 217.,
    ...                             pressures=[1e3, 3.5e3])
    """

    # One set of semi-axes and one pressure per damage level
    a_perp, a_front, a_behind, pressures = [
        np.atleast_1d(np.asarray(a, dtype=float))
        for a in (a_perp, a_front, a_behind, pressures)]
    if not (a_perp.shape == a_front.shape == a_behind.shape
            == pressures.shape):
        raise ValueError('semi-axes and pressures must have the same length')
    if min(a.min(initial=0.) for a in (a_perp, a_front, a_behind)) < 0:
        raise ValueError('semi-axes must be non-negative')
    if populations is not None and len(populations) != len(pressures):
        raise ValueError('populations must have one value per damage level')

    if fmap is None:
        fmap = folium.Map(location=[blat, blon], control_scale=True)

    # Damage level of each pressure, as in plot_damage_zones
    level = np.searchsorted(_LEVEL_PRESSURES, pressures, side='right') - 1
    level = np.clip(level, 0, len(_LEVEL_PRESSURES) - 1)

    # Largest zone (by area) first, so the smaller ones are drawn on top
    area = a_perp * (a_front + a_behind)
    for i in np.argsort(area)[::-1]:
        if area[i] <= 0:      # level not reached on the ground: no zone
            continue
        colour = _LEVEL_COLOURS[level[i]]
        name = f'Level {level[i] + 1}: {pressures[i] / 1e3:g} kPa'
        population = None if populations is None else populations[i]
        size = (f'Semi-axes: {a_perp[i] / 1e3:.1f} km across, '
                f'{a_front[i] / 1e3:.1f} km ahead, '
                f'{a_behind[i] / 1e3:.1f} km behind')
        popup = _zone_popup(level[i], pressures[i], size, population)
        outline = _ellipse_outline(blat, blon, a_perp[i], a_front[i],
                                   a_behind[i], bearing)
        layer = folium.FeatureGroup(name=f'{name} (ellipse)').add_to(fmap)
        folium.Polygon(outline, color=colour, weight=2, fill=True,
                       fill_color=colour, fill_opacity=0.25, tooltip=name,
                       popup=popup).add_to(layer)

    # Surface zero: the point on the ground directly below the airburst
    folium.Marker([blat, blon], tooltip='Surface zero',
                  popup=f'Surface zero<br>{blat:.4f}, {blon:.4f}',
                  icon=folium.Icon(color='black', icon='crosshairs',
                                   prefix='fa')).add_to(fmap)

    # Legend, layer control, and zoom to fit the largest zone
    _finish_map(fmap)
    r_max = max(a.max(initial=0.) for a in (a_perp, a_front, a_behind))
    _fit_to_circle(fmap, blat, blon, r_max)

    return fmap


def _postcode_sector(postcodes):
    """
    Postcode sector of each postcode, e.g. 'SW7 2AZ' -> 'SW7 2' and
    'SW195AE' -> 'SW19 5'.

    The sector is the outward code (everything before the last three
    characters) and the first digit of the inward code.

    Parameters
    ----------
    postcodes : pandas.Series
        Postcodes (str)

    Returns
    -------
    pandas.Series
    """
    return postcodes.str[:-3].str.strip() + ' ' + postcodes.str[-3]


def plot_risk(probability, postcodes=None, fmap=None):
    """
    Plot the impact risk on a map, one dot per postcode sector, coloured
    by the highest probability of any postcode in the sector (creating a
    new folium map instance if necessary).

    Parameters
    ----------
    probability : pandas.DataFrame
        Output of impact_risk: indexed by postcode, with a column
        'probability' (between 0 and 1)
    postcodes : pandas.DataFrame, optional
        Postcode locations with columns 'Postcode', 'Latitude' and
        'Longitude', e.g. GeospatialLocator().postcodes. If not given,
        a GeospatialLocator is created to read them (a few seconds).
    fmap : folium.Map, optional
        Existing map to draw on

    Returns
    -------
    folium.Map
        The map with one dot per affected sector, a colour scale from
        0 to 1 and a layer control. Hovering over a dot shows the sector
        and its highest probability; clicking it also shows the mean
        probability over all postcodes in the sector and how many of
        them are affected.

    Notes
    -----
    Sectors (for example 'SW7 2', about 180 postcodes) keep the map
    readable and fast: a 10 km zone covers about 76,000 postcodes but
    only about 700 sectors. The colour is the highest probability in the
    sector, so a sector is never shown as safer than any part of it.
    The dot is at the mean location of the sector's postcodes; the data
    has no sector boundaries. Implementation aided by Claude (Anthropic).
    """

    if 'probability' not in probability.columns:
        raise ValueError("probability must have a column 'probability'")
    if postcodes is None:
        from .locator import GeospatialLocator
        postcodes = GeospatialLocator().postcodes

    # Every postcode with its sector and probability (0 if never hit)
    table = postcodes[['Postcode', 'Latitude', 'Longitude']].copy()
    table['sector'] = _postcode_sector(table['Postcode'])
    table['probability'] = (table['Postcode'].map(probability['probability'])
                            .fillna(0.))

    # Only sectors with at least one affected postcode
    hit = table.loc[table['probability'] > 0, 'sector'].unique()
    table = table[table['sector'].isin(hit)]

    # One row per sector: mean location, highest and mean probability,
    # and how many of its postcodes are affected
    sectors = table.groupby('sector').agg(
        lat=('Latitude', 'mean'), lon=('Longitude', 'mean'),
        p_max=('probability', 'max'), p_mean=('probability', 'mean'),
        affected=('probability', lambda p: int((p > 0).sum())),
        total=('probability', 'size'))

    if fmap is None:
        fmap = folium.Map(location=[51.5, -1.5], zoom_start=6,
                          control_scale=True)

    # Fixed scale from 0 to 1, so the same colour always means the same
    # probability, whatever the scenarios
    colours = LinearColormap(_LEVEL_COLOURS, vmin=0., vmax=1.,
                             caption='Probability of being in the '
                                     'damage zone (highest in sector)')
    colours.add_to(fmap)

    # Lowest probability first, so the riskiest sectors are drawn on top
    layer = folium.FeatureGroup(name='Risk by postcode sector').add_to(fmap)
    for row in sectors.sort_values('p_max').itertuples():
        name = row.Index
        popup = folium.Popup(
            f'<b>Sector {name}</b><br>'
            f'Highest probability: {row.p_max:.0%}<br>'
            f'Mean probability: {row.p_mean:.0%}<br>'
            f'Postcodes affected: {row.affected} of {row.total}',
            max_width=300)
        folium.CircleMarker([row.lat, row.lon], radius=6, weight=1,
                            color=colours(row.p_max), fill=True,
                            fill_color=colours(row.p_max), fill_opacity=0.8,
                            tooltip=f'{name}: {row.p_max:.0%}',
                            popup=popup).add_to(layer)

    _add_layer_control(fmap)
    if len(sectors):
        fmap.fit_bounds([[sectors.lat.min(), sectors.lon.min()],
                         [sectors.lat.max(), sectors.lon.max()]])
    return fmap


def plot_trajectory(lat, lon, bearing, distance, fmap=None, n=50):
    """
    Plot the ground track of the meteoroid, from its entry point to the
    surface zero point, on a map (creating a new folium map instance if
    necessary).

    Parameters
    ----------
    lat : float
        Latitude of the meteoroid entry point (degrees)
    lon : float
        Longitude of the meteoroid entry point (degrees)
    bearing : float
        Bearing of the trajectory (degrees clockwise from north)
    distance : float
        Distance along the ground from the entry point to surface zero
        (m), e.g. outcome['burst_distance']
    fmap : folium.Map, optional
        Existing map to draw on
    n : int, optional
        Number of points on the track

    Returns
    -------
    folium.Map
        The map with the track (arrows show the direction of travel), a
        marker at the entry point and a layer control, zoomed to fit the
        track.

    Raises
    ------
    ValueError
        If distance is negative.

    Notes
    -----
    The track follows the great circle from the entry point along the
    bearing. Its points come from surface_zero, the same function that
    damage_zones uses, so the track ends exactly at surface zero.
    Implementation aided by Claude (Anthropic).

    Examples
    --------
    >>> fmap = plot_trajectory(55.2, -2.5, 217., 90e3)
    """

    if distance < 0:
        raise ValueError('distance must be non-negative')

    if fmap is None:
        fmap = folium.Map(location=[lat, lon], control_scale=True)

    # Points along the great circle, from the entry point (0 m) to
    # surface zero (distance)
    track = [list(surface_zero(lat, lon, bearing, d))
             for d in np.linspace(0., distance, n)]

    layer = folium.FeatureGroup(name='Trajectory').add_to(fmap)
    line = folium.PolyLine(track, color='black', weight=3, dash_array='8 6',
                           tooltip=f'Ground track (bearing {bearing:g}°, '
                                   f'{distance / 1e3:.1f} km)')
    line.add_to(layer)
    # Arrows along the line show the direction of travel
    PolyLineTextPath(line, '\u25BA', repeat=True, offset=6,
                     attributes={'fill': 'black', 'font-size': '14'}
                     ).add_to(layer)

    # Entry point: where the meteoroid enters the atmosphere
    folium.Marker([lat, lon], tooltip='Atmospheric entry',
                  popup=f'Atmospheric entry<br>{lat:.4f}, {lon:.4f}<br>'
                        f'Bearing {bearing:g}°',
                  icon=folium.Icon(color='gray', icon='arrow-right',
                                   prefix='fa')).add_to(layer)

    _add_layer_control(fmap)
    lats, lons = zip(*track)
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    return fmap
