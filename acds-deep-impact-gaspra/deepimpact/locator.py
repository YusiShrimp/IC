"""Module dealing with postcode information."""

import os

import numpy as np
import pandas as pd

# Folder holding the data files: <repo>/resources
_RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'resources')

__all__ = ['GeospatialLocator', 'great_circle_distance']


def great_circle_distance(latlon1, latlon2):
    """
    Calculate the great circle distance (in metres) between pairs of
    points specified as latitude and longitude on a spherical Earth
    (with radius 6371 km).

    Parameters
    ----------

    latlon1: arraylike
        latitudes and longitudes of first point (as [n, 2] array for n points,
        or a single point as a 1-D array of length 2)
    latlon2: arraylike
        latitudes and longitudes of second point (as [m, 2] array for m points,
        or a single point as a 1-D array of length 2)

    Returns
    -------

    numpy.ndarray
        Distance in metres between each pair of points (as an n x m array).
        If latlon1 or latlon2 is a single 1-D point, that axis is removed,
        e.g. a 1-D latlon2 gives shape (n,). Pass [[lat, lon]] to keep it.

    Raises
    ------

    ValueError
        If any latitude is outside [-90, 90] or any longitude is outside
        [-180, 180] degrees.

    Notes
    -----

    Uses the haversine formula (DamageMapper.ipynb), vectorised with numpy
    broadcasting. Checked against the reference values in
    tests/test_deepimpact.py. Implementation aided by Claude (Anthropic).

    Examples
    --------

    >>> import numpy
    >>> fmt = lambda x: numpy.format_float_scientific(x, precision=3)
    >>> with numpy.printoptions(formatter={'all': fmt}):
    ...     print(great_circle_distance([[54.0, 0.0], [55, 0.0]], [55, 1.0]))
    [1.286e+05 6.378e+04]
    """

    # Haversine formula. Vectorised with numpy broadcasting.
    R = 6371e3  # Earth radius in metres

    # Transform inputs: float arrays, at least 2-D (still in degrees)
    p1 = np.atleast_2d(np.asarray(latlon1, dtype=float))
    p2 = np.atleast_2d(np.asarray(latlon2, dtype=float))

    # Check inputs are valid latitudes and longitudes
    for p in (p1, p2):
        if np.any(np.abs(p[:, 0]) > 90) or np.any(np.abs(p[:, 1]) > 180):
            raise ValueError('latitude must be in [-90, 90] and '
                             'longitude in [-180, 180] degrees')

    # Degrees -> radians
    p1, p2 = np.radians(p1), np.radians(p2)

    # First set as column vectors (n, 1), second set as row vectors (1, m)
    lat1, lon1 = p1[:, 0][:, None], p1[:, 1][:, None]
    lat2, lon2 = p2[:, 0][None, :], p2[:, 1][None, :]

    # Haversine term, broadcast to shape (n, m)
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    a = np.clip(a, 0.0, 1.0)  # guard against rounding just above 1

    # Central angle (2 * arcsin(sqrt(a))) times radius gives arc length
    distance = 2 * R * np.arcsin(np.sqrt(a))
    # A single 1-D point removes that axis (as in the docstring example);
    # pass [[lat, lon]] instead to keep the full (n, m) shape
    if np.ndim(latlon2) == 1:
        distance = distance[:, 0]
    if np.ndim(latlon1) == 1:
        distance = distance[0, ...]
    return distance


def _initial_bearing(X, points):
    """
    Initial bearing (forward azimuth) from X to each of a set of points.

    Parameters
    ----------
    X : arraylike
        Latitude-longitude pair of the start point (degrees).
    points : arraylike
        Latitudes and longitudes of the end points (degrees), as an
        [N, 2] array.

    Returns
    -------
    numpy.ndarray
        Bearing of each point as seen from X, shape (N,), in degrees
        clockwise from north (0 = north, 90 = east), between 0 and 360.

    Notes
    -----
    Uses the forward azimuth formula in DamageMapper.ipynb, written with
    arctan2 so that the angle is in the correct quadrant. A point at X
    itself gets bearing 0. Implementation aided by Claude (Anthropic).
    """

    # Start point as two scalars, end points as two (N,) arrays; radians
    lat1, lon1 = np.radians(np.ravel(np.asarray(X, dtype=float)))
    pts = np.radians(np.atleast_2d(np.asarray(points, dtype=float)))
    lat2, lon2 = pts[:, 0], pts[:, 1]
    dlon = lon2 - lon1

    # Forward azimuth (beta in DamageMapper.ipynb): arctan2(east, north)
    num = np.sin(dlon) * np.cos(lat2)
    den = (np.cos(lat1) * np.sin(lat2)
           - np.sin(lat1) * np.cos(lat2) * np.cos(dlon))

    # Radians in (-pi, pi] -> degrees, then shift negatives into [0, 360)
    return np.degrees(np.arctan2(num, den)) % 360


def _in_bounding_box(X, points, r):
    """
    Find the points inside a latitude-longitude box around X that
    contains every point closer than r to X.

    Parameters
    ----------
    X : arraylike
        Latitude-longitude pair of the centre (degrees).
    points : numpy.ndarray
        Latitudes and longitudes of the points (degrees), shape (N, 2).
    r : float
        Radius (m) of the circle the box must contain.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape (N,), True for the points inside the box.

    Notes
    -----
    The box assumes great-circle distances on a sphere of radius
    6371 km, as in great_circle_distance. Moving along a meridian is the
    shortest way to change latitude, so |lat - lat0| <= r / R. The
    longitude half-width is arcsin(sin(r / R) / cos(lat0)), the widest
    point of the circle; if the circle contains a pole, longitude is not
    filtered. Points outside the box are certainly not closer than r, so
    filtering first does not change any result. Suggested in the project
    briefing; implementation aided by Claude (Anthropic).
    """

    R = 6371e3  # Earth radius (m)
    lat0, lon0 = np.ravel(np.asarray(X, dtype=float))

    # Radius as an angle seen from the Earth's centre (radians),
    # made a tiny bit larger so rounding never drops a point
    ang = r / R * (1 + 1e-9)

    # Latitude: no point in the circle is more than ang north or south of X
    keep = np.abs(points[:, 0] - lat0) <= np.degrees(ang)

    # Longitude: allow more degrees away from the equator, where meridians
    # are closer together (no limit if the circle covers a pole)
    s = np.sin(ang) / np.cos(np.radians(lat0))
    if ang < np.pi / 2 and s < 1:
        dlon = np.degrees(np.arcsin(s))
        diff = np.abs(points[:, 1] - lon0)
        # Near the 180th meridian, go the short way round
        if lon0 - dlon < -180 or lon0 + dlon > 180:
            diff = np.minimum(diff, 360 - diff)
        keep &= diff <= dlon
    return keep


class GeospatialLocator(object):
    """
    Class to interact with a postcode database file and a population grid file.
    """

    def __init__(self, postcode_file='',
                 census_file='',
                 norm=great_circle_distance):
        """
        Parameters
        ----------

        postcode_file : str, optional
            Filename of a .csv file containing geographic
            location data for postcodes. Defaults to
            resources/full_postcodes.csv. Postcodes without a real
            location (placeholder latitude 99.999999) are dropped.

        census_file :  str, optional
            Filename of a .asc file containing census data on a
            latitude-longitude grid. Defaults to
            resources/UK_residential_population_2011_latlon.asc.
            Cells with the NODATA value (-9999) are dropped, so
            self.pop_lat, self.pop_lon and self.pop are 1-D arrays
            of the cells with data.

        norm : function
            Python function defining the distance between points in
            latitude-longitude space.

        Notes
        -----
        Data loading aided by Claude (Anthropic).

        """
        if not postcode_file:
            postcode_file = os.path.join(_RESOURCES, 'full_postcodes.csv')
        if not census_file:
            census_file = os.path.join(
                _RESOURCES, 'UK_residential_population_2011_latlon.asc')

        self.norm = norm

        # Postcodes: one row per postcode. 12,687 postcodes (mostly Isle of
        # Man and Channel Islands) have the placeholder location
        # (99.999999, 0.0) instead of a real one, so they are dropped.
        postcodes = pd.read_csv(postcode_file)
        has_loc = ((postcodes['Latitude'].abs() <= 90)
                   & (postcodes['Longitude'].abs() <= 180))
        self.postcodes = postcodes[has_loc].reset_index(drop=True)

        # The same postcodes as numpy arrays for fast distance queries
        self._pc_latlon = self.postcodes[['Latitude', 'Longitude']].to_numpy()
        self._pc_codes = self.postcodes['Postcode'].to_numpy()

        # Census header (6 lines): ncols, nrows, NODATA_value, 3 array names
        with open(census_file) as f:
            header = [next(f).split() for _ in range(6)]
        ncols, nrows = int(header[0][1]), int(header[1][1])
        nodata = float(header[2][1])

        # Data: 3 stacked arrays (lat, lon, population), each nrows x ncols
        data = np.loadtxt(census_file, skiprows=6)
        lat, lon, pop = data.reshape(3, nrows, ncols)

        # Keep only cells with data (drops sea / NODATA cells)
        valid = pop != nodata
        self.pop_lat = lat[valid]
        self.pop_lon = lon[valid]
        self.pop = pop[valid]

        # Cell centres as an (N, 2) array of (lat, lon), built once
        self._pop_latlon = np.column_stack([self.pop_lat, self.pop_lon])

    def _candidates(self, X, points, r):
        """
        Boolean mask of the points that could be closer than r to X.

        Uses _in_bounding_box when self.norm is great_circle_distance.
        For any other distance function the box may not be valid, so
        every point is kept.
        """
        if self.norm is great_circle_distance:
            return _in_bounding_box(X, points, r)
        return np.ones(len(points), dtype=bool)

    def get_postcodes_by_radius(self, X, radii):
        """
        Return postcodes within specific distances of
        input location.

        Parameters
        ----------
        X : arraylike
            Latitude-longitude pair of centre location
        radii : arraylike
            array of radial distances from X

        Returns
        -------
        list of lists
            Contains the lists of postcodes closer than the elements
            of radii to the location X.

        Raises
        ------
        ValueError
            If any radius is negative.

        Notes
        -----
        Postcodes without a location in the data file (placeholder
        latitude 99.999999) are never returned. Distances use self.norm.
        Implementation aided by Claude (Anthropic).

        Examples
        --------

        >>> locator = GeospatialLocator()
        >>> locator.get_postcodes_by_radius((51.4981, -0.1773), [1.5e3])
        >>> locator.get_postcodes_by_radius((51.4981, -0.1773),
        ...                                 [1.5e3, 4.0e3])
        """

        # Radii as a 1-D float array; negative distances make no sense
        radii = np.atleast_1d(np.asarray(radii, dtype=float))
        if np.any(radii < 0):
            raise ValueError('radii must be non-negative')

        # Skip postcodes outside the box around the largest circle
        near = self._candidates(X, self._pc_latlon, radii.max(initial=0.))
        codes = self._pc_codes[near]

        # Distance from X to each remaining postcode
        dist = np.ravel(self.norm(X, self._pc_latlon[near]))

        # One list of postcodes per radius, in the same order as radii
        return [codes[dist < r].tolist() for r in radii]

    def get_population_by_radius(self, X, radii, method='naive'):
        """
        Return the population within specific distances of input location.

        Parameters
        ----------
        X : arraylike
            Latitude-longitude pair of centre location
        radii : arraylike
            array of radial distances from X
        method : str, optional
            Method to use for population calculation. Default is naive, and
            this is what the scorer will use. You may implement more
            sophisticated methods if you wish as an extension task, and
            discuss them in your presentation. If you do so, add a
            description of the method in the docstring, including the
            string to pass in to this kwarg.
            Note you should not change the default method.

        Returns
        -------
        list
            Contains the population closer than the elements of radii to
            the location X. Output should be the same shape as the radii array.

        Raises
        ------
        ValueError
            If any radius is negative.

        Notes
        -----
        method='naive': a grid cell (1 km x 1 km) is counted in full if
        its centre is closer than the radius to X, otherwise not at all.
        Implementation aided by Claude (Anthropic).

        Examples
        --------
        >>> loc = GeospatialLocator()
        >>> loc.get_population_by_radius((51.4981, -0.1773), [1e2, 5e2, 1e3])

        """

        if method != 'naive':
            raise NotImplementedError(
                f"method={method!r} is not implemented; use 'naive'")

        # Radii as a 1-D float array; negative distances make no sense
        radii = np.atleast_1d(np.asarray(radii, dtype=float))
        if np.any(radii < 0):
            raise ValueError('radii must be non-negative')

        # Skip cells outside the box around the largest circle
        near = self._candidates(X, self._pop_latlon, radii.max(initial=0.))
        pop = self.pop[near]

        # Distance from X to each remaining cell centre
        dist = np.ravel(self.norm(X, self._pop_latlon[near]))

        # Naive method: a cell counts fully if its centre is inside the radius
        return [int(pop[dist < r].sum()) for r in radii]

    def _in_ellipse_zones(self, X, points, a_perp, a_front, a_behind,
                          bearing):
        """
        Find the points inside each angle-dependent blast zone.

        Each zone is two half-ellipses centred on X and aligned with the
        trajectory: semi-axis a_front ahead of X, a_behind behind X, and
        a_perp crossrange (see DamageMapper.ipynb).

        Parameters
        ----------
        X : arraylike
            Latitude-longitude pair of the surface zero location (degrees)
        points : numpy.ndarray
            Latitudes and longitudes of the points to test, shape (N, 2)
        a_perp, a_front, a_behind : arraylike or float
            Crossrange, downrange and uprange semi-axes (m), one per zone
        bearing : float
            Bearing of the trajectory (degrees clockwise from north)

        Returns
        -------
        list of numpy.ndarray
            One boolean array of shape (N,) per zone, True for the points
            strictly inside that zone.

        Raises
        ------
        ValueError
            If the semi-axes have different lengths or any is negative.

        Notes
        -----
        Each point is placed in polar coordinates around X (distance from
        self.norm, bearing from _initial_bearing) and rotated so that x is
        the distance along the trajectory and y the distance across it.
        With all three semi-axes equal to r, a zone is the circle of radius
        r used by the radius queries. Implementation aided by Claude
        (Anthropic).
        """

        # Semi-axes as 1-D float arrays, one entry per zone
        a_perp, a_front, a_behind = [np.atleast_1d(np.asarray(a, dtype=float))
                                     for a in (a_perp, a_front, a_behind)]
        if not a_perp.shape == a_front.shape == a_behind.shape:
            raise ValueError('a_perp, a_front and a_behind must have '
                             'the same length')
        for a in (a_perp, a_front, a_behind):
            if np.any(a < 0):
                raise ValueError('semi-axes must be non-negative')

        # Skip points outside the box around the largest zone (each zone
        # fits inside a circle of radius max(a_perp, a_front, a_behind))
        r_max = max(a.max(initial=0.) for a in (a_perp, a_front, a_behind))
        near = self._candidates(X, points, r_max)
        pts = points[near]

        # Polar coordinates around X: distance (m) and angle from the bearing
        dist = np.ravel(self.norm(X, pts))
        alpha = np.radians(_initial_bearing(X, pts) - bearing)

        # Along-track (x > 0 ahead of X) and crossrange (y) distances, (N,)
        x2 = (dist * np.cos(alpha)) ** 2
        y2 = (dist * np.sin(alpha)) ** 2
        ahead = np.cos(alpha) >= 0

        # (x / a)^2 + (y / a_perp)^2 < 1, multiplied by (a * a_perp)^2
        # to avoid 0 / 0 when a semi-axis is zero
        masks = []
        for ap, af, ab in zip(a_perp, a_front, a_behind):
            a = np.where(ahead, af, ab)
            # Full-length mask: False for every point outside the box
            mask = np.zeros(len(points), dtype=bool)
            mask[near] = x2 * ap**2 + y2 * a**2 < (a * ap) ** 2
            masks.append(mask)
        return masks

    def get_postcodes_by_ellipse(self, X, a_perp, a_front,
                                 a_behind, bearing):
        """
        Return postcodes inside the angle-dependent blast-zone (made of
        two half-ellipses), centred on the input location.

        NOTE: This is an extension task.

        Parameters
        ----------
        X : arraylike
            Latitude-longitude pair of the surface zero location
        a_perp : arraylike or float
            Crossrange semi-axes (m), one per damage level
        a_front : arraylike or float
            Downrange semi-axes (m), ahead of X along the bearing
        a_behind : arraylike or float
            Uprange semi-axes (m), behind X
        bearing : float
            Bearing of the trajectory (degrees clockwise from north),
            which orients the zones

        Returns
        -------
        list of lists
            Contains the lists of postcodes inside each zone.

        Raises
        ------
        ValueError
            If the semi-axes have different lengths or any is negative.

        Notes
        -----
        A postcode is inside a zone if it is strictly inside the two
        half-ellipses (see _in_ellipse_zones). With all three semi-axes
        equal to r, the result is the same as get_postcodes_by_radius.
        Implementation aided by Claude (Anthropic).
        """

        # One boolean mask per damage zone over all postcodes with a location
        masks = self._in_ellipse_zones(X, self._pc_latlon, a_perp, a_front,
                                       a_behind, bearing)

        # One list of postcodes per zone, in the same order as the semi-axes
        return [self._pc_codes[mask].tolist() for mask in masks]

    def get_population_by_ellipse(self, X, a_perp, a_front, a_behind, bearing,
                                  method='naive'):
        """
        Return the population within the angle-dependent blast-zone (made of
        two half-ellipses), centred on the input location.

        NOTE: This is an extension task.

        Parameters
        ----------
        X : arraylike
            Latitude-longitude pair of surface zero location
        a_perp : arraylike or float
            Crossrange semi-axes (m), one per damage level
        a_front : arraylike or float
            Downrange semi-axes (m), ahead of X along the bearing
        a_behind : arraylike or float
            Uprange semi-axes (m), behind X
        bearing : float
            Bearing of the trajectory (degrees clockwise from north),
            which orients the zones
        method : str, optional
            Method to use for population calculation. Default is naive.
            You may implement more sophisticated methods if you wish,
            and discuss them in your presentation. If you do so, add a
            description of the method in the docstring, including the
            string to pass in to this kwarg.
            Note you should not change the default method.

        Returns
        -------
        list
            Contains the population inside each zone. Output has the
            same length as the semi-axis arrays.

        Raises
        ------
        ValueError
            If the semi-axes have different lengths or any is negative.
        NotImplementedError
            If method is not 'naive'.

        Notes
        -----
        method='naive': a grid cell (1 km x 1 km) is counted in full if
        its centre is strictly inside the zone, otherwise not at all, as
        in get_population_by_radius. Implementation aided by Claude
        (Anthropic).
        """

        if method != 'naive':
            raise NotImplementedError(
                f"method={method!r} is not implemented; use 'naive'")

        # Naive method: sum the full population of cells centred in the zone
        masks = self._in_ellipse_zones(X, self._pop_latlon, a_perp, a_front,
                                       a_behind, bearing)
        return [int(self.pop[mask].sum()) for mask in masks]
