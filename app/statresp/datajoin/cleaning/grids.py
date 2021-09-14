# -*- coding: utf-8 -*-

import json

import numpy as np
import pandas as pd
from pyproj import Proj
from shapely.geometry import Point

__author__ = 'Yasas Senarath'






(-87.03820999999998, 36.00783), (-86.56824, 36.34003)


sw, ne = (-87.03820999999998, 36.00783), (-86.56824, 36.34003)
grid_array = GridArray(*sw, *ne, delta=1000)
print('Number of Regions: {}'.format(len(grid_array)))
p = (sw[0] + ne[0]) / 2, (sw[1] + ne[1]) / 2
print('grid of {} is {}'.format(p, grid_array.search(p)))



class GridArray:
    """Grid array gets a region rectangle and splits that area to grid regions.
    Search function will help to get which grid a point belongs.

    Example usage:
    >>> from tacci.regions import GridArray
    >>> sw, ne = (-87.05365006365614, 35.964741961820465), (-87.03599790090553, 35.97940067224588)
    >>> grid_array = GridArray(*sw, *ne, res=1000)
    >>> print('Number of Regions: {}'.format(len(grid_array)))
    >>> p = (sw[0] + ne[0]) / 2, (sw[1] + ne[1]) / 2
    >>> print('grid of {} is {}'.format(p, grid_array.search(p)))
    """
    proj = Proj(
        '+proj=lcc +lat_1=36.41666666666666 +lat_2=35.25 +lat_0=34.33333333333334 +lon_0=-86 '
        '+x_0=600000 +y_0=0 +ellps=GRS80 +datum=NAD83 +no_defs',
    )

    def __init__(self, x1, y1, x2, y2, delta=1000):
        """Initialize Grid Array.

        :param x1: South-West Longitude
        :param y1: South-West Latitude
        :param x2: North-East Longitude
        :param y2: North-East Latitude
        :param delta: resolution of the grid
        """
        co_sw, co_ne = Point(x1, y1), Point(x2, y2)
        sw, ne = Point(GridArray.proj.transform(co_sw.x, co_sw.y)), Point(
            GridArray.proj.transform(co_ne.x, co_ne.y))
        self._delta = delta
        self.xs = np.arange(sw.x - self._delta / 2,
                            ne.x + self._delta, self._delta)
        self.ys = np.arange(sw.y - self._delta / 2,
                            ne.y + self._delta, self._delta)

    def search(self, pt):
        """Search the grid where `pt` is located.
        TODO: Handle points outside of grid.

        :param pt: Point
        :return: Tuple(row, col) of pt
        """
        if isinstance(pt, Point):
            x, y = pt.x, pt.y
        elif isinstance(pt, pd.Series):
            x, y = pt.longitude, pt.latitude
        else:
            x, y = pt
        pj = Point(GridArray.proj(x, y))
        # `np.searchsorted` - finds indices where elements should be inserted to maintain order
        col, row = np.searchsorted(self.xs, pj.x) - 1, np.searchsorted(self.ys, pj.y) - 1
        return col, row

    def reverse_search(self, col, row):
        """Search the South-West location of grid provided by col, and row.

        :param col: grid column
        :param row: grid row
        :return: South-West location of grid
        """
        return self.proj(self.xs[col], self.ys[row], inverse=True)

    @property
    def delta(self):
        """Gets delta

        :return: delta parameter
        """
        return self._delta

    def __len__(self):
        """Returns total number of grids in provided region"""
        return len(self.xs) * len(self.ys)

    @property
    def shape(self):
        """Gets shape of grid.

        :return: shape of grid
        """
        return len(self.xs), len(self.ys)

    @property
    def bbox(self):
        """Gets bounding box of grid.

        :return:
        """
        x0, y0, x1, y1 = min(self.xs), min(self.ys), max(self.xs), max(self.ys)
        return [*self.proj(x0, y0, inverse=True), *self.proj(x1, y1, inverse=True)]

    def grids(self):
        """Generator for grids.

        :return: Generator for Tuple(Grid Index, South-West Pt, North-West Pt)
        """
        ys, xs = np.zeros(len(self.ys)), np.zeros(len(self.xs))
        for iy, ya in enumerate(self.ys):
            for ix, xa in enumerate(self.xs):
                xs[ix], ys[iy] = self.proj(xa, ya, inverse=True)
        for row, (y0, y1) in enumerate(zip(ys, ys[1:])):
            for col, (x0, x1) in enumerate(zip(xs, xs[1:])):
                yield [(col, row), (x0, y0), (x1, y1)]

    def to_geojson(self, path_or_buf=None):
        """Convert the object to a GeoJSON string.

        @path_or_buf: File path or object. If not specified, the result is only returned as a string.
        """
        features = []
        for (x, y), sw, ne in self.grids():
            south_west = [sw[0], sw[1]]
            south_east = [ne[0], sw[1]]
            north_east = [ne[0], ne[1]]
            north_west = [sw[0], ne[1]]
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [
                        [
                            south_west, south_east, north_east, north_west, south_west
                        ]
                    ],
                },
                'properties': {
                    'index': [x, y],
                },
            })
        features = {
            'type': 'FeatureCollection',
            'bbox': self.bbox,
            'features': features
        }
        if path_or_buf is not None:
            with open(path_or_buf, 'w', encoding='utf-8') as fp:
                json.dump(features, fp)
        return json.dumps(features)
