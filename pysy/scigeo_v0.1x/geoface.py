# coding: utf-8
# author: soonyenju@outlook.com
# date: 11/28/2019

from pathlib import Path
from shapely.geometry import Polygon
import rasterio as rio
from rasterio.mask import mask
from rasterio.enums import Resampling
import geopandas as gpd
import warnings
import numpy as np

class Raster(object):
    """
    the wrapper of rasterio
    """
    def __init__(self, path):
        super() # or: super(Raster, self).__init__()
        self.path = path

    def __del__(self):
        if hasattr(self, "src"):
            self.src.closed

    def read(self):
        warnings.filterwarnings("ignore")
        with rio.open(self.path) as src:
            array = src.read()
            profile = src.profile
        return {"array": array, "meta": profile}

    def fopen(self, src_only = True):
        """
        c, a, b, f, d, e = src.transform
        gt = rasterio.transform.Affine.from_gdal(c, a, b, f, d, e)
        proj = src.crs
        count = src.count
        name = src.name
        mode = src.mode
        closed = src.closed
        width = src.width
        height = src.height
        bounds = src.bounds
        idtypes = {i: dtype for i, dtype in zip(
            src.indexes, src.dtypes)}
        meta = src.meta
        src = src.affine
        """
        self.src = rio.open(self.path)
        if not src_only:
            self.data = self.src.read()
            self.profile = self.src.profile

    def write(self, array, fulloutpath, profile, dtype = rio.float64):
        count=profile["count"]
        # bug fix, can't write a 3D array having a shape like (1, *, *)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0, :, :]
        profile.update(dtype = dtype, count = count, compress='lzw')
        with rio.open(fulloutpath, 'w', **profile) as dst:
            dst.write(array.astype(dtype), count)

    def clip(self, polygon):
        self.clip_arr, self.clip_transform = mask(self.src, polygon, crop=True)

    def resampling(self, new_shape):
        """
        new_shape format: height, width, count in order
        Resample: default Resampling.bilinear, other choice includes Resampling.average
        """
        height, width, count = new_shape
        resampled_data = self.src.read(
            out_shape = (height, width, count),
            resampling = Resampling.bilinear
        )
        return resampled_data

    def get_pntpairs(self, **kwargs):
        # compatible with affine format, rather than geotransform
        if not kwargs:
            # print("Inside data")
            affine = self.profile["transform"]
            cols = self.profile["width"]
            rows = self.profile["height"]
            data = self.data.ravel()
        else:
            # print("Outside data")
            # NOTICE: try to transform the geotransform to affine.
            affine = kwargs["affine"]
            # NOTICE: the first dimension of rasterio array is band.
            cols = kwargs["data"].shape[2]
            rows = kwargs["data"].shape[1]
            data = kwargs["data"].ravel()
        # print(affine)
        # print(profile)
        lats = [idx * affine[4] + affine[5] for idx in range(rows)]
        lons = [idx * affine[0] + affine[2] for idx in range(cols)]

        lons, lats = np.meshgrid(lons, lats)

        pntpairs = np.vstack([lons.ravel(), lats.ravel()]).T
        return pntpairs, data

class Vector(object):
    """docstring for Vector"""

    def __init__(self, **kwargs):
        super(Vector, self).__init__()
        if "path" in kwargs.keys():
            vector_path = kwargs["path"]
            try:
                self.path = vector_path.as_posix()
            except Exception as e:
                print(e)
                self.path = vector_path

    def __del__(self):
        pass

    def read(self):
        gdf = gpd.read_file(self.path)
        return gdf

    def write(self, gpd, fulloutpath):
        # filetype = fulloutpath.split('.')[-1]
        gpd.to_file(fulloutpath)

    def create_polygon(self, coors, epsg_code = "4326"):
        polygon_geom = Polygon(coors)
        crs = {"init": "epsg:" + epsg_code}
        poly = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_geom])  # gdf   
        # gjs = poly.to_json()
        return poly