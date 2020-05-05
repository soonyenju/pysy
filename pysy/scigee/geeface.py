import os
import ee
import requests
import numpy as np

class EEarth(object):
    """
    Earth engine object
    """
    def __init__(self, dataset_name):
        super()
        ee.Initialize()
        self.dataset_name = dataset_name

    def fetch_image(self):
        self.image = ee.Image(self.dataset_name)

    def fetch_collection(self, date_range = [], roi = None):
        collection = ee.ImageCollection(self.dataset_name)
        # filter by date:
        if date_range:
            start_date, end_date = date_range
            collection = collection.filterDate(start_date, end_date)
        # filter by bounds:
        if roi:
            collection = collection.filterBounds(roi)
        self.__length__ = collection.size().getInfo()
        self.collection = collection

    # def fetch_collection(self, date_range = [], roi = None, to_list = False):
    # 	collection = ee.ImageCollection(self.dataset_name)
    # 	# filter by date:
    # 	if date_range:
    # 		start_date, end_date = date_range
    # 		collection = collection.filterDate(start_date, end_date)
    # 	# filter by bounds:
    # 	if roi:
    # 		collection = collection.filterBounds(roi)
    # 	self.__length__ = collection.size().getInfo()
    # 	# covert ee collection to list?
    # 	if to_list: 
    # 		collection = collection.toList(collection.size())
    # 	self.collection = collection
    
    # def get_image_by_index(self, idx):
    # 	self.image = ee.Image(self.collection.get(idx))

    # def reduce_collection(self, band = None, label = "DATA", band_reducer = ee.Reducer.mean(), spatial_reducer = ee.Reducer.median()):
    # 	if band:
    # 		image = self.collection.select(band).reduce(spatial_reducer).rename(band)
    # 	else:
    # 		image = self.collection.map(
    # 				lambda image: image.reduce(band_reducer)
    # 			).reduce(spatial_reducer).rename(label)
    # 	self.image = image

class Ecolbox(EEarth):
    """
    Ecolbox: Earth engine Collection Toolbox
    """
    def __init__(self, collection_name, date_range = [], roi = None):
        EEarth.__init__(self, collection_name)
        self.fetch_collection(date_range, roi)
        self.__length__ = self.collection.size().getInfo()

    def __call__(self, collection):
        # update collection box
        self.__length__ = collection.size().getInfo()
        self.collection = collection

    def __to_ee_list(self):
        self.__image_list__ = self.collection.toList(self.collection.size())

    def to_list(self, collection):
        if not collection:
            collection = self.collection
        return collection.toList(collection.size()).getInfo()

    def get_image_by_index(self, collection = None, idx = 0):
        # default: first.
        if collection:
            collection = collection.toList(collection.size())
            image = ee.Image(collection.get(idx))
        else:
            collection = self.collection
            if not isinstance(collection, ee.ee_list.List):
                self.__to_ee_list()
                image = ee.Image(self.__image_list__.get(idx))
        return image

    # def fmap(self, func, *args,**kwargs):
    #     # simplify the map function in the future
    #     return self.collection.map(
    #         lambda image: func(image, *args,**kwargs)
    #     )

    def reduce_collection(self, collection = None, band = None, label = "DATA", band_reducer = ee.Reducer.mean(), spatial_reducer = ee.Reducer.median()):
        if not collection:
            collection = self.collection
        if band:
            image = collection.select(band).reduce(spatial_reducer).rename(band)
        else:
            image = collection.map(
                    lambda image: image.reduce(band_reducer)
                ).reduce(spatial_reducer).rename(label)
        return image

class Emagebox(object):
    """
    Emagebox: Earth engine Image Toolbox
    """
    def __init__(self, image, scale = 30, max_pixels = 1e8, default_value = -9999):
        self.image = image
        self.scale = scale
        self.max_pixels = max_pixels
        self.default_value = default_value

    def set_scale(self, scale):
        self.scale = scale
    
    def set_max_pixels(self, max_pixels):
        self.max_pixels = max_pixels
    
    def set_default_value(self, default_value):
        self.default_value = default_value

    def get_band_names(self):
        return self.image.bandNames().getInfo()

    def get_stats(self, roi, reducer = ee.Reducer.mean()):
        stat = self.image.reduceRegion(
            reducer = reducer,
            geometry = roi,
            scale = self.scale,
            maxPixels = self.max_pixels
        )
        return stat.getInfo()
    
    def get_date(self):
        date = ee.Date(self.image.get('system:time_start'))
        return date.format('Y-M-d').getInfo()
    
    def get_proj(self, band, get_info = False):
        # // Get projection information from band.
        proj = self.image.select(band).projection()
        if get_info:
            proj = proj.getInfo()
        return proj

    def get_scale(self, band):
        # // Get scale (in meters) information from band.
        scale = self.image.select(band).projection().nominalScale()
        return scale.getInfo()

    def reproject(self, proj):
        return self.image.reproject(proj)

    def clip(self, roi):
        return self.image.clip(roi)

    def mask_value(self):
        mask = self.image.eq(self.default_value)
        self.image = self.image.updateMask(mask)

    def unmask(self):
        self.image = self.image.unmask(ee.Image.constant(self.default_value))

    def get_value(self, band, point, reducer = ee.Reducer.first()):
        value = self.image.select(band).reduceRegion(reducer, point, self.scale).get(band)
        value = ee.Number(value)
        return value.getInfo()

    def get_values(self, band, roi):
        # getInfo() is limited to 5000 records
        # ee.ee_exception.EEException: Array: No numbers in 'values', must provide a type.
        # see more at: https://gis.stackexchange.com/questions/321560/getting-dem-values-as-numpy-array-in-earth-engine
        # for rectangele: bounds = [-97.94, 26.81, -96.52, 26.84] ## sample land / sea bounds
        # for polygon: bounds = [[[105.532,19.059],[105.606,19.058],[105.605,19.108],[105.530,19.110],[105.532,19.059]]]
        latlng = ee.Image.pixelLonLat().addBands(self.image.clip(roi).unmask(ee.Image.constant(self.default_value)))
        latlng = latlng.reduceRegion(
            reducer = ee.Reducer.toList(), 
            geometry = roi, 
            maxPixels = self.max_pixels, 
            scale = self.scale
        )
        lats = np.array((ee.Array(latlng.get("latitude")).getInfo()))
        lons = np.array((ee.Array(latlng.get("longitude")).getInfo()))
        try:
            values = np.array((ee.Array(latlng.get(band)).getInfo()))
        except:
            values = np.full_like(lats, np.nan, dtype = np.float64)
        # self.values = list(values) ## print as list to check

        if not (values.shape == lats.shape == lons.shape):
            raise Exception(
                f"SizeError: " +
                f"values shape is {values.shape}, " + 
                f"lats shape is {lats.shape}, " + 
                f"lons shape is {lons.shape}."
            )
        return {
            "values":  values,
            "lons": lons,
            "lats": lats
        }

    def localize(self, save_name, save_folder = ".", crs_epsg = "4326"):
        url = self.image.getDownloadURL({
            "name": save_name,
            "crs": "EPSG:" + crs_epsg,
            "scale": self.scale
        })
        if not os.path.exists(save_folder): os.makedirs(save_folder)
        save_dir = f"{save_folder}/{save_name}.zip"

        # Download the subset
        r = requests.get(url, stream = True)
        with open(save_dir, 'wb') as fd:
            for chunk in r.iter_content(chunk_size = 1024):
                fd.write(chunk)

class VI(object):
    def __init__(self):
        pass

    def __call__(self, vi, image, **kwargs):
        if vi == "evi":
            image = self.calc_evi(image, **kwargs)
        elif vi == "cire":
            image = self.calc_cire(image, **kwargs)

        return image

    @staticmethod
    def calc_evi(image, **kwargs):
        nir = kwargs["nir"]
        red = kwargs["red"]
        blue = kwargs["blue"]

        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
            'NIR': image.select(nir),
            'RED': image.select(red),
            'BLUE': image.select(blue)
        })
        return image.addBands(evi.rename("EVI"))

    @staticmethod
    def calc_ndi(image, **kwargs):
        band_a = kwargs["b1"]
        band_b = kwargs["b2"]

        ndi = image.expression(
            '(band_a - band_b) / (band_a + band_b)', {
            'band_a': image.select(band_a),
            'band_b': image.select(band_b)
        })
        return image.addBands(ndi.rename("NDI"))

    @staticmethod
    def calc_cire(image, **kwargs):
        re2 = kwargs["re2"]
        re3 = kwargs["re3"]

        cire = image.expression(
            '(RE3/RE2) - 1', {
            'RE2': image.select(re2),
            'RE3': image.select(re3)
        })
        return image.addBands(cire.rename("CIRE"))

class Point(object):
    def __init__(self):
        pass

    def __call__(self, bounds = None, lon = None, lat = None):
        if bounds:
            lon, lat = bounds
        assert lon, "No longitude input..."
        assert lat, "No latitude input..."
        point = ee.Geometry.Point(lon, lat)
        return point

    @classmethod
    def get_circle_buffer(self, point, buffer_size = 100):
        return point.buffer(buffer_size)

    @classmethod
    def get_rect_buffer(self, point, buffer_size = 0.5):
        # buffer_size unit eq proj unit, e.g., degree for WGS84
        lon, lat = point.getInfo()["coordinates"]
        # example: [-97.94, 26.81, -96.52, 26.84]
        bounds = [lon - buffer_size, lat - buffer_size, lon + buffer_size, lat + buffer_size]
        return ee.Geometry.Rectangle(bounds)

class Polygon(object):
    def __init__(self):
        pass

    def __call__(self, bounds):
        # for rectangele: bounds = [-97.94, 26.81, -96.52, 26.84] ## sample land / sea bounds
        # for polygon: bounds = [[[105.532,19.059],[105.606,19.058],[105.605,19.108],[105.530,19.110],[105.532,19.059]]]

        if isinstance(bounds, list):
            if np.array(bounds).ndim == 1:
                region = ee.Geometry.Rectangle(bounds)
            else:
                region = ee.Geometry.Polygon(bounds)
        return region

class Geometry(Point, Polygon):
    def __init__(self, bounds = None, lon = None, lat = None):
        Point.__init__(self)
        Polygon.__init__(self)
        self.bounds = bounds
        self.lon = lon
        self.lat = lat

    def __call__(self, geom_type):
        if geom_type == 0 or geom_type == "point":
            geom = Point.__call__(self, bounds = self.bounds, lon = self.lon, lat = self.lat)
        elif geom_type == 2 or geom_type == "polygon":
            geom = Polygon.__call__(self, bounds = self.bounds)
        return geom


class Utils(VI):
    def __init__(self):
        VI.__init__(self)

    @classmethod
    def calc_vi(self, vi, image, **kwargs):
        return VI.__call__(self, vi, image, **kwargs)

    @classmethod
    def sentinel_2_cloud_mask(self, image, mask_out = True):
        """
        javascript code:

        # function maskS2clouds(image) {
        #   var qa = image.select('QA60');

        #   // Bits 10 and 11 are clouds and cirrus, respectively.
        #   var cloudBitMask = 1 << 10;
        #   var cirrusBitMask = 1 << 11;

        #   // Both flags should be set to zero, indicating clear conditions.
        #   var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
        #       .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

        #   return image.updateMask(mask).divide(10000);
        # }

        European Space Agency (ESA) clouds from 'QA60', i.e. Quality Assessment band at 60m
        
        parsed by Nick Clinton
        """

        qa = image.select('QA60')

        # bits 10 and 11 are clouds and cirrus
        cloudBitMask = int(2**10)
        cirrusBitMask = int(2**11)

        # both flags set to zero indicates clear conditions.
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(\
            qa.bitwiseAnd(cirrusBitMask).eq(0))
        
        if mask_out:
            return image.updateMask(mask)#.divide(10000)
        else:
            # clouds is not clear
            cloud = mask.Not().rename(['ESA_clouds'])

            # return the masked and scaled data.
            return image.addBands(cloud)

    # @classmethod
    # def coors2roi(self, bounds):
    #     # for rectangele: bounds = [-97.94, 26.81, -96.52, 26.84] ## sample land / sea bounds
    #     # for polygon: bounds = [[[105.532,19.059],[105.606,19.058],[105.605,19.108],[105.530,19.110],[105.532,19.059]]]
    #     if isinstance(bounds, list):
    #         if np.array(bounds).ndim == 1:
    #             roi = ee.Geometry.Rectangle(bounds)
    #         else:
    #             roi = ee.Geometry.Polygon(bounds)
    #     return roi

    # @classmethod
    # def coor2point(self, coors = None, lon = None, lat = None):
    #     if coors:
    #         lon, lat = coors
    #     assert lon, "No longitude input..."
    #     assert lat, "No latitude input..."
    #     point = ee.Geometry.Point(lon, lat)
    #     return point

    # @classmethod
    # def get_circle_buffer(self, point, buffer_size = 100):
    #     return point.buffer(buffer_size)

    # @classmethod	
    # def get_rect_buffer(self, point, buffer_size = 0.5):
    #     # buffer_size unit eq proj unit, e.g., degree for WGS84
    #     lon, lat = point.getInfo()["coordinates"]
    #     # example: [-97.94, 26.81, -96.52, 26.84]
    #     bounds = [lon - buffer_size, lat - buffer_size, lon + buffer_size, lat + buffer_size]
    #     return ee.Geometry.Rectangle(bounds)