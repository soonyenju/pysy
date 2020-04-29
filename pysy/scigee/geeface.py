import ee
import numpy as np
import requests, zipfile
from pathlib import Path
from pysy.toolbox.sysutil import *

class Earth(object):
	def __init__(self, ds_name, collection = True, MUTE = True):
		super()
		self.MUTE = MUTE
		ee.Initialize()
		if collection:
			self.image_collection = ee.ImageCollection(ds_name)
		else:
			self.image = ee.Image(ds_name)
   
	def filter_by_date_bounds(self, date_range, area_coordinates_list = []):
		start_date, end_date = date_range
		image_collection = self.image_collection.filterDate(start_date, end_date)
		if area_coordinates_list:
			roi = ee.Geometry.Polygon(area_coordinates_list)
			image_collection = image_collection.filterBounds(roi)
		return image_collection

	def collection_to_image_list(self, image_collection):
		self.image_list = image_collection.toList(image_collection.size())
		self.__length__ = image_collection.size().getInfo()
		return self.image_list

	def get_image_by_order(self, image_list, order):
		return ee.Image(image_list.get(order))

	def get_image_band_names(self, image):
		return image.bandNames().getInfo()

	def get_image_proj_scale(self, image, band):
		# // Get projection information from band.
		proj = image.select(band).projection()

		# // Get scale (in meters) information from band.
		scale = image.select(band).projection().nominalScale()
		return proj.getInfo(), scale.getInfo()

	def reproject(self, image, proj):
		return image.reproject(proj)

	def get_point_value(self, image, band, coors = None, lon = None, lat = None, scale = 10):
		if coors:
			lon, lat = coors
		assert lon, "No longitude input..."
		assert lat, "No latitude input..."
		point = ee.Geometry.Point(lon, lat)
		value = image.select(band).reduceRegion(ee.Reducer.first(), point, scale).get(band)
		value = ee.Number(value)
		return value.getInfo()
  
	def create_point_buffer(self, lon, lat, buffer_size = 100):
		return ee.Geometry.Point(lon,lat).buffer(buffer_size)
  
	def average_collection(self, image_collection, label = "DATA", band_select = None):
		if band_select:
			image = image_collection.select(band_select).reduce(ee.Reducer.median()).rename(label)
		else:
			image = image_collection.map(
				lambda image: image.reduce(ee.Reducer.mean())
				).reduce(ee.Reducer.median()).rename(label)
		return image

	def mask_by_value(self, image, value = -9999):
		mask = image.eq(value)
		image = image.updateMask(mask)
		return image

	def filter_image(self, date_range, label = "DATA", band_select = None, area_coordinates_list = []):
		start_date, end_date = date_range
		image_collection = self.image_collection.filterDate(start_date, end_date)
		if area_coordinates_list:
			roi = ee.Geometry.Polygon(area_coordinates_list)
			image_collection = image_collection.filterBounds(roi)
		if band_select:
			image = image_collection.select(band_select).reduce(ee.Reducer.median()).rename(label)
		else:
			image = image_collection.map(
				lambda image: image.reduce(ee.Reducer.mean())
				).reduce(ee.Reducer.median()).rename(label)
		if not self.MUTE: print("retrieved images...")
		return image
	
	def retrievel_single_image(self, band):
		return self.image.select(band)

	def get_values(self, image, bounds, label = "DATA", max_pixels = 1e8, scale = 20):
		# getInfo() is limited to 5000 records
		# ee.ee_exception.EEException: Array: No numbers in 'values', must provide a type.
		# see more at: https://gis.stackexchange.com/questions/321560/getting-dem-values-as-numpy-array-in-earth-engine
		# for rectangele: bounds = [-97.94, 26.81, -96.52, 26.84] ## sample land / sea bounds
		# for polygon: bounds = [[[105.532,19.059],[105.606,19.058],[105.605,19.108],[105.530,19.110],[105.532,19.059]]]
		if isinstance(bounds, list):
			if np.array(bounds).ndim == 1:
				area = ee.Geometry.Rectangle(bounds)
				if not self.MUTE: print("bounds are a rectangle list...")
			else:
				area = ee.Geometry.Polygon(bounds)
				if not self.MUTE: print("bounds are a polygon list...")
		else:
			area = bounds
			if not self.MUTE: print("bounds are a pre-defined ee region...")
		image = image.clip(area)
		image = image.unmask(ee.Image.constant(-9999))
		latlng = ee.Image.pixelLonLat().addBands(image)
		latlng = latlng.reduceRegion(
			reducer=ee.Reducer.toList(), 
			geometry = area, 
			maxPixels = max_pixels, 
			scale = scale
		)
		self.lats = np.array((ee.Array(latlng.get("latitude")).getInfo()))
		self.lons = np.array((ee.Array(latlng.get("longitude")).getInfo()))
		try:
			self.data = np.array((ee.Array(latlng.get(label)).getInfo()))
		except:
			self.data = np.full_like(self.lats, np.nan, dtype=np.float64)
		# self.data = list(data) ## print as list to check

		if not (self.data.shape == self.lats.shape == self.lons.shape):
			raise Exception(
				f"SizeError: " +
				f"data shape is {self.data.shape}, " + 
				f"lats shape is {self.lats.shape}, " + 
				f"lons shape is {self.lons.shape}."
			)
		if not self.MUTE: print("retrieved data...")


	def to_2d_tif(self):
		# NOTICE: only one band for now
		# get the unique coordinates
		uniqueLats = np.unique(self.lats)
		uniqueLons = np.unique(self.lons)

		# get number of columns and rows from coordinates
		ncols = len(uniqueLons)    
		nrows = len(uniqueLats)
		
		# determine pixelsizes
		ys = uniqueLats[1] - uniqueLats[0] 
		xs = uniqueLons[1] - uniqueLons[0]

		# create an array with dimensions of image
		array = np.zeros([nrows, ncols], np.float32) #-9999
		
		# fill the array with values
		counter = 0
		for y in range(0, len(array), 1):
			for x in range(0, len(array[0]), 1):
				if self.lats[counter] == uniqueLats[y] and self.lons[counter] == uniqueLons[x] and counter < len(self.lats)-1:
					counter+=1
					array[len(uniqueLats)-1-y,x] = self.data[counter] # we start from lower left corner

		# set the 
		#SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
		transform = (np.min(uniqueLons),xs,0,np.max(uniqueLats),0,-ys)
		return {"array": array, "transform": transform}

	def localize_image(self, image, savefolder, image_name = "temp", filename = "temp.zip", new_folder = False, zip_folder = None, crs_epsg = "4326", scale = 1000):    
		url = image.getDownloadURL({'name': image_name, 'crs': 'EPSG:' + crs_epsg, 'scale': scale})
		
		if not isinstance(savefolder, Path):
			savefolder = Path(savefolder)
		filename = savefolder.joinpath(zip_folder).joinpath(filename)
		create_all_parents(filename, flag = "f")
		# print(url)

		# Download the subset
		r = requests.get(url, stream=True)
		with open(filename.as_posix(), 'wb') as fd:
			for chunk in r.iter_content(chunk_size = 1024):
				fd.write(chunk)

		# Extract the GeoTIFF for the zipped download
		# z = zipfile.ZipFile(filename)
		# z.extractall()
		unzip(filename, savefolder, new_folder = new_folder, delete = False)