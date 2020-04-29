import os
import ee
import folium
import requests
import numpy as np
from pathlib import Path

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

	def fetch_collection(self, date_range = [], roi = None, to_list = False):
		collection = ee.ImageCollection(self.dataset_name)
		# filter by date:
		if date_range:
			start_date, end_date = date_range
			collection = collection.filterDate(start_date, end_date)
		# filter by bounds:
		if roi:
			collection = collection.filterBounds(roi)
		self.__length__ = collection.size().getInfo()
		# covert ee collection to list?
		if to_list: 
			collection = collection.toList(collection.size())
		self.collection = collection
	
	def get_image_by_index(self, idx):
		self.image = ee.Image(self.collection.get(idx))

	def reduce_collection(self, band = None, label = "DATA", band_reducer = ee.Reducer.mean(), spatial_reducer = ee.Reducer.median()):
		if band:
			image = self.collection.select(band).reduce(spatial_reducer).rename(band)
		else:
			image = self.collection.map(
					lambda image: image.reduce(band_reducer)
				).reduce(spatial_reducer).rename(label)
		self.image = image

class Emagebox(object):
	"""
	Emagebox: Earth engine Image Toolbox
	"""
	def __init__(self, image):
		self.image = image

	def get_band_names(self):
		return self.image.bandNames().getInfo()

	def get_stats(self, roi, reducer = ee.Reducer.mean(), scale = 30, max_pixels = 1e8):
		stat = image.reduceRegion(
			reducer = reducer,
			geometry = roi,
			scale = scale,
			maxPixels = max_pixels
		)
		return stat.getInfo()
	
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

	def mask_value(self, value = -9999):
		mask = self.image.eq(value)
		self.image = self.image.updateMask(mask)

	def unmask(self, value = -9999):
		self.image = self.image.unmask(ee.Image.constant(value))

	def get_value(self, band, point, scale = 10, reducer = ee.Reducer.first()):
		value = self.image.select(band).reduceRegion(reducer, point, scale).get(band)
		value = ee.Number(value)
		return value.getInfo()

	def get_values(self, roi, band, unmask = -9999, max_pixels = 1e8, scale = 20):
		# getInfo() is limited to 5000 records
		# ee.ee_exception.EEException: Array: No numbers in 'values', must provide a type.
		# see more at: https://gis.stackexchange.com/questions/321560/getting-dem-values-as-numpy-array-in-earth-engine
		# for rectangele: bounds = [-97.94, 26.81, -96.52, 26.84] ## sample land / sea bounds
		# for polygon: bounds = [[[105.532,19.059],[105.606,19.058],[105.605,19.108],[105.530,19.110],[105.532,19.059]]]
		latlng = ee.Image.pixelLonLat().addBands(self.image.clip(roi).unmask(ee.Image.constant(unmask)))
		latlng = latlng.reduceRegion(
			reducer = ee.Reducer.toList(), 
			geometry = roi, 
			maxPixels = max_pixels, 
			scale = scale
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

	def localize(self, save_name, save_folder = ".", crs_epsg = "4326", scale = 1000):
		url = self.image.getDownloadURL({
			"name": save_name,
			"crs": "EPSG:" + crs_epsg,
			"scale": scale
		})
		if not os.path.exists(save_folder): os.makedirs(save_folder)
		save_dir = f"{save_folder}/{save_name}.zip"

		# Download the subset
		r = requests.get(url, stream = True)
		with open(save_dir, 'wb') as fd:
			for chunk in r.iter_content(chunk_size = 1024):
				fd.write(chunk)

class Utils(object):
	def __init__(self):
		pass

	def coors2roi(self, bounds):
		# for rectangele: bounds = [-97.94, 26.81, -96.52, 26.84] ## sample land / sea bounds
		# for polygon: bounds = [[[105.532,19.059],[105.606,19.058],[105.605,19.108],[105.530,19.110],[105.532,19.059]]]
		if isinstance(bounds, list):
			if np.array(bounds).ndim == 1:
				roi = ee.Geometry.Rectangle(bounds)
			else:
				roi = ee.Geometry.Polygon(bounds)
		return roi

	def coor2point(self, coors = None, lon = None, lat = None):
		if coors:
			lon, lat = coors
		assert lon, "No longitude input..."
		assert lat, "No latitude input..."
		point = ee.Geometry.Point(lon, lat)
		return point

	def get_circle_buffer(self, point, buffer_size = 100):
		return point.buffer(buffer_size)
	
	def get_rect_buffer(self, point, buffer_size = 0.5):
		# buffer_size unit eq proj unit, e.g., degree for WGS84
		lon, lat = point.getInfo()["coordinates"]
		# example: [-97.94, 26.81, -96.52, 26.84]
		bounds = [lon - buffer_size, lat - buffer_size, lon + buffer_size, lat + buffer_size]
		return ee.Geometry.Rectangle(bounds)
	
class Canvas(object):
	def __init__(self):
		pass

	def draw(self, images, vis_params, layer_names = [], location=[20, 0], zoom_start = 3, height = 500):
		# draw using folium
		# Add EE drawing method to folium.
		folium.Map.add_ee_layer = self.add_ee_layer

		# Create a folium map object.
		the_map = folium.Map(location = location, zoom_start = zoom_start, height = height)

		for image, vis_param, layer_name in zip(images, vis_params, layer_names):
			# Add the elevation model to the map object.
			the_map.add_ee_layer(image, vis_param, layer_name)

		# Add a layer control panel to the map.
		the_map.add_child(folium.LayerControl())

		# # Display the map.
		# display(the_map)
		return the_map

	# Define a method for displaying Earth Engine image tiles to folium map.
	def add_ee_layer(self, image, vis_params, name):
		if not isinstance(image, ee.image.Image):
			image = ee.Image(image)
		map_id_dict = image.getMapId(vis_params)
		folium.raster_layers.TileLayer(
			tiles = map_id_dict['tile_fetcher'].url_format,
			attr = "Map Data © Google Earth Engine",
			name = name,
			overlay = True,
			control = True
		).add_to(self)

	# easy visualizaiton parameter generator
	def viz(self, vmin, vmax, palette):
		return {
			"min": vmin,
			"max": vmax,
			"palette": palette
		}
