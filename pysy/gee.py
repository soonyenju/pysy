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
		stat = self.image.reduceRegion(
			reducer = reducer,
			geometry = roi,
			scale = scale,
			maxPixels = max_pixels
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

	def mask_value(self, value = -9999):
		mask = self.image.eq(value)
		self.image = self.image.updateMask(mask)

	def unmask(self, value = -9999):
		self.image = self.image.unmask(ee.Image.constant(value))

	def get_value(self, band, point, scale = 10, reducer = ee.Reducer.first()):
		value = self.image.select(band).reduceRegion(reducer, point, scale).get(band)
		value = ee.Number(value)
		return value.getInfo()

	def get_values(self, band, roi, unmask = -9999, max_pixels = 1e8, scale = 20):
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

	@classmethod
	def coors2roi(self, bounds):
		# for rectangele: bounds = [-97.94, 26.81, -96.52, 26.84] ## sample land / sea bounds
		# for polygon: bounds = [[[105.532,19.059],[105.606,19.058],[105.605,19.108],[105.530,19.110],[105.532,19.059]]]
		if isinstance(bounds, list):
			if np.array(bounds).ndim == 1:
				roi = ee.Geometry.Rectangle(bounds)
			else:
				roi = ee.Geometry.Polygon(bounds)
		return roi

	@classmethod
	def coor2point(self, coors = None, lon = None, lat = None):
		if coors:
			lon, lat = coors
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
	

class Canvas(object):
	def __init__(self):
		self.cmap = {
			"veg": ['FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A901',
					'66A000', '529400', '3E8601', '207401', '056201', '004C00', '023B01',
					'012E01', '011D01', '011301'
					],
			"grey": ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']
		}
		# Add custom basemaps to folium
		self.basemaps = {
			'Google Maps': folium.TileLayer(
				tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
				attr = 'Google',
				name = 'Google Maps',
				overlay = True,
				control = True
			),
			'Google Satellite': folium.TileLayer(
				tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
				attr = 'Google',
				name = 'Google Satellite',
				overlay = True,
				control = True
			),
			'Google Terrain': folium.TileLayer(
				tiles = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
				attr = 'Google',
				name = 'Google Terrain',
				overlay = True,
				control = True
			),
			'Google Satellite Hybrid': folium.TileLayer(
				tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
				attr = 'Google',
				name = 'Google Satellite',
				overlay = True,
				control = True
			),
			'Esri Satellite': folium.TileLayer(
				tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
				attr = 'Esri',
				name = 'Esri Satellite',
				overlay = True,
				control = True
			)
		} 

	def draw(self, images, vis_params, layer_names = [], location=[20, 0], zoom_start = 3, height = 500):
		# draw using folium
		# Add EE drawing method to folium.
		folium.Map.add_ee_layer = self.add_ee_layer

		# Create a folium map object.
		self.m = folium.Map(location = location, zoom_start = zoom_start, height = height)

		# Add custom basemaps
		self.basemaps['Google Maps'].add_to(self.m)
		self.basemaps['Google Satellite Hybrid'].add_to(self.m)
		self.basemaps['Google Terrain'].add_to(self.m)

		# show coors interactively
		self.mouse_position()

		# # click show coors:
		# self._click_popup_coors()

		# click to place a popup marker
		self.m.add_child(folium.ClickForMarker(popup="Hello :)"))

		for image, vis_param, layer_name in zip(images, vis_params, layer_names):
			# Add the elevation model to the map object.
			self.m.add_ee_layer(image, vis_param, layer_name)

		# Add a layer control panel to the map.
		self.m.add_child(folium.LayerControl())

		# Add fullscreen button
		folium.plugins.Fullscreen().add_to(self.m)

		# # Display the map.
		# display(m)

	def save(self, filename):
		# example: m.save("filename.png")
		self.m.save(filename)

	# Define a method for displaying Earth Engine image tiles on a folium map.
	def add_ee_layer(self, ee_object, vis_params, name):
		
		try:    
			# display ee.Image()
			if isinstance(ee_object, ee.image.Image):    
				map_id_dict = ee.Image(ee_object).getMapId(vis_params)
				folium.raster_layers.TileLayer(
				tiles = map_id_dict['tile_fetcher'].url_format,
				attr = 'Google Earth Engine',
				name = name,
				overlay = True,
				control = True
				).add_to(self.m)
			# display ee.ImageCollection()
			elif isinstance(ee_object, ee.imagecollection.ImageCollection):    
				ee_object_new = ee_object.mosaic()
				map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
				folium.raster_layers.TileLayer(
				tiles = map_id_dict['tile_fetcher'].url_format,
				attr = 'Google Earth Engine',
				name = name,
				overlay = True,
				control = True
				).add_to(self.m)
			# display ee.Geometry()
			elif isinstance(ee_object, ee.geometry.Geometry):    
				folium.GeoJson(
				data = ee_object.getInfo(),
				name = name,
				overlay = True,
				control = True
			).add_to(self.m)
			# display ee.FeatureCollection()
			elif isinstance(ee_object, ee.featurecollection.FeatureCollection):  
				ee_object_new = ee.Image().paint(ee_object, 0, 2)
				map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
				folium.raster_layers.TileLayer(
				tiles = map_id_dict['tile_fetcher'].url_format,
				attr = 'Google Earth Engine',
				name = name,
				overlay = True,
				control = True
			).add_to(self.m)
		
		except:
			print("Could not display {}".format(name))

	def mouse_position(self):
		formatter = "function(num) {return L.Util.formatNum(num, 3) + ' deg';};"

		folium.plugins.MousePosition(
			position='topright',
			separator=' | ',
			empty_string='NaN',
			lng_first=True,
			num_digits=20,
			prefix='Coordinates:',
			lat_formatter=formatter,
			lng_formatter=formatter,
		).add_to(self.m)

	def _click_popup_coors(self):
		folium.LatLngPopup().add_to(self.m)

	# easy visualizaiton parameter generator
	def viz(self, vmin, vmax, palette):
		return {
			"min": vmin,
			"max": vmax,
			"palette": palette
		}