import ee, folium
from pathlib import Path

# Define a method for displaying Earth Engine image tiles to folium map.
def add_ee_layer(self, ee_image_object, vis_params, name):
  map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
  folium.raster_layers.TileLayer(
    tiles = map_id_dict['tile_fetcher'].url_format,
    attr = "Map Data © Google Earth Engine",
    name = name,
    overlay = True,
    control = True
  ).add_to(self)

# draw using folium
def folium_draw(images, vis_params, layer_names = [], location=[20, 0], zoom_start=3, height=500):
  # Add EE drawing method to folium.
  folium.Map.add_ee_layer = add_ee_layer

  # Create a folium map object.
  my_map = folium.Map(location=location, zoom_start=zoom_start, height=height)

  for image, vis_param, layer_name in zip(images, vis_params, layer_names):
    # Add the elevation model to the map object.
    my_map.add_ee_layer(image, vis_param, layer_name)

  # Add a layer control panel to the map.
  my_map.add_child(folium.LayerControl())

  # Display the map.
  display(my_map)

# easy visualizaiton parameter generator
def viz(vmin, vmax):
  return {
      "min": vmin,
      "max": vmax,
      "palette": ["blue", "purple", "cyan", "green", "yellow", "red"]
  }
