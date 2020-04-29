import folium, time
import numpy as np
from pathlib import Path
from pysy.scigeo.geoface import *
from pysy.scigeo.geobox import *
from pysy.scigee.geeface import *
from pysy.scigee.utilizes import  *
from pysy.scigee.drive import mount_drive
from pysy.scigee.vis_params import vis_params_dict

import ee
ee.Initialize()

date_range = ["2014-5-1", "2014-5-31"]
coor_list = [[110, 20], [110, 30], [112, 30], [112, 20]]
region = ee.Geometry.Polygon(coor_list)

cur_dir = mount_drive()
print(cur_dir)
koppen_dir = cur_dir.joinpath("workspace/project_data/Beck_KG_V1/Beck_KG_V1_present_0p0083.tif")
print(koppen_dir)

lat_point_list = [50.854457, 52.518172, 50.072651, 48.853033, 50.854457]
lon_point_list = [4.377184, 13.407759, 14.435935, 2.349553, 4.377184]
coors = zip(lon_point_list, lat_point_list)

ras = Raster(koppen_dir)
ras.fopen(src_only = False)

vec = Vector()
poly = vec.create_polygon(coors)

ras.clip(poly.geometry)
# print(ras.clip_arr.shape)
pntpairs, values = ras.get_pntpairs(affine = ras.clip_transform, data = ras.clip_arr)

lat_tar_list = [51, 50, 49]
lon_tar_list = [4, 12, 7]
pnts = np.array(list(zip(lon_tar_list, lat_tar_list)))
# print(pnts.ndim)
# print(pnts.shape)
idw = IDW(pntpairs, values)
res = idw(pnts)
print(res)


earth = Earth("MODIS/006/MCD15A3H")
fpar = earth.filter_image(date_range, band_select = "Fpar")
fpar = fpar.clip(region)
# folium_draw([fpar], vis_params_dict["fpar"], layer_names = ["fpar"])

earth = Earth("MODIS/006/MOD13A1")
ndvi = earth.filter_image(date_range, band_select = "NDVI")
# folium_draw(ndvi, vis_params_dict["ndvi"], layer_name = "ndvi")

earth = Earth("MODIS/006/MOD17A2H")
gpp = earth.filter_image(date_range, band_select = "Gpp")

earth = Earth("MODIS/006/MCD12Q1")
lc_type1 = earth.filter_image(["2014-01-01", "2014-12-31"], band_select = "LC_Type1")

srtm = ee.Image('CGIAR/SRTM90_V4')
elevation = srtm.select('elevation')
slope = ee.Terrain.slope(elevation)

earth = Earth("MODIS/006/MCD43A3")
albedo = earth.filter_image(date_range, band_select = "Albedo_BSA_Band1")

earth = Earth("MODIS/006/MCD19A2_GRANULES")
aod = earth.filter_image(date_range, band_select = "Optical_Depth_047")

earth = Earth("NOAA/CFSV2/FOR6H")

temperature = earth.filter_image(date_range, band_select = "Temperature_height_above_ground")
shortwave = earth.filter_image(date_range, band_select = "Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average")
soil_moisture = earth.filter_image(date_range, band_select = "Volumetric_Soil_Moisture_Content_depth_below_surface_layer_5_cm")
max_humidity = earth.filter_image(date_range, band_select = "Maximum_specific_humidity_at_2m_height_above_ground_6_Hour_Interval")
min_humidity = earth.filter_image(date_range, band_select = "Minimum_specific_humidity_at_2m_height_above_ground_6_Hour_Interval")
uwind = earth.filter_image(date_range, band_select = "u-component_of_wind_height_above_ground")
vwind = earth.filter_image(date_range, band_select = "v-component_of_wind_height_above_ground")
precipitation = earth.filter_image(date_range, band_select = "Precipitation_rate_surface_6_Hour_Average")

folium_draw(
      [
       fpar, 
       ndvi, 
       gpp, 
       lc_type1, 
       slope,
       albedo,
       aod,
       temperature,
       shortwave,
       soil_moisture,
       max_humidity,
       min_humidity,
       uwind,
       vwind,
       precipitation,
      ], 
      [
       vis_params_dict["fpar"], 
       vis_params_dict["ndvi"], 
       vis_params_dict["gpp"], 
       vis_params_dict["lc_type1"], 
       vis_params_dict["slope"],
       vis_params_dict["blackSkyAlbedoVis"],
       vis_params_dict["aod"],
       vis_params_dict["temperature"],
       viz(0, 800),
       viz(0.02, 1),
       viz(0.0, 0.1),
       viz(0.0, 0.02),
       viz(-57.2,	57.99),
       viz(-53.09,	57.11),
       viz(0, 0.03),
      ], 
      layer_names = [
                     "fpar", 
                     "ndvi", 
                     "gpp", 
                     "lc_type1", 
                     "slope",
                     "albedo",
                     "aod",
                     "temperature",
                     "shortwave",
                     "soil_moisture",
                     "max_humidity",
                     "min_humidity",
                     "uwind",
                     "vwind",
                     "precipitation",
                     ]
    )



#***********************************************************************************************************************************#
print(f"Cell runs at {time.ctime()}")
# or:
# print(f"All set successfully at {strftime("%Y-%m-%d %H:%M:%S", gmtime())}")

# https://mygeoblog.com/2017/10/06/from-gee-to-numpy-to-geotiff/
# https://gis.stackexchange.com/questions/303381/google-earth-engine-python-api-clipped-export-bad-behavior