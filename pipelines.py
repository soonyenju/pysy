from pysy.scigee.geeface import *
from pysy.scigee.utilizes import *
from pysy.scigee.config import *
from pysy.scigeo.geoface import *
from pysy.scigeo.geobox import *
from pysy.toolbox.sysutil import *
from pysy.toolbox.timeutil import *
from time import gmtime, strftime, ctime
from pathlib import Path
from itertools import product
import pickle, ee
import numbers
import pandas as pd
import numpy as np
import pickle

def main():
    ee.Initialize()

    root_dir = Path.cwd()
    cur_path = root_dir.joinpath("project_data")
    fluxnet_dir = cur_path.joinpath("fluxnet2015_save.pkl")
    create_all_parents(fluxnet_dir)
    with open(fluxnet_dir, "rb") as f:
        fluxnet = pickle.load(f)

    config = Yaml(cur_path.joinpath("config.yaml")).load()
    cfg_ds = config["datasets"]
    montre = Montre() # process time and dates

    results = []
    # Each site
    for idx, (site_name, site_info) in enumerate(fluxnet.items()):
        each_site = {}
        each_site["name"] = site_name
        print(idx, site_name)
        # print(site_info)
        lat = site_info["lat"]
        lon = site_info["lon"]

        if not isinstance(lat, numbers.Number):
            lat = float(lat)
        if not isinstance(lon, numbers.Number):
            lon = float(lon)
        lats = [int(lat) -1, int(lat) + 1]
        lons = [int(lon) -1, int(lon) + 1]
        bounds = [lons[0], lats[0], lons[1], lats[1]]
        # pntpairs = list(product(lats, lons, repeat = 1)) # map each lat to each lon
        # lats, lons = list(zip(*pntpairs))
        #   print(lats, lons)
        flux_series = site_info["values"]
        flux_series["TIMESTAMP"] = flux_series["TIMESTAMP"].astype(str)
        # flux_series["TIMESTAMP"] = pd.to_datetime(flux_series["TIMESTAMP"])
        save_folder = root_dir.joinpath("temp_save")

        climate = koppen_func(cur_path, config, bounds, lon, lat)
        each_site["climate"] = climate
        dem = strm_func(site_name, bounds, lon, lat, save_folder)
        each_site["dem"] = dem
        print(f"Climate type {climate}")
        print(f"Dem is {dem}")

        all_day = []
        # Each day
        for nrow, row in flux_series.iterrows():
            each_day = []
            date = row["TIMESTAMP"]
            each_day.append(date)
            print(row)
            today = montre.to_date(date, format = r"%Y%m%d")
            yesterday = montre.manage_time(today, days = -1)
            today = montre.to_str(today)
            yesterday = montre.to_str(yesterday)
            date_range = [yesterday, today]
            modis = cfg_ds["modis"]
            # NOTICE: CHANGE date_range
            # date_range = ["2008-01-01", "2008-1-31"]
            #
            for nds, dataset in enumerate(modis):
                ds_name = dataset["name"]
                band_list = dataset["band"]
                
                for band in band_list:
                    res = gee_pipeline(ds_name, bounds, lon, lat, date_range, band, save_folder)
                    print(band, date_range[1], res)
                    each_day.append(res)
            cfsv2 = cfg_ds["cfsv2"]
            for nds, dataset in enumerate(cfsv2):
                ds_name = dataset["name"]
                band_list = dataset["band"]
                
                for band in band_list:
                    res = gee_pipeline(ds_name, bounds, lon, lat, date_range, band, save_folder)
                    print(band, date_range[1], res)
                    each_day.append(res)
            all_day.append(each_day)
            # exit(0)
        each_site["values"] = all_day
    results.append(each_site)
    with open("alldata.pkl", "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

def gee_pipeline(ds_name, bounds, lon, lat, date_range, band, save_folder):
    earth = Earth(ds_name)
    region = ee.Geometry.Rectangle(bounds)
    image = earth.filter_image(date_range, band_select = band, label = band)
    image = image.clip(region)
    try:
        earth.localize_image(
            image, 
            save_folder,
            image_name = date_range[0]
            )
        save_path = save_folder.joinpath(f"{date_range[0]}.{band}.tif")
        ras = Raster(save_path)
        ras.fopen(src_only = False)

        # NOTICE: if ras.profile["transform"] matches the format required in ras.get_pntpairs
        pntpairs, values = ras.get_pntpairs(affine = ras.profile["transform"], data = ras.data)
        idw = IDW(pntpairs, values)
        res = idw([lon, lat])
        return res[0]

    except Exception as e:
        print(e)
        return -9999

def strm_func(site_name, bounds, lon, lat, save_folder):
    band = 'elevation'
    srtm = Earth('CGIAR/SRTM90_V4', collection = False)
    image = srtm.retrievel_single_image(band)
    region = ee.Geometry.Rectangle(bounds)
    image = image.clip(region)
    try:
        srtm.localize_image(
            image, 
            save_folder,
            image_name = site_name
            )
        save_path = save_folder.joinpath(f"{site_name}.{band}.tif")
        ras = Raster(save_path)
        ras.fopen(src_only = False)

        # NOTICE: if ras.profile["transform"] matches the format required in ras.get_pntpairs
        pntpairs, values = ras.get_pntpairs(affine = ras.profile["transform"], data = ras.data)
        idw = IDW(pntpairs, values)
        res = idw([lon, lat])
        return res[0]

    except Exception as e:
        print(e)
        return -9999

def koppen_func(path, config, bounds, lon, lat):
    koppen_dir = path.joinpath("Beck_KG_V1/Beck_KG_V1_present_0p0083.tif")
    # print(koppen_dir)

    lats = [bounds[1], bounds[3]]
    lons = [bounds[0], bounds[2]]
    coors = list(product(lats, lons, repeat = 1)) # map each lat to each lon
    # lats, lons = list(zip(*pntpairs))

    ras = Raster(koppen_dir)
    ras.fopen(src_only = False)

    vec = Vector()
    poly = vec.create_polygon(coors)

    ras.clip(poly.geometry)
    # print(ras.clip_arr.shape)
    pntpairs, values = ras.get_pntpairs(affine = ras.clip_transform, data = ras.clip_arr)

    idw = IDW(pntpairs, values)
    res = idw([lon, lat])
    return res[0]


"""
value search example:
# earth.get_values(image, bounds, scale = 1000)
# print(bounds, lats, lons)
# ds = earth.to_2d_tif()
# print(earth.data.shape, earth.lats.shape, earth.lons.shape)
# data = ds["array"]
# gt = ds["transform"]
# print(data.shape)
# print(gt)
"""

if __name__ == "__main__":
    main()