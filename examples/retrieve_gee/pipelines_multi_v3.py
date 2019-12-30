# New function: resume from break-point.
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
import pickle, ee, sys
import numbers
import pandas as pd
import numpy as np
import pickle
import multiprocessing

def main():
    # preparations: {
    # initialize Google Earth Engine:
    ee.Initialize()

    # load configuration:
    config = Yaml("config_v3.yaml").load()
    # data input/output path:
    root = Path(config["others"]["root"])
    # load fluxnet:
    fluxnet_dir = root.joinpath(config["others"]["subdirs"]["fluxnet"])
    with open(fluxnet_dir, "rb") as f:
        fluxnet = pickle.load(f)

    # to be processed datasets, including gee's and local's:
    # datasets is a dict, it has two keys and their values are dicts as well.
    datasets = config["datasets"]
    # dir of folders to save:
    save = root.joinpath(config["others"]["subdirs"]["save"])
    create_all_parents(save)
    # load site names and data for iteration,
    # check if user specified fluxnet sites:
    try:
        usr_site_names = config["others"]["my_sites"]
        used_sites = ((name, fluxnet[name]) for name in usr_site_names)
    except Exception as e:
        print(e)
        used_sites = fluxnet.items()
    # }
    
    # start iteration: {
    for idx, (site_name, site_info) in enumerate(used_sites):
        # site_name: name of each fluxnet site;
        # site_info: a dict of each fluxnet site:
        #   - lat
        #   - lon
        #   - info: a dataframe of each fluxnet site including date and GPP value
        
        # result pkl path
        pkl_path = save.joinpath(f"{site_name}.pkl")
        # resume mechanism if timeseries of one site is too long:
        temp_pkl_path = save.joinpath(f"temp_{site_name}.pkl")
        save_folder = save.joinpath(f"{site_name}")

        # proc status: {
        # skip already processed site:
        if pkl_path.exists():
            print(f"skipping, {site_name}.pkl already exists.")
            continue
        # resume mechanism:
        elif temp_pkl_path.exists():
            print(f"resuming {site_name}...")
            # cur_site keys:
            #   - name
            #   - climate
            #   - dem
            #   - values
            with open(temp_pkl_path, "rb") as f:
                cur_site = pickle.load(f)
        else:
            print(f"starting {site_name}...")
            cur_site = {}
            cur_site["name"] = site_name
        # }

        # region range, 1 * 1 degree rectangle 
        # whose center is curr_site coordinates: {
        lat = site_info["lat"]
        lon = site_info["lon"]

        if not isinstance(lat, numbers.Number):
            lat = float(lat)
        if not isinstance(lon, numbers.Number):
            lon = float(lon)
        lats = [int(lat) - 1, int(lat) + 1]
        lons = [int(lon) - 1, int(lon) + 1]
        bounds = [lons[0], lats[0], lons[1], lats[1]] 
        # }

        # make timestamps to iterate, drop already exist ones: {
        # in case cur_site has now key named values:
        if not "values" in cur_site.keys():
            cur_site["values"] = []
            # keep writing to temp file
            with open(temp_pkl_path, "wb") as f:
                pickle.dump(cur_site, f, protocol = pickle.HIGHEST_PROTOCOL)
        # drop used rows:
        flux_series = site_info["values"]
        flux_series["TIMESTAMP"] = flux_series["TIMESTAMP"].astype(str)
        series_len = len(flux_series)
        for record in cur_site["values"]:
            if record[0] in flux_series["TIMESTAMP"].values:
                flux_series = flux_series[flux_series["TIMESTAMP"] != record[0]]
        remainder_len = len(flux_series)
        print(f"{remainder_len} of {series_len} to go...")
        # }

        # prepare multicore: {
        rows = flux_series.values.tolist()
        # other parameters which're used every batch.
        params = {
            "datasets": datasets,
            "bounds": bounds,
            "lon": lon,
            "lat": lat,
            "save_folder": save_folder,
            "temp_pkl_path": temp_pkl_path
        }
        # temporally save in avoid to pass them to function:
        with open(f"temp_params.pkl", "wb") as f:
            pickle.dump(params, f, protocol = pickle.HIGHEST_PROTOCOL)

        # initialize multiprocessor
        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes = cores)
        # }

        # start fetch each day: {
        cnt = 0
        for _ in pool.imap_unordered(get_eachday, rows):
            print(f"{cnt} of {len(rows)} done...")
            cnt += 1
        # }

        # fetch climate type and dem, they are not time dependent: {
        koppen_dir = root.joinpath("Beck_KG_V1/Beck_KG_V1_present_0p0083.tif")
        climate = koppen_func(koppen_dir, bounds, lon, lat)
        cur_site["climate"] = climate

        dem = strm_func(site_name, bounds, lon, lat, save_folder)
        cur_site["dem"] = dem

        with open(temp_pkl_path, "wb") as f:
            pickle.dump(cur_site, f, protocol = pickle.HIGHEST_PROTOCOL)
        # }

        # delete temporal params file
        Path("temp_params.pkl").unlink()
        # run over, rename temp pkl to pkl.
        temp_pkl_path.rename(pkl_path)

    # }

def get_eachday(row):
    # parse params: {
    with open("temp_params.pkl", "rb") as f:
        params = pickle.load(f)

    modis = params["datasets"]["gee"]["modis"]
    cfsv2 = params["datasets"]["gee"]["cfsv2"]
    bounds = params["bounds"]
    lon = params["lon"]
    lat = params["lat"]
    save_folder = params["save_folder"]
    temp_pkl = params["temp_pkl_path"]
    # }

    montre = Montre() # process time and dates
    # one record:
    record = []
    record.extend(row)
    date, _ = row

    # make date range, [yesterday, today]:{
    today = montre.to_date(date, format = r"%Y%m%d")
    yesterday = montre.manage_time(today, days = -1)
    today = montre.to_str(today)
    yesterday = montre.to_str(yesterday)
    date_range = [yesterday, today]
    # }
    
    # retrieve data:{
    # modis:
    for nds, dataset in enumerate(modis):
        ds_name = dataset["name"]
        band_list = dataset["band"]
        
        for band in band_list:
            res = gee_pipeline(ds_name, bounds, lon, lat, date_range, band, save_folder)
            # print(f"{band}, {date_range[1]}, {res}")
            record.append(res)
    # cfsv2:
    for nds, dataset in enumerate(cfsv2):
        ds_name = dataset["name"]
        band_list = dataset["band"]
        
        for band in band_list:
            res = gee_pipeline(ds_name, bounds, lon, lat, date_range, band, save_folder)
            # print(f"{band}, {date_range[1]}, {res}")
            record.append(res)
    # }
    # update temp_pkl:{
    with open(temp_pkl, "rb") as f:
        each_site = pickle.load(f)
    each_site["values"].append(record)
    with open(temp_pkl.as_posix(), "wb") as f:
        pickle.dump(each_site, f, protocol = pickle.HIGHEST_PROTOCOL)
    # }

def gee_pipeline(ds_name, bounds, lon, lat, date_range, band, save_folder):
    try:
        earth = Earth(ds_name)
        region = ee.Geometry.Rectangle(bounds)
        image = earth.filter_image(date_range, band_select = band, label = band)
        image = image.clip(region)

        earth.localize_image(
            image, 
            save_folder,
            image_name = date_range[1],
            filename = f"temp_{str(np.random.randint(0, 1e5)).zfill(5)}.zip",
            zip_folder = f"temp_delete"
            )
        save_path = save_folder.joinpath(f"{date_range[1]}.{band}.tif")
        ras = Raster(save_path)
        ras.fopen(src_only = False)

        # NOTICE: if ras.profile["transform"] matches the format required in ras.get_pntpairs
        pntpairs, values = ras.get_pntpairs(affine = ras.profile["transform"], data = ras.data)
        idw = IDW(pntpairs, values)
        res = idw([lon, lat])
        return res[0]

    except Exception as e:
        # print(e)
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
            image_name = site_name,
            filename = f"temp_{str(np.random.randint(0, 1e5)).zfill(5)}.zip",
            zip_folder = f"temp_delete"
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
        # print(e)
        return -9999

def koppen_func(koppen_dir, bounds, lon, lat):
    lats = [bounds[1], bounds[3]]
    lons = [bounds[0], bounds[2]]
    coors = list(product(lats, lons, repeat = 1)) # map each lat to each lon
    # lats, lons = list(zip(*pntpairs))

    ras = Raster(koppen_dir)
    ras.fopen(src_only = False)

    vec = Vector()
    poly = vec.create_polygon(coors)

    try:
        ras.clip(poly.geometry)
        pntpairs, values = ras.get_pntpairs(affine = ras.clip_transform, data = ras.clip_arr)
        idw = IDW(pntpairs, values)
        res = idw([lon, lat])
        return res[0]
    except Exception as e:
        # print(f"{e}")
        py, px = ras.src.index(lon, lat)
        array = ras.read()["array"]
        res = array[:, py, px]
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