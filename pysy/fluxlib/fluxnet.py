import pickle
import multiprocessing
import numpy as np
import pandas as pd
from pathlib import Path
from pysy.toolbox.sysutil import Yaml, unzip, create_all_parents, pbar

class Fluxnet2015(object):
    def __init__(self, path_cfg):
        super()
        fyaml = Yaml(path_cfg)
        self.config = fyaml.load()
        self.root = self.config["dirs"]["root"] # root dir for all fluxnet2015 data
        if not isinstance(self.root, Path):
            self.root = Path(self.root)

    def untar(self):
        # initialize multicores
        cores = multiprocessing.cpu_count()
        pool  = multiprocessing.Pool(processes=cores)

        # unzip use multiprocessors 
        src_dir = self.root.joinpath(self.config["dirs"]["src"])
        dst_dir = self.root.joinpath(self.config["dirs"]["dst"])
        create_all_parents(dst_dir)
        paths = list(Path(src_dir).glob(r"*.zip"))
        lengs = len(paths)
        out_dirs = [dst_dir for leng in range(lengs)]
        print(f"core number is: {cores}")
        pool.starmap(unzip, zip(paths, out_dirs))
    
    def get_meta(self):
        # generate necessary info, extract site ID, longitude, latitude as a dataframe
        meta_dir = self.root.joinpath(self.config["dirs"]["meta"])
        meta = pd.read_excel(meta_dir, usecols = ["SITE_ID", "VARIABLE", "DATAVALUE"])
        lats = meta[meta["VARIABLE"] == "LOCATION_LAT"]
        lats = lats.rename(columns={"DATAVALUE": "LAT"})
        lons = meta[meta["VARIABLE"] == "LOCATION_LONG"]
        lons = lons.rename(columns={"DATAVALUE": "LON"})
        meta = pd.merge(lats, lons, on='SITE_ID')
        self.meta = meta.loc[:, ["SITE_ID", "LAT", "LON"]]
        # print(meta)
        # site_ids = meta["SITE_ID"].values
        # print(site_ids)
        print(f"meta data was accessed...")

    def search_file(self):
        freq = self.config["params"]["freq"]
        dst_dir = self.root.joinpath(self.config["dirs"]["dst"])
        paths = Path(dst_dir).rglob("*.csv")
        # print(list(paths))

        # paths = [p for p in paths if "HH" in p.stem.split("_")]
        paths_ = []
        for p in paths:
            temp = p.stem.split("_")
            if freq == "HH":
                if "HH" in temp or "HR" in temp: # HR rather than HH in US-UMB
                    paths_.append(p)
            if freq == "DD":
                if "DD" in temp:
                    paths_.append(p)
            if freq == "MM":
                if "MM" in temp:
                    paths_.append(p)
        paths = paths_
        site_names_in_paths = np.array([Path(p).stem.split("_")[1] for p in paths])

        data_dict = {}

        # iterate each row (each site) of meta which is a dataframe
        for index, row in self.meta.iterrows():
            print(index, row["SITE_ID"])
            si = row["SITE_ID"]
            data_dict[si] = {}
            data_dict[si]["LON"] = row["LON"]
            data_dict[si]["LAT"] = row["LAT"]
            idxs = np.where(site_names_in_paths == si)[0]
            data_dict[si]["PATHS"] = [paths[i].as_posix() for i in idxs]
            # print(data_dict)
            # exit(0)

        site_info_dir = self.root.joinpath(self.config["dirs"]["site_info"])
        fyaml = Yaml(site_info_dir)
        fyaml.dump(data_dict)
        print(f"fluxnet files were iterated...")
        
    def retrieve(self):
        site_names = self.config["params"]["site_name"]
        savefile = self.root.joinpath(self.config["dirs"]["fpkl"])

        site_info_dir = self.root.joinpath(self.config["dirs"]["site_info"])
        fyaml = Yaml(site_info_dir)
        info = fyaml.load()

        if not site_names:
            site_names = list(info.keys())

        vars_req = self.config["params"]["vars"]
        # read each fluxnet file
        records = {}
        for count, (site_name, cell) in enumerate(info.items()):
            if not site_name in site_names:
                continue
            print(site_name)
            lat = cell["LAT"]
            lon = cell["LON"]
            paths = cell["PATHS"]
            # aux = pd.read_csv(paths[0]) # ERAI auxiliary data
            data = pd.read_csv(paths[1]) # fullset data
            # check the variable names
            # print(aux.columns.values)
            # print(data.columns.values)
            try:
                # print the time range of each fluxnet data
                print(Path(paths[1]).stem.split("_")[-2])
            except Exception as e:
                print(e)
                print(paths)
            df = data[vars_req]
            pbar(count, len(info.items()))
            item = {
                "lat":  lat,
                "lon": lon,
                "values": df
            }
            records[site_name] = item
        with open(savefile, "wb") as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
