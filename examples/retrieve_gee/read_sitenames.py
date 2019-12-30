# coding: utf-8
import pickle
from pathlib import Path


def main():
    root = Path.cwd()
    path = root.joinpath("project_data/fluxnet2015_save.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
        site_names = list(data.keys())
        print(site_names)
        print(len(site_names))

if __name__ == "__main__":
    main()
