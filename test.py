from scigeo.geoface import Raster, Vector
from scigeo.geobox import IDW
from pathlib import Path
import numpy as np

def main():
    lat_point_list = [50.854457, 52.518172, 50.072651, 48.853033, 50.854457]
    lon_point_list = [4.377184, 13.407759, 14.435935, 2.349553, 4.377184]
    coors = zip(lon_point_list, lat_point_list)
    p = "Beck_KG_V1_present_0p0083.tif"
    ras = Raster(p)
    # print(p)

    # data = raster.read()
    # print(data)
    ras.fopen(src_only = False)

    vec = Vector()
    poly = vec.create_polygon(coors)


    ras.clip(poly.geometry)
    # print(ras.clip_arr.shape)
    pntpairs, values = ras.get_pntpairs(affine = ras.clip_transform, data = ras.clip_arr)

    lat_tar_list = [51, 50, 49]
    lon_tar_list = [4, 12, 7]
    pnts = np.array(list(zip(lon_tar_list, lat_tar_list)))
    pnts = [[51, 12]]
    # print(pnts.ndim)
    # print(pnts.shape)
    idw = IDW(pntpairs, values)
    res = idw(pnts)
    print(res)

    vec = Vector(path = "vector.shp")

if __name__ == "__main__":
    main()