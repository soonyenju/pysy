#coding: utf-8
from __future__ import print_function
import gdal, osr, ogr
import numpy as np
import pandas as pd
import scipy as sp
import h5py
from scipy import spatial
from scipy.spatial import distance
from netCDF4 import Dataset
try:
    from pyproj import Proj, Geod, transform
except ImportError:
    Proj, Geod, transform = None, None, None
    print("Error: No module named pyproj")
try:
    from pyhdf.HDF import *
    from pyhdf.V   import *
    from pyhdf.VS  import *
    from pyhdf.SD  import *
except ImportError:
    pyhdf = None
    print("Error: No module named pyhdf")

# from scipy import interpolate.grid
# from scipy.ndimage.interpolation import map_coordinates

def raster_info(fullpath):
    f = gdal.Open(fullpath)
    files = f.GetFileList()
    meta = f.GetMetadata_List()
    sds = f.GetSubDatasets()
    cols = f.RasterXSize
    rows = f.RasterYSize
    bands = f.RasterCount
    gt = f.GetGeoTransform()
    proj = f.GetProjection()
    return {"FileLists": files, "metadata": meta, "subdatasets": sds,
        "shape": [rows, cols, bands], "gt": gt, "proj": [proj]}

def read_tif(fullpath, dim = 2, b = 0):
    """
    In gdal, band starts from 1
    parameters: fullpath, dim = 2, b = 0
    return : data, gt, proj
    """
    gdal.SetConfigOption('GDAL_FILENAME_IS_UTF8', 'NO')
    gdal.SetConfigOption('SHAPE_ENCODING', 'gb2312')
    f = gdal.Open(fullpath)
    cols = f.RasterXSize
    rows = f.RasterYSize
    proj = f.GetProjection()
    gt = f.GetGeoTransform()

    if dim == 2:
        data = f.ReadAsArray(0, 0, cols, rows)

    if dim == 3:
        if b == 0:
            array = f.GetVirtualMemArray()
        else:
            band = f.GetRasterBand(b)
            array = band.ReadAsArray(0, 0, cols, rows)

    return data, gt, proj

def rm_val(np_data, m = "nan", fill_val = 0):
    """
    three modes: m == "nan", m == "inf", m == special value
    parameter: np_data, m = "nan", fill_val = 0
    return: np_data
    """
    if m == "nan":
        np_data[np.where(np.isnan(np_data) == True)] = fill_val
    elif m == "inf":
        np_data[np.where(np.isfinite(np_data) == False)] = fill_val
    else:
        val = np.float64(m)
        np_data[np.where(np_data >= val)] = fill_val
    return np_data

def draw_tif(array, proj, fullpath, gt = "global025", datatype = gdal.GDT_Float32):
    """
    parameter: array, gt = "global025", proj, fullpath, datatype = gdal.GDT_Float32
    return: NONE
    datatype: gdal.Byte, gdal.GDT_UInt16, gdal.GDT_Float32
    """
    if gt != "global025":
        cols = array.shape[1]
        rows = array.shape[0]
        originX = gt[0]
        originY = gt[3]
        pixelWidth = gt[1]
        pixelHeight = gt[5]
    else:
        cols = 1440
        rows = 720
        originX = -179.88
        originY = -89.88
        pixelWidth = 0.25
        pixelHeight = 0.25

    if len(array.shape) == 3:
        bands = array.shape[2]
    else:
        bands = 1

    dr = gdal.GetDriverByName('GTiff')
    name = fullpath
    ds = dr.Create(name, cols, rows, 1, datatype)
    ds.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    # ds.SetGeoTransform(gt)
    srs = osr.SpatialReference()
    try:
        srs.ImportFromEPSG(4326)
    except Exception as e:
        print(Exception, ":", e, " Can not found gcs.csv")
        p = Proj(init="epsg:4326")
        srs.ImportFromProj4(p.srs)
    ds.SetProjection(srs.ExportToWkt())

    if bands == 1:
        outband = ds.GetRasterBand(1)
        outband.WriteArray(array)
        outband.FlushCache()
    else:
        for i in range(bands):
            outband = ds.GetRasterBand(i + 1)
            outband.WriteArray(array[i, :, :])
    del(ds)

def df2pntshp(df, filename, datatype = ogr.OFTString):
    """
    df, filename, datatype = ogr.OFTString
    df must have columns "lon" and "lat"
    """
    dr = ogr.GetDriverByName("ESRI Shapefile")
    ds = dr.CreateDataSource(filename)
    srs = osr.SpatialReference()
    try:
        srs.ImportFromEPSG(4326)
    except Exception as e:
        print(Exception, ":", e, " Can not found gcs.csv")
        p = Proj(init="epsg:4326")
        srs.ImportFromProj4(p.srs)
    geotype = ogr.wkbPoint

    lyr = ds.CreateLayer(filename[:-4], srs = srs, geom_type = geotype)

    lons = df["lon"].values
    lats = df["lat"].values

    fieldnames = df.columns.values
    for fieldName in fieldnames:
        idField = ogr.FieldDefn(str(fieldName), datatype)
        lyr.CreateField(idField)

    pnts = ogr.Geometry(ogr.wkbPoint)
    for i in range(lons.shape[0]):
        pnts.AddPoint(lons[i], lats[i])

        featureDefn = lyr.GetLayerDefn()
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(pnts)
        for fieldName in fieldnames:
            fieldValue = str(df[fieldName].values[i])
            outFeature.SetField(str(fieldName), fieldValue)
        lyr.CreateFeature(outFeature)

def read_nc(fullpath):
    f = Dataset(fullpath)
    data = {}

    for var in f.variables.keys():
        data[var] = f.variables[var][:]
    data["path"] = f.filepath()
    return data

def coor2pos(lon, lat, gt):
    """
    calculate row and col from lat and lon
    parameter: lon, lat, gt
    return row, col
    """
    col = np.round((lon - gt[0]) / gt[1])
    row = np.round((lat - gt[3]) / gt[5])
    return row, col

def pos2coor(row, col, gt):
    """
    calculate lon and lat from col and row
    parameter: row, col, gt
    return lon, lat
    """
    lon = gt[0] + gt[1]*col
    lat = gt[3] + gt[5]*row
    return lon, lat

def rec_by_coor(data_dict, gt, col3df, colname, ws = 3):
    """
    serch by lon and lat
    data_dict, gt, col3df, colname, ws = 3
    data_dict is a dict, keys are date (mon/day), values are 2-dimension arraies
    col3df has only three columns in order: colname, lon, lat
    """
    dates = np.sort(data_dict.keys())
    record = np.empty([col3df[colname].shape[0], dates.shape[0]])
    for i in col3df.index:
        _, lon, lat = col3df.iloc[i, :]
        for j in range(dates.shape[0]):
            date = dates[j]
            data = data_dict[date]
            col, row = coor2pos(lon, lat, gt)
            # window size floor division
            vals = data[row - (ws//2): row + (ws//2) + 1, col - (ws//2) : col + (ws//2) + 1]
            vals = vals[np.where(np.isnan(vals) == False)]
            if vals.size:
                val = vals.mean()
                record[i, j] = val
            else:
                record[i, j] = 0
    df = pd.DataFrame(record.T, columns = col3df[colname], index = dates)
    return df

def retrieveSD(fullpath, SDName, r = [0, -1], filter = ""):
    f = gdal.Open(fullpath)
    sds = f.GetSubDatasets()

    tagrefs = {}
    for idx, sd in enumerate(sds):
        # p = sd[0]
        info = sd[1]
        tagrefs[info] = idx

    tags = tagrefs.keys()
    refs = tagrefs.values()
    tag_, ref_ = [], []
    for ref, tag in enumerate(tags):
        if filter in tag and SDName in tag:
            tag_.append(tag)
            ref_.append(ref)

    assert len(tag_) >= 1
    print("techincally it's should be only one dim, at line 224")
    tag, ref = tag_[0], ref_[0]
    sd = sds[ref][0]
    ds = gdal.Open(sd)

    cols = ds.RasterXSize
    rows = ds.RasterYSize
    bans = ds.RasterCount
    array = []

    if r[-1] == -1:
        for i in range(bans)[r[0]::]:
            b = ds.GetRasterBand(i + 1)
            array.append(b.ReadAsArray(0, 0, cols, rows))

    else:
        for i in range(bans)[r[0]: r[-1] + 1]:
            b = ds.GetRasterBand(i + 1)
            array.append(b.ReadAsArray(0, 0, cols, rows))
    return np.array(array)

def tagrefs_info(fullpath):
    f = gdal.Open(fullpath)
    sds = f.GetSubDatasets()
    tagrefs = {}
    for idx, sd in enumerate(sds):
        # ps = sd[0]
        info = sd[1]
        tagrefs[info] = idx
    return(tagrefs)

def h4read(path, swath, sdname = "RadianceMantissa", info = "n"):
    refs_dict = _h4lookup(path, swath)
    array = _h4read(path, refs_dict["RadianceMantissa"])
    if info == "y":
        shape = _query(path, refs_dict[sdname])[2]
        return array, shape
    else:
        return array

def _h4read(path, ref):
    '''
    only capable of reading datasets, vdata is not.
    '''
    sd = SD(path)
    sds = sd.select(sd.reftoindex(ref))
    data = np.float64(sds.get())
    sds.endaccess(); sd.end()
    return data

def _query(path, ref):
    sd = SD(path)
    sds = sd.select(sd.reftoindex(ref))
    info = sds.info()
    sds.endaccess(); sd.end()
    return info

def _h4lookup(path, swath = "Earth UV-2 Swath"):
    '''
    only look-up datasets, ignore vdata and
    "WavelengthReferenceColumn" is that.
    '''
    hdf = HDF(path)
    v = hdf.vgstart()
    s2_vg = v.attach(swath)
    geo_tag, geo_ref = s2_vg.tagrefs()[0]
    dat_tag, dat_ref = s2_vg.tagrefs()[1]
    s2_vg.detach()
    #--------------------------------------------
    # found geoloaction & data fields
    #--------------------------------------------
    geo_vgs = v.attach(geo_ref); dat_vgs = v.attach(dat_ref)
    gvg_tagrefs = geo_vgs.tagrefs(); dvg_tagrefs = dat_vgs.tagrefs()
    geo_vgs.detach(); dat_vgs.detach()
    tagrefs_list = gvg_tagrefs + dvg_tagrefs
    refs_dict = {}
    #--------------------------------------------
    # create dict in which keys are names in hdf and values are refs
    #--------------------------------------------
    sd = SD(path)
    for tr in tagrefs_list:
        tag, ref = tr
        if tag == HC.DFTAG_NDG:
            sds = sd.select(sd.reftoindex(ref))
            refs_dict[sds.info()[0]] = ref
    sds.endaccess(); sd.end(); v.end(); hdf.close()
    return refs_dict

def h5read(fullpath, route = "/HDFEOS/SWATHS/OMI Total Column Amount HCHO/Data Fields/ColumnAmount"):
    f = h5py.File(fullpath)
    sd = f[route]
    array =  sd.value
    return array


class h5write:
    def __init__(self, fullpath):
        self.file = h5py.File(fullpath,'w')
    def __del__(self):
        pass

    def write_ds(self, pos, data = None):
        ds = self.file.create_dataset(pos, data = data)
        return ds
    def write_grp(self, pos):
        grp = self.file.create_group(pos)
        return grp
    def add_attr(self, dsGrp, name, content):
        dsGrp.attrs[name] = content


def w_mean(data, row, col, ws = 3):
    vals = data[row - (ws//2): row + (ws//2) + 1, col - (ws//2) : col + (ws//2) + 1]
    vals = vals[np.where(np.isnan(vals) == False)]
    if vals.size:
        val = vals.mean()
        return val
    else:
        return 0

def spearman(vec1, vec2):
    vec1 = np.array(vec1); vec2 = np.array(vec2)
    vec1_ = vec1.copy(); vec1_ = np.sort(vec1_)[::-1]
    vec2_ = vec2.copy(); vec2_ = np.sort(vec2_)[::-1]

    if np.isfinite(vec1).all() == False or np.isfinite(vec2).all() == False:
        s = 0
    else:
        trace_1 = np.array([np.where(vec1_ == vec1[i])[0][0] for i in range(vec1.shape[0])])
        trace_2 = np.array([np.where(vec2_ == vec2[i])[0][0] for i in range(vec2.shape[0])])
        dif_t = trace_1 - trace_2
        n = dif_t.shape[0]
        s = 1 - 6 * np.float(np.sum((dif_t)**2)) / np.float((n * (n**2 - 1)))

    return s

def pearson(vec1, vec2):
    lens = len(vec1)
    if lens != 0 and np.any((vec1 + vec2) != 0):
        vec1 = (vec1 - vec1.min()) / (vec1.max() - vec1.min())
        vec2 = (vec2 - vec2.min()) / (vec2.max() - vec2.min())
        s_cross = sum([vec1[i] * vec2[i] for i in range(lens)])
        s_vec1 = sum(vec1); s_vec2 = sum(vec2)
        s_vec1sq = sum([vec1[i] * vec1[i] for i in range(lens)])
        s_vec2sq = sum([vec2[i] * vec2[i] for i in range(lens)])

        p_numerator = s_cross - (s_vec1 * s_vec2) / lens
        p_denominator_l = np.sqrt(np.abs(s_vec1sq - ((s_vec1)**2/lens)))
        p_denominator_r = np.sqrt(np.abs(s_vec2sq - ((s_vec2)**2/lens)))

        p = p_numerator / (p_denominator_l * p_denominator_r)
    else:
        p = 0
    return p

def moran(array, k = 3):
    import pysal
    array[np.where(~np.isfinite(array) == True)] = -1
    array[np.where(array < 0)] = np.float("nan")
    #------
    y, x = np.where(np.isfinite(array))

    lats = np.array([gt[3] + gt[5] * i for i in y])
    lons = np.array([gt[0] + gt[1] * i for i in x])
    data = array[y, x]

    coor = np.vstack((lons, lats)).T
    w = pysal.knnW(coor, k = k)
    mr = pysal.Moran_Local(data, w).p_sim
    array[np.where(np.isfinite(array))] = mr
    array[np.where(~np.isfinite(array))] = 0.
    return array

def cal_seasons(vals = 0):
    if mode == 1:
        spr = (vals[2::12] + vals[3::12] + vals[4::12]) / 3
        smr = (vals[5::12] + vals[6::12] + vals[7::12]) / 3
        fal = (vals[8::12] + vals[9::12] + vals[10::12]) / 3
        wtr = (vals[12::12] + vals[13::12] + vals[11: -1 :12]) / 3
        fir = (vals[0] + vals[1]) / 2
        wtr = np.r_[fir, wtr]
        return spr, smr, fal, wtr

    elif mode == 2:
        seasons = np.empty(mons, dtype = "S32")
        seasons[0::12] = "wtr"; seasons[1::12] = "wtr"; seasons[11::12] = "wtr"
        seasons[2::12] = "spr"; seasons[3::12] = "spr"; seasons[4::12] = "spr"
        seasons[5::12] = "smr"; seasons[6::12] = "smr"; seasons[7::12] = "smr"
        seasons[8::12] = "fal"; seasons[9::12] = "fal"; seasons[10::12] = "fal"
        return seasons

def reproject(shapefile):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shapefile, 0)
    inLayer = dataSource.GetLayer()

    inSpatialRef = inLayer.GetSpatialRef()
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(4326)
    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

    geom_type = inLayer.GetGeomType()

    # get the input layer

    # create the output layer
    outputShapefile = "out.shp"
    if os.path.exists(outputShapefile):
        driver.DeleteDataSource(outputShapefile)
    outDataSet = driver.CreateDataSource(outputShapefile)
    outLayer = outDataSet.CreateLayer("proj_4326",  srs = outSpatialRef, geom_type = geom_type)

    # add fields
    inLayerDefn = inLayer.GetLayerDefn()
    for i in range(0, inLayerDefn.GetFieldCount()):
        fieldDefn = inLayerDefn.GetFieldDefn(i)
        outLayer.CreateField(fieldDefn)

    # get the output layer's feature definition
    outLayerDefn = outLayer.GetLayerDefn()

    # loop through the input features
    inFeature = inLayer.GetNextFeature()
    while inFeature:
        # get the input geometry
        geom = inFeature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(coordTrans)
        # create a new feature
        outFeature = ogr.Feature(outLayerDefn)
        # set the geometry and attribute
        outFeature.SetGeometry(geom)
        for i in range(0, outLayerDefn.GetFieldCount()):
            outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
        # add the feature to the shapefile
        outLayer.CreateFeature(outFeature)
        # destroy the features and get the next input feature
        outFeature.Destroy()
        inFeature.Destroy()
        inFeature = inLayer.GetNextFeature()

    # close the shapefiles
    dataSource.Destroy()
    outDataSet.Destroy()

def Rasterize(raster_fn, shapefile):
    pixel_size = 0.25
    NoData_value = -9999

    # raster_fn = r"G:\Pys_HCHO_Workshop\Map\test1.tif"
    # shapefile = r"out.shp"
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shapefile, 0)
    inLayer = dataSource.GetLayer()
    x_min, x_max, y_min, y_max = inLayer.GetExtent()

    # Create the destination data source
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)
    target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(NoData_value)

    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    target_ds.SetProjection(outRasterSRS.ExportToWkt())

    im_width = target_ds.RasterXSize # col number
    im_height = target_ds.RasterYSize # row number
    array = np.ones([im_height, im_width])
    target_ds.GetRasterBand(1).WriteArray(array)
    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], inLayer, burn_values=[0])

def clip(raster_fn, tiff, name):
    mask = gdal.Open(raster_fn)
    # mask = gdal.Open(r"G:\Pys_HCHO_Workshop\Map\test1.tif")
    mask_geotrans = mask.GetGeoTransform()
    mask_proj = mask.GetProjection()
    mask_data = mask.ReadAsArray(0, 0, mask.RasterXSize, mask.RasterYSize)

    # ds = gdal.Open("test.tif")
    ds = gdal.Open(tiff)
    ds_geotrans = ds.GetGeoTransform()
    ds_proj = ds.GetProjection()
    array = ds.ReadAsArray(0, 0, ds.RasterXSize, ds.RasterYSize)

    rows = np.where(mask_data == 0)[0]; cols = np.where(mask_data == 0)[1]
    lons = mask_geotrans[0] + 0.25 * cols
    lats = mask_geotrans[3] - 0.25 * rows

    array[np.where(np.isnan(array))] = 0
    for i in range(rows.shape[0]):
        mask_data[rows[i], cols[i]] = \
            array[np.ceil((lats[i] - ds_geotrans[3])/0.25), np.ceil((lons[i] - ds_geotrans[0])/0.25)]

    cols = mask.RasterXSize
    rows = mask.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    # newRasterfn = "clip.tif"
    newRasterfn = name
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform(mask_geotrans)
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(mask_data)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

def interp_(x, y, new_x, mode = "s"):
    from scipy.interpolate import splev, splrep, pchip, UnivariateSpline
    try:
        y = y[np.where(np.isfinite(y))]
        x = x[np.where(np.isfinite(y))]
        if mode == "s":
            tck = splrep(x, y)
            return splev(new_x, tck)
        elif mode == "p":
            curve = pchip(x, y)
            return curve(new_x)
        elif mode == "u":
            spline = UnivariateSpline(x, y, k = 3, s = 8)
            return spline(new_x)
    except:
        return np.ones(new_x.shape) * np.float("nan")

def polfitdif(speran, val):
    try:
        speran_ = speran[np.where(np.isfinite(val))]
        val_ = val[np.where(np.isfinite(val))]
        fCurve3p = sp.polyfit(speran_, val_, 3)
        fCurve3 = sp.polyval(fCurve3p, speran)
        dif = val - fCurve3
        return dif
    except:
        return np.ones(val.shape) * np.float("nan")

def lstsquare(x, y):
    from scipy import linalg
    a = np.vstack([x.T, y]).T
    p = np.unique(np.where(~np.isfinite(a))[0])
    x_ = np.delete(x, p, 0)
    y_ = np.delete(y, p, 0)
    try:
        return linalg.lstsq(x_, y_)[0]
    except:
        return np.empty(x.shape[1]) * np.float("nan")

def spatial_interp(array, inputlons, inputlats, resX = 0.25, resY = 0.25, threshold = 0.125):
    outputlons, outputlats = _cal_coorRange(inputlons, inputlats, resX, resY)
    lons, lats = np.meshgrid(outputlons, outputlats)
    tars = np.c_[lons.ravel(), lats.ravel()]
    inputpnts = np.vstack([inputlons.ravel(), inputlats.ravel()]).T

    kd = spatial.cKDTree(inputpnts)
    dists, idxs = kd.query(tars, 10, n_jobs = -1)
    arr = np.empty(idxs.shape[0])
    dists[np.where(dists > threshold)] = None
    for i, dist in enumerate(dists):
        pos = np.where(np.isfinite(dist) == True)
        dist = dist[pos]
        idx = idxs[i][pos]
        if len(dist) > 0:
            _, weight = IDW_weight(dist)
            vals = np.array([array.ravel()[i] for i in idx])
            val = np.average(vals, weights = weight)
            arr[i] = val
        else:
            arr[i] = 0
        arr = arr.reshape([outputlats.shape[0], outputlons.shape[0]])
        return arr


def IDW_weight(distance ,p = 4):
    idw = [i**-p for i in distance]
    index, weight = [], []
    for inx, val in enumerate(idw):
        w = round(idw[inx]/sum(idw), 2)
        index.append(inx); weight.append(w)
    return index, weight

def _cal_coorRange(lons, lats, resX, resY):
    lon_l = np.round(lons.min(), 0)
    lon_r = np.round(lons.max(), 0)
    lat_d = np.round(lats.min(), 0)
    lat_u = np.round(lats.max(), 0)
    lonNum = (lon_r - lon_l)/resX
    latNum = (lat_u - lat_d)/resY
    lons_ = np.linspace(lon_l, lon_r - resX, lonNum)
    lats_ = np.linspace(lat_d, lat_u - resY, latNum)
    return lons_, lats_
