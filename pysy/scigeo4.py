# coding: utf-8
# Date: 2018-12-12 12:42
import gdal, osr, ogr
import warnings
import numpy as np
try:
	import rasterio
except ImportError:
	print("No module named rasterio, using original gdal instead")
try:
	from pyproj import Proj, Geod, transform
except ImportError:
	Proj, Geod, transform = None, None, None
	print("No module named pyproj")


class Raster(object):
	"""docstring for Raster"""
	def __init__(self, raster_path):
		super(Raster, self).__init__()
		gdal.SetConfigOption('GDAL_FILENAME_IS_UTF8', 'NO')
		gdal.SetConfigOption('SHAPE_ENCODING', 'gb2312')
		try:
			self.path = raster_path.as_posix()
		except Exception as identifier:
			self.path = raster_path
		
	def __del__(self):
		pass

	def read(self):
		try:
			array, gt, proj = self.gdal_read()
		except Exception as identifier:
			array, gt, proj = self.rio_read()
		finally:
			return array, gt, proj

	def gdal_read(self):
		src = gdal.Open(self.path)
		cols = src.RasterXSize
		rows = src.RasterYSize
		bands = src.RasterCount
		proj = src.GetProjection()
		gt = src.GetGeoTransform()
		if bands == 1:
			array = src.ReadAsArray(0, 0, cols, rows)
		else:
			try:
				array = src.GetVirtualMemArray()
			except Exception as identifier:
				array = []
				for band in np.arange(bands) + 1:
					band_raster = src.GetRasterBand(band)
					data = band_raster.ReadAsArray(0, 0, cols, rows)
					array.append(data)
				array = np.array(array)
		return array, gt, proj

	def gdal_read_info(self):
		src = gdal.Open(self.path)
		files = src.GetFileList()
		meta = src.GetMetadata_List()
		sds = src.GetSubDatasets()
		cols = src.RasterXSize
		rows = src.RasterYSize
		bands = src.RasterCount
		gt = src.GetGeoTransform()
		proj = src.GetProjection()
		return {"FileLists": files, "metadata": meta, "subdatasets": sds,
				"shape": [rows, cols, bands], "gt": gt, "proj": [proj]}

	def gdal_readAll(self):
		data, _, _ = self.gdal_read()
		info = self.gdal_read_info()
		return {"data": data, "info": info}

	def rio_read(self):
		warnings.filterwarnings("ignore")
		with rasterio.open(self.path) as src:
			array = src.read()
			gt = src.transform
			proj = src.crs
		return array, gt, proj

	def rio_readAll(self):
		"""
		c, a, b, f, d, e = src.transform
		gt = rasterio.transform.Affine.from_gdal(c, a, b, f, d, e)
		proj = src.crs
		bands = src.count
		name = src.name
		mode = src.mode
		closed = src.closed
		width = src.width
		height = src.height
		bounds = src.bounds
		idtypes = {i: dtype for i, dtype in zip(
			src.indexes, src.dtypes)}
		meta = src.meta
		src = src.affine
		"""
		src = rasterio.open(self.path)
		return src


class Craftsman(object):
	def __init__(self, array):
		super(Craftsman, self).__init__()
		if type(array) == list:
			array = np.array(array)
		if len(array.shape) == 1:
			self.array = array[:, np.newaxis]
		else:
			self.array = array

	def __del__(self):
		pass

	def clean_array(self, FILLVAL = 0, SPECIFICVAL = None, AutoSelected = False):
		self.array[np.where(np.isnan(self.array) == True)] = FILLVAL
		self.array[np.where(np.isfinite(self.array) == False)] = FILLVAL
		if SPECIFICVAL:
			self.array[np.where(self.array == np.float64(SPECIFICVAL))] = FILLVAL
		elif AutoSelected:
			SPECIFICVAL = sorted([(np.sum(self.array == i), i)
                         for i in set(self.array.flat)])[-1][1]
			self.array[np.where(self.array == np.float64(SPECIFICVAL))] = FILLVAL
			SPECIFICVAL = None
		else:
			self.array[np.where(self.array <= 0)] = FILLVAL
	
	def flip_array(self, ROT = True, ROTANG = None, UPSIDEDOWN = False, LEFTSIDERIGHT = False):
		if ROT:
			# rotate clockwise 180
			self.array = self.array[::-1]

		if UPSIDEDOWN:
			# upside down
			nRow = self.array.shape[0]
			for row in range(nRow // 2):
				self.array[row], self.array[nRow-1 -
								i] = self.array[nRow-1-row], self.array[row]
		
		if LEFTSIDERIGHT:
			# leftside right
			nCol = self.array.shape[1]
			for each_row in self.array:
				for col in range(nCol // 2):
					each_row[col], each_row[nCol-1-col] = each_row[nCol-1-col], each_row[col]

		
			
		


# def main():
# 	raster = Raster('test.tif')
# 	raster.read()

if __name__ == '__main__':
	main()
	print("ok")
