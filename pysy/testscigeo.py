# coding: utf-8
from pathlib import Path
from pysy import scigeo
from scigeo4 import *
import geojsonio
from matplotlib import pyplot as plt


def main():
	'''
	paths = Path('vcd').glob(r'*.tif')
	for path in list(paths)[0: 2]:
		print(path)
		ras = Raster(path)
		dataset = ras.read()
		# array = dataset["data"]
		# print(array)
		craft = Craftsman(array)
		craft.clean_array()
		dataset["data"] = (craft.array)
		# ras.write(dataset, "test.tif", gt=dataset["info"]["gt"])
		# dataset = ras.rio_read()
		# ras.rio_write(dataset["data"], "test2.tif", dataset["info"])
	'''
	paths = Path('shp').glob(r'*.shp')
	for path in paths:
		print(path)
		vec = Vector(path)
		# dataset = vec.ogr_read_shp()
		# print(dataset)
		# vec.ogr_write_shp(dataset, 'test.shp')
		
		dataset = vec.ogr_read_shp()
		print(dataset)
		vec.ogr_write_shp(dataset, "sss.shp")
		# dataset = dataset.to_json()
		# dataset.plot()
		# print(dataset)
		# plt.show()
		# geojsonio.display(dataset)
		# print(type(dataset))
		# vec.write(dataset, "sss.shp")
		
		# to_json


if __name__ == "__main__":
	main()
