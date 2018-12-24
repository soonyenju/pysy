#!/usr/bin/env python
#-*- coding:utf-8 -*-

#############################################
# File Name: setup.py
# Author: soonyenju
# Mail: soonyenju@foxmail.com
# Created Time:  2018-10-23 13:28:34
#############################################


from setuptools import setup, find_packages

setup(
	name = "pysy",
	version = "0.0.6",
	keywords = ("pip", "geo-processing","GDAL", "raster-file", "shapefile", "soonyenju"),
	description = "For faster proccessing geofile",
	long_description = "Read/write and process rs/gis related data, especially atmospheric rs data.",
	license = "MIT Licence",

	url="https://github.com/soonyenju/pysy",
	author = "soonyenju",
	author_email = "soonyenju@foxmail.com",

	packages = find_packages(),
	include_package_data = True,
	platforms = "any",
	install_requires=[
            'numpy == 1.14.2',
            'pandas == 0.20.1',
            'GDAL == 2.2.4',
            'pyproj == 1.9.5.1',
            'PySAL == 1.14.3',
            'h5py == 2.8.0rc1',
            'scipy == 0.19.0',
            'netCDF4 == 1.4.2',
            'osr == 0.0.1'
	]
)

