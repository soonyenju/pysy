#!/usr/bin/env python
#-*- coding:utf-8 -*-

#############################################
# File Name: setup.py
# Author: soonyenju
# Mail: soonyenju@foxmail.com
# Created Time:  2018-10-23 13:28:34
#############################################


from setuptools import setup, find_packages


def parse_requirements(filename):
	""" load requirements from a pip requirements file """
	lineiter = (line.strip() for line in open(filename))
	return [line for line in lineiter if line and not line.startswith("#")]


setup(
	name = "pysy",
	version = "0.0.1",
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
	install_requires=parse_requirements("requirements.txt")
)

