import ee
import numpy as np

class Earth(object):
  def __init__(self, collection_name):
    super()
    self.image_collection = ee.ImageCollection(collection_name)

  def filter_image(self, date_range, label = "DATA", band_select = None, area_coordinates_list = []):
    start_date, end_date = date_range
    image_collection = self.image_collection.filterDate(start_date, end_date)
    if area_coordinates_list:
      roi = ee.Geometry.Polygon(area_coordinates_list)
      image_collection = image_collection.filterBounds(roi)
    if band_select:
      image = image_collection.select(band_select).reduce(ee.Reducer.median()).rename(label)
    else:
      image = image_collection.map(
              lambda image: image.reduce(ee.Reducer.mean())
            ).reduce(ee.Reducer.median()).rename(label)
    print("retrieved images...")
    return image

  def get_values(self, image, area_coordinates_list, label = "DATA", max_pixels = 1e8, scale = 20):
    # coor format: clockwise
    # for example:
    # area_coordinates_list = [[[105.532,19.059],[105.606,19.058],[105.605,19.108],[105.530,19.110],[105.532,19.059]]]
    area = ee.Geometry.Polygon(area_coordinates_list)

    latlon = ee.Image.pixelLonLat().addBands(image)
    # apply reducer to list
    latlon = latlon.reduceRegion(
      reducer=ee.Reducer.toList(),
      geometry=area,
      maxPixels=max_pixels,
      scale=scale)
    # print(latlon)
    self.data = np.array((ee.Array(latlon.get(label)).getInfo()))
    self.lats = np.array((ee.Array(latlon.get("latitude")).getInfo()))
    self.lons = np.array((ee.Array(latlon.get("longitude")).getInfo()))
    print("retrieved data...")

  def to_geotif(self):
    # NOTICE: only one band for now
    # get the unique coordinates
    uniqueLats = np.unique(self.lats)
    uniqueLons = np.unique(self.lons)

    # get number of columns and rows from coordinates
    ncols = len(uniqueLons)    
    nrows = len(uniqueLats)
    
    # determine pixelsizes
    ys = uniqueLats[1] - uniqueLats[0] 
    xs = uniqueLons[1] - uniqueLons[0]

    # create an array with dimensions of image
    array = np.zeros([nrows, ncols], np.float32) #-9999
    
    # fill the array with values
    counter = 0
    for y in range(0, len(array), 1):
        for x in range(0, len(array[0]), 1):
            if self.lats[counter] == uniqueLats[y] and self.lons[counter] == uniqueLons[x] and counter < len(self.lats)-1:
                counter+=1
                array[len(uniqueLats)-1-y,x] = self.data[counter] # we start from lower left corner

    # set the 
    #SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    transform = (np.min(uniqueLons),xs,0,np.max(uniqueLats),0,-ys)
    return {"array": array, "transform": transform}