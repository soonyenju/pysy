def gen_buffer(lon, lat, step, shape = "rectangle"):
    if shape == "rectangle":
        # clockwise
        coors = [
                 [lon - step, lat + step], # upper left
                 [lon + step, lat + step], # upper right
                 [lon + step, lat - step], # lower right
                 [lon - step, lat - step], # lower left
        ]

    return coors