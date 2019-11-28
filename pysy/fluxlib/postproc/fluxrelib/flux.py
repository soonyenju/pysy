# import sys
# import dask
# import numpy as np
import pandas as pd

# import holoviews as hv
# import hvplot.pandas
# import geoviews.tile_sources as gts

# import intake
# from pathlib import Path

# import dask.dataframe as dd
from utilizes import Craftman, VizPad

def main():
    craft = Craftman()
    urls, keys = craft.parse_yaml()
    for url in urls:

        df = craft.clean_read(url, keys = keys)
        print(df)
        vizpad = VizPad(df)
        vizpad.interactive()
        exit(0)






if __name__ == "__main__":
    # pd.options.display.max_columns = 10
    # hv.extension('bokeh', width=80)
    main()