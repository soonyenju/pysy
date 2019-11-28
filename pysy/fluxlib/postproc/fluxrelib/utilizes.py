import re, yaml

import numpy as np
import pandas as pd

from pathlib import Path

from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool

class Craftman(object):
    super(object)
    def __init__(self):
        pass

    def __del__(self):
        pass

    def parse_yaml(self, cur_dir = None, FILE_NAME = "*"):
        """
        directory:
        |-
        |--data
        |--config.yaml
        catalog.yaml should be at the same dir with data folder.
        """
        if not cur_dir:
            CUR_DIR = Path.cwd().parent
        else:
            CUR_DIR = cur_dir
        CATALOG_DIR = CUR_DIR.joinpath('config.yaml')

        with open(CATALOG_DIR.as_posix(), 'r') as stream:
            try:
                catlog = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        url_tempalte = catlog["sources"]["flux"]["args"]["urlpath"]
        # iterate the parameter place holders.
        pattern = r"{{ \S* }}"
        # print(url_tempalte)
        results = re.findall(pattern, url_tempalte)
        # print(results)
        for res in results:
            if "CATALOG_DIR" in res:
                url_tempalte = url_tempalte.replace(res, CATALOG_DIR.parent.as_posix())
            elif "FILE_NAME" in res:
                url = url_tempalte.replace(res, FILE_NAME)
            else:
                raise Exception("Wrong format!")
        del(url_tempalte)

        # iterate the csv urls
        url = Path(url)
        urls = url.parent.glob(url.name)
        keys = catlog["sources"]["flux"]["args"]["keys"]
        keys = [key.strip() for key in keys.split(",")]
        return urls, keys

        # "trasfer string 2 float"
        # cur_dir = Path.cwd()
        # cat_dir = cur_dir.parent.joinpath('catalog.yaml').as_posix()
        # cat = intake.open_catalog(cat_dir)
        # print(list(cat))
        # print(help(cat.flux().read))
        # metadata = cat.flux().read()
        # metadata.sample(5)
        # print(metadata)

        # p = r"G:\OneDrive - University of Exeter\Workspace\after_EdiRe\data\Roth_N.csv"
        # df = dd.read_csv(p)
        # print(df)


    def clean_read(self, url, keys = "*"):
        # read only pre-selected columns
        if keys != "*":
            # header = pd.read_csv(url, index_col=0, dtype = object, nrows=1).columns.values
            # clean_header = [name.strip() for name in header]
            # for key in keys:
            #     if key in clean_header:
            #         print(clean_header.index(key))
            # fields = [header[clean_header.index(key)] for key in keys if key in clean_header]
            # del(header, clean_header)
            # df = pd.read_csv(url, index_col=0, dtype = object, dayfirst = True, usecols=fields)
            ## codes above equals to the line below:
            df = pd.read_csv(url, index_col=0, dtype = object, dayfirst = True, skipinitialspace=True, usecols=keys)
        else:
            df = pd.read_csv(url, index_col=0, dtype = object, dayfirst = True)
        
        df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.strip()
        # df = df.replace(" ", "-9999")
        # To replace white space at the beginning: '^ +'
        # To replace white space at the end: ' +$'
        # To replace white space at both ends: '^ +| +$'
        df = df.replace(' +$', '-9999', regex=True)
        # # debug the space revmoal:
        # print(df["cH2Oxcor"][2: 15])
        # print(len(df["cH2Oxcor"][10]))
        # print(df["cH2Oxcor"][10] == " ")
        # print(type(df["cH2Oxcor"][10]))

        # print(df.dtypes) # check the data types
        df = df.astype(np.float)

        return df

class VizPad(object):
    super(object)
    def __init__(self, df):
        # import holoviews as hv
        # import hvplot.pandas
        # import geoviews.tile_sources as gts
        # pd.options.display.max_columns = 10
        # hv.extension('bokeh', width=80)
        self.df = df
    
    def interactive(self):
        # output to static HTML file
        output_file("log_lines.html")
        sample = self.df.sample(5000)
        source = ColumnDataSource(sample)
        p = figure(x_axis_type='datetime', y_range=(-100, 100))
        p.circle(x='Date/Time', y='Hc',
                source=source,
                size=10, color='green')

        p.title.text = 'Hc and date'
        p.xaxis.axis_label = 'Date'
        p.yaxis.axis_label = 'Hc'


        hover = HoverTool()
        hover.tooltips=[
            ('Date', '@Date/Time'),
            ('Hc', '@Hc'),
        ]

        p.add_tools(hover)

        show(p)
