import bz2
import pickle
import _pickle as cPickle
import plotly.express as px
from jupyter_dash import JupyterDash
from datetime import datetime as dt
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import zipfile

# Pickle a file and then compress it into a file with extension 
def compressed_pickle(title, data):
 with bz2.BZ2File(title + '.pbz2', 'w') as f: 
 	cPickle.dump(data, f)


# Load any compressed pickle file
def decompress_pickle(file):
 data = bz2.BZ2File(file, 'rb')
 data = cPickle.load(data)
 return data



# load data 
with zipfile.ZipFile('geo_out_july_2020_central_time.csv.zip', 'r') as zip_ref:
    zip_ref.extractall()
df = pd.read_csv('geo_out_july_2020_central_time.csv', index_col=0)
df['emdCardNumber'] = df['emdCardNumber'].str.upper()

compressed_pickle('geo_out_july_2020_central_time', df) 



