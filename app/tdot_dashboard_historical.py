# %%
from colormap import rgb2hex
import matplotlib.colors
import time
from resource import *
import traceback
import dateparser
from matplotlib import cm, colors
import dash
import flask
import pyarrow.parquet as pq
import os
# from flask_caching import Cache
from dask import dataframe as dd
from random import randint
from flask_caching import Cache
from dash import dcc
from dash import html
import pandas as pd
import numpy as np
import plotly.express as px
from dash.dependencies import Input, Output, State
from plotly import graph_objs as go
from plotly.graph_objs import *
from datetime import datetime as dt
from datetime import time as tt
import dash_daq as daq
# import dataextract
import datetime
import os
import sys
# import resource
import geopandas as gpd
import dash_bootstrap_components as dbc
from statresp.datajoin.cleaning.categorizer import categorize_numerical_features, Filter_Combo_Builder, FILTER_calculator
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
import shapely.geometry


def returnlat(feature):
    lats = []
    if isinstance(feature, shapely.geometry.linestring.LineString):
        linestrings = [feature]
    elif isinstance(feature, shapely.geometry.multilinestring.MultiLineString):
        linestrings = feature.geoms
    for linestring in linestrings:
        x, y = linestring.xy
        for entry in y:
            lats.append(entry)
    return lats


def returnlon(feature):
    lon = []
    if isinstance(feature, shapely.geometry.linestring.LineString):
        linestrings = [feature]
    elif isinstance(feature, shapely.geometry.multilinestring.MultiLineString):
        linestrings = feature.geoms
    for linestring in linestrings:
        x, y = linestring.xy
        for entry in x:
            lon.append(entry)
    return lon


# set configurations
mapbox_style = "light"
mapbox_access_token = "pk.eyJ1Ijoidmlzb3ItdnUiLCJhIjoiY2tkdTZteWt4MHZ1cDJ4cXMwMnkzNjNwdSJ9.-O6AIHBGu4oLy3dQ3Tu2XA"
MAPBOX_ACCESS_TOKEN = mapbox_access_token
pd.set_option('display.max_columns', None)

seg = pd.read_pickle('data/tdot/grouped_inrix.pkl')
seg = seg[seg.frc == 0]
predRF = pd.read_pickle('data/tdot/DF_likelihood_spacetime_RF.pkl')
predRF = predRF.merge(
    seg[['grouped_xdsegid', 'county_inrix', 'geometry']], on='grouped_xdsegid')
predRF = predRF[['RF', 'county_inrix',
                 'time_local', 'geometry', 'grouped_xdsegid']]
predRF['day_of_week'] = predRF['time_local'].dt.dayofweek
predRF['month'] = predRF['time_local'].dt.month
predRF['year'] = predRF['time_local'].dt.year
seg = None


latInitial = 36.16228
lonInitial = -86.774372
metadata = {}
metadata['merged_pickle_address'] = 'data/tdot/merged_4h_2017-04-01_to_2021-06-01_top20percent_segments_grouped_nona'
metadata['incident_pickle_address'] = 'data/tdot/incident'
metadata['pred_name_TF'] = 'incident_occurred'
available_features = ['is_weekend', 'window',
                      'speed_mean', 'average_speed_mean', 'reference_speed_mean', 'congestion_mean', 'miles', 'lanes',
                      'isf_length', 'slope_median', 'ends_ele_diff', 'temp_mean', 'wind_spd_mean', 'vis_mean', 'precip_mean',
                      'mean_incidents_last_7_days', 'mean_incidents_last_4_weeks', 'mean_incidents_over_all_windows',
                      ]
df_merged = pd.read_parquet(metadata['merged_pickle_address'], columns=available_features+[
                            'time_local', 'county', 'month', 'incident_occurred'])
if not 'day_of_week' in df_merged.columns:
    df_merged['day_of_week'] = df_merged['time_local'].dt.dayofweek
df = pd.read_parquet(metadata['incident_pickle_address'], columns=['frc', 'incident_id', 'county_inrix',
                                                                   'time_local', 'day_of_week', 'month', 'gps_coordinate_longitude', 'gps_coordinate_latitude'])
df = df[df['frc'] == 0]
df['month-year'] = df['time_local'].dt.strftime('%b %Y')
df = df.rename(columns={'gps_coordinate_longitude': 'longitude',
                        'gps_coordinate_latitude': 'latitude', 'incident_id': 'incidentNumber'})

startdate = df_merged['time_local'].dt.date.min()
enddate = df_merged['time_local'].dt.date.max()
MIN_YR = startdate.year
MAX_YR = enddate.year


counties = df.county_inrix.drop_duplicates()
tncounties = pd.read_csv('counties.csv')
tncounties['county'] = tncounties['county'].str.lower()

counties = counties.tolist()
counties.sort()
cluster_list = [1, 2]


external_stylesheets = [
    {
        'href': 'https://use.fontawesome.com/releases/v5.8.1/css/all.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-50oBUHEmvpQ+1lW4y57PTFmhCaXp0ML5d60M1M7uH2+nqUivzIebhndOJK28anvf',
        'crossorigin': 'anonymous'
    },
    dbc.themes.BOOTSTRAP,
    "https://scopelab.ai/files/lightstyle.css"
]

# set the server and global parameters
server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
app = dash.Dash(__name__, title='Incident Dashboard', update_title=None, external_stylesheets=external_stylesheets, server=server, meta_tags=[
                {"name": "viewport", "content": "width=device-width"}])
app.title = 'Incident Dashboard'

cache = Cache(app.server,
              config=dict(CACHE_TYPE='filesystem', CACHE_DEFAULT_TIMEOUT=10000, CACHE_DIR='cache-directory'))

mapbox_access_token = "pk.eyJ1Ijoidmlzb3ItdnUiLCJhIjoiY2tkdTZteWt4MHZ1cDJ4cXMwMnkzNjNwdSJ9.-O6AIHBGu4oLy3dQ3Tu2XA"
px.set_mapbox_access_token(mapbox_access_token)

# Layout of Dash App
app.layout = html.Div(id='container-div', className="container-fluid bg-white text-dark", children=[html.Div(
    children=[
        html.Div(
            className="row",
            children=[
                # Column for user controls
                html.Div(
                    className="col-12 col-lg-3",
                    children=[
                        dbc.Row(id='initial-div', form=True, no_gutters=True,
                                className="p-0 m-0 bg-white", style={"border": "none"},
                                children=[
                                    dbc.Col(
                                        dcc.Markdown(
                                            '''# [Statresp.ai](https://statresp.ai) | TN Highway Incident Dashboard''',
                                            className="p-0 m-0", style={"margin": "0", "padding": "0"}),
                                        width=10,
                                    ),
                                    dbc.Row(form=True, no_gutters=True, children=[
                                        dbc.Col(
                                            html.I(id='sun', className='fas fa-sun p-0 m-0', style={"margin": "0", "padding": "0"}),  style={"margin": "0", "padding": "0"},
                                        ),
                                        dbc.Col(daq.BooleanSwitch(
                                            id='theme-toggle',
                                            persistence=True,
                                            color='blue',
                                            className="p-0 m-0",
                                            style={"margin": "0",
                                                   "padding": "0"}
                                        ),
                                        ),
                                        dbc.Col(
                                            html.I(id='moon', className='fas fa-moon p-0 m-0', style={"margin": "0", "padding": "0"}),  style={"margin": "0", "padding": "0"},

                                        )]
                                    )
                                ],
                                ),
                        html.Div(id='year-div', children=[
                            dcc.Markdown(
                                '''  # Choose Year Range''', style={"margin": "0", "padding": "0"}),
                            dcc.RangeSlider(
                                id="year_slider",
                                min=MIN_YR,
                                max=MAX_YR,
                                count=1,
                                step=1,
                                value=[MIN_YR, MAX_YR],
                                marks={yr: str(yr) for yr in range(
                                    MIN_YR, MAX_YR + 1)},
                            )
                        ],
                        ),
                        html.Div(id='start-date-div', className="card p-1 m-1 bg-white text-dark", children=[
                            dcc.Markdown(
                                '''  # Start Date''', style={"margin": "0", "padding": "0"}),
                            dcc.DatePickerSingle(
                                id="date-picker",
                                min_date_allowed=startdate,
                                number_of_months_shown=4,
                                max_date_allowed=enddate,
                                initial_visible_month=startdate,
                                date=startdate,
                                display_format="MMMM D, YYYY",
                                style={"border": "0px solid black"},
                            )],
                        ),

                        html.Div(id='end-date-div',
                                 className="card p-1 m-1 bg-white text-dark",
                                 children=[
                                     dcc.Markdown(
                                         '''## End Date''', style={"margin": "0", "padding": "0"}),
                                     dcc.DatePickerSingle(
                                         id="date-picker-end",
                                         number_of_months_shown=4,
                                         min_date_allowed=startdate,
                                         max_date_allowed=enddate,
                                         initial_visible_month=enddate,
                                         date=enddate,
                                         display_format="MMMM D, YYYY",
                                         style={"border": "0px solid black"},
                                     )
                                 ],
                                 ),

                        html.Div(id='county-selector-div',
                                 className="card p-1 m-1 bg-white text-dark",
                                 children=[
                                     dcc.Markdown(
                                         '''## County'''),
                                     dcc.Dropdown(
                                         options=[{'label': name, 'value': name}
                                                  for name in counties],
                                         value=['davidson'],
                                         id='county',
                                         multi=True,
                                     ),
                                 ],
                                 ),

                        html.Div(id='cluster-selector-div',
                                 className="card p-1 m-1 bg-white text-dark", style={"display": "none"},
                                 children=[
                                     dcc.Markdown(
                                         '''## Cluster'''),
                                     dcc.Dropdown(
                                         options=[{'label': name, 'value': name}
                                                  for name in cluster_list],
                                         value=[],
                                         id='cluster',
                                         multi=True,
                                     ),
                                 ],
                                 ),
                        # Change to side-by-side for mobile layout
                        html.Div(id='month-selector-div',
                                 className="card p-1 m-1 btn-group bg-white text-dark",
                                 children=[
                                     dcc.Markdown('''## Month'''),
                                     # Dropdown to select times
                                     dcc.Dropdown(
                                         id="month-selector",
                                         options=[

                                             {'label': 'Jan', 'value': 1},
                                             {'label': 'Feb', 'value': 2},
                                             {'label': 'Mar', 'value': 3},
                                             {'label': 'Apr', 'value': 4},
                                             {'label': 'May', 'value': 5},
                                             {'label': 'June', 'value': 6},
                                             {'label': 'July', 'value': 7},
                                             {'label': 'Aug', 'value': 8},
                                             {'label': 'Sep', 'value': 9},
                                             {'label': 'Oct', 'value': 10},
                                             {'label': 'Nov', 'value': 11},
                                             {'label': 'Dec', 'value': 12},
                                         ], multi=True,
                                     )
                                 ],
                                 ),
                        html.Div(id='day-of-week-selector-div',
                                 className="card p-1 m-1 bg-white text-dark",
                                 children=[
                                     dcc.Markdown(
                                         '''## Days of Week '''),
                                     # Dropdown to select times
                                     dcc.Dropdown(
                                         id="day-selector",
                                         options=[

                                             {'label': 'Mon', 'value': 0},
                                             {'label': 'Tue', 'value': 1},
                                             {'label': 'Wed', 'value': 2},
                                             {'label': 'Thur', 'value': 3},
                                             {'label': 'Fri', 'value': 4},
                                             {'label': 'Sat', 'value': 5},
                                             {'label': 'Sun', 'value': 6},
                                         ],
                                         multi=True,
                                     )
                                 ],
                                 ),

                        html.Div(id="feature-selector",
                                 className="card p-1 m-1 bg-white text-dark",
                                 children=[
                                     dcc.Markdown(
                                         '''## Select Feature for Likelihood Analysis'''),
                                     dcc.Dropdown(
                                         options=[
                                             {'value': 'is_weekend',
                                              'label': 'Weekend'},
                                             {'value': 'window',
                                              'label': 'Time of Day'},
                                             {'value': 'reference_speed_mean',
                                              'label': 'Reference Speed'},
                                             {'value': 'congestion_mean',
                                              'label': 'Traffic Congestion'},
                                             {'value': 'lanes',
                                              'label': "Number of Lanes"},
                                             {'value': 'temp_mean',
                                              'label': "Temperature"},
                                             {'value': 'wind_spd_mean',
                                              'label': "Windspeed"},
                                             {'value': 'vis_mean',
                                              'label': "Visibility"},
                                             {'value': 'precip_mean',
                                              'label': "Precipitation"},
                                         ],
                                         value='temp_mean',
                                         id='feature',
                                         multi=False,
                                     ),
                                 ],
                                 ),

                        html.Div(id='time-slider-div', className="card p-1 m-1 bg-white text-dark",
                                 children=[
                                     dcc.Markdown(
                                         '''## Incident Time'''),
                                     dcc.RangeSlider(
                                         id='time-slider',
                                         min=0,
                                         max=24,
                                         step=1,
                                         value=[0, 24],
                                         marks={i: '{}:00'.format(
                                                str(i).zfill(2)) for i in range(0, 25, 4)},
                                     ),

                                 ]
                                 ),

                        html.Div(id='radius-div', className="card p-1 m-1 bg-white text-dark",
                                 children=[
                                     dcc.Markdown(
                                         '''## Heatmap density radius.'''),
                                     dcc.Slider(
                                         id='map-graph-radius',
                                         min=1,
                                         max=10,
                                         step=0.05,
                                         marks={i: '{}'.format(i)
                                                for i in range(1, 11)},
                                         value=2
                                     ),
                                 ]
                                 ),



                        html.Div(id='incidents-count-div', className="card p-1 m-1 bg-white text-dark",
                                 children=[html.P('Incidents', id='incident-text', style={'text-align': 'left', 'font-weight': 'bold'}),
                                           html.P(
                                     'Months', id='month-text', style={'text-align': 'left', 'font-weight': 'bold'}),
                                     html.P(
                                     'Time', id='time-text', style={'text-align': 'left', 'font-weight': 'bold'}),
                                     html.P(
                                     'Response', id='response-text', style={'text-align': 'left', 'font-weight': 'bold'}),
                                 ],
                                 ),
                    ],
                ),
                # Column for app graphs and plots
                html.Div(
                    className="col-12 col-lg-9 p-0 m-0 container-fluid",
                    children=[
                        html.Div(id='maps-tabs-div', className="p-0 m-0 card  bg-white text-dark",
                                 children=[
                                     dbc.Tabs(id='tabs', active_tab="incidents", children=[
                                         dbc.Tab(label='Incidents Total', tab_id='incidents', children=[dcc.Loading(
                                             id="loading-icon1", children=[dcc.Graph(id="map-graph"), ], type='default')]),
                                         dbc.Tab(label='Incidents by Month', tab_id='incidents-month', children=[dcc.Loading(
                                             id="loading-icon-incidents-month", children=[dcc.Graph(id="map-incidents-month"), ], type='default')]),
                                         dbc.Tab(label='Predictions', tab_id='incidents-predictions', children=[dcc.Loading(
                                             id="loading-icon-incidents-predictions", children=[dcc.Graph(id="map-incidents-predictions"), ], type='default')]),
                                     ]
                                     ),
                                 ]),

                        html.Div(id='histogram-basis-div', className="p-0 m-0 card bg-white text-dark", children=[
                            dbc.Tabs(id='histogram-basis', active_tab="month", children=[
                                 dbc.Tab(label='Incident Frequency',
                                         tab_id='totals'),
                                 dbc.Tab(label='Incidents by Month',
                                         tab_id='month'),
                                 dbc.Tab(label='Incidents by Weekday',
                                         tab_id='day'),
                                 dbc.Tab(label='Incidents by Time of Day',
                                         tab_id='hour'),
                                 dbc.Tab(label='Comparitive Likelihood by Feature',
                                         tab_id='incidentsfeature'),
                                 dbc.Tab(label='Comparitive Likelihood',
                                         tab_id='incidentsfeaturecombo'),

                                 ]),
                            dcc.Loading(id="loading-icon2", className="flex-grow-1",
                                        children=[
                                            dcc.Graph(id="histogram"), ], type='default'),
                        ]
                        ),

                    ],
                ),
            ],
        ), ],
),

    html.Div(className="row p-0 m-0 bg-white text-dark", id='footer-div',
             children=[dcc.Markdown('''Site designed by [ScopeLab](http://scopelab.ai/). Data source: TDOT. '''
                                    '''Funding by TDOT and National Science Foundation.''', id='footer', className="col p-0 m-0"), ]),
    html.Div(id="blank", children=[dcc.Markdown(
        '''dark theme''')], style={"display": "none"}),
]
)

app.clientside_callback(
    """
    function(darktheme) {
        var stylesheets = document.querySelectorAll('link[rel=stylesheet][href^="https://scopelab.ai"]')
        // Update the url of the main stylesheet.
        var url="https://scopelab.ai/files/darkstyle.css"
        if(darktheme === true)
        {
           //stylesheets[0].href = "https://scopelab.ai/files/darkstyle.css""
           url="https://scopelab.ai/files/darkstyle.css"
           
        }
        else{
           // stylesheets[0].href = "https://scopelab.ai/files/lightstyle.css"
            url="https://scopelab.ai/files/lightstyle.css"
           
        }
        setTimeout(function() {stylesheets[0].href = url;}, 1000);

        // Delay update of the url of the buffer stylesheet.
      return stylesheets[0].href
    }
    """,
    Output("blank", "children"),
    Input("theme-toggle", "on"),
)

# Incident Filterings


@app.callback(
    [Output("date-picker", "date"), Output("date-picker-end", "date")],
    [Input("year_slider", "value")],
    [State("date-picker", "date"), State("date-picker-end", "date")],
)
def update_date_range(slider_dates, date_range_start, date_range_end):
    start_yr, end_yr = slider_dates[0], slider_dates[1]

    if date_range_start is not None:
        date_range_start = str(start_yr) + date_range_start[4:]

    if date_range_end is not None:
        date_range_end = str(end_yr) + date_range_end[4:]

    new_start_date = dateparser.parse(date_range_start)
    new_end_date = dateparser.parse(date_range_end)

    if isinstance(new_start_date, datetime.datetime):
        new_start_date = new_start_date.date()
    if isinstance(new_end_date, datetime.datetime):
        new_end_date = new_end_date.date()

    if new_start_date > enddate:
        new_start_date = enddate
    if new_start_date < startdate:
        new_start_date = startdate

    if new_end_date > enddate:
        new_end_date = enddate
    if new_end_date < startdate:
        new_end_date = startdate

    return str(new_start_date), str(new_end_date)


@ cache.memoize()
def return_predictions(start_date, end_date, counties, months, timerange, days):
    start_date = dateparser.parse(start_date)
    end_date = dateparser.parse(end_date)
    if True:
        date_condition = ((predRF['time_local'] >= start_date)
                          & (predRF['time_local'] <= end_date))
        if days is None or len(days) == 0:
            weekday_condition = (True)
        else:
            weekday_condition = ((predRF['day_of_week']).isin(days))
        if months is None or len(months) == 0:
            month_condition = True
        else:
            month_condition = ((predRF['month'].isin(months)))
        timemin, timemax = timerange
        hourmax = int(timemax)
        hourmin = int(timemin)

        if counties is None or len(counties) == 0:
            county_condition = True
        else:
            county_condition = ((predRF['county_inrix'].isin(counties)))
        starttime = tt(hour=hourmin)
        if hourmax != 24:
            endtime = tt(hour=max(0, hourmax-1), second=59)
        else:
            endtime = tt(hour=23, second=59)
        timecondition = ((predRF['time_local'].dt.time >= starttime)
                         & (predRF['time_local'].dt.time <= endtime))
        result = predRF[timecondition & date_condition &
                        month_condition & weekday_condition & county_condition]
        result = result.sort_values(by=['time_local'])
    else:
        print("Exception in user code:")
        traceback.print_exc(file=sys.stdout)
    return result


@ cache.memoize()
def return_incidents(start_date, end_date, counties, months, timerange,   days):
    start_date = dateparser.parse(start_date)
    end_date = dateparser.parse(end_date)

    if start_date is None or end_date is None or start_date > end_date:
        return None

    if True:
        date_condition = ((df['time_local'] >= start_date)
                          & (df['time_local'] <= end_date))
        if days is None or len(days) == 0:
            weekday_condition = (True)
        else:
            weekday_condition = ((df['day_of_week']).isin(days))
        if months is None or len(months) == 0:
            month_condition = True
        else:
            month_condition = ((df['month'].isin(months)))
        timemin, timemax = timerange
        hourmax = int(timemax)
        hourmin = int(timemin)

        if counties is None or len(counties) == 0:
            county_condition = True
        else:
            county_condition = ((df['county_inrix'].isin(counties)))
        starttime = tt(hour=hourmin)
        if hourmax != 24:
            endtime = tt(hour=max(0, hourmax-1), second=59)
        else:
            endtime = tt(hour=23, second=59)
        timecondition = ((df['time_local'].dt.time >= starttime)
                         & (df['time_local'].dt.time <= endtime))
        result = df[timecondition & date_condition &
                    month_condition & weekday_condition & county_condition]
        result = result.sort_values(by=['time_local'])
    else:
        print("Exception in user code:")
        traceback.print_exc(file=sys.stdout)
    return result


def empty_fig(text="No Result"):
    return {
        "layout": {
            "xaxis": {
                "visible": False
            },
            "yaxis": {
                "visible": False
            },
            "annotations": [
                {
                    "text": text,
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 28
                    }
                }
            ]
        }
    }


@cache.memoize()
def return_merged(start_date, end_date, counties, months, timerange, days):
    start_date = dateparser.parse(start_date)
    end_date = dateparser.parse(end_date)
    '''
    print('start_date:',type(start_date),start_date)
    print('alarm_datetime:', type(
        df[['alarm_datetime']]), df['alarm_datetime'].iloc[0])
    print(df['alarm_datetime'] >= start_date)
    print(len(df))
    '''
    # try:

    if start_date is None or end_date is None:
        return empty_fig()

    if True:
        date_condition = ((df_merged['time_local'] >= start_date)
                          & (df_merged['time_local'] <= end_date))
        if days is None or len(days) == 0:
            weekday_condition = (True)
        else:
            weekday_condition = ((df_merged['day_of_week']).isin(days))
        if months is None or len(months) == 0:
            month_condition = True
        else:
            month_condition = ((df_merged['month'].isin(months)))
        timemin, timemax = timerange
        hourmax = int(timemax)
        hourmin = int(timemin)

        if counties is None or len(counties) == 0:
            county_condition = True
        else:
            county_condition = ((df_merged['county'].isin(counties)))
        starttime = tt(hour=hourmin)
        if hourmax != 24:
            endtime = tt(hour=max(0, hourmax-1), second=59)
        else:
            endtime = tt(hour=23, second=59)
        timecondition = ((df_merged['time_local'].dt.time >= starttime)
                         & (df_merged['time_local'].dt.time <= endtime))
        result = df_merged[timecondition & date_condition &
                           month_condition & weekday_condition & county_condition]
    else:
        print("Exception in user code:")
        traceback.print_exc(file=sys.stdout)

    return result


# county-selector-div,
# cluster-selector-div,
# month-selector-div,
# day-of-week-selector-div,
# feature-selector,
# time-slider-div,
# radius-div,
# incidents-count-div,
# maps-tabs-div,
# histogram-basis-div,
# footer-div,


@app.callback(
    [Output('container-div', "className"),
     Output('initial-div', "className"),
     Output('start-date-div', "className"),
     Output('end-date-div', "className"),
     Output('county-selector-div', "className"),
     Output('cluster-selector-div', "className"),
     Output('month-selector-div', "className"),
     Output('day-of-week-selector-div', "className"),
     Output('feature-selector', "className"),
     Output('time-slider-div', "className"),
     Output('radius-div', "className"),
     Output('incidents-count-div', "className"),
     Output('maps-tabs-div', "className"),
     Output('histogram-basis-div', "className"),
     Output('footer-div', "className"),
     Output('footer', "className"),
     ],
    [Input("theme-toggle", "on")]
)
def update_theme(dark_theme):
    if dark_theme:
        return "container-fluid bg-dark text-white",\
            "p-0 m-0 bg-dark row",\
            "card p-1 m-1 bg-dark text-white",\
            "card p-1 m-1 bg-dark text-white",\
            "card p-1 m-1 bg-dark text-white",\
            "card p-1 m-1 bg-dark text-white",\
            "card p-1 m-1 btn-group bg-dark text-white",\
            "card p-1 m-1 bg-dark text-white",\
            "card p-1 m-1 bg-dark text-white",\
            "card p-1 m-1 bg-dark text-white",\
            "card p-1 m-1 bg-dark text-white",\
            "card p-1 m-1 bg-dark text-white",\
            "p-0 m-0 card  bg-dark text-white",\
            "p-0 m-0 card bg-dark text-white",\
            "row p-0 m-0 bg-dark text-white",\
            "col p-0 m-0 bg-dark text-white"
    else:
        return "container-fluid bg-white text-dark",\
            "p-0 m-0 bg-white row",\
            "card p-1 m-1 bg-white text-dark",\
            "card p-1 m-1 bg-white text-dark",\
            "card p-1 m-1 bg-white text-dark",\
            "card p-1 m-1 bg-white text-dark",\
            "card p-1 m-1 btn-group bg-white text-dark",\
            "card p-1 m-1 bg-white text-dark",\
            "card p-1 m-1 bg-white text-dark",\
            "card p-1 m-1 bg-white text-dark",\
            "card p-1 m-1 bg-white text-dark",\
            "card p-1 m-1 bg-white text-dark",\
            "p-0 m-0 card  bg-white text-dark",\
            "p-0 m-0 card bg-white text-dark",\
            "row p-0 m-0 bg-white text-dark",\
            "col p-0 m-0 bg-white text-dark"


@app.callback(
    [Output('incident-text', "children"), Output('month-text', "children"), Output('time-text', "children"), Output('response-text', "children"), Output(component_id='response-text',
                                                                                                                                                         component_property='style'), Output(component_id='time-text', component_property='style'), Output(component_id='month-text', component_property='style')],
    [Input("date-picker", "date"),
     Input("date-picker-end", "date"),
     Input('county', 'value'),
     Input("month-selector", "value"),
     Input("time-slider", "value"),  Input("day-selector", "value"),
     ]
)
def update_incidents(start_date, end_date, counties, datemonth, timerange,   days):

    result = return_incidents(
        start_date, end_date, counties, datemonth, timerange,   days)
    
    timemin, timemax = timerange
    responsefilter = -1
    if result is None or len(result) == 0:
        return "Incidents: %d" % (0), "Months: %s" % (str(datemonth)), "Time %s:00 to %s:00" % (timerange[0], timerange[1]), "Response Time >%s minutes" % (str(responsefilter)), ({'display': 'none'}, {'display': 'block', 'text-align': 'left', 'font-weight': 'bold'})[responsefilter > 0], ({'display': 'none'}, {'display': 'block', 'text-align': 'left', 'font-weight': 'bold'})[timemin > 0 or timemax < 24], ({'display': 'none'}, {'display': 'block', 'text-align': 'left', 'font-weight': 'bold'})[datemonth is not None and len(datemonth) != 0]
   

    return "Incidents: %d" % (len(result)), "Months: %s" % (str(datemonth)), "Time %s:00 to %s:00" % (timerange[0], timerange[1]), "Response Time >%s minutes" % (str(responsefilter)), ({'display': 'none'}, {'display': 'block', 'text-align': 'left', 'font-weight': 'bold'})[responsefilter > 0], ({'display': 'none'}, {'display': 'block', 'text-align': 'left', 'font-weight': 'bold'})[timemin > 0 or timemax < 24], ({'display': 'none'}, {'display': 'block', 'text-align': 'left', 'font-weight': 'bold'})[datemonth is not None and len(datemonth) != 0]


cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "gyr", [[0., 'green'], [0.5, 'yellow'], [1.0, 'red']], N=101)
discr_map = {}
for i in range(0, 101, 1):
    discr_map.update(
        {i: rgb2hex(int(255 * cmap(i)[0]), int(255 * cmap(i)[1]), int(255 * cmap(i)[2]))})


def plotly_linestring(linestring, grouped_xdsegid, prediction):
    colorval = discr_map[int(prediction*100)]
    return go.Scattermapbox(
        lat=np.array(np.array(linestring.coords)[:, 1]),
        lon=np.array(np.array(linestring.coords)[:, 0]),
        mode='lines',
        name="group {}".format(grouped_xdsegid),
        line={'color': colors.to_hex(colorval, keep_alpha=False), 'width': 4},
        text="likelihood {0}".format(prediction)
    )


def create_prediction_frame(pred):
    figures = []
    for feature, name, prediction in zip(pred.geometry, pred.grouped_xdsegid, pred.RF):
        if isinstance(feature, shapely.geometry.linestring.LineString):
            linestrings = [feature]
        elif isinstance(feature, shapely.geometry.multilinestring.MultiLineString):
            linestrings = feature.geoms
        else:
            continue
        for linestring in linestrings:
            figures.append(plotly_linestring(linestring, name, prediction))
    return figures


@cache.memoize()
@app.callback(
    Output('map-incidents-predictions', 'figure'),
    [Input("date-picker", "date"),
     Input("date-picker-end", "date"),
     Input("map-graph-radius", "value"),
     Input('county', 'value'),
     Input("month-selector", "value"),
     Input("time-slider", "value"),   Input("day-selector", "value"), Input("theme-toggle", "on"), ]
)
def update_map_incident_predictions(start_date, end_date, radius, counties, datemonth, timerange,   days, darktheme):
    start_date2 = dateparser.parse(start_date)
    end_date2 = dateparser.parse(end_date)

    if start_date2 is None or end_date2 is None:
        return empty_fig()

    # print(end_date2-start_date2)

    if (end_date2-start_date2).days >= 7 or len(counties) != 1:
        return empty_fig("Select a period of less than 7 days and select one county")

    result = return_predictions(
        start_date, end_date, counties, datemonth, timerange, days).copy()

    latone = latInitial
    lonone = lonInitial
    zoomvalue = 9
    if counties == None or len(counties) != 1:
        latone = latInitial
        lonone = lonInitial
        zoomvalue = 6
    else:
        # find center of one county
        onecounty = counties[0]
        try:
            latone = tncounties[tncounties.county ==
                                onecounty].latitude.iloc[0]
            lonone = tncounties[tncounties.county ==
                                onecounty].longitude.iloc[0]
        except:
            pass

    if result is None or len(result) == 0:
        return empty_fig()

    # result['lat']=result.geometry.apply(returnlat)
    # result['lon']=result.geometry.apply(returnlon)
    # result['color']=result.RF.apply(lambda x: int(x*100))
    result['time_local'] = result['time_local'].apply(str)
    # result=result.drop('geometry',axis=1)
    # result=result.explode(['lat','lon'])
    time_local = result['time_local'].unique().tolist()
    # for k in time_local:
    #     print(k)

    fig = go.Figure(data=create_prediction_frame(result[result.time_local == time_local[0]]),
                    frames=[go.Frame(data=create_prediction_frame(result[result.time_local == k]))
                            for k in time_local]
                    )

    # fig = px.line_mapbox(result,lat='lat', lon='lon', hover_name='RF',color='color',
    #                     line_group='grouped_xdsegid',color_discrete_map=discr_map,
    #                  mapbox_style="stamen-terrain",  zoom=11,animation_frame='time_local')

    fig.update_layout(autosize=True,
                      margin=go.layout.Margin(l=0, r=35, t=0, b=0),
                      showlegend=False,
                      mapbox=dict(
                          accesstoken=mapbox_access_token,
                          center=dict(lat=latone, lon=lonone),
                          style="dark" if darktheme else "light",
                          bearing=0,
                          zoom=zoomvalue,
                      ),
                      updatemenus=[
                          dict(
                              buttons=(
                                  [
                                      dict(
                                          args=[
                                              {
                                                  "mapbox.zoom": zoomvalue,
                                                  "mapbox.center.lon": lonone,
                                                  "mapbox.center.lat": latone,
                                                  "mapbox.bearing": 0,
                                                  "mapbox.style": "dark" if darktheme else "light",
                                              }
                                          ],
                                          label="Reset Zoom",
                                          method="relayout",
                                      ),
                                      dict(label="Play",
                                           method="animate",
                                           args=[None, {'frame': {'duration': 2000}}]),
                                      {
                                          "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                                            "mode": "immediate",
                                                            "transition": {"duration": 0}}],
                                          "label": "Pause",
                                          "method": "animate"
                                      }
                                  ]
                              ),
                              direction="left",
                              pad={"r": 0, "t": 0, "b": 0, "l": 0},
                              showactive=False,
                              type="buttons",
                              x=0.45,
                              y=0.02,
                              xanchor="left",
                              yanchor="bottom",
                              bgcolor="#1E1E1E" if darktheme else "#f8f9fa",
                              borderwidth=1,
                              bordercolor="#6d6d6d",
                              font=dict(
                                  color="white" if darktheme else "#1E1E1E"),
                          ),
                      ],
                      )

    #fig.layout.updatemenus[0].buttons[1].args[1]["frame"]["duration"] = 4000

    # fig['layout']['updatemenus'][0]['pad'] = dict(r=0, t=0)
    # fig["layout"]["updatemenus"] = [
    #     {
    #         "buttons": [
    #             {
    #                 "args": [
    #                     {
    #                         "mapbox.zoom": zoomvalue,
    #                         "mapbox.center.lon": lonone,
    #                         "mapbox.center.lat": latone,
    #                         "mapbox.bearing": 0,
    #                         "mapbox.style": "dark" if darktheme else "light",
    #                     }
    #                 ],
    #                 "label": "Reset Zoom",
    #                 "method": "relayout"

    #             },
    #             {
    #                 "args": [None, {"fromcurrent": True}],
    #                 "label": "Play",
    #                 "method": "animate"
    #             },
    #             {
    #                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
    #                                   "mode": "immediate",
    #                                   "transition": {"duration": 0}}],
    #                 "label": "Pause",
    #                 "method": "animate"
    #             }
    #         ],
    #         "direction": "left",
    #         "pad": {"r": 0, "t": 0, "b": 0, "l": 0},
    #         "showactive": False,
    #         "type": "buttons",
    #         "x": 0.45,
    #         "xanchor": "left",
    #         "y": 0.02,
    #         "yanchor": "bottom",
    #         "bgcolor":  "#1E1E1E" if darktheme else "#f8f9fa",
    #         "borderwidth": 1,
    #         "bordercolor": "#6d6d6d",
    #         "font": {"color": "white" if darktheme else "#1E1E1E"}
    #     }
    # ]
    for k in range(len(fig.frames)):
        fig.frames[k]['layout'].update(
            title_text=f'<b>{str(time_local[k])}</b> <br> Only top 20 percent shown.')
        fig.frames[k]['layout'].update(title_x=0.1)
        fig.frames[k]['layout'].update(
            title_font=dict(family='Arial Black', size=18, color="white" if darktheme else "#1E1E1E"))
        fig.frames[k]['layout'].update(title_y=0.92)
        fig.frames[k]['layout'].update(title_yanchor="bottom")
        fig.frames[k]['layout'].update(title_xanchor="left")
    return fig


@cache.memoize()
@app.callback(
    Output('map-graph', 'figure'),
    [Input("date-picker", "date"),
     Input("date-picker-end", "date"),
     Input("map-graph-radius", "value"),
     Input('county', 'value'),
     Input("month-selector", "value"),
     Input("time-slider", "value"),   Input("day-selector", "value"), Input("theme-toggle", "on"), ]
)
def update_map_graph(start_date, end_date, radius, counties, datemonth, timerange,   days, darktheme):
    result = return_incidents(
        start_date, end_date, counties, datemonth, timerange, days)

    if result is None or len(result) == 0:
        return empty_fig()
    latone = result['latitude'].iloc[0]
    lonone = result['longitude'].iloc[0]
    zoomvalue = 10

    if counties == None or len(counties) != 1:
        latone = latInitial
        lonone = lonInitial
        zoomvalue = 6
    else:
        # find center of one county
        onecounty = counties[0]
        try:
            latone = tncounties[tncounties.county ==
                                onecounty].latitude.iloc[0]
            lonone = tncounties[tncounties.county ==
                                onecounty].longitude.iloc[0]
        except:
            pass

    fig = px.density_mapbox(result, lat="latitude", lon="longitude",  hover_data=['incidentNumber'],
                            mapbox_style="open-street-map", radius=radius)
    fig.update_layout(
        autosize=True,
        margin=go.layout.Margin(l=0, r=35, t=0, b=0),
        showlegend=False,
        mapbox=dict(
            accesstoken=mapbox_access_token,
            center=dict(lat=latone, lon=lonone),
            style="dark" if darktheme else "light",
            bearing=0,
            zoom=zoomvalue,
        ),
        updatemenus=[
            dict(
                buttons=(
                    [
                        dict(
                            args=[
                                {
                                    "mapbox.zoom": zoomvalue,
                                    "mapbox.center.lon": lonone,
                                    "mapbox.center.lat": latone,
                                    "mapbox.bearing": 0,
                                    "mapbox.style": "dark" if darktheme else "light",
                                }
                            ],
                            label="Reset Zoom",
                            method="relayout",
                        )
                    ]
                ),
                direction="left",
                pad={"r": 0, "t": 0, "b": 0, "l": 0},
                showactive=False,
                type="buttons",
                x=0.45,
                y=0.02,
                xanchor="left",
                yanchor="bottom",
                bgcolor="#1E1E1E" if darktheme else "#f8f9fa",
                borderwidth=1,
                bordercolor="#6d6d6d",
                font=dict(color="white" if darktheme else "#1E1E1E"),
            ),
        ],
    )
    return fig


@cache.memoize()
@app.callback(
    Output('map-incidents-month', 'figure'),
    [Input("date-picker", "date"),
     Input("date-picker-end", "date"),
     Input("map-graph-radius", "value"),
     Input('county', 'value'),
     Input("month-selector", "value"),
     Input("time-slider", "value"), Input("day-selector", "value"), Input("theme-toggle", "on"), ]
)
def update_map_incidents_month(start_date, end_date, radius, counties, datemonth, timerange,   days, darktheme):

    result = return_incidents(
        start_date, end_date, counties, datemonth, timerange,   days)
    if result is None or len(result) == 0:
        return empty_fig()
    monthlist = result['month-year'].unique().tolist()

    if len(monthlist) <= 1:
        return empty_fig("Not available with only one month selection")

    if result is None or len(result) == 0:
        return empty_fig()

    latone = result['latitude'].iloc[0]
    lonone = result['longitude'].iloc[0]
    zoomvalue = 10

    if counties == None or len(counties) != 1:
        latone = latInitial
        lonone = lonInitial
        zoomvalue = 6
    else:
        # find center of one county
        onecounty = counties[0]
        try:
            latone = tncounties[tncounties.county ==
                                onecounty].latitude.iloc[0]
            lonone = tncounties[tncounties.county ==
                                onecounty].longitude.iloc[0]
        except:
            pass
    fig = px.density_mapbox(result, animation_frame='month-year', lat="latitude", lon="longitude",  hover_data=['incidentNumber'],
                            mapbox_style="open-street-map", radius=radius)
    fig.update_layout(autosize=True,
                      margin=go.layout.Margin(l=0, r=35, t=0, b=0),
                      showlegend=False,
                      mapbox=dict(
                          accesstoken=mapbox_access_token,
                          center=dict(lat=latone, lon=lonone),
                          style="dark" if darktheme else "light",
                          bearing=0,
                          zoom=zoomvalue,
                      ),
                      updatemenus=[
                          dict(
                              buttons=(
                                  [
                                      dict(
                                          args=[
                                              {
                                                  "mapbox.zoom": zoomvalue,
                                                  "mapbox.center.lon": lonone,
                                                  "mapbox.center.lat": latone,
                                                  "mapbox.bearing": 0,
                                                  "mapbox.style": "dark" if darktheme else "light",
                                              }
                                          ],
                                          label="Reset Zoom",
                                          method="relayout",
                                      ),
                                      dict(label="Play",
                                           method="animate",
                                           args=[None]),
                                  ]
                              ),
                              direction="left",
                              pad={"r": 0, "t": 0, "b": 0, "l": 0},
                              showactive=False,
                              type="buttons",
                              x=0.45,
                              y=0.02,
                              xanchor="left",
                              yanchor="top",
                              bgcolor="#1E1E1E" if darktheme else "#f8f9fa",
                              borderwidth=1,
                              bordercolor="#6d6d6d",
                              font=dict(
                                  color="white" if darktheme else "#1E1E1E"),
                          ),
                      ],
                      )
    fig['layout']['updatemenus'][0]['pad'] = dict(r=0, t=0)
    fig['layout']['sliders'][0]['pad'] = dict(r=0, t=0, b=0, l=0)
    fig['layout']['sliders'][0]['bgcolor'] = "white" if darktheme else "#f8f9fa"
    fig["layout"].pop("sliders")
    fig["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [
                        {
                            "mapbox.zoom": zoomvalue,
                            "mapbox.center.lon": lonone,
                            "mapbox.center.lat": latone,
                            "mapbox.bearing": 0,
                            "mapbox.style": "dark" if darktheme else "light",
                        }
                    ],
                    "label": "Reset Zoom",
                    "method": "relayout"


                },
                {
                    "args": [None, {"fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 0, "t": 0, "b": 0, "l": 0},
            "showactive": False,
            "type": "buttons",
            "x": 0.45,
            "xanchor": "left",
            "y": 0.02,
            "yanchor": "bottom",
            "bgcolor":  "#1E1E1E" if darktheme else "#f8f9fa",
            "borderwidth": 1,
            "bordercolor": "#6d6d6d",
            "font": {"color": "white" if darktheme else "#1E1E1E"}
        }
    ]
    for k in range(len(fig.frames)):
        fig.frames[k]['layout'].update(title_text=f'<b>{monthlist[k]}</b>')
        fig.frames[k]['layout'].update(title_x=0.1)
        fig.frames[k]['layout'].update(
            title_font=dict(family='Arial Black', size=18, color="white" if darktheme else "#1E1E1E"))
        fig.frames[k]['layout'].update(title_y=0.92)
        fig.frames[k]['layout'].update(title_yanchor="bottom")
        fig.frames[k]['layout'].update(title_xanchor="left")
    return fig


@cache.memoize()
def hourhist(result, datemonth, darktheme):
    result['hour'] = result['time_local'].dt.hour
    result = result.groupby(['hour']).count().reset_index()
    result['count'] = result['incidentNumber']
    result['h'] = result['hour'].astype(int)
    xVal = result['hour']
    yVal = result['count'].fillna(0)
    colorVal = ["#2202d1"]*25
    layout = go.Layout(
        bargap=0.05,
        autosize=True,
        bargroupgap=0,
        barmode="group",
        margin=go.layout.Margin(l=10, r=0, t=0, b=30),
        showlegend=False,
        plot_bgcolor="#1E1E1E" if darktheme else "#f8f9fa",
        paper_bgcolor="#1E1E1E" if darktheme else "#f8f9fa",
        title_font_color="white" if darktheme else "#1E1E1E",
        dragmode="select",
        font_color="white" if darktheme else "#1E1E1E",
        font=dict(color="white" if darktheme else "#1E1E1E"),
        xaxis=dict(
            range=[-1, 25],
            showgrid=False,
            tickvals=[x for x in range(0, 24)],
            ticktext=['12 AM', '1 AM', '2 AM', '3 AM', '4 AM', '5 AM', '6 AM', '7 AM', '8 AM', '9 AM', '10 AM', '11 AM', '12 PM', '1 PM', '2 PM',
                      '3 PM', '4 PM', '5 PM', '6 PM', '7 PM', '8 PM', '9 PM', '10 PM', '11 PM'],
            fixedrange=True,
            ticksuffix="",
        ),
        yaxis=dict(
            range=[0, max(yVal) + max(yVal) / 4],
            showticklabels=False,
            showgrid=True,
            fixedrange=True,
            rangemode="nonnegative",
            zeroline=False,
        ),
        annotations=[
            dict(
                x=xi,
                y=yi,
                text=(str((yi))) if yi != 0 else "",
                xanchor="center",
                yanchor="top",
                showarrow=False,
                font=dict(color="white"),
            )
            for xi, yi in zip(xVal, yVal)
        ],
    )

    return go.Figure(
        data=[
            go.Bar(x=xVal, y=yVal, marker=dict(
                color=np.array(colorVal)), hoverinfo="x"),
        ],
        layout=layout,
    )


@cache.memoize()
def dayhist(result, datemonth, darktheme):
    result = result.groupby(['day_of_week']).count().reset_index()
    result['count'] = result['incidentNumber']
    colorVal = ["#2202d1"]*25
    xVal = result['day_of_week']
    yVal = result['count'].fillna(0)
    layout = go.Layout(
        bargap=0.05,
        autosize=True,
        bargroupgap=0,
        barmode="group",
        margin=go.layout.Margin(l=10, r=0, t=0, b=30),
        showlegend=False,
        plot_bgcolor="#1E1E1E" if darktheme else "#f8f9fa",
        paper_bgcolor="#1E1E1E" if darktheme else "#f8f9fa",
        dragmode="select",
        font=dict(color="white" if darktheme else "#1E1E1E"),
        xaxis=dict(
            range=[-1, 8],
            showgrid=False,
            tickvals=[0, 1, 2, 3, 4, 5, 6],
            ticktext=['Mon', 'Tues', 'Wed', 'Thur', 'Friday', 'Sat', 'Sun'],
            fixedrange=True,
            ticksuffix="",
        ),
        yaxis=dict(
            range=[0, max(yVal) + max(yVal) / 4],
            showticklabels=False,
            showgrid=True,
            fixedrange=True,
            rangemode="nonnegative",
            zeroline=False,
        ),
        annotations=[
            dict(
                x=xi,
                y=yi,
                text=(str((yi))) if yi != 0 else "",
                xanchor="center",
                yanchor="top",
                showarrow=False,
                font=dict(color="white"),
            )
            for xi, yi in zip(xVal, yVal)
        ],
    )

    return go.Figure(
        data=[
            go.Bar(x=xVal, y=yVal, marker=dict(
                color=np.array(colorVal)), hoverinfo="x"),
        ],
        layout=layout,
    )


@cache.memoize()
def monthhist(result, datemonth, darktheme):
    colorVal = ["#2202d1"]*25
    result = result.groupby(['month']).count().reset_index()
    result['count'] = result['incidentNumber']
    result['m'] = result['month'].astype(int)
    monthindex = range(1, 13)
    for month in list(range(1, 13)):
        if month not in result['m'].values:
            new_row = {'m': month, 'month': str(month), 'count': 0}
            result = result.append(new_row, ignore_index=True)
    result = result.sort_values('m')
    xVal = result['month']
    yVal = result['count'].fillna(0)
    layout = go.Layout(
        bargap=0.1,
        bargroupgap=0,
        barmode="group",
        margin=go.layout.Margin(l=10, r=0, t=0, b=30),
        showlegend=False,
        plot_bgcolor="#1E1E1E" if darktheme else "#f8f9fa",
        paper_bgcolor="#1E1E1E" if darktheme else "#f8f9fa",
        dragmode="select",
        font=dict(color="white" if darktheme else "#1E1E1E"),
        xaxis=dict(
            range=[0, 13],
            showgrid=False,
            tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            ticktext=['Jan', 'Feb', 'March', 'Apr', 'May', 'June',
                      'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            fixedrange=True,
            ticksuffix="",
        ),
        yaxis=dict(
            range=[0, max(yVal) + max(yVal) / 4],
            showticklabels=False,
            showgrid=True,
            fixedrange=True,
            rangemode="nonnegative",
            zeroline=False,
        ),
        annotations=[
            dict(
                x=xi,
                y=yi,
                text=(str((yi))) if yi != 0 else "",
                xanchor="center",
                yanchor="top",
                showarrow=False,
                font=dict(color="white"),
            )
            for xi, yi in zip(xVal, yVal)
        ],
    )

    return go.Figure(
        data=[
            go.Bar(x=xVal, y=yVal, marker=dict(
                color=np.array(colorVal)), hoverinfo="x"),
        ],
        layout=layout,
    )


@cache.memoize()
def incidentsfeature(df_merged, feature, darktheme):
    colorVal = ["#2202d1"]*25
    feature_cat = feature+'_cat_'
    df_merged_ = categorize_numerical_features(
        df_merged[[feature, metadata['pred_name_TF']]])
    df_incidentcount = df_merged_[[metadata['pred_name_TF'], feature_cat]].groupby(
        feature_cat).agg(['mean', 'sum'])
    df_incidentcount.columns = ['mean', 'sum']
    df_incidentcount = df_incidentcount.reset_index()
    xVal = df_incidentcount[feature_cat].tolist()
    yVal = df_incidentcount['mean'].fillna(0).tolist()

    if feature == 'window':
        for index in range(len(xVal)):
            x = xVal[index]
            if x == "0":
                xVal[index] = "12AM-4AM"
            elif x == "1":
                xVal[index] = "4AM-8AM"
            elif x == "2":
                xVal[index] = "8AM-12PM"
            elif x == "3":
                xVal[index] = "12PM-4PM"
            elif x == "4":
                xVal[index] = "4PM-8PM"
            elif x == "5":
                xVal[index] = "8PM-12AM"

    if feature == 'lanes':
        for index in range(len(xVal)):
            x = xVal[index]
            if x == "0":
                xVal[index] = "Zero"
            elif x == "1":
                xVal[index] = "One"
            elif x == "2":
                xVal[index] = "Two"
            elif x == "3":
                xVal[index] = "Three"
            elif x == "4":
                xVal[index] = "Four"
            elif x == "5":
                xVal[index] = "Five"
            elif x == "6":
                xVal[index] = "Six"
            elif x == "7":
                xVal[index] = "Seven"

    layout = go.Layout(
        bargap=0.1,
        bargroupgap=0,
        barmode="group",
        margin=go.layout.Margin(l=10, r=0, t=10, b=10),
        showlegend=False,
        plot_bgcolor="#1E1E1E" if darktheme else "#f8f9fa",
        paper_bgcolor="#1E1E1E" if darktheme else "#f8f9fa",
        dragmode="select",
        font=dict(color="white" if darktheme else "#1E1E1E"),
        yaxis=dict(
            range=[0, max(yVal) + max(yVal) / 4],
            showticklabels=False,
            showgrid=True,
            fixedrange=True,
            rangemode="nonnegative",
            zeroline=False,
        ),
        annotations=[
            dict(
                x=xi,
                y=yi,
                text=(str("{:.3f}".format(yi))) if yi != 0 else "",
                xanchor="center",
                yanchor="top",
                showarrow=False,
                font=dict(color="white"),
            )
            for xi, yi in zip(xVal, yVal)
        ],
    )

    return go.Figure(
        data=[
            go.Bar(x=xVal, y=yVal, marker=dict(
                color=np.array(colorVal))),  # , hoverinfo="x"
        ],
        layout=layout,
    )


@cache.memoize()
def incidentsfeaturecombo(result, datemonth, darktheme):
    colorVal = ["#2202d1"]*25
    All_Possible_Dic, All_Possible_Filters = Filter_Combo_Builder()
    result = categorize_numerical_features(
        result[['incident_occurred', 'congestion_mean', 'temp_mean', 'precip_mean', 'is_weekend']])
    result = FILTER_calculator(
        result[['incident_occurred'] + list(All_Possible_Filters.keys())], All_Possible_Dic, 'All')
    result = result.rename(
        columns={'Average_Number_Incidents': 'Comparitive Likelihood', "No_Filter": "Baseline"})
    # drawing

    xVal = result['Tag']
    yVal = result['Comparitive Likelihood'].fillna(0)

    layout = go.Layout(
        bargap=0.1,
        bargroupgap=0,
        barmode="group",
        margin=go.layout.Margin(l=10, r=0, t=10, b=10),
        showlegend=False,
        plot_bgcolor="#1E1E1E" if darktheme else "#f8f9fa",
        paper_bgcolor="#1E1E1E" if darktheme else "#f8f9fa",
        dragmode="select",
        font=dict(color="white" if darktheme else "#1E1E1E"),

        yaxis=dict(
            range=[0, max(yVal) + max(yVal) / 4],
            showticklabels=False,
            showgrid=True,
            fixedrange=True,
            rangemode="nonnegative",
            zeroline=False,
        ),
        annotations=[
            dict(
                x=xi,
                y=yi,
                text=(str("{:.3f}".format(yi))) if yi != 0 else "",
                xanchor="center",
                yanchor="top",
                showarrow=False,
                font=dict(color="white"),
            )
            for xi, yi in zip(xVal, yVal)
        ],
    )

    return go.Figure(
        data=[
            go.Bar(x=xVal, y=yVal, marker=dict(
                color=np.array(colorVal))),  # , hoverinfo="x"
        ],
        layout=layout,
    )


@cache.memoize()
def incidenttotals(result, datemonth, darktheme):
    resultgroup = result.groupby(
        'month-year').agg({'incidentNumber': 'count', 'time_local': 'first'})
    resultgroup = resultgroup.reset_index()
    resultgroup = resultgroup.sort_values(by=['time_local'])
    fig = px.line(resultgroup, x="month-year",
                  y="incidentNumber", title='Total Incidents')
    fig.update_xaxes(
        showgrid=True
    )
    fig.update_yaxes(
        showgrid=False
    )
    fig.update_layout(plot_bgcolor="#1E1E1E" if darktheme else "#f8f9fa", yaxis_title_text='Count', margin=go.layout.Margin(
        l=10, r=0, t=0, b=30), paper_bgcolor="#1E1E1E" if darktheme else "#f8f9fa", font=dict(color="white" if darktheme else "#1E1E1E"))

    return fig

# %%


@app.callback(
    Output('histogram', 'figure'),
    [Input("date-picker", "date"),
     Input("date-picker-end", "date"),
     Input('county', 'value'), Input("month-selector", "value"), Input("histogram-basis",
                                                                       "active_tab"), Input("time-slider", "value"),  Input("day-selector", "value"),
     Input('feature', 'value'),
     Input("theme-toggle", "on"),
     ]
)
def update_bar_chart(start_date, end_date, counties, datemonth, histogramkind, timerange,   days, feature, darktheme):

    if histogramkind == "incidentsfeature" or histogramkind == "incidentsfeaturecombo":
        result = return_merged(start_date, end_date,
                               counties, datemonth, timerange, days)
    else:
        result = return_incidents(
            start_date, end_date, counties, datemonth, timerange,   days)

    if result is None or len(result) == 0:
        return empty_fig()

    if histogramkind == "month":
        return monthhist(result, datemonth, darktheme)
    elif histogramkind == "totals":
        return incidenttotals(result, datemonth, darktheme)
    elif histogramkind == "day":
        return dayhist(result, datemonth, darktheme)
    elif histogramkind == "hour":
        return hourhist(result, datemonth, darktheme)
    elif histogramkind == "incidentsfeature":
        return incidentsfeature(result, feature, darktheme)
    elif histogramkind == "incidentsfeaturecombo":
        return incidentsfeaturecombo(result, datemonth, darktheme)


# %%
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
    # app.server.run(threaded=True)

# %%
