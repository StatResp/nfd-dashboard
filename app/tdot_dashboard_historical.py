# %%
import traceback
import dateparser
from matplotlib import cm, colors
import dash
import flask
import pyarrow.parquet as pq
import os
#from flask_caching import Cache
from dask import dataframe as dd
from random import randint
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.express as px
from dash.dependencies import Input, Output
from plotly import graph_objs as go
from plotly.graph_objs import *
from datetime import datetime as dt
from datetime import time as tt
#import dataextract
import os
import sys
#import resource
import geopandas as gpd
import dash_bootstrap_components as dbc
from statresp.datajoin.cleaning.categorizer import categorize_numerical_features, Filter_Combo_Builder, FILTER_calculator
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os


# set configurations
mapbox_style = "light"
mapbox_access_token = "pk.eyJ1Ijoidmlzb3ItdnUiLCJhIjoiY2tkdTZteWt4MHZ1cDJ4cXMwMnkzNjNwdSJ9.-O6AIHBGu4oLy3dQ3Tu2XA"
MAPBOX_ACCESS_TOKEN = mapbox_access_token
pd.set_option('display.max_columns', None)

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
df_merged = pd.read_parquet(metadata['merged_pickle_address'],
                            columns=available_features+['time_local', 'county', 'incident_occurred'])
if metadata['incident_pickle_address'][-4:] == '.pkl':
    df = pd.read_pickle(metadata['incident_pickle_address'])
else:
    df = pd.read_parquet(metadata['incident_pickle_address'])

df = df[df['frc'] == 0]
# df=df.iloc[0:1000]
df['month-year'] = df['time_local'].dt.strftime('%b %Y')
df = df.rename(columns={'gps_coordinate_longitude': 'longitude',
                        'gps_coordinate_latitude': 'latitude', 'incident_id': 'incidentNumber'})

startdate = df['time_local'].dt.date.min()
enddate = df['time_local'].dt.date.max()
counties = df.county_inrix.drop_duplicates()
tncounties = pd.read_csv('counties.csv')
tncounties['county'] = tncounties['county'].str.lower()

counties = counties.tolist()
counties.sort()
cluster_list = [1, 2]

# set the server and global parameters
server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
app = dash.Dash(__name__, title='Incident Dashboard', update_title=None, external_stylesheets=[dbc.themes.BOOTSTRAP], server=server, meta_tags=[
                {"name": "viewport", "content": "width=device-width"}])
app.title = 'Incident Dashboard'

# cache = Cache(app.server,
#               config=dict(CACHE_TYPE='filesystem', CACHE_DEFAULT_TIMEOUT=10000, CACHE_DIR='cache-directory'))


mapbox_access_token = "pk.eyJ1Ijoidmlzb3ItdnUiLCJhIjoiY2tkdTZteWt4MHZ1cDJ4cXMwMnkzNjNwdSJ9.-O6AIHBGu4oLy3dQ3Tu2XA"
px.set_mapbox_access_token(mapbox_access_token)

# setup the app frameworks and main layout
# Layout of Dash App
app.layout = html.Div(className="container-fluid bg-dark text-white", children=[html.Div(
    children=[
        html.Div(
            className="row",
            children=[
                # Column for user controls
                html.Div(
                    className="col-12 col-lg-3",
                    children=[
                        dcc.Markdown(
                            '''# [Statresp.ai](https://statresp.ai) | TN Highway Incident Dashboard'''),
                        html.Div(
                            className="card p-0 m-0 bg-dark", style={"border": "none"},
                            children=[
                                dcc.Markdown(
                                    '''Configure the options below for filtering the data.'''),
                            ],
                        ),
                        html.Div(className="card p-1 m-1 bg-dark", children=[
                            dcc.Markdown(
                                '''  # Start Date''', className="p-0 m-0", style={"margin": "0", "padding": "0"}),
                            dcc.DatePickerSingle(
                                id="date-picker",
                                min_date_allowed=startdate,
                                max_date_allowed=enddate,
                                initial_visible_month=startdate,
                                date=startdate,
                                display_format="MMMM D, YYYY",
                                style={"border": "0px solid black"},
                            )],
                        ),

                        html.Div(
                            className="card p-1 m-1 bg-dark",
                            children=[
                                dcc.Markdown(
                                    '''## End Date''', style={"margin": "0", "padding": "0"}),
                                dcc.DatePickerSingle(
                                    id="date-picker-end",
                                    min_date_allowed=startdate,
                                    max_date_allowed=enddate,
                                    initial_visible_month=enddate,
                                    date=enddate,
                                    display_format="MMMM D, YYYY",
                                    style={"border": "0px solid black"},
                                )
                            ],
                        ),

                        html.Div(
                            className="card p-1 m-1 bg-dark",
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

                        html.Div(
                            className="card p-1 m-1 bg-dark", style={"display": "none"},
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
                        html.Div(
                            className="card p-1 m-1 btn-group bg-dark",
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

                                    # labelStyle={
                                    #     'display': 'inline-block', 'padding-left': '0.4em'}
                                )
                            ],
                        ),
                        html.Div(
                            className="card p-1 m-1 bg-dark",
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
                        html.Div(className="card p-1 m-1 bg-dark",
                                 children=[
                                     dcc.Markdown(
                                         '''## Incident Time'''),
                                     #html.P("""Select Time Range""",style={'text-align': 'left' ,'font-weight':'bold'}),
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

                        html.Div(className="card p-1 m-1 bg-dark",
                                 children=[
                                     dcc.Markdown(
                                         '''Adjust Slider below to configure the heatmap intensity.'''),
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
                        html.Div(className="card p-1 m-1 bg-dark", children=[html.P('Incidents', id='incident-text', style={'text-align': 'left', 'font-weight': 'bold'}),
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
                        html.Div(className="p-2 m-2 card  bg-dark",
                                 children=[
                                     dbc.Tabs(id='tabs', active_tab="incidents", children=[
                                         dbc.Tab(label='Incidents Total', tab_id='incidents', className="bg-dark text-white", children=[dcc.Loading(
                                             id="loading-icon1", children=[dcc.Graph(id="map-graph"), ], type='default')]),
                                         dbc.Tab(label='Incidents by Month', tab_id='incidents-month', className="bg-dark text-white", children=[dcc.Loading(
                                             id="loading-icon-incidents-month", children=[dcc.Graph(id="map-incidents-month"), ], type='default')]),

                                     ]
                                     ),
                                 ]),

                        html.Div(className="p-0 m-0 card bg-dark", children=[
                            dbc.Tabs(id='histogram-basis', active_tab="month", children=[
                                 dbc.Tab(label='Incident Frequency',
                                         tab_id='totals', className="bg-dark text-white"),
                                 dbc.Tab(label='Incidents by Month',
                                         tab_id='month', className="bg-dark text-white"),
                                 dbc.Tab(label='Incidents by Weekday',
                                         tab_id='day', className="bg-dark text-white"),
                                 dbc.Tab(label='Incidents by Time of Day',
                                         tab_id='hour', className="bg-dark text-white"),
                                 dbc.Tab(label='Incidents by Feature',
                                         tab_id='incidentsfeature', className="bg-dark text-white"),
                                 dbc.Tab(label='Incidents by Feature Combo',
                                         tab_id='incidentsfeaturecombo', className="bg-dark text-white"),
                                 ]),
                            dcc.Loading(id="loading-icon2", className="flex-grow-1",
                                        children=[
                                            html.Div(
                                                className="card p-1 m-1 bg-dark",style={"display": "none"},
                                                children=[
                                                    dcc.Markdown(
                                                        '''## Feature'''),
                                                    dcc.Dropdown(
                                                        options=[{'label': name, 'value': name}
                                                                 for name in available_features],
                                                        value='temp_mean',
                                                        id='feature',
                                                        multi=False,
                                                    ),
                                                ],
                                            ),
                                            dcc.Graph(id="histogram"), ], type='default'),
                        ]
                        ),

                    ],
                ),
            ],
        ), ],
),
    html.Div(className="row", children=[dcc.Markdown(""),
                                        dcc.Markdown('''Site designed by [ScopeLab](http://scopelab.ai/). Data source: TDOT. '''
                                                     '''Funding by TDOT and National Science Foundation.''', id='footer', className="col"), ]),
]
)


# Incident Filterings


def return_incidents(start_date, end_date, counties, months, timerange,   days):
    start_date = dateparser.parse(start_date)
    end_date = dateparser.parse(end_date)

    #print('start_date  :',type(start_date),start_date)
    # print('time_local  :', type(df[['time_local']]), df['time_local'].iloc[0])
    #print(df['time_local'] >= start_date)
    # print(len(df))

    # try:
    if True:
        date_condition = ((df['time_local'] >= start_date)
                          & (df['time_local'] <= end_date))
        # date_condition=True
        #print("set date condition")
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

        #print("going to set time condition")

        starttime = tt(hour=hourmin)
        if hourmax != 24:
            endtime = tt(hour=max(0, hourmax-1), second=59)
        else:
            endtime = tt(hour=23, second=59)
        timecondition = ((df['time_local'].dt.time >= starttime)
                         & (df['time_local'].dt.time <= endtime))
        # timecondition=True
        #result = df[timecondition & date_condition & month_condition & weekday_condition & county_condition].compute()
        result = df[timecondition & date_condition &
                    month_condition & weekday_condition & county_condition]
        result = result.sort_values(by=['time_local'])

    # except Exception:
    else:
        print("Exception in user code:")
        traceback.print_exc(file=sys.stdout)

    # print(result)
    return result


def return_merged(start_date, end_date, counties, months, timerange, days):
    start_date = dateparser.parse(start_date)
    end_date = dateparser.parse(end_date)
    '''
    print('start_date:',type(start_date),start_date)
    print('alarm_datetime:', type(df[['alarm_datetime']]), df['alarm_datetime'].iloc[0])
    print(df['alarm_datetime'] >= start_date)
    print(len(df))
    '''
    # try:
    if True:
        date_condition = ((df_merged['time_local'] >= start_date)
                          & (df_merged['time_local'] <= end_date))
        # date_condition=True
        # print("set date condition")
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

        # print("going to set time condition")

        starttime = tt(hour=hourmin)
        if hourmax != 24:
            endtime = tt(hour=max(0, hourmax-1), second=59)
        else:
            endtime = tt(hour=23, second=59)
        timecondition = ((df_merged['time_local'].dt.time >= starttime)
                         & (df_merged['time_local'].dt.time <= endtime))
        # timecondition=True
        # result = df[timecondition & date_condition & month_condition & weekday_condition & county_condition].compute()
        result = df_merged[timecondition & date_condition &
                           month_condition & weekday_condition & county_condition]

    # except Exception:
    else:
        print("Exception in user code:")
        traceback.print_exc(file=sys.stdout)

    # print(result)
    return result


# Process

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

    return "Incidents: %d" % (len(result)), "Months: %s" % (str(datemonth)), "Time %s:00 to %s:00" % (timerange[0], timerange[1]), "Response Time >%s minutes" % (str(responsefilter)), ({'display': 'none'}, {'display': 'block', 'text-align': 'left', 'font-weight': 'bold'})[responsefilter > 0], ({'display': 'none'}, {'display': 'block', 'text-align': 'left', 'font-weight': 'bold'})[timemin > 0 or timemax < 24], ({'display': 'none'}, {'display': 'block', 'text-align': 'left', 'font-weight': 'bold'})[datemonth is not None and len(datemonth) != 0]


def return_empty_fig():
    return {
        "layout": {
            "xaxis": {
                "visible": False
            },
            "plot_bgcolor": "#1E1E1E",
            "paper_bgcolor": "#1E1E1E",
            "yaxis": {
                "visible": False
            },
            "annotations": [
                {
                    "text": "No matching data found",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 28,
                        "color": "white"
                    }
                }
            ]
        }
    }


# %%
@app.callback(
    Output('map-graph', 'figure'),
    [Input("date-picker", "date"),
     Input("date-picker-end", "date"),
     Input("map-graph-radius", "value"),
     Input('county', 'value'),
     Input("month-selector", "value"),
     Input("time-slider", "value"),   Input("day-selector", "value")]
)
def update_map_graph(start_date, end_date, radius, counties, datemonth, timerange,   days):

    # if counties == None or len(counties) != 1:
    #     return {
    #         "layout": {
    #             "xaxis": {
    #                 "visible": False
    #             },
    #             "yaxis": {
    #                 "visible": False
    #             },
    #             "annotations": [
    #                 {
    #                     "text": "This map can be drawn for one and only one county.",
    #                     "xref": "paper",
    #                     "yref": "paper",
    #                     "showarrow": False,
    #                     "font": {
    #                         "size": 28
    #                     }
    #                 }
    #             ]
    #         }
    #     }

    result = return_incidents(
        start_date, end_date, counties, datemonth, timerange,   days)

    if len(result) == 0:
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
                        "text": "No data found.",
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
                            #  color="responsetime",range_color=[0,40], hover_data=['incidentNumber','latitude','longitude','alarm_datetime','responsetime'],color_continuous_scale=px.colors.sequential.Hot
                            mapbox_style="open-street-map", radius=radius)
    fig.update_layout(
        autosize=True,
        margin=go.layout.Margin(l=0, r=35, t=0, b=0),
        showlegend=False,
        mapbox=dict(
            accesstoken=mapbox_access_token,
            # 40.7272  # -73.991251
            center=dict(lat=latone, lon=lonone),
            style=mapbox_style,
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
                                    "mapbox.style": mapbox_style,
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
                bgcolor="#1E1E1E",
                borderwidth=1,
                bordercolor="#6d6d6d",
                font=dict(color="#FFFFFF"),
            ),
        ],
    )
    return fig


@app.callback(
    Output('map-incidents-month', 'figure'),
    [Input("date-picker", "date"),
     Input("date-picker-end", "date"),
     Input("map-graph-radius", "value"),
     Input('county', 'value'),
     Input("month-selector", "value"),
     Input("time-slider", "value"), Input("day-selector", "value")]
)
def update_map_incidents_month(start_date, end_date, radius, counties, datemonth, timerange,   days):

    result = return_incidents(
        start_date, end_date, counties, datemonth, timerange,   days)
    monthlist = result['month-year'].unique().tolist()

    if len(result) == 0:
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
                        "text": "No data found.",
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
                            #  color="responsetime",range_color=[0,40], hover_data=['incidentNumber','latitude','longitude','alarm_datetime','responsetime'],color_continuous_scale=px.colors.sequential.Hot
                            mapbox_style="open-street-map", radius=radius)
    fig.update_layout(autosize=True,
                      margin=go.layout.Margin(l=0, r=35, t=0, b=0),
                      showlegend=False,
                      mapbox=dict(
                          accesstoken=mapbox_access_token,
                          # 40.7272  # -73.991251
                          center=dict(lat=latone, lon=lonone),
                          style=mapbox_style,
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
                                                  "mapbox.style": mapbox_style,
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
                              yanchor="bottom",
                              bgcolor="#1E1E1E",
                              borderwidth=1,
                              bordercolor="#6d6d6d",
                              font=dict(color="#FFFFFF"),
                          ),
                      ],
                      )
    fig['layout']['updatemenus'][0]['pad'] = dict(r=0, t=0)
    fig['layout']['sliders'][0]['pad'] = dict(r=0, t=0, b=0, l=0)
    fig['layout']['sliders'][0]['bgcolor'] = "#1E1E1E"
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
                            "mapbox.style": mapbox_style,
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
            "bgcolor": "#1E1E1E",
            "borderwidth": 1,
            "bordercolor": "#6d6d6d",
            "font": {"color": "#FFFFFF"}
        }
    ]
    for k in range(len(fig.frames)):
        fig.frames[k]['layout'].update(title_text=f'<b>{monthlist[k]}</b>')
        fig.frames[k]['layout'].update(title_x=0.1)
        fig.frames[k]['layout'].update(
            title_font=dict(family='Arial Black', size=18))
        fig.frames[k]['layout'].update(title_y=0.92)
        fig.frames[k]['layout'].update(title_yanchor="bottom")
        fig.frames[k]['layout'].update(title_xanchor="left")
    return fig


def hourhist(result, datemonth):

    result['hour'] = result['time_local'].dt.hour
    result = result.groupby(['hour']).count().reset_index()
    result['count'] = result['incidentNumber']
    result['h'] = result['hour'].astype(int)

    # for hour in list(range(0, 24)):
    #     if hour not in result['h'].values:
    #         new_row = {'h': hour, 'hour': str(hour), 'count': 0}
    #         result = result.append(new_row, ignore_index=True)

    xVal = result['hour']
    yVal = result['count']
    colorVal = ["#2202d1"]*25
    layout = go.Layout(
        bargap=0.05,
        autosize=True,
        bargroupgap=0,
        barmode="group",
        margin=go.layout.Margin(l=10, r=0, t=0, b=30),
        showlegend=False,
        plot_bgcolor="#31302F",
        paper_bgcolor="#31302F",
        dragmode="select",
        font=dict(color="white"),
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
            showgrid=False,
            fixedrange=True,
            rangemode="nonnegative",
            zeroline=False,
        ),
        annotations=[
            dict(
                x=xi,
                y=yi,
                text=str(yi),
                xanchor="center",
                yanchor="bottom",
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


def dayhist(result, datemonth):
    result = result.groupby(['day_of_week']).count().reset_index()
    result['count'] = result['incidentNumber']
    # dayindex = range(0, 7)
    # result = result.reindex(dayindex, fill_value=0)
    # print(result.index)
    # print(result)
    colorVal = ["#2202d1"]*25
    xVal = result['day_of_week']
    yVal = result['count']
    layout = go.Layout(
        bargap=0.05,
        autosize=True,
        bargroupgap=0,
        barmode="group",
        margin=go.layout.Margin(l=10, r=0, t=0, b=30),
        showlegend=False,
        plot_bgcolor="#31302F",
        paper_bgcolor="#31302F",
        dragmode="select",
        font=dict(color="white"),
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
            showgrid=False,
            fixedrange=True,
            rangemode="nonnegative",
            zeroline=False,
        ),
        annotations=[
            dict(
                x=xi,
                y=yi,
                text=str(yi),
                xanchor="center",
                yanchor="bottom",
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


def responsehist(result, datemonth):

    fig = px.histogram(result, x="responsetime",   labels={
                       'responsetime': 'Response Time (min)', 'y': 'Count'},  opacity=0.8, marginal="rug")
    fig.update_xaxes(
        showgrid=True
    )
    fig.update_yaxes(
        showgrid=False
    )
    fig.update_layout(plot_bgcolor="#31302F", yaxis_title_text='Count', margin=go.layout.Margin(
        l=10, r=0, t=0, b=30), paper_bgcolor="#31302F", font=dict(color="white"))
    return fig


def responsetimebymonth(result, datemonth):
    fig = px.box(result, x="month",  y="responsetime")
    fig.update_xaxes(
        showgrid=True
    )
    fig.update_yaxes(
        showgrid=False
    )
    fig.update_layout(plot_bgcolor="#31302F", yaxis_title_text='Response time (min)', margin=go.layout.Margin(
        l=10, r=0, t=0, b=30), paper_bgcolor="#31302F", font=dict(color="white"), xaxis=dict(
            range=[0, 13],
            showgrid=False,
            tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            ticktext=['Jan', 'Feb', 'March', 'Apr', 'May', 'June',
                      'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            fixedrange=True,
            ticksuffix="",
    ))
    return fig


def responsetimebytod(result, datemonth):
    result['hour'] = result['time_local'].dt.hour
    fig = px.box(result, x="hour",  y="responsetime")
    fig.update_xaxes(
        showgrid=True
    )
    fig.update_yaxes(
        showgrid=False
    )
    fig.update_layout(plot_bgcolor="#31302F", yaxis_title_text='Response time (min)', margin=go.layout.Margin(
        l=10, r=0, t=0, b=30), paper_bgcolor="#31302F", font=dict(color="white"), xaxis=dict(
            range=[-1, 25],
            showgrid=False,
            tickvals=[x for x in range(0, 24)],
            ticktext=['12 AM', '1 AM', '2 AM', '3 AM', '4 AM', '5 AM', '6 AM', '7 AM', '8 AM', '9 AM', '10 AM', '11 AM', '12 PM', '1 PM', '2 PM',
                      '3 PM', '4 PM', '5 PM', '6 PM', '7 PM', '8 PM', '9 PM', '10 PM', '11 PM'],
            fixedrange=True,
            ticksuffix="",
    ),)
    return fig


def responsetimebyweekday(result, datemonth):
    fig = px.box(result, x="day_of_week",  y="responsetime")
    fig.update_xaxes(
        showgrid=True
    )
    fig.update_yaxes(
        showgrid=False
    )
    fig.update_layout(plot_bgcolor="#31302F", yaxis_title_text='Response time (min)', margin=go.layout.Margin(
        l=10, r=0, t=0, b=30), paper_bgcolor="#31302F", font=dict(color="white"), xaxis=dict(
            range=[-1, 8],
            showgrid=False,
            tickvals=[0, 1, 2, 3, 4, 5, 6],
            ticktext=['Mon', 'Tues', 'Wed', 'Thur', 'Friday', 'Sat', 'Sun'],
            fixedrange=True,
            ticksuffix="",
    ))
    return fig


def monthhist(result, datemonth):
    #result['month'] = result['month'].astype(str)
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
    yVal = result['count']

    layout = go.Layout(
        bargap=0.1,
        bargroupgap=0,
        barmode="group",
        margin=go.layout.Margin(l=10, r=0, t=0, b=30),
        showlegend=False,
        plot_bgcolor="#31302F",
        paper_bgcolor="#31302F",
        dragmode="select",
        font=dict(color="white"),
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
            showgrid=False,
            fixedrange=True,
            rangemode="nonnegative",
            zeroline=False,
        ),
        annotations=[
            dict(
                x=xi,
                y=yi,
                text=str(yi),
                xanchor="center",
                yanchor="bottom",
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


def incidentsfeature(df_merged, feature):
    #feature = available_features[6]

    # it converts the selected feature to categorical using a predifined function called categorize_numerical_features
    feature_cat = feature+'_cat_'
    df_merged_ = categorize_numerical_features(
        df_merged[[feature, metadata['pred_name_TF']]])
    # making sure all subplots will have the same category order
    category_orders = {feature_cat: df_merged_[feature_cat].cat.categories}

    # #Drawing the first figure; histogram of the feature
    # if len(df_merged)<1e6:
    #     fig1 = px.histogram(df_merged_[[feature]], x=feature, nbins=20)
    #     #fig1 = px.histogram(df_merged[[feature,metadata['pred_name_TF']]].sample(n=100, random_state=0), x=feature, nbins=20)
    #     #fig1.write_html('hist_test1.html')
    # else:
    #     print('The size of the data is huge so the plot just shows 1% of it.')
    #     #fig1 = px.histogram(df_merged[[feature,metadata['pred_name_TF']].sample(frac=0.001,random_state=0), x=feature, nbins=20)
    #     fig1 = px.histogram(df_merged_[[feature,metadata['pred_name_TF']]].sample(n=1000000, random_state=0), x=feature, nbins=20)

    # Drawing the second figure; histogram of the categorized feature
    df_merged_cat_barplot = df_merged_[[metadata['pred_name_TF'], feature_cat]].groupby(
        feature_cat).agg('count').rename(columns={metadata['pred_name_TF']: 'freq'}).reset_index()
    #fig2 = px.bar(df_merged_cat_barplot, x=feature_cat, y='freq', category_orders=category_orders)
    # fig2.write_html('hist_test2.html')
    # Drawing the third figure; total number of incidents given each category of the selected feature
    df_incidentcount = df_merged_[[metadata['pred_name_TF'], feature_cat]].groupby(
        feature_cat).agg(['mean', 'sum'])
    df_incidentcount.columns = ['mean', 'sum']
    df_incidentcount = df_incidentcount.reset_index()
    #fig3 = px.bar(df_incidentcount, x=feature_cat, y='sum',category_orders=category_orders)
    # fig3.write_html('hist_test3.html')
    # Drawing the fourth figure; mean (normalized) number of incidents given each category of the selected feature
    fig = px.bar(df_incidentcount, x=feature_cat,
                 y='mean', category_orders=category_orders)
    # fig4.write_html('hist_test4.html')

    # putting all figures in one
    # fig = make_subplots(rows=3, cols=1, subplot_titles=("Histogram of the categorized Data", "Total number of incidents in each category", "Comparitive Likelihood"))
    # # for trace in fig1.data:
    # #     fig.add_trace(trace, 1, 1)
    # for trace in fig2.data:
    #     fig.add_trace(trace, 1, 1)
    # for trace in fig3.data:
    #     fig.add_trace(trace, 2, 1)
    # for trace in fig4.data:
    #     fig.add_trace(trace, 3, 1)

    layout = go.Layout(
        bargap=0.05,
        autosize=True,
        bargroupgap=0,
        # barmode="group",
        yaxis_title_text='Comparitive Likelihood per category of '+feature,
        xaxis_title_text='Categories of '+feature,
        margin=go.layout.Margin(l=10, r=0, t=10, b=10),
        showlegend=False,
        plot_bgcolor="#31302F",
        paper_bgcolor="#31302F",
        dragmode="select",
        font=dict(color="white"),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            fixedrange=True,
            rangemode="nonnegative",
            zeroline=False,
        ),
    )

    #fig.update_layout(title_text="Using update_layout() With Graph Object Figures", title_font_size=30)
    fig.update_layout(layout)
    return fig


def incidentsfeaturecombo(result, datemonth):
    All_Possible_Dic, All_Possible_Filters = Filter_Combo_Builder()
    result = categorize_numerical_features(
        result[['incident_occurred', 'congestion_mean', 'temp_mean', 'precip_mean', 'is_weekend']])
    result = FILTER_calculator(
        result[['incident_occurred'] + list(All_Possible_Filters.keys())], All_Possible_Dic, 'All')
    result = result.rename(
        columns={'Average_Number_Incidents': 'Comparitive Likelihood'})
    # drawing
    fig = px.bar(result, x='Tag', y='Comparitive Likelihood')
    layout = go.Layout(
        bargap=0.05,
        autosize=True,
        bargroupgap=0,
        # barmode="group",
        margin=go.layout.Margin(l=10, r=0, t=0, b=30),
        showlegend=False,
        plot_bgcolor="#31302F",
        paper_bgcolor="#31302F",
        dragmode="select",
        font=dict(color="white"),
        yaxis=dict(
            showticklabels=True,
            showgrid=True,
            fixedrange=True,
            rangemode="nonnegative",
            zeroline=False,
        ),
    )
    fig.update_layout(layout)
    return fig


def totals(result, datemonth):
    #result['month'] = result['month'].astype(str)
    # colorVal = ["#2202d1"]*25

    resultgroup = result.groupby(
        'month-year').agg({'incidentNumber': 'count', 'time_local': 'first'})
    resultgroup = resultgroup.reset_index()
    resultgroup = resultgroup.sort_values(by=['time_local'])
    #result = result.groupby(['month-year']).count().reset_index()
    #result['count'] = result['incidentNumber']
    fig = px.line(resultgroup, x="month-year",
                  y="incidentNumber", title='Total Incidents')
    fig.update_xaxes(
        showgrid=True
    )
    fig.update_yaxes(
        showgrid=False
    )
    fig.update_layout(plot_bgcolor="#31302F", yaxis_title_text='Count', margin=go.layout.Margin(
        l=10, r=0, t=0, b=30), paper_bgcolor="#31302F", font=dict(color="white"))
    # xVal = result['month-year']
    # yVal = result['count']
    # fig.update_layout
    # layout = go.Layout(
    #     bargap=0.1,
    #     bargroupgap=0,
    #     barmode="group",
    #     margin=go.layout.Margin(l=10, r=0, t=0, b=30),
    #     showlegend=False,
    #     plot_bgcolor="#31302F",
    #     paper_bgcolor="#31302F",
    #     dragmode="select",
    #     font=dict(color="white"),
    #     yaxis=dict(
    #         range=[0, max(yVal) + max(yVal) / 4],
    #         showticklabels=False,
    #         showgrid=False,
    #         fixedrange=True,
    #         rangemode="nonnegative",
    #         zeroline=False,
    #     ),
    #     annotations=[
    #         dict(
    #             x=xi,
    #             y=yi,
    #             text=str(yi),
    #             xanchor="center",
    #             yanchor="bottom",
    #             showarrow=False,
    #             font=dict(color="white"),
    #         )
    #         for xi, yi in zip(xVal, yVal)
    #     ],
    # )

    # return go.Figure(
    #     data=[
    #         go.Bar(x=xVal, y=yVal, marker=dict(
    #             color=np.array(colorVal)), hoverinfo="x"),
    #     ],
    #     layout=layout,
    # )
    return fig

# %%


@app.callback(
    Output('histogram', 'figure'),
    [Input("date-picker", "date"),
     Input("date-picker-end", "date"),
     Input('county', 'value'), Input("month-selector", "value"), Input("histogram-basis",
                                                                       "active_tab"), Input("time-slider", "value"),  Input("day-selector", "value"),
     Input('feature', 'value'),
     ]
)
def update_bar_chart(start_date, end_date, counties, datemonth, histogramkind, timerange,   days, feature):

    if histogramkind == "incidentsfeature" or histogramkind == "incidentsfeaturecombo":
        result = return_merged(start_date, end_date,
                               counties, datemonth, timerange, days)
    else:
        result = return_incidents(
            start_date, end_date, counties, datemonth, timerange,   days)

    if histogramkind == "month":
        return monthhist(result, datemonth)
    elif histogramkind == "totals":
        return totals(result, datemonth)
    elif histogramkind == "day":
        return dayhist(result, datemonth)
    elif histogramkind == "hour":
        return hourhist(result, datemonth)
    # elif histogramkind == "response":
    #     return responsehist(result, datemonth)
    # elif histogramkind == "responsetimebytod":
    #     return responsetimebytod(result, datemonth)
    elif histogramkind == "incidentsfeature":
        return incidentsfeature(result, feature)
    elif histogramkind == "incidentsfeaturecombo":
        return incidentsfeaturecombo(result, datemonth)


# %%
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
    # app.server.run(threaded=True)

# %%
