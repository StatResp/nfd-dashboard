# %%
import traceback
import dateparser
from matplotlib import cm, colors
import dash
import flask
import pyarrow.parquet as pq
import os
from flask_caching import Cache
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
import dataextract
import os
import sys
import resource
import geopandas as gpd
import dash_bootstrap_components as dbc

# set configurations
mapbox_style = "light"
mapbox_access_token = "pk.eyJ1Ijoidmlzb3ItdnUiLCJhIjoiY2tkdTZteWt4MHZ1cDJ4cXMwMnkzNjNwdSJ9.-O6AIHBGu4oLy3dQ3Tu2XA"
MAPBOX_ACCESS_TOKEN = mapbox_access_token
pd.set_option('display.max_columns', None)

latInitial = 36.16228
lonInitial = -86.774372

#table2 = pq.read_table('data/nfd/incidents_july_2020_2.parquet')

# df = dataextract.decompress_pickle(
#    'data/nfd/geo_out_july_2020_central_time.pbz2')

#

table2 = pq.read_table('tdot_incidents.parquet')
df =table2.to_pandas()
#df = dd.read_parquet('tdot_incidents.parquet', engine='pyarrow')
startdate = df.alarm_date.min()#.compute()
enddate = df.alarm_date.max()#.compute()
counties = df.county_incident.drop_duplicates()#.compute()
counties = counties.tolist()
counties.sort()
# print(counties)

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
                            '''# [Statresp.ai](https://statresp.ai) | TDOT Incident Dashboard'''),
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
                                         step=0.25,
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
                                  dbc.Tab(label='Incidents by Weather Conditions',
                                         tab_id='incidentsweather', className="bg-dark text-white"),
                                  dbc.Tab(label='Incidents by Light Conditions',
                                         tab_id='incidentslight', className="bg-dark text-white"),
                                 ]),
                            dcc.Loading(id="loading-icon2", className="flex-grow-1",
                                        children=[dcc.Graph(id="histogram"), ], type='default'),
                        ]
                        ),

                    ],
                ),
            ],
        ), ],
),
    html.Div(className="row", children=[dcc.Markdown(""),
                                        dcc.Markdown('''Site designed by [ScopeLab](http://scopelab.ai/). Data source: TDOT.'''
                                                     '''Funding by TDOT and National Science Foundation.''', id='footer', className="col"), ]),
]
)


# Incident Filterings


 
def return_incidents(start_date, end_date, counties, months, timerange,   days):
    start_date = dateparser.parse(start_date)
    end_date = dateparser.parse(end_date)

    # print(type(start_date),start_date)

    # print(df.dtypes)

    try:
        date_condition = ((df['alarm_date'] >= start_date)
                          & (df['alarm_date'] <= end_date))

        #print("set date condition")
        if days is None or len(days) == 0:
            weekday_condition = (True)
        else:
            weekday_condition = ((df['dayofweek']).isin(days))
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
            county_condition = ((df['county_incident'].isin(counties)))

        #print("going to set time condition")

        if hourmax == 24:
            endtime = (tt(23, 59, 59))
            endtime = 23*3600+59*60 + 59
        else:
            minutesmax = int(60*(timemax-hourmax))
            endtime = hourmax*3600+minutesmax*60
        minutesmin = int(60*(timemin-hourmin))
        starttime = hourmin*3600+minutesmin*60

        # print(type(starttime),starttime)

        timecondition = ((df['time'] >= starttime)
                         & (df['time'] <= endtime))
        result = df[timecondition & date_condition & month_condition &
                    weekday_condition & county_condition]#.compute()
    except Exception:
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

    if counties == None or len(counties) != 1:
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
                        "text": "This map can be drawn for one and only one county.",
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

    result = return_incidents(
        start_date, end_date, counties, datemonth, timerange,   days)


    if len(result)==0:
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
    latone=result['latitude'].iloc[0]
    lonone=result['longitude'].iloc[0]

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
            zoom=10,
        ),
        updatemenus=[
            dict(
                buttons=(
                    [
                        dict(
                            args=[
                                {
                                    "mapbox.zoom": 10,
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

    if len(result)==0:
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

    latone=result['latitude'].iloc[0]
    lonone=result['longitude'].iloc[0]
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
                          zoom=9,
                      ),
                      updatemenus=[
                          dict(
                              buttons=(
                                  [
                                      dict(
                                          args=[
                                              {
                                                  "mapbox.zoom": 9,
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
                            "mapbox.zoom": 9,
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

    result['hour'] = result['alarm_datetime'].dt.hour
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
    result = result.groupby(['dayofweek']).count().reset_index()
    result['count'] = result['incidentNumber']
    # dayindex = range(0, 7)
    # result = result.reindex(dayindex, fill_value=0)
    # print(result.index)
    # print(result)
    colorVal = ["#2202d1"]*25
    xVal = result['dayofweek']
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
    result['hour'] = result['alarm_datetime'].dt.hour
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
    fig = px.box(result, x="dayofweek",  y="responsetime")
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


def incidentslight(result, datemonth):
    result = result.groupby(['light_conditions']).count().reset_index()
    result['count'] = result['incidentNumber']
    colorVal = ["#2202d1"]*25
    xVal = result['light_conditions']
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
def incidentsweather(result, datemonth):
    result = result.groupby(['weather_conditions']).count().reset_index()
    result['count'] = result['incidentNumber']
    colorVal = ["#2202d1"]*25
    xVal = result['weather_conditions']
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


def totals(result, datemonth):
    #result['month'] = result['month'].astype(str)
    # colorVal = ["#2202d1"]*25

    resultgroup = result.groupby(
        'month-year').agg({'incidentNumber': 'count', 'alarm_datetime': 'first'})
    resultgroup = resultgroup.reset_index()
    resultgroup = resultgroup.sort_values(by=['alarm_datetime'])
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
     Input('county', 'value'), Input("month-selector", "value"), Input("histogram-basis", "active_tab"), Input("time-slider", "value"),  Input("day-selector", "value")]
)
def update_bar_chart(start_date, end_date, counties, datemonth, histogramkind, timerange,   days):
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
    elif histogramkind == "response":
        return responsehist(result, datemonth)
    elif histogramkind == "responsetimebytod":
        return responsetimebytod(result, datemonth)
    elif histogramkind == "incidentslight":
        return incidentslight(result, datemonth)
    elif histogramkind == "incidentsweather":
        return incidentsweather(result, datemonth)


# %%
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
    # app.server.run(threaded=True)

# %%
