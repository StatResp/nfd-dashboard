# %%
from matplotlib import cm, colors
import dash
import flask
import os
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

def transform_severity(Type_of_Crash):
    if Type_of_Crash=='Prop Damage (under)':
        return '1'
    elif Type_of_Crash=='Prop Damage (over)':
        return '2'
    elif Type_of_Crash=='Suspected Minor Injury':
        return '3'
    elif Type_of_Crash=='Suspected Serious Injury':
        return '4'
    elif Type_of_Crash=='Fatal':
        return '5'
    else:
        return '0'

'''
df1 = dataextract.decompress_pickle('data/nfd/geo_out_july_2020_central_time.pbz2')
#df.to_csv("data/nfd/geo_out_july_2020_central_time.pbz2.csv")
df = pd.read_csv('data/nfd/geo_out_july_2020_central_time.pbz2.csv')
df['alarm_datetime']=pd.to_datetime(df['alarm_datetime'],format="%Y-%m-%d  %H:%M:%S.%f")

for i in df1.columns:
    print(i, type(df1.iloc[0][i]))
    if  type(df.iloc[0][i])!= type(df1.iloc[0][i]): print(i ,'is not the same #########################')
    #else: print('checked')
#sys.exit()
'''

'''
df = dataextract.decompress_pickle('data/nfd/incident_weather_FRC0_traffic.pbz2')
df['ID'] = df['ID'].astype(str)
df['Type of Crash_2']=df.apply(lambda ROW: transform_severity(ROW['Type of Crash']), axis=1)
df['alarm_datetime']=df.apply(lambda ROW: ROW['time'].replace(tzinfo=None), axis=1)
#df['alarm_date']= df['alarm_datetime'].dt.date 
df['alarm_date']=df['alarm_datetime'].apply(lambda x: x.strftime("%Y-%m-%d"))
#df['alarm_time']= df['alarm_datetime'].dt.time 
df['alarm_time']=df['alarm_datetime'].apply(lambda x: x.strftime("%H:%M:%S.%f"))
df=df.rename(columns={'day_of_week': 'dayofweek' })
df=df.rename(columns={'ID': '_id' , 'GPS Coordinate Longitude': 'longitude', 'GPS Coordinate Latitude': 'latitude',
                       'Type of Crash_2': 'emdCardNumber', 'year': 'year'})

#df['alarmDateTime']=df['alarm_datetime'].strftime("%Y-%m-%d  %H:%M:%S.%f")
df['alarmDateTime']=df['alarm_datetime'].apply(lambda x: x.strftime("%Y-%m-%d  %H:%M:%S.%f"))
df['arrivalDateTime']=df['alarmDateTime']
df['weather']='s'
df['location']='Points'
df['location.coordinates']='Points'
df['respondingVehicles']='F'
df['fireZone']='F'
df['responsetime']=0
df['severity']=0
#df['emdCardNumber']='0'
print(df)

df1=                 df[['_id','latitude','longitude','emdCardNumber','alarmDateTime','arrivalDateTime','weather','location','location.coordinates',
                        'alarm_datetime','alarm_date','alarm_time','year','month','dayofweek','severity']]
dataextract.compressed_pickle('data/nfd/incident_weather_FRC0_traffic_nfdFormat', df1)

for i in df.columns:
    print(i, type(df.iloc[0][i]))                       
#sys.exit()
'''

'''
df1 = dataextract.decompress_pickle('data/nfd/geo_out_july_2020_central_time.pbz2')
for i in df1.columns:
    print(i, type(df1.iloc[0][i])) 
print('---------------------------------------')
'''

df = dataextract.decompress_pickle('data/nfd/incident_weather_FRC0_traffic_nfdFormat.pbz2')
print(df)
for i in df.columns:
    print(i, type(df.iloc[0][i])) 
#sys.exit()



# configure the dates
startdate = df.alarm_date.min()
enddate = df.alarm_date.max()
#df['alarm_datetime'] = pd.to_datetime(df.alarm_datetime)

print(startdate, enddate)





# set the server and global parameters
server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
app = dash.Dash(__name__, title='Incident Dashboard', update_title=None, external_stylesheets=[dbc.themes.BOOTSTRAP], server=server, meta_tags=[
                {"name": "viewport", "content": "width=device-width"}])
app.title = 'Incident Dashboard'

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
                            '''# [Statresp.ai](https://statresp.ai) | Incident Dashboard'''),
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
                                    '''## Incident Categories'''),
                                dcc.Dropdown(
                                    options=[
                                        #{'label': 'Automatic Crash Notifications', 'value': '34'},
                                        {'label': 'Prop Damage (under)',     'value': '1'},
                                        {'label': 'Prop Damage (over)',      'value': '2'},
                                        {'label': 'Suspected Minor Injury',  'value': '3'},
                                        {'label': 'Suspected Serious Injury', 'value': '4'},
                                        {'label': 'Fatal', 'value': '5'},
                                    ],
                                    value=[],
                                    id='emd-card-num-dropdown',
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

                                        {'label': 'Jan', 'value': '1'},
                                        {'label': 'Feb', 'value': '2'},
                                        {'label': 'Mar', 'value': '3'},
                                        {'label': 'Apr', 'value': '4'},
                                        {'label': 'May', 'value': '5'},
                                        {'label': 'June', 'value': '6'},
                                        {'label': 'July', 'value': '7'},
                                        {'label': 'Aug', 'value': '8'},
                                        {'label': 'Sep', 'value': '9'},
                                        {'label': 'Oct', 'value': '10'},
                                        {'label': 'Nov', 'value': '11'},
                                        {'label': 'Dec', 'value': '12'},


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

                                        {'label': 'Mon', 'value': '0'},
                                        {'label': 'Tue', 'value': '1'},
                                        {'label': 'Wed', 'value': '2'},
                                        {'label': 'Thur', 'value': '3'},
                                        {'label': 'Fri', 'value': '4'},
                                        {'label': 'Sat', 'value': '5'},
                                        {'label': 'Sun', 'value': '6'},
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
                                         dbc.Tab(label='Incidents Distribution', tab_id='incidents', className="bg-dark text-white", children=[dcc.Loading(
                                             id="loading-icon1", children=[dcc.Graph(id="map-graph"), ], type='default')]),
                                     ]
                                     ),
                                 ]),

                        html.Div(className="p-0 m-0 card bg-dark", children=[
                            dbc.Tabs(id='histogram-basis', active_tab="month", children=[
                                 dbc.Tab(label='Distribution by Year',
                                         tab_id='year', className="bg-dark text-white"),                                
                                 dbc.Tab(label='Distribution by Month',
                                         tab_id='month', className="bg-dark text-white"),
                                 dbc.Tab(label='Distribution by Weekday',
                                         tab_id='day', className="bg-dark text-white"),
                                 dbc.Tab(label='Distribution by Time of Day',
                                         tab_id='hour', className="bg-dark text-white"),
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
                                        dcc.Markdown('''Site designed by [ScopeLab](http://scopelab.ai/). Data source: Nashville Fire Department. Funding for this work has been provided by the National Science Foundation under awards [CNS-1640624](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1640624) and  [IIS-1814958](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1814958).'''
                                                     '''
         Funding by Department of Energy Vehicles Technology Office and National Science Foundation.''', id='footer', className="col"), ]),
]
)


## Incident Filterings

def return_incidents(start_date, end_date, emd_card_num, months, timerange, days):
    print(start_date, end_date, emd_card_num, months, timerange, days)
    date_condition = ((df['alarm_date'] >= start_date)
                      & (df['alarm_date'] <= end_date))
    if days is None or len(days) == 0:
        weekday_condition = (True)
    else:
        weekday_condition = ((df['dayofweek']).isin(days))
    if emd_card_num is None or len(emd_card_num) == 0:
        emd_card_condition = (True)
    else:
        emd_card_condition = (df['emdCardNumber'].isin(emd_card_num))
    if months is None or len(months) == 0:
        month_condition = True
    else:
        month_condition = ((df['month'].isin(months)))
    timemin, timemax = timerange
    hourmax = int(timemax)
    hourmin = int(timemin)
    if hourmax == 24:
        endtime = (tt(23, 59, 59))
    else:
        minutesmax = int(60*(timemax-hourmax))
        endtime = (tt(hourmax, minutesmax, 0))
    minutesmin = int(60*(timemin-hourmin))
    starttime = (tt(hourmin, minutesmin, 0))
    timecondition = ((df['alarm_datetime'].dt.time >= starttime)
                     & (df['alarm_datetime'].dt.time <= endtime))

    return df[emd_card_condition & date_condition & month_condition & weekday_condition & timecondition]


############ Process

@app.callback(
    [Output('incident-text', "children"), Output('month-text', "children"), Output('time-text', "children"), Output(component_id='time-text', component_property='style'), Output(component_id='month-text', component_property='style')],
    [Input("date-picker", "date"),
     Input("date-picker-end", "date"),
     Input('emd-card-num-dropdown', 'value'),
     Input("month-selector", "value"),
     Input("time-slider", "value"),
     Input("day-selector", "value"),
     ]
)
def update_incidents(start_date, end_date, emd_card_num, datemonth, timerange, days):
    result = return_incidents(start_date, end_date, emd_card_num, datemonth, timerange, days)
    timemin, timemax = timerange
    return "Incidents: %d" % (result.size), "Months: %s" % (str(datemonth)), "Time %s:00 to %s:00" % (timerange[0], timerange[1]), ({'display': 'none'}, {'display': 'block', 'text-align': 'left', 'font-weight': 'bold'})[timemin > 0 or timemax < 24], ({'display': 'none'}, {'display': 'block', 'text-align': 'left', 'font-weight': 'bold'})[datemonth is not None and len(datemonth) != 0]


viridis = cm.get_cmap('RdYlGn', 20000)


def plotly_linestring(vis_shape_row, minenergy, maxenergy):
    normalized = (1-(vis_shape_row.energy_consumed_kwh_per_mile -
                     minenergy)/(maxenergy-minenergy))
    colorval = viridis(normalized)
    return go.Scattermapbox(
        lat=np.array(np.array(vis_shape_row['geometry'].coords)[:, 1]),
        lon=np.array(np.array(vis_shape_row['geometry'].coords)[:, 0]),
        mode='lines',
        name="route {}".format(vis_shape_row.route_id),
        line={'color': colors.to_hex(
            colorval, keep_alpha=False), 'width': 4},
        text="route {0}, Average KWH per Mile {1}".format(
            vis_shape_row.route_id, vis_shape_row.energy_consumed_kwh_per_mile)
    )


def set_map_layout(fig):
    return fig.update_layout(
        autosize=True,
        margin=go.layout.Margin(l=0, r=0, t=0, b=0),
        hovermode='closest',
        plot_bgcolor="#1E1E1E",
        paper_bgcolor="#1E1E1E",
        legend=dict(
            orientation="h",
            y=0,
            yanchor="bottom",
            xanchor="center",
            x=0.5,
            traceorder="reversed",
            title_font_family="Times New Roman",
            font=dict(
                family="Courier",
                size=14, color="white",
            ),
            bgcolor="rgba(0, 0, 0, 0.5)",
            bordercolor="Black",
            borderwidth=0,
        ),
        mapbox=dict(
            accesstoken=MAPBOX_ACCESS_TOKEN,
            bearing=0, style=mapbox_style,
            center=dict(
                lat=latInitial,
                lon=lonInitial
            ),
            pitch=0,
            zoom=11
        ),
    )


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
     Input('emd-card-num-dropdown', 'value'),
     Input("month-selector", "value"),
     Input("time-slider", "value"),
      Input("day-selector", "value")]
)
def update_map_graph(start_date, end_date, radius, emd_card_num, datemonth, timerange, days):
    result = return_incidents(
        start_date, end_date, emd_card_num, datemonth, timerange, days)
    fig = go.Figure(go.Densitymapbox(lat=result['latitude'], lon=result['longitude'],
                                     #customdata=result[['incidentNumber','responsetime','alarm_datetime','severity','emdCardNumber']],
                                     #hovertemplate="%{lat},%{lon} <br> Incidentid: %{customdata[0]} <br> EmdCardNum: %{customdata[4]} <br> ResponseTime: %{customdata[1]} min. <br> Alarm Time: %{customdata[2]}<br> Severity  %{customdata[3]}",
                                     radius=radius), layout=Layout(
        autosize=True,
        margin=go.layout.Margin(l=0, r=35, t=0, b=0),
        showlegend=False,
        mapbox=dict(
            accesstoken=mapbox_access_token,
            # 40.7272  # -73.991251
                    center=dict(lat=latInitial, lon=lonInitial),
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
                                    "mapbox.center.lon": lonInitial,
                                    "mapbox.center.lat": latInitial,
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
    ),
    )
    return fig


def hourhist(result, datemonth):

    result['hour'] = result['alarm_datetime'].dt.hour
    result = result.groupby(['hour']).count().reset_index()
    result['count'] = result['_id']
    result['h'] = result['hour'].astype(int)

    for hour in list(range(0, 24)):
        if hour not in result['h'].values:
            new_row = {'h': hour, 'hour': str(hour), 'count': 0}
            result = result.append(new_row, ignore_index=True)

    xVal = result['hour']
    yVal = result['count']
    colorVal = ["#a10000"]*25
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
    result['count'] = result['_id']
    dayindex = range(0, 7)
    result = result.reindex(dayindex, fill_value=0)
    #print(result.index)
    #print(result)
    colorVal = ["#a10000"]*25
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


def yearhist(result, datemonth):
    #result['month'] = result['month'].astype(str)
    colorVal = ["#a10000"]*25
    result = result.groupby(['year']).count().reset_index()
    print(result)
    result['count'] = result['_id']
    result['y'] = result['year'].astype(int)
    monthindex = range(2017, 2021)
    for year in list(range(2017, 2021)):
        if year not in result['y'].values:
            new_row = {'y': year, 'year': str(year), 'count': 0}
            result = result.append(new_row, ignore_index=True)
    print(result)
    result = result.sort_values('y')
    xVal = result['year']
    yVal = result['count']
    print(xVal)
    print(yVal)

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
            range=[2016, 2021],
            showgrid=False,
            tickvals=[2017, 2018, 2019, 2020],
            ticktext=['2017', '2018', '2019', '2020'],
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


def monthhist(result, datemonth):
    #result['month'] = result['month'].astype(str)
    colorVal = ["#a10000"]*25
    result = result.groupby(['month']).count().reset_index()
    result['count'] = result['_id']
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

# %%
@app.callback(
    Output('histogram', 'figure'),
    [Input("date-picker", "date"),
     Input("date-picker-end", "date"),
     Input('emd-card-num-dropdown', 'value'),
     Input("month-selector", "value"),
     Input("histogram-basis", "active_tab"),
     Input("time-slider", "value"),
     Input("day-selector", "value")]
)
def update_bar_chart(start_date, end_date, emd_card_num, datemonth, histogramkind, timerange, days):
    result = return_incidents(start_date, end_date, emd_card_num, datemonth, timerange, days)
    if histogramkind == "month":
        return monthhist(result, datemonth)
    elif histogramkind == "day":
        return dayhist(result, datemonth)
    elif histogramkind == "hour":
        return hourhist(result, datemonth)
    elif histogramkind == "year":
        return yearhist(result, datemonth)


# %%
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
    # app.server.run(threaded=True)

# %%
