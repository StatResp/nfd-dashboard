# %%
import dash
import flask
import os
from random import randint
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.express as px
import dataextract
from dash.dependencies import Input, Output
from plotly import graph_objs as go
from plotly.graph_objs import *
from datetime import datetime as dt
import zipfile

server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
app = dash.Dash(__name__, server=server,meta_tags=[{"name": "viewport", "content": "width=device-width"}])
app.title='Incident Dashboard'




# %%

# Plotly mapbox public token
#mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"
 

# # %%
# # load data 
# with zipfile.ZipFile('geo_out_july_2020_central_time.csv.zip', 'r') as zip_ref:
#     zip_ref.extractall()
 

# load data
df=dataextract.decompress_pickle('geo_out_july_2020_central_time.pbz2')


   
#df = pd.read_pickle('geo_out_july_2020_central_time.pkl')
#df['emdCardNumber'] = df['emdCardNumber'].str.upper()

# %%

# Layout of Dash App
app.layout = html.Div(
    children=[
        html.Div(
            className="row",
            children=[
                # Column for user controls
                html.Div(
                    className="four columns div-user-controls",
                    children=[
                        dcc.Markdown('''# [Statresp.ai](https://statresp.ai) | Incident Dashboard'''),
                        html.Div(
                            className="div-for-dropdown",
                            children=[
                                #html.P("""Select Start Date """,style={'text-align': 'left' ,'font-weight':'bold'}),
                                dcc.Markdown('''## Select Start Date'''),
                                dcc.DatePickerSingle(
                                    id="date-picker",
                                    min_date_allowed=dt(2019, 1, 1),
                                    max_date_allowed=dt(2020, 7, 9),
                                    initial_visible_month=dt(2019, 1, 1),
                                    date=dt(2019, 1, 1).date(),
                                    display_format="MMMM D, YYYY",
                                    style={"border": "0px solid black"},
                                )
                            ],
                        ),
                        html.Div(
                            className="div-for-dropdown",
                            children=[                                
                                dcc.Markdown('''## Select End Date'''),
                                dcc.DatePickerSingle(
                                    id="date-picker-end",
                                    min_date_allowed=dt(2019, 1, 1),
                                    max_date_allowed=dt(2020, 7, 9),
                                    initial_visible_month=dt(2020, 7, 9),
                                    date=dt(2020, 7, 9).date(),
                                    display_format="MMMM D, YYYY",
                                    style={"border": "0px solid black"},
                                )
                            ],
                        ),
                        html.Div(
                            className="div-for-dropdown",
                            children=[                                
                                dcc.Markdown('''## Select Incident Categories'''),
                                dcc.Checklist(
                                        options=[
                                              {'label': 'All', 'value': '1002'},
                                          #{'label': 'Automatic Crash Notifications', 'value': '34'},
                                                {'label': 'Motor Vehicle Accidents', 'value': '29'},
                                            {'label': 'Breathing Problems', 'value': '6'},
                                            {'label': 'Burns', 'value': '7'},
                                            {'label': 'Cardiac Problems', 'value': '9'},
                                            {'label': 'Chest Pain', 'value': '10'},
                                                 {'label': 'Stab/Gunshot', 'value': '27'},    
                                            #{'label': 'Pandemic Flu', 'value': '36'},
                                            {'label': 'Structure Fire', 'value': '69'},
                                            {'label': 'Outside Fire', 'value': '67'},
                                            {'label': 'Vehicle Fire', 'value': '71'},      
                                        ],
                                        value=['29'],
                                        id='emd-card-num-dropdown' ,
                                         labelStyle={'display': 'inline-block'} ,
                                    ),                                                         
                            ],
                        ),
                        # Change to side-by-side for mobile layout
                        html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        dcc.Markdown('''## Filter Incidents By Month'''),
                                        #html.P("""Select from the list to filter data by month.""",style={'text-align': 'left' ,'font-weight':'bold'}),
                                        # Dropdown to select times
                                        dcc.Checklist(
                                            id="bar-selector",
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
                                                    {'label': 'Dec', 'value': '11'},
                                                
                                                
                                            ],
                                            
                                            labelStyle={'display': 'inline-block'}                                         
                                        )
                                    ],
                                ),
                        html.Div(className="div-for-dropdown",
                                    children=[
                                    dcc.Markdown('''## Filter Incidents By Time of Day'''),
                                    #html.P("""Select Time Range""",style={'text-align': 'left' ,'font-weight':'bold'}),
                                    dcc.RangeSlider(
                                        id='time-slider',
                                        min=0,
                                        max=24,
                                        step=1,
                                        value=[0, 24],
                                        marks={i: '{}:00'.format(str(i).zfill(2)) for i in range(0, 25,4)},
                                    ),
                                    ]
                                    ),
                        html.Div(
                                    className="div-for-dropdown",                                    
                                    children=[
                                        dcc.Markdown('''## Include Incident Severity'''),
                                        dcc.RadioItems( id='severity-basis',
                                                options=[                                             
                                                    {'label': 'Yes', 'value': 'severity'},
                                                    {'label': 'No', 'value': 'none'},
                                                    ],
                                                labelStyle={'display': 'inline-block'} ,     
                                                value='severity'
                                            ),                                        
                                    ]
                                ),
                                
                                                
                        
                    ],
                ),
                # Column for app graphs and plots
                html.Div(
                    className="eight columns div-for-charts bg-grey",
                    children=[
                        
                        dcc.Markdown('''Adjust Slider below to configure the heatmap intensity.''') ,
                        html.Div(children=[dcc.Slider(
                                                    id='map-graph-radius',
                                                    min=1,
                                                    max=10,
                                                    step=0.05,
                                                    marks={i: '{}'.format(i) for i in range(1, 11)},
                                                    value=2
                                                ),],),    
                        dcc.Graph(id="map-graph"),                        
                        dcc.RadioItems( id='histogram-basis',
                                          options=[                                             
                                              {'label': 'Group By Month', 'value': 'month'},
                                              {'label': 'Group By Day of The Week', 'value': 'day'},
                                              {'label': 'Group By Hour of The Day', 'value': 'hour'}],
                                        labelStyle={'display': 'inline-block'} ,     
                                        value='month',style={'text-align': 'center'},
                                    ),
                        #html.Div(className="div-for-dropdown", children=[html.P(id='heatmap-text',style={'text-align': 'center'})]),                       
                        html.P('Histogram by Month',id='heatmap-text',style={'text-align': 'center'}),                                                
                        dcc.Graph(id="histogram"),
                    ],
                ),
            ],
        ),
        html.Div(
            className="row",
            children=[html.Div(className="div-for-dropdown", style={'text-align': 'center'},
                                            children=[
                                                dcc.Markdown('Site designed by [ScopeLab from Vanderbilt University](http://scopelab.ai/) starting from [the Uber Ride Demo from Plotly](https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-uber-rides-demo). Data source: Nashville Fire Department.')]),                                   
            ]
        ),
    ]
)



# %%
# Selected Data in the Histogram updates the Values in the DatePicker
# @app.callback(
#     Output("bar-selector", "value"),
#     [Input("histogram", "selectedData"), Input("histogram", "clickData"),Input("histogram-basis","value")],
# )
# def update_bar_selector(value, clickData,histogramkind):
#     holder = []
#     if(histogramkind=="month"):
#         if clickData:
#             holder.append(str(int(clickData["points"][0]["x"])))
#         if value:
#             for x in value["points"]:
#                 holder.append(str(int(x["x"])))
#     return list(set(holder))

# %%
mapbox_access_token = "pk.eyJ1Ijoidmlzb3ItdnUiLCJhIjoiY2tkdTZteWt4MHZ1cDJ4cXMwMnkzNjNwdSJ9.-O6AIHBGu4oLy3dQ3Tu2XA"

@app.callback(
    Output('heatmap-text', "children"),
    [Input("date-picker", "date"),
    Input("date-picker-end", "date"),    
    Input('emd-card-num-dropdown', 'value'),
    Input("bar-selector", "value"),
    Input("time-slider", "value"),  Input("severity-basis", "value")
    ]
)
def update_incidents(start_date, end_date, emd_card_num, datemonth, timerange,severity):
  
    if '1002' in emd_card_num:
        emd_card_num=range(1,136)
    elif '29' in emd_card_num:
        emd_card_num.append('34')        
    date_condition = ((df['alarm_date'] >= start_date) & (df['alarm_date'] <= end_date))
    string = '[A-Z]'
    updatedlist = [str(x) + string for x in emd_card_num]
    separator = '|'
    search_str = '^' + separator.join(updatedlist)
    emd_card_condition = (df.emdCardNumber.str.contains(search_str))
    result = df.loc[date_condition & emd_card_condition][['alarm_datetime','latitude','longitude']]  
    result['hour'] = pd.to_datetime(result['alarm_datetime']).dt.hour
    timemin,timemax=timerange
    timemin=int(timemin)
    timemax=int(timemax)
    if(timemin>0 or timemax<24):
        time_condition=((result['hour']>=timemin)&(result['hour']<=timemax))
        result = result.loc[time_condition][['alarm_datetime','latitude','longitude']]  
    if datemonth is not None and len(datemonth)!=0:            
            result['month'] = pd.to_datetime(result['alarm_datetime']).dt.month
            month_condition = ((result['month'].isin(datemonth)))
            result = result.loc[month_condition][['latitude','longitude']]   
    # if 'severity' in severity:
    #    return "Incident distribution (with severity from 1 to 5) from %s to %s within %s:00 to %s:00 hours. Total %d incidents."%(start_date,end_date,timerange[0],timerange[1],result.size)
    #else:
    return "Showing %d incidents from %s to %s within %s:00 to %s:00 hours. "%(result.size,start_date,end_date,timerange[0],timerange[1])
  
def transform_severity(emdCardNumber):
    if 'A' in emdCardNumber:
         return 2
    elif 'B' in emdCardNumber:
         return 3
    elif 'C' in emdCardNumber:
         return 4
    elif 'D' in emdCardNumber:
         return 5
    elif 'E' in emdCardNumber:
         return 6
    elif 'Ω' in emdCardNumber:
         return 1
    else:
         return 0
    
# %%
@app.callback(
    Output('map-graph', 'figure'),
    [Input("date-picker", "date"),
    Input("date-picker-end", "date"),
    Input("map-graph-radius", "value"),
    Input('emd-card-num-dropdown', 'value'),
    Input("bar-selector", "value"),
    Input("time-slider", "value"), Input("severity-basis", "value")]
)
def update_map_graph(start_date, end_date, radius, emd_card_num, datemonth, timerange,severity):
  
    if '1002' in emd_card_num:
        emd_card_num=range(1,136)
    elif '29' in emd_card_num:
        emd_card_num.append('34')           
    date_condition = ((df['alarm_date'] >= start_date) & (df['alarm_date'] <= end_date))
    string = '[A-Z]'
    updatedlist = [str(x) + string for x in emd_card_num]
    separator = '|'
    search_str = '^' + separator.join(updatedlist)
    emd_card_condition = (df.emdCardNumber.str.contains(search_str))
    result = df.loc[date_condition & emd_card_condition][['alarm_datetime','latitude','longitude','emdCardNumber']]  
    result['hour'] = pd.to_datetime(result['alarm_datetime']).dt.hour
    result['severity']=result['emdCardNumber'].apply(lambda x: transform_severity(x))
    timemin,timemax=timerange
    timemin=int(timemin)
    timemax=int(timemax)
    if(timemin>0 or timemax<24):
        time_condition=((result['hour']>=timemin)&(result['hour']<=timemax))
        result = result.loc[time_condition][['alarm_datetime','latitude','longitude','severity']]  
    if datemonth is not None and len(datemonth)!=0:            
            result['month'] = pd.to_datetime(result['alarm_datetime']).dt.month
            month_condition = ((result['month'].isin(datemonth)))
            result = result.loc[month_condition][['latitude','longitude','severity']]  
    #use go.Densitymapbox(lat=quakes.Latitude, lon=quakes.Longitude, z=quakes.Magnitude,
    latInitial=36.16228
    lonInitial=-86.774372
    
    if 'severity' in severity:
        fig = go.Figure(go.Densitymapbox(lat=result['latitude'], lon=result['longitude'],z=result['severity'],radius=radius),layout=Layout(
                autosize=True,
                margin=go.layout.Margin(l=0, r=35, t=0, b=0),
                showlegend=False,
                mapbox=dict(
                    accesstoken=mapbox_access_token,
                    center=dict(lat=latInitial, lon=lonInitial),  # 40.7272  # -73.991251
                    style="light",
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
                                            "mapbox.style": "light",
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
    else:
        fig = go.Figure(go.Densitymapbox(lat=result['latitude'], lon=result['longitude'],radius=radius),layout=Layout(
                autosize=True,
                margin=go.layout.Margin(l=0, r=35, t=0, b=0),
                showlegend=False,
                mapbox=dict(
                    accesstoken=mapbox_access_token,
                    center=dict(lat=latInitial, lon=lonInitial),  # 40.7272  # -73.991251
                    style="light",
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
                                            "mapbox.style": "light",
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

colorVal = ["#2202d1"]*25

def hourhist(result,datemonth):
    if datemonth is not None and len(datemonth)!=0:            
            result['month'] = pd.to_datetime(result['alarm_datetime']).dt.month
            month_condition = ((result['month'].isin(datemonth)))
            result = result.loc[month_condition][['alarm_datetime']]              
    result['hour'] = result['alarm_datetime'].dt.hour 
    result = result.groupby(['hour']).count().reset_index()
    result.columns = ['hour', 'count']
    hourindex = range(0,24)
    result=result.reindex(hourindex,fill_value=0)
    #print(result.index)
    #print(result)
    xVal=result['hour']
    yVal=result['count']
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
            tickvals = [ x for x in range(0,24)],       
            ticktext = ['12 AM','1 AM', '2 AM', '3 AM', '4 AM', '5 AM', '6 AM','7 AM','8 AM', '9 AM', '10 AM', '11 AM', '12 PM', '1 PM', '2 PM',
             '3 PM', '4 PM', '5 PM', '6 PM','7 PM','8 PM', '9 PM', '10 PM', '11 PM'],           
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
            go.Bar(x=xVal, y=yVal,marker=dict(color=np.array(colorVal)), hoverinfo="x"),
        ],
        layout=layout,
    )

def dayhist(result,datemonth):
    if datemonth is not None and len(datemonth)!=0:            
            result['month'] = pd.to_datetime(result['alarm_datetime']).dt.month
            month_condition = ((result['month'].isin(datemonth)))
            result = result.loc[month_condition][['alarm_datetime']]              
    result['dayofweek'] = result['alarm_datetime'].dt.dayofweek 
    result = result.groupby(['dayofweek']).count().reset_index()
    result.columns = ['dayofweek', 'count']
    dayindex = range(0,7)
    result=result.reindex(dayindex,fill_value=0)
    #print(result.index)
    #print(result)
    xVal=result['dayofweek']
    yVal=result['count']
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
            tickvals = [0,1, 2, 3,4, 5,6],    
            ticktext = ['Mon','Tues', 'Wed', 'Thur', 'Friday', 'Sat', 'Sun'],      
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
            go.Bar(x=xVal, y=yVal,marker=dict(color=np.array(colorVal)), hoverinfo="x"),
        ],
        layout=layout,
    )

def monthhist(result,selection):
    result['month_year'] = result['alarm_datetime'].dt.month 
    result['month_year'] = result['month_year'].astype(str)
    

    result = result.groupby(['month_year']).count().reset_index()
    result.columns = ['month', 'count']
    monthindex = range(0,12)
    result=result.reindex(monthindex,fill_value=0)
    result['m']=result['month'].astype(int)
    result=result.sort_values('m')
    
    #print(result.index)
    #print(result)
    xVal=result['month']
    yVal=result['count']

    if selection is not None:        
        xSelected = [int(x) for x in selection]
        #print (selection)
        for i in range(12):        
            if i+1 in xSelected: 
                #print ("setting to white " + str(i))         
                colorVal[i]= "#FFFFFF"
            else:
                colorVal[i]= "#2202d1"
    #print(colorVal)        

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
            tickvals = [1, 2, 3,4, 5,6, 7,8, 9,10, 11,12],    
            ticktext = ['Jan','Feb', 'March', 'Apr', 'May', 'June', 'July','Aug','Sep','Oct','Nov','Dec'],      
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
            go.Bar(x=xVal, y=yVal,marker=dict(color=np.array(colorVal)), hoverinfo="x"),
        ],
        layout=layout,
    )
# @app.callback(
#     Output('histogram-text',"children"),
#     [Input("histogram-basis","value"),Input("date-picker", "date"),
#     Input("date-picker-end", "date"),           
#     Input("time-slider", "value")]
# )
# def updatehistogramtext(histogramkind,start_date,end_date,timerange):
#     outputkind=''
#     if histogramkind=="month":
#         outputkind= "Histogram by month"
#     elif histogramkind=="day":
#         outputkind= "Histogram by day of the week"
#     elif histogramkind=="hour":
#         outputkind= "Histogram by hour of the day"
#     return "Histogram by %s from %s to %s within %s:00 to %s:00 hours."%(outputkind,start_date,end_date,timerange[0],timerange[1])



# %%
@app.callback(
    Output('histogram', 'figure'),
    [Input("date-picker", "date"),
    Input("date-picker-end", "date"),
    Input('emd-card-num-dropdown', 'value'), Input("bar-selector", "value"),Input("histogram-basis","value"),Input("time-slider", "value")]
)
def update_bar_chart(start_date, end_date, emd_card_num,selection,histogramkind,timerange):
    if '1002' in emd_card_num:
        emd_card_num=range(1,136)
    elif '29' in emd_card_num:
        emd_card_num.append('34')     
    date_condition = ((df['alarm_date'] >= start_date) & (df['alarm_date'] <= end_date))
    string = '[A-Z]'
    updatedlist = [str(x) + string for x in emd_card_num]
    separator = '|'
    search_str = '^' + separator.join(updatedlist)
    emd_card_condition = (df.emdCardNumber.str.contains(search_str))
    
    result = df.loc[date_condition & emd_card_condition][['alarm_datetime']]    
    result['alarm_datetime'] = pd.to_datetime(result['alarm_datetime'])
    timemin,timemax=timerange
    timemin=int(timemin)
    timemax=int(timemax)
    if(timemin>0 or timemax<24):
        result['hour'] = pd.to_datetime(result['alarm_datetime']).dt.hour    
        time_condition=((result['hour']>=timemin)&(result['hour']<=timemax))
        result = result.loc[time_condition][['alarm_datetime']]  

    if histogramkind=="month":
        return monthhist(result,selection)
    elif histogramkind=="day":
        return dayhist(result,selection)
    elif histogramkind=="hour":
        return hourhist(result,selection)

   

# %%
if __name__ == '__main__':
	#app.run_server(host='0.0.0.0', port=8080, debug=True, use_reloader=False)  
    app.server.run(threaded=True)

# %%
