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

from dash.dependencies import Input, Output
from plotly import graph_objs as go
from plotly.graph_objs import *
from datetime import datetime as dt
import zipfile

server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
app = dash.Dash(__name__, server=server,meta_tags=[{"name": "viewport", "content": "width=device-width"}])



# %%
colorVal = [
        "#F4EC15",
        "#DAF017",
        "#BBEC19",
        "#9DE81B",
        "#80E41D",
        "#66E01F",
        "#4CDC20",
        "#34D822",
        "#24D249",
        "#25D042",
        "#26CC58",
        "#28C86D",
        "#29C481",
        "#2AC093",
        "#2BBCA4",
        "#2BB5B8",
        "#2C99B4",
        "#2D7EB0",
        "#2D65AC",
        "#2E4EA4",
        "#2E38A4",
        "#3B2FA0",
        "#4E2F9C",
        "#603099",
    ]

# %%

# Plotly mapbox public token
#mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"
 

# %%
# load data 
with zipfile.ZipFile('geo_out_july_2020_central_time.csv.zip', 'r') as zip_ref:
    zip_ref.extractall()
    
df = pd.read_csv('geo_out_july_2020_central_time.csv', index_col=0)
df['emdCardNumber'] = df['emdCardNumber'].str.upper()

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
                                html.P("""Select Start Date """),
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
                                html.P("""Select End Date """),
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
                                html.P("""Select different incident categories"""),
                                dcc.Checklist(
                                        options=[
                                              {'label': 'All', 'value': '1002'},
                                          {'label': 'Automatic Crash Notifications', 'value': '34'},
                                                {'label': 'Motor Vehicle Accidents', 'value': '29'},
                                            {'label': 'Breathing Problems', 'value': '6'},
                                            {'label': 'Burns', 'value': '7'},
                                            {'label': 'Cardiac Problems', 'value': '9'},
                                            {'label': 'Chest Pain', 'value': '10'},
                                                 {'label': 'Stab/Gunshot', 'value': '27'},    
                                            {'label': 'Pandemic Flu', 'value': '36'},
                                            {'label': 'Structure Fire', 'value': '69'},
                                            {'label': 'Outside Fire', 'value': '67'},
                                            {'label': 'Vehicle Fire', 'value': '71'},      
                                        ],
                                        value=['29', '34'],
                                        id='emd-card-num-dropdown' 

                                    ),  
                            ],
                        ),
                        html.Div(
                            className="row",
                            children=[
                                html.P("""Select Radius for heatmap """),
                                dcc.Slider(
                                            id='heatmap-radius',
                                            min=1,
                                            max=10,
                                            step=0.05,
                                            marks={i: '{}'.format(i) for i in range(1, 11)},
                                            value=2
                                        ),
                            ],
                        ),                        
                        # Change to side-by-side for mobile layout
                        html.Div(
                            className="row",
                            children=[
                                
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        html.P("""Select months to filter data."""),
                                        # Dropdown to select times
                                        dcc.Dropdown(
                                            id="bar-selector",
                                            options=[
                                                {
                                                    "label": str(n),
                                                    "value": str(n),
                                                }
                                                for n in range(1,13)
                                            ],
                                            multi=True,                                            
                                        )
                                    ],
                                ),
                            ],
                        ),
                        
                    ],
                ),
                # Column for app graphs and plots
                html.Div(
                    className="eight columns div-for-charts bg-grey",
                    children=[
                        dcc.Graph(id="heatmap"),
                        html.Div(
                            className="text-padding",
                            children=[
                                  html.P("""Select any of the bars on the histogram to filter incidents by month"""),
                                  html.P(""""""),
                                  html.P(""""""),
                               
                            ],
                        ),
                        dcc.Graph(id="bar-chart"),
                    ],
                ),
            ],
        )
    ]
)



# %%
# Selected Data in the Histogram updates the Values in the DatePicker
@app.callback(
    Output("bar-selector", "value"),
    [Input("bar-chart", "selectedData"), Input("bar-chart", "clickData")],
)
def update_bar_selector(value, clickData):
    holder = []
    if clickData:
        holder.append(str(int(clickData["points"][0]["x"])))
    if value:
        for x in value["points"]:
            holder.append(str(int(x["x"])))
    return list(set(holder))

# %%
mapbox_access_token = "pk.eyJ1Ijoidmlzb3ItdnUiLCJhIjoiY2tkdTZteWt4MHZ1cDJ4cXMwMnkzNjNwdSJ9.-O6AIHBGu4oLy3dQ3Tu2XA"

# %%
@app.callback(
    Output('heatmap', 'figure'),
    [Input("date-picker", "date"),
    Input("date-picker-end", "date"),
    Input("heatmap-radius", "value"),
    Input('emd-card-num-dropdown', 'value'),
    Input("bar-selector", "value")]
)
def update_heatmap(start_date, end_date, radius, emd_card_num, datemonth):
    if '1002' in emd_card_num:
        emd_card_num=range(1,136)        
    date_condition = ((df['alarm_date'] >= start_date) & (df['alarm_date'] <= end_date))
    string = '[A-Z]'
    updatedlist = [str(x) + string for x in emd_card_num]
    separator = '|'
    search_str = '^' + separator.join(updatedlist)
    emd_card_condition = (df.emdCardNumber.str.contains(search_str))
    result = df.loc[date_condition & emd_card_condition][['alarm_datetime','latitude','longitude']]  
    
    if len(datemonth)!=0:            
            result['month'] = pd.to_datetime(result['alarm_datetime']).dt.month
            month_condition = ((result['month'].isin(datemonth)))
            result = result.loc[month_condition][['latitude','longitude']]  
            
             
    #return px.density_mapbox(result, lat='latitude', lon='longitude', radius=radius,
    #                    center=dict(lat=36.16228, lon=-86.774372), zoom=10,
    #                    mapbox_style="stamen-terrain")
    #use go.Densitymapbox(lat=quakes.Latitude, lon=quakes.Longitude, z=quakes.Magnitude,
    fig = go.Figure(go.Densitymapbox(lat=result['latitude'], lon=result['longitude'],radius=radius))
    #fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=-86.774372,mapbox_center_lat=36.16228)
    #fig.update_layout(margin={"r":35,"t":0,"l":0,"b":0}, showlegend=False,zoom:10,)
    fig.update_layout(
        title='Incident heatmap',
        autosize=True,
        plot_bgcolor="#323130",
        paper_bgcolor="#323130",
        #height=420,
        margin=dict(l=0, r=35, t=0, b=0),
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            style='streets',
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=36.16228,
                lon=-86.774372
            ),
            pitch=0,
            zoom=10
        )
    )
    return fig

# %%
@app.callback(
    Output('bar-chart', 'figure'),
    [Input("date-picker", "date"),
    Input("date-picker-end", "date"),
    Input('emd-card-num-dropdown', 'value')]
)
def update_bar_chart(start_date, end_date, emd_card_num):
    if '1002' in emd_card_num:
        emd_card_num=range(1,136)
    date_condition = ((df['alarm_date'] >= start_date) & (df['alarm_date'] <= end_date))
    string = '[A-Z]'
    updatedlist = [str(x) + string for x in emd_card_num]
    separator = '|'
    search_str = '^' + separator.join(updatedlist)
    emd_card_condition = (df.emdCardNumber.str.contains(search_str))
    
    result = df.loc[date_condition & emd_card_condition][['alarm_datetime']]    
    result['alarm_datetime'] = pd.to_datetime(result['alarm_datetime'])

    result['month_year'] = result['alarm_datetime'].dt.month 
    result['month_year'] = result['month_year'].astype(str)

    result = result.groupby(['month_year']).count().reset_index()
    result.columns = ['month', 'count']
    xVal=result['month']
    yVal=result['count']
    layout = go.Layout(
        bargap=0.1,
        bargroupgap=0,
        barmode="group",
        margin=go.layout.Margin(l=10, r=0, t=0, b=50),
        showlegend=False,
        plot_bgcolor="#323130",
        paper_bgcolor="#323130",
        dragmode="select",
        font=dict(color="white"),
        xaxis=dict(
            range=[0, 13],
            showgrid=False,
            title='Month',
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
            go.Bar(x=xVal, y=yVal  ),
        ],
        layout=layout,
    )

# %%
if __name__ == '__main__':
    app.server.run(debug=True, threaded=True)

# %%
