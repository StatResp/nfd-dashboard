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
                                html.P("""Select incident categories""",style={'text-align': 'center'}),
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

                        # Change to side-by-side for mobile layout
                        html.Div(
                            className="row",
                            children=[
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        html.P("""Select from the list or click on histogram to filter data by month.""",style={'text-align': 'center'}),
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
                                    html.P("""Select Time Range""",style={'text-align': 'center'}),
                                    dcc.RangeSlider(
                                        id='time-slider',
                                        min=0,
                                        max=24,
                                        step=1,
                                        value=[0, 24],
                                        marks={
                                            0: '12 AM',
                                            2: '2 AM',
                                            4: '4 AM',
                                            6: '6 AM',
                                            8: '8 AM',
                                            10: '10 AM',
                                            12: '12 PM',
                                            14: '2 PM',
                                            16: '4 PM',
                                            18: '6 PM',
                                            20: '8 PM',
                                            22: '10 PM',
                                            24: '12 AM',                                        
                                        },
                                    ),
                                    ]
                                    ), 
                                html.Div(
                                    className="div-for-dropdown",
                                    
                                    children=[
                                        html.P('Select radius to configure heatmap',style={'text-align': 'center'}),
                                        dcc.Slider(
                                                    id='map-graph-radius',
                                                    min=1,
                                                    max=10,
                                                    step=0.05,
                                                    marks={i: '{}'.format(i) for i in range(1, 11)},
                                                    value=2
                                                ),
                                    ]
                                )
                            ],
                        ),
                        
                        
                    ],
                ),
                # Column for app graphs and plots
                html.Div(
                    className="eight columns div-for-charts bg-grey",
                    children=[           
                         html.P(id='heatmap-text',style={'text-align': 'center','padding-top':'20px'}),               
                        dcc.Graph(id="map-graph"),
                        html.P('Click on a month in the histogram below to filter data',id='histogram-text',style={'text-align': 'center','padding-top':'20px'}),                        
                        dcc.Graph(id="histogram"),
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
    [Input("histogram", "selectedData"), Input("histogram", "clickData")],
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

# Update the total number of rides Tag
@app.callback(Output("heatmap-text", "children"),   [Input("date-picker", "date"),Input("date-picker-end", "date"),Input("time-slider", "value")])
def update_total_rides(datestart,dateend,timevalue):
    return "Incident distribution from: %s to %s within %s:00 to %s:00 hours."%(datestart,dateend,timevalue[0],timevalue[1])
    

# %%
@app.callback(
    Output('map-graph', 'figure'),
    [Input("date-picker", "date"),
    Input("date-picker-end", "date"),
    Input("map-graph-radius", "value"),
    Input('emd-card-num-dropdown', 'value'),
    Input("bar-selector", "value"),
    Input("time-slider", "value")]
)
def update_map_graph(start_date, end_date, radius, emd_card_num, datemonth, timerange):
  
    if '1002' in emd_card_num:
        emd_card_num=range(1,136)        
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
    if len(datemonth)!=0:            
            result['month'] = pd.to_datetime(result['alarm_datetime']).dt.month
            month_condition = ((result['month'].isin(datemonth)))
            result = result.loc[month_condition][['latitude','longitude']]  
            
             
    #return px.density_mapbox(result, lat='latitude', lon='longitude', radius=radius,
    #                    center=dict(lat=36.16228, lon=-86.774372), zoom=10,
    #                    mapbox_style="stamen-terrain")
    #use go.Densitymapbox(lat=quakes.Latitude, lon=quakes.Longitude, z=quakes.Magnitude,
    latInitial=36.16228
    lonInitial=-86.774372
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
                )
            ],
        ),
        )
    #fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=-86.774372,mapbox_center_lat=36.16228)
    #fig.update_layout(margin={"r":35,"t":0,"l":0,"b":0}, showlegend=False,zoom:10,)
    # fig.update_layout(
    #     title='Incident map-graph',
    #     autosize=True,
    #     plot_bgcolor="#1E1E1E",
    #     paper_bgcolor="#1E1E1E",
    #     #height=420,
    #     margin=dict(l=0, r=35, t=0, b=0),
    #     mapbox=go.layout.Mapbox(
    #         accesstoken=mapbox_access_token,
    #         style='light',
    #         bearing=0,
    #         center=go.layout.mapbox.Center(
    #             lat=36.16228,
    #             lon=-86.774372
    #         ),
    #         pitch=0,
    #         zoom=10
    #     )
    # )
    return fig

colorVal = [
        "#2202d1",
        "#2202d1",
        "#2202d1",
        "#FFFFFF",
        "#2202d1",
        "#2202d1",
        "#2202d1",
        "#2202d1",
        "#2202d1",
        "#2202d1",
        "#2202d1",
        "#2202d1",
    ]
# %%
@app.callback(
    Output('histogram', 'figure'),
    [Input("date-picker", "date"),
    Input("date-picker-end", "date"),
    Input('emd-card-num-dropdown', 'value'), Input("bar-selector", "value")]
)
def update_bar_chart(start_date, end_date, emd_card_num,selection):
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
    result['m']=result['month'].astype(int)
    result=result.sort_values('m')
    #print(result)
    xVal=result['month']
    yVal=result['count']
    
    xSelected = [int(x) for x in selection]
    #print (selection)
    for i in range(12):        
        if i+1 in xSelected:          
            colorVal[i]= "#FFFFFF"
        else:
            colorVal[i]= "#2202d1"
    #print(colorVal)        

    layout = go.Layout(
        bargap=0.1,
        bargroupgap=0,
        barmode="group",
        margin=go.layout.Margin(l=10, r=0, t=0, b=50),
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

# %%
if __name__ == '__main__':
	#app.run_server(host='0.0.0.0', port=8080, debug=True, use_reloader=False)  
    app.server.run(threaded=True)

# %%
