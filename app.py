#!/usr/bin/env python
# coding: utf-8

# In[1]:

import flask
import os
from random import randint
import plotly.express as px
from datetime import datetime as dt
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import zipfile


# In[2]:


# load data 
with zipfile.ZipFile('geo_out_july_2020_central_time.csv.zip', 'r') as zip_ref:
    zip_ref.extractall()
    
df = pd.read_csv('geo_out_july_2020_central_time.csv', index_col=0)
df['emdCardNumber'] = df['emdCardNumber'].str.upper()


# In[3]:


# preview dataset
df.head()


# In[4]:

server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
app = dash.Dash(__name__, server=server)


app.layout = html.Div([
    html.H1("Incident Reports"),
    html.Label([
       "Date Range",
        dcc.DatePickerRange(
            id='date-picker-range',
            min_date_allowed=dt(2019, 1, 1),
            max_date_allowed=dt(2020, 7, 9),
            start_date=dt(2019, 1, 1).date(),
            end_date=dt(2020, 6, 30).date()
        ),
    ]),
    html.Div(),
    html.Label([
        "Radius",
        dcc.Slider(
            id='heatmap-radius',
            min=1,
            max=10,
            step=0.05,
            marks={i: '{}'.format(i) for i in range(1, 11)},
            value=3
        ) 
    ]), 
    html.Div(),
    html.Label([
        "Incident Categories",
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
   
)  
        
    ]),
    
     html.P(
                            """Select specific months to show data from."""
                        ),
    
    
           html.Div(
                                    className="div-for-dropdown",
                                    children=[
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
                                            placeholder="Select certain months",
                                        )
                                    ],
                                ),
          html.P(
                            """Heatmap of the incidents."""
                        ),
    
    dcc.Graph(
        id='heatmap',
        style={
            'height': 700
        }
    ),
      html.P(
                            """Select different the month from the historgram to narrow down the heatmap shown above."""
                        ),
    dcc.Graph(
        id='bar-chart',
        style={
            'height': 300
        }
    )
])


# In[5]:


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


# In[6]:


@app.callback(
    Output('heatmap', 'figure'),
    [Input("date-picker-range", "start_date"),
    Input("date-picker-range", "end_date"),
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
            
             
    return px.density_mapbox(result, lat='latitude', lon='longitude', radius=radius,
                        center=dict(lat=36.16228, lon=-86.774372), zoom=10,
                        mapbox_style="stamen-terrain")


# In[7]:


@app.callback(
    Output('bar-chart', 'figure'),
    [Input("date-picker-range", "start_date"),
    Input("date-picker-range", "end_date"),
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

    return px.bar(result, x='month', y='count', text='count')


# In[8]:



# Run the Dash app
if __name__ == '__main__':
    app.server.run(debug=True, threaded=True)

# In[ ]:





# In[ ]:




