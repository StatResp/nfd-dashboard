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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import pandas as pd4
from statresp.utils.read_data import prepare_merged_dataset, check_trainmodel_exist
import os

from statresp.ml.forecasters.ml_models import ml_predict
from statresp.plotting.figuring_prediction import Figure_Generator
from statresp.utils.readConfig import metadata_creator
from statresp.utils.read_data import Save_training_results, Savedata
from statresp.utils.main_utils import Resuluts_Combiner, add_to_metadata_prediction
from statresp.ml.forecasters.modules.base import Forecaster
import warnings

# from forecasters.ml_models import learn
from statresp.simulation.simulation_allocation import get_allocation, get_distance, Data_collector, \
    simulation_HyperParams_creator, simulation_ExperimentInput_creator
from statresp.simulation.allocation.Allocation import Dispaching_Scoring, Find_DemandSupply_Location, Weight_and_Merge, \
    Responders_Location, Graph_Distance_Metric
from statresp.simulation.Reading_All_Distances import read_performance_simulation, average_performance_simulation
from statresp.plotting.figuring_simulation import simulation_bar_chart, simulation_alpha_chart
from statresp.utils.read_data import read_model_list
from statresp.utils.readConfig import metadata_creator
from statresp.utils.checkers import Time_checker, feature_checker
from multiprocessing import Pool, cpu_count
import time
import pandas as pd
import os
import shutil

warnings.filterwarnings("ignore")


def run_simulation(metadata):
    print('\nbeggining:')
    #metadata['simulation']['num_cores'] = 1
    if metadata['rewirte_Tag'] == True:
        if os.path.exists(metadata['Output_Address'] + '/simulation_example'):
            # os.remove(metadata['Output_Address']+'/simulation_example')
            shutil.rmtree(metadata['Output_Address'] + '/simulation_example')

    if 'Future' == Time_checker(
            pd.Timestamp(metadata['start_time_predict']).floor(freq='{}H'.format(int(metadata['window_size'] / 3600)))):
        raise ValueError('Simulation for future dates are not possible.')

    DF_Test_spacetime, df_grouped, df_incident, All_seg_incident, time_range = Data_collector(metadata)

    possible_facility_locations, demand_nodes, Distant_Dic, All_seg_incident, Grid_center = Find_DemandSupply_Location(
        df_grouped, df_incident, All_seg_incident, metadata, width=metadata['simulation']['grid_size'],
        height=metadata['simulation']['grid_size'], Source_crs='EPSG:4326', Intended_crs='EPSG:3310')

    metadata['simulation']['model_list'] = read_model_list(metadata)
    HyperParams = simulation_HyperParams_creator(metadata)

    experimental_inputs = simulation_ExperimentInput_creator(HyperParams, metadata, possible_facility_locations,
                                                             DF_Test_spacetime, All_seg_incident, Grid_center,
                                                             df_incident, demand_nodes, Distant_Dic, time_range)
    # %Finding the best location for the responders
    print('\nstarting Allocation')
    start_time = time.time()
    if metadata['simulation']['num_cores'] > 1:
        with Pool(processes=metadata['simulation']['num_cores']) as pool:
            pool.map(get_allocation, experimental_inputs)
    elif metadata['simulation']['num_cores'] == 1:
        for experimental_inputs_i in experimental_inputs:
            get_allocation(experimental_inputs_i)
    print('computation time for allocation: {}'.format(time.time() - start_time))

    # %Running a simulation
    print('\nstarting Simulation')
    start_time = time.time()
    for experimental_inputs_i in experimental_inputs:
        Map = get_distance(experimental_inputs_i, Save_Tag = False)
    # for args in experimental_inputs:
    #     get_dist_metric_for_allocation(args)
    print('computation time for simulation: {}'.format(time.time() - start_time))
    #Map.write_html('test_sim.html')



    DF_metric_allmethod_time = read_performance_simulation(HyperParams, metadata)
    DF_Distance_DistanceTravel_mean, DF_NotResponded_TotalNumAccidentsNotResponded_mean,            DF_NotResponded_TotalNumAccidentsNotResponded_max, DF_DistancePerAccident_DistanceTravelPerAccident_mean= \
        DF_Distance_DistanceTravel_mean, \
        DF_NotResponded_TotalNumAccidentsNotResponded_mean, \
        DF_NotResponded_TotalNumAccidentsNotResponded_max, \
        DF_DistancePerAccident_DistanceTravelPerAccident_mean = average_performance_simulation(DF_metric_allmethod_time, metadata)

    Distance_DistanceTravel_mean = DF_Distance_DistanceTravel_mean[(metadata['simulation']['num_ambulances'][0], metadata['simulation']['alpha'][0])].values[0]
    TotalNumAccidentsNotResponded_mean = DF_NotResponded_TotalNumAccidentsNotResponded_mean[(metadata['simulation']['num_ambulances'][0], metadata['simulation']['alpha'][0])].values[0]
    TotalNumAccidentsNotResponded_max = DF_NotResponded_TotalNumAccidentsNotResponded_max[(metadata['simulation']['num_ambulances'][0], metadata['simulation']['alpha'][0])].values[0]
    DistanceTravelPerAccident_mean = DF_DistancePerAccident_DistanceTravelPerAccident_mean[(metadata['simulation']['num_ambulances'][0], metadata['simulation']['alpha'][0])].values[0]

    fig = go.Figure(data=[go.Table(header=dict(values=['Metric', 'Value'],align='left',line_color='darkslategray',fill_color='lightskyblue' ),
                                   cells=dict(values=[['Total Distance (km) Travelled by Responders in 4-h', 'Total Distance (km) Travelled by Responders per Accidents in 4-h',
                                                       'Mean Number of Unattended Accidents among all 4-h windows', 'Max Number of Unattended Accidents among all 4-h windows'],
                                                      [Distance_DistanceTravel_mean.round(2), DistanceTravelPerAccident_mean.round(2), TotalNumAccidentsNotResponded_mean.round(2), TotalNumAccidentsNotResponded_max.round(2)]],
                                              line_color='darkslategray', fill_color="#31302F" ,align='left'))
                          ])

    layout = go.Layout(
        margin=go.layout.Margin(l=10, r=0, t=0, b=30),
        showlegend=False,
        plot_bgcolor="#31302F",
        paper_bgcolor="#31302F",
        font=dict(color="white"),
    )

    #fig.update_layout(title_text="Using update_layout() With Graph Object Figures", title_font_size=30)
    fig.update_layout(layout)
    #fig.write_html('test-peformance.html')


    if metadata['figure_tag'] == True:
        simulation_bar_chart(DF_metric_allmethod_time, metadata)
        simulation_alpha_chart(DF_metric_allmethod_time, metadata)

    print('run_simulation is done.')
    return Map, fig




def run_prediction(metadata, df_merged):
    print('\nbeggining:')
    results = {}
    model = Forecaster()
    model = model.load(metadata['Input_Address'] + '/trained_models/models/' + metadata['model_to_predict'])
    Window_Number_i = ' '
    learn_results, metadata = add_to_metadata_prediction(metadata, model, Window_Number_i)
    results['model'], results['df_predict'], results['results'] = ml_predict(metadata, model, df_merged)
    learn_results[Window_Number_i][model.metadata['model_type']] = results
    # %%saving
    DF_Pred_metric_time, DF_performance, DF_Test_spacetime, model_set = Resuluts_Combiner(learn_results, metadata,
                                                                                          Type='df_predict')
    Save_training_results(DF_Pred_metric_time, DF_performance[DF_performance['rolling_window_group'] == 'Mean'],
                          DF_Test_spacetime, model_set, metadata['Output_Address'] + '/prediction_example', 'pkl')
    Savedata(df_predict=results['df_predict'], Name=model.metadata['Data_Name'], directory=metadata['Output_Address'])

    Window_Number_i  # It will automaically gives you the results for the last window; however, you can change it here
    plt1, plt2, Map1, Map2= Figure_Generator(Window_Number_i, learn_results, metadata, Type='df_predict', Address = metadata['Output_Address'] + '/prediction_example/figures',Save_Tag=False)
    print('run_prediction is done.')
    return results, plt1, plt2, Map1, Map2, DF_performance[DF_performance['rolling_window_group']=='Mean'][['accuracy', 'precision','recall', 'f1','spearman_corr', 'pearson_corr', 'Correctness']]





# set configurations
mapbox_style = "light"
mapbox_access_token = "pk.eyJ1Ijoidmlzb3ItdnUiLCJhIjoiY2tkdTZteWt4MHZ1cDJ4cXMwMnkzNjNwdSJ9.-O6AIHBGu4oLy3dQ3Tu2XA"
MAPBOX_ACCESS_TOKEN = mapbox_access_token
pd.set_option('display.max_columns', None)

latInitial = 36.16228
lonInitial = -86.774372


metadata = metadata_creator()
metadata['rewirte_Tag'] = True
#df_merged = pd.read_parquet(metadata['merged_pickle_address'], columns=['time_local','county'])
df_merged = prepare_merged_dataset(metadata)
if 'day_of_week' in df_merged.columns:
    df_merged.drop('day_of_week', axis=1, inplace=True)
df_merged['day_of_week'] = df_merged['time_local'].dt.dayofweek
if not 'month' in df_merged.columns:
    df_merged['day_of_week'] = df_merged['time_local'].dt.month
#results, plt1, plt2, Map1, Map2, DF_performance = run_prediction(metadata, df_merged)
#run_simulation(metadata)
'''
#results, plt1, plt2, Map1, Map2, DF_performance= run_prediction(metadata, df_merged)
#model_i   = learn_results[Rolling_Window_Num_str][m]['model'].metadata['Name']
model_i='RF+RUS1.0+KM2'
Window_Number_i=' '
Rolling_Window_Num_str=Window_Number_i
'''
time_range=df_merged['time_local'].drop_duplicates().sort_values().tolist()  #


#table2 = pq.read_table('data/nfd/incidents_july_2020_2.parquet')

# df = dataextract.decompress_pickle(
#    'data/nfd/geo_out_july_2020_central_time.pbz2')
df_merged["month"]=df_merged['time_local'].dt.month
#df =table2.to_pandas()
#df = dd.read_parquet('tdot_incidents.parquet', engine='pyarrow')

startdate = df_merged['time_local'].min()
enddate =  df_merged['time_local'].max()

startdate_hostorical = metadata['start_time_predict']
enddate_hostorical = metadata['end_time_predict']

# if you design a radio button for future and historical, we can use the following for future time range.
startdate_future = pd.Timestamp.now().floor(freq='4H')
enddate_future = pd.Timestamp.now().floor(freq='4H')+pd.DateOffset(days=5)


counties = df_merged.county.drop_duplicates()
#df = dd.from_pandas(df,  npartitions=3)
#startdate = df.alarm_date.min().compute()
#enddate = df.alarm_date.max().compute()
#counties = df.county_inrix.drop_duplicates().compute()

counties = counties.tolist()
counties.sort()
print(counties)
cluster_list=[1,2]

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
                                date=startdate_hostorical,
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
                                    date=enddate_hostorical,
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

                        html.Div(className="card p-1 m-1 bg-dark",
                                 children=[
                                     dcc.Markdown(
                                         '''Adjust Slider below to configure the number of responders (ambulances).'''),
                                     dcc.Slider(
                                         id='num_ambulances',
                                         min=0,
                                         max=30,
                                         step=1,
                                         marks={i: '{}'.format(i)
                                                for i in [1, 5, 10, 15, 20, 25, 30]},
                                         value=5
                                     ),
                                 ]
                                 ),

                        html.Div(className="card p-1 m-1 bg-dark",
                                 children=[
                                     dcc.Markdown(
                                         '''Adjust Slider below to configure the parameter alpha in the modified p-median problem'''),
                                     dcc.Slider(
                                         id='alpha',
                                         min=0,
                                         max=2,
                                         step=0.25,
                                         marks={i: '{}'.format(i)
                                                for i in [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]},
                                         value=0
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
                        html.Div(className="p-2 m-2 card  bg-dark",children=[
                             dbc.Tabs(id='tabs', active_tab="incidents", children=[
                                 dbc.Tab(label='Predicted likelihood', tab_id='incidents', className="bg-dark text-white", children=[dcc.Loading(
                                     id="loading-icon1", children=[dcc.Graph(id="map-graph"), ], type='default')]),
                                 dbc.Tab(label='Clusters', tab_id='clusters', className="bg-dark text-white", children=[dcc.Loading(
                                     id="loading-icon-cluster", children=[dcc.Graph(id="map-clusters"), ], type='default')]),
                                 dbc.Tab(label='Location of Accidents and Responders in each 4-h', tab_id='sims', className="bg-dark text-white",children=[dcc.Loading(
                                             id="loading-icon-sim", children=[dcc.Graph(id="map-sims"), ], type='default')]),
                             ]),
                         ]),
                        html.Div(className="p-0 m-0 card bg-dark", children=[
                            dbc.Tabs(id = 'histogram-basis', active_tab = "radar", children=[
                                dbc.Tab(label='Performance', tab_id='radar', className="flex-grow-1",  children=[dcc.Loading(
                                    id="loading-icon2", children=[dcc.Graph(id="map-radar"), ], type='default')]),
                                dbc.Tab(label='Simulation Performance', tab_id='sim', className="flex-grow-1", children = [dcc.Loading(
                                    id="loading-icon3", children=[dcc.Graph(id="map-sim_table"), ], type='default')]),
                            ]),
                        ]),
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

        starttime=tt(hour=hourmin)
        if hourmax!=24:
            endtime = tt(hour=max(0,hourmax-1),second=59)
        else:
            endtime = tt(hour=23, second=59)
        timecondition = ((df_merged['time_local'].dt.time >= starttime)
                         & (df_merged['time_local'].dt.time <= endtime))


        # timecondition=True
        # result = df[timecondition & date_condition & month_condition & weekday_condition & county_condition].compute()
        result     = df_merged[timecondition & date_condition & month_condition & weekday_condition & county_condition]

    # except Exception:
    else:
        print("Exception in user code:")
        traceback.print_exc(file=sys.stdout)

    #print(result)
    return result

def return_metadata(start_date, end_date, counties, months, timerange, days, num_ambulances, alpha):

    metadata['start_time_predict']= pd.Timestamp(dateparser.parse(start_date))+pd.DateOffset(seconds=0)
    metadata['end_time_predict']= pd.Timestamp(dateparser.parse(end_date))+pd.DateOffset(seconds=0)
    metadata['county_list_pred']= counties
    metadata['segment_list_pred'] = []
    metadata['simulation']['num_ambulances'] = [num_ambulances]
    metadata['simulation']['alpha'] = [alpha]
    return metadata

# Process


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
    [Output('map-graph', 'figure'),Output('map-clusters', 'figure'),Output('map-radar', 'figure'),
    Output('map-sims', 'figure'),Output('map-sim_table', 'figure'),
     ],
    [Input("date-picker", "date"),
     Input("date-picker-end", "date"),
     Input("map-graph-radius", "value"),
     Input('county', 'value'),
     Input("month-selector", "value"),
     Input("time-slider", "value"),   Input("day-selector", "value"),
     Input("num_ambulances", "value"),  Input("alpha", "value"),
     ]
)

def update_map_graph(start_date, end_date, radius, counties, months, timerange,   days, num_ambulances, alpha):
    'Beggining of Prediction Dashboard:'
    metadata_new = return_metadata(start_date, end_date, counties, months, timerange, days, num_ambulances, alpha)
    print("metadata['simulation']['num_ambulances']",metadata['simulation']['num_ambulances'])
    print("metadata['simulation']['alpha']", metadata['simulation']['alpha'])
    df_merged_new = return_merged(start_date, end_date, counties, months, timerange, days)
    results, plt1, plt2, Map1, Map2, DF_performance = run_prediction(metadata_new, df_merged_new)
    if len(results[ 'df_predict']['time_local'].unique())<=42:
        if len(results[ 'df_predict']['time_local'].unique())>=6:
            print('Simulation may take a while')
        Map_sim, table = run_simulation(metadata)
        Radar = radar_chart(DF_performance)
        print('DF_performance:', DF_performance)
    if len(results['df_predict']['time_local'].unique()) > 42:
        Map_sim= return_empty_fig()
        table= return_empty_fig()
        Radar = return_empty_fig()


    '''
    Map1.update_layout(
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
    '''
    return Map1, Map2, Radar, Map_sim, table









def radar_chart(result_performance):
    result_performance=result_performance.rename(columns={'precision':'Precision',
                                                      'recall': 'Recall',
                                                      'f1': 'F1',
                                                      'spearman_corr': 'Spearman Cor.',
                                                      'pearson_corr': 'Pearson Cor.',
                                                      'Correctness': 'Exactness'})

    result_performance.drop('accuracy',axis=1,inplace=True)
    value = result_performance.iloc[0].tolist()
    value = [*value, value[0]]
    categories = result_performance.columns.tolist()
    categories = [*categories, categories[0]]

    fig = go.Figure(
        data=[
            go.Scatterpolar(r=value, theta=categories, fill='toself'),
        ],
        layout=go.Layout(
            autosize=True,
            margin=go.layout.Margin(l=10, r=10, t=10, b=30),
            showlegend=False,
            plot_bgcolor="#31302F",
            paper_bgcolor="#31302F",
            dragmode="select",
            font=dict(color="white"),
            polar={'radialaxis': {'visible': True, 'range': [0, 1]}},
        )
    )
    return fig


# %%
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
    # app.server.run(threaded=True)

# %%
