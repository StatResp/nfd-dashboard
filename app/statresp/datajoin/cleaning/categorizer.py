import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def Categorizer_Dic(df_: pd.DataFrame):
    m_in_mile: double = 1609.344
    #%% The following section will categorize each feature based on the intervals and labels you defined below
    Features=dict()


    temp_min=-10
    temp_max=30
    if 'temp_min' in df_.columns:
        temp_min = min(temp_min, df_['temp_min'].min())
        temp_max = max(temp_max, df_['temp_min'].max())
    if 'temp_mean' in df_.columns:
        temp_min = min(temp_min, df_['temp_mean'].min())
        temp_max = max(temp_max, df_['temp_mean'].max())
    if 'temp_max' in df_.columns:
        temp_min = min(temp_min, df_['temp_max'].min())
        temp_max = max(temp_max, df_['temp_max'].max())
    if 'temp_min' in df_.columns:
        Features['temp_min'] = {'bins': [temp_min,0,10,25,temp_max],'labels': ['Freezing','Cold','Mild','Hot']}
    if 'temp_mean' in df_.columns:
        Features['temp_mean'] ={'bins': [temp_min,0,10,25,temp_max],'labels': ['Freezing','Cold','Mild','Hot']}
    if 'temp_max' in df_.columns:
        Features['temp_max'] = {'bins': [temp_min,0,10,25,temp_max],'labels': ['Freezing','Cold','Mild','Hot']}

    wind_min = 0
    wind_max = 10
    if 'wind_spd_min' in df_.columns:
        wind_min = min(wind_min, df_['wind_spd_min'].min())
        wind_max = max(wind_max, df_['wind_spd_min'].max())
    if 'wind_spd_mean' in df_.columns:
        wind_min = min(wind_min, df_['wind_spd_mean'].min())
        wind_max = max(wind_max, df_['wind_spd_mean'].max())
    if 'wind_spd_max' in df_.columns:
        wind_min = min(wind_min, df_['wind_spd_max'].min())
        wind_max = max(wind_max, df_['wind_spd_max'].max())
    if 'wind_spd_min' in df_.columns:
        Features['wind_spd_min'] = { 'bins': [wind_min,3,7,wind_max],'labels': ['No_Wind','Mild','Windy']}
    if 'wind_spd_mean' in df_.columns:
        Features['wind_spd_mean'] ={ 'bins': [wind_min,3,7,wind_max],'labels': ['No_Wind','Mild','Windy']}
    if 'wind_spd_max' in df_.columns:
        Features['wind_spd_max'] = { 'bins': [wind_min,3,7,wind_max],'labels': ['No_Wind','Mild','Windy']}

    vis_min = 0
    vis_max = 5
    if 'vis_min' in df_.columns:
        vis_min = min(vis_min, df_['vis_min'].min())
        vis_max = max(vis_max, df_['vis_min'].max())
    if 'vis_mean' in df_.columns:
        vis_min = min(vis_min, df_['vis_mean'].min())
        vis_max = max(vis_max, df_['vis_mean'].max())
    if 'vis_max' in df_.columns:
        vis_min = min(vis_min, df_['vis_max'].min())
        vis_max = max(vis_max, df_['vis_max'].max())
    if 'vis_min' in df_.columns:
        Features['vis_min'] = { 'bins': [vis_min, 0.8,3, vis_max],'labels': ['Low','Fair','Clear']}
    if 'vis_mean' in df_.columns:
        Features['vis_mean'] ={ 'bins': [vis_min, 0.8,3, vis_max],'labels': ['Low','Fair','Clear']}
    if 'vis_max' in df_.columns:
        Features['vis_max'] = { 'bins': [vis_min, 0.8,3, vis_max],'labels': ['Low','Fair','Clear']}


    precip_min = 0
    precip_max = 2
    if 'precip_min' in df_.columns:
        precip_min = min(precip_min, df_['precip_min'].min())
        precip_max = max(precip_max, df_['precip_min'].max())
    if 'precip_mean' in df_.columns:
        precip_min = min(precip_min, df_['precip_mean'].min())
        precip_max = max(precip_max, df_['precip_mean'].max())
    if 'precip_max' in df_.columns:
        precip_min = min(precip_min, df_['precip_max'].min())
        precip_max = max(precip_max, df_['precip_max'].max())
    if 'precip_min' in df_.columns:
        Features['precip_min'] = { 'bins': [precip_min,0.01,1,precip_max],'labels': ['No_Rain','Mild','Heavy']}
    if 'precip_mean' in df_.columns:
        Features['precip_mean'] ={ 'bins': [precip_min,0.01,1,precip_max],'labels': ['No_Rain','Mild','Heavy']}
    if 'precip_max' in df_.columns:
        Features['precip_max'] = { 'bins': [precip_min,0.01,1,precip_max],'labels': ['No_Rain','Mild','Heavy']}



    speed_max = 80
    if 'speed_min' in df_.columns:
        speed_max = max(speed_max, df_['speed_min'].max())
    if 'speed_mean' in df_.columns:
        speed_max = max(speed_max, df_['speed_mean'].max())
    if 'speed_max' in df_.columns:
        speed_max = max(speed_max, df_['speed_max'].max())
    if 'speed_min' in df_.columns:
        Features['speed_min'] = { 'bins': [0,20,40,60,speed_max],'labels': ['[0-20)','[20-40)', '[40-60)','[60-max]']}
    if 'speed_mean' in df_.columns:
        Features['speed_mean'] = { 'bins': [0,20,40,60,speed_max],'labels': ['[0-20)','[20-40)', '[40-60)','[60-max]']}
    if 'speed_max' in df_.columns:
        Features['speed_max'] = { 'bins': [0,20,40,60,speed_max],'labels': ['[0-20)','[20-40)', '[40-60)','[60-max]']}

    reference_speed_max = 80
    if 'reference_speed_min' in df_.columns:
        reference_speed_max = max(reference_speed_max, df_['reference_speed_min'].max())
    if 'reference_speed_mean' in df_.columns:
        reference_speed_max = max(reference_speed_max, df_['reference_speed_mean'].max())
    if 'reference_speed_max' in df_.columns:
        reference_speed_max = max(reference_speed_max, df_['reference_speed_max'].max())
    if 'reference_speed_min' in df_.columns:
        Features['reference_speed_min'] = {'bins': [0, 20, 40, 60, reference_speed_max],'labels': ['[0-20)', '[20-40)', '[40-60)', '[60-max]']}
    if 'reference_speed_mean' in df_.columns:
        Features['reference_speed_mean'] = {'bins': [0, 20, 40, 60, reference_speed_max], 'labels': ['[0-20)', '[20-40)', '[40-60)', '[60-max]']}
    if 'reference_speed_max' in df_.columns:
        Features['reference_speed_max'] = {'bins': [0, 20, 40, 60, reference_speed_max],'labels': ['[0-20)', '[20-40)', '[40-60)', '[60-max]']}

    average_speed_max = 80
    if 'average_speed_min' in df_.columns:
        average_speed_max = max(average_speed_max, df_['average_speed_min'].max())
    if 'average_speed_mean' in df_.columns:
        average_speed_max = max(average_speed_max, df_['average_speed_mean'].max())
    if 'average_speed_max' in df_.columns:
        average_speed_max = max(average_speed_max, df_['average_speed_max'].max())
    if 'average_speed_min' in df_.columns:
        Features['average_speed_min'] = {'bins': [0, 20, 40, 60, average_speed_max],'labels': ['[0-20)', '[20-40)', '[40-60)', '[60-max]']}
    if 'average_speed_mean' in df_.columns:
        Features['average_speed_mean'] = {'bins': [0, 20, 40, 60, average_speed_max], 'labels': ['[0-20)', '[20-40)', '[40-60)', '[60-max]']}
    if 'average_speed_max' in df_.columns:
        Features['average_speed_max'] = {'bins': [0, 20, 40, 60, reference_speed_max],'labels': ['[0-20)', '[20-40)', '[40-60)', '[60-max]']}


    congestion_min = 0
    congestion_max = 1
    '''
    if 'congestion_min' in df_.columns:
        congestion_min = min(congestion_min, df_['congestion_min'].min())
        congestion_max = max(congestion_max, df_['congestion_min'].max())
    if 'congestion_mean' in df_.columns:
        congestion_min = min(congestion_min, df_['congestion_mean'].min())
        congestion_max = max(congestion_max, df_['congestion_mean'].max())
    if 'congestion_max' in df_.columns:
        congestion_min = min(congestion_min, df_['congestion_max'].min())
        congestion_max = max(congestion_max, df_['congestion_max'].max())
    '''
    if 'congestion_min' in df_.columns:
        Features['congestion_min'] = {'bins': [congestion_min, 0.1, 0.5, congestion_max], 'labels': ['Light', 'Medium', 'Congested']}
    if 'congestion_mean' in df_.columns:
        Features['congestion_mean']= {'bins': [congestion_min, 0.1, 0.5, congestion_max], 'labels': ['Light', 'Medium', 'Congested']}
    if 'congestion_max' in df_.columns:
        Features['congestion_max'] = {'bins': [congestion_min, 0.1, 0.5, congestion_max], 'labels': ['Light', 'Medium', 'Congested']}

    if 'isf_length' in df_.columns:
        Features['isf_length'] = { 'bins': [0,  0.8,0.9,0.95, 1], 'labels': ['[min-0.8)','[0.8-09)','[0.9-0.95)','[0.95-max]']}
    if 'isf_length_min' in df_.columns:
        Features['isf_length_min'] = { 'bins': [0,  0.8,0.9,0.95, 1], 'labels': ['[min-0.8)','[0.8-09)','[0.9-0.95)','[0.95-max]']}

    if 'length_m' in df_.columns:
        Features['length_m'] = { 'bins': [0,500,1000,1500, df_['length_m'].max()], 'labels': ['[0-500)','[500-1000)','[1000-1500)','[1500-max]']}

    if 'miles' in df_.columns:
        Features['miles'] = { 'bins': [0, (500/m_in_mile),(1000/m_in_mile),(1500/m_in_mile),  df_['miles'].max()], 'labels': ['[0-500/m_in_mile)','[500/m_in_mile-1000/m_in_mile)', '[1000/m_in_mile-1500/m_in_mile)','[1500/m_in_mile-max]']}

    if 'lanes' in df_.columns:
        Features['lanes'] = {'bins': [0, 1.5, 2.5, 3.5, 4.5, np.inf], 'labels': ['1', '2', '3', '4', '5']}

    if 'slope' in df_.columns:
        Features['slope'] = {'bins': [-np.inf, -0.3, -0.1, 0.1, 0.3, np.inf], 'labels': ['-inf--.1', '-.1--.01', '-.01-.01', '.01-.1', '.1-inf']}

    if 'slope_median' in df_.columns:
        Features['slope_median'] = { 'bins': [-np.inf,-0.3,-0.1,0.1,0.3,np.inf], 'labels': ['-inf--.1','-.1--.01','-.01-.01','.01-.1','.1-inf']}

    if 'ends_ele_diff' in df_.columns:
        Features['ends_ele_diff'] = { 'bins': [-np.inf, -100,-50,50,100,  np.inf],  'labels': ['[-inf--100)','[-100--50)','[-50-50)','[50-100)','(100-inf]']}

    if 'max_ele_diff' in df_.columns:
        Features['max_ele_diff'] = { 'bins': [0, 25,50,75,100,  np.inf],  'labels': ['[-inf-25)','[25-50)','[50-75)','[75-100)','(100-inf]']}

    if 'window' in df_.columns:
        Features['window'] = { 'bins': [0,0.5,1.5,2.5,3.5,4.5,5], 'labels': ['0','1','2','3','4','5']}

    if 'window_peak' in df_.columns:
        Features['window_peak'] = { 'bins': [0,0.5,1], 'labels': ['not_peak','peak']} #windos 0,2,3,5 are not peak; 1 an 4 are peak

    if 'is_weekend' in df_.columns:
        Features['is_weekend'] = { 'bins': [0,0.5,1], 'labels': ['Weekday','Weekend']}

    if 'year' in df_.columns:
        Features['year'] = { 'bins': [2016.5, 2017.5, 2018.5, 2019.5, 2020.5],'labels': ['2017','2018','2019','2020']}

    if 'month' in df_.columns:
        Features['month'] = { 'bins': [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5], 'labels': ['1','2','3','4','5','6','7','8','9','10','11','12']}

    if 'count_incidents_past_window' in df_.columns:
        Features['count_incidents_past_window'] = { 'bins': [0, 0.1, 0.5, 1], 'labels': ['0-0.1','0.1-0.5', '0.5-1']}

    if 'count_incidents_exact_yesterday' in df_.columns:
        Features['count_incidents_exact_yesterday'] = { 'bins': [0, 0.1, 0.5, 1], 'labels': ['0-0.1','0.1-0.5', '0.5-1']}

    if 'count_incidents_exact_last_week' in df_.columns:
        Features['count_incidents_exact_last_week'] = { 'bins': [0, 0.1, 0.5, 1], 'labels': ['0-0.1','0.1-0.5', '0.5-1']}

    if 'count_incidents_exact_last_month' in df_.columns:
        Features['count_incidents_exact_last_month'] = { 'bins': [0, 0.1, 0.5, 1], 'labels': ['0-0.1','0.1-0.5', '0.5-1']}

    if 'mean_incidents_last_24_hours' in df_.columns:
        Features['mean_incidents_last_24_hours'] = { 'bins': [0, 0.1, 0.5, 1], 'labels': ['0-0.1','0.1-0.5', '0.5-1']}

    if 'mean_incidents_last_7_days' in df_.columns:
        Features['mean_incidents_last_7_days'] = { 'bins': [0, 0.1, 0.5, 1], 'labels': ['0-0.1','0.1-0.5', '0.5-1']}

    if 'mean_incidents_last_4_weeks' in df_.columns:
        Features['mean_incidents_last_4_weeks'] = { 'bins': [0, 0.1, 0.5, 1], 'labels': ['0-0.1','0.1-0.5', '0.5-1']}

    if 'mean_incidents_over_all_windows' in df_.columns:
        Features['mean_incidents_over_all_windows'] = { 'bins': [0, 0.1, 0.5, 1], 'labels': ['0-0.1','0.1-0.5', '0.5-1']}

    return Features

def categorize_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    features = Categorizer_Dic(df)
    cat_features = []

    for i in list(features.keys()):
        print('           ',i)
        try:
            df[i+'_cat_']=pd.cut(df[i],bins=features[i]['bins'],labels=features[i]['labels'],include_lowest=True, right=True)
            cat_features += [i + '_cat_']
            #df = df.drop([i], axis = 1)
        except Exception as e:
            print(e)
            print(f"min: {df[i].min()}")
            print(f"max: {df[i].max()}")
            print(features[i]['bins'])
            print("\n")
    return df




def Filter_Combo_Builder(All_Possible_Filters=None):
    '''
    All_Possible_Filters=  {'temp_mean_Cat':[['Hot'],['']],
                            'precip_mean_Cat':[['Heavy'],['']],
                            'congestion_mean_Cat':[['Mild','Heavy'],['']],
                            'past_incidents_all_last4weeks_Cat':[['Many'],['']],
                            'weekend_Cat':[['Weekday'],['']],
                            }
    '''

    '''   
    All_Possible_Dic={0: {'temp_mean_Cat':['Hot']},
                     1: {'precip_mean_Cat':['Heavy']}, 
                     2: {'congestion_mean_Cat':['Mild','Heavy']},  
                     3: {'past_incidents_all_last4weeks_Cat':['Many']},  
                     4: {'weekend_Cat':['Weekday']},
                     }
    '''
    if All_Possible_Filters is None:
        All_Possible_Filters = {'congestion_mean_cat_': [['Congested'],['']],
                                'temp_mean_cat_':[['Hot'], ['Freezing'],['']],
                                'precip_mean_cat_': [['Heavy'], ['']],
                                'is_weekend_cat_': [['Weekday'], ['']],
        }

    def Recursive(Intial_Dic, Length_i, Counter, All_Possible_Dic):
        if Length_i == Length:
            print(Counter, Length_i, Intial_Dic)
            All_Possible_Dic[Counter] = Intial_Dic.copy()
            return Counter + 1, All_Possible_Dic
        else:
            for j in All_Possible_Filters[Key_List[Length_i]]:
                # print(j,Counter)
                # print(Length_i+1,)
                # print(Counter+1)
                Intial_Dic[Key_List[Length_i]] = j
                Counter, All_Possible_Dic = Recursive(Intial_Dic, Length_i + 1, Counter, All_Possible_Dic)
            return Counter, All_Possible_Dic


    # The following section calculates all possible combinations
    Counter = 0
    Key_List = list(All_Possible_Filters.keys())
    Length = len(Key_List)
    Intial_Dic = dict()
    All_Possible_Dic = dict()
    for i in All_Possible_Filters.keys():
        Intial_Dic[i] = ''

    Counter, All_Possible_Dic = Recursive(Intial_Dic, 0, Counter, All_Possible_Dic)
    return All_Possible_Dic, All_Possible_Filters





def FILTER_plotter(df_All_Filtered):
    fig = px.bar(df_All_Filtered, x='Tag', y='Average_Number_Incidents')
    layout = go.Layout(
        bargap=0.05,
        autosize=True,
        bargroupgap=0,
        #barmode="group",
        margin=go.layout.Margin(l=10, r=0, t=0, b=30),
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
    fig.update_layout(layout)
    return fig


def FILTER_calculator(df_, All_Possible_Dic, Title):
    '''
    Based on All_Filter_Dict, this function generate a DF and plots that shows the total number of observation and average incident_TF for each filter key.
    Parameters
    ----------
    df_ : DF
        The DF that includes all the features including categorical features created based on continues ones and incident_occurred.
    All_Filter_Dict : Dictionary
        This is a nested dictionary. The first layer indicates the set of filteration and the second layer includes the catagories and key for filteration .
    Returns
    -------
    df_All_Filtered : DF
        it includes the total number of observation and average incident_TF for each filter key.
    '''
    df_All_Filtered = pd.DataFrame()
    for j in list(All_Possible_Dic.keys()):
        print(j, ', ', end='')
        Filter_Dict = All_Possible_Dic[j]

        df_new = df_[list(Filter_Dict.keys()) + ['incident_occurred']].copy()
        for i in list(Filter_Dict.keys()):
            if len(Filter_Dict[i]) > 1:
                # print(i, ': len(Filter_Dict[i])>1')
                df_new[i] = df_new[i].astype('str')
                df_new[i] = df_new[i].mask(df_new[i].isin(Filter_Dict[i]), '|'.join(Filter_Dict[i]))
            elif Filter_Dict[i] == ['']:
                Filter_Dict.pop(i)

        if len(Filter_Dict.keys()) > 0:
            # if there is at least 1 key to filter
            df_Filtered = df_new.groupby(list(Filter_Dict.keys())).agg(
                {'incident_occurred': ['mean', 'sum', 'count']})
            df_Filtered.columns = ['Average_Number_Incidents', 'Total_Number_Accidents', 'Total_Number_Observation']
            df_Filtered = df_Filtered.reset_index()
            df_Filtered['Tag'] = ''
            for i in list(Filter_Dict.keys()):
                df_Filtered = df_Filtered[df_Filtered[i].isin(['|'.join(Filter_Dict[i])])]
                df_Filtered['Tag'] = df_Filtered['Tag'].astype(str) + '+' + df_Filtered[i].astype(
                    str)  # '|'.join(Filter_Dict[i])
        else:
            # if there is no key to folter
            df_Filtered['Average_Number_Incidents'] = df_new['incident_occurred'].sum() / len(df_new)
            df_Filtered['Total_Number_Observation'] = len(df_new)
            df_Filtered['Tag'] = '+No_Filter'
        df_All_Filtered = df_All_Filtered.append(df_Filtered, ignore_index=True)

    df_All_Filtered['Tag'] = df_All_Filtered['Tag'].str.slice(start=1)

    return df_All_Filtered
