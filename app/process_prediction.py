import pandas as pd

seg=pd.read_pickle('data/tdot/grouped_inrix.pkl')
seg=seg[seg.frc==0]
predRF=pd.read_pickle('data/tdot/DF_likelihood_spacetime_RF.pkl')
predRF=predRF.merge(seg[['grouped_xdsegid','county_inrix','geometry']],on='grouped_xdsegid')
predRF=predRF[['RF','time_local','geometry','grouped_xdsegid']]
predRF['day_of_week'] = predRF['time_local'].dt.dayofweek
predRF['month'] = predRF['time_local'].dt.month
predRF['year'] = predRF['time_local'].dt.year

