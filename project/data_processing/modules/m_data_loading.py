import pandas as pd
import datetime


# loading data
# ------------------------------------------------------------------------------

def get_data_for_location(city_name='Stadt Freiburg', counter_site='FR1 Dreisam / Otto-Wels-Str.', channel_name='FR1 Dreisam / Hindenburgstr.', start_date=datetime.date(2021, 1, 1), end_date=datetime.date(2021, 12, 31)):
    '''
    Load data for specific counting station.

    Parameters
    ----------
    city_name : str
        specifies the city within which the counting station is lcoated
    counter_site : str
        specifies the location within the city, e.g., street name
    channel_name : str
        specifies a time series measured at the counting station
    start_date : datetime
        start of time period for which to load data
    end_date : datetime
        end of time period for which to load data       
    
    Returns
    -------
    pd.DataFrame with at least the columns 'temperature', 'is_busday' and 'rider_count'
    '''
    # import data
    combined_daily_dat = pd.read_pickle('./../../data/processed/combined_daily_dat.pkl')

    # extract data for counting station
    combined_daily_dat = combined_daily_dat[
        (combined_daily_dat.standort == city_name) &
        (combined_daily_dat.counter_site == counter_site) &
        (combined_daily_dat.channel_name == channel_name) &
        (combined_daily_dat.date >= start_date) &
        (combined_daily_dat.date <= end_date)
    ]
    
    return combined_daily_dat