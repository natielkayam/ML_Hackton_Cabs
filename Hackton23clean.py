import numpy as np
import pandas as pd
df = pd.read_csv('CABS.csv',low_memory=False)
df.info()

#clean payment colls
df.drop(columns= ["payment_type", "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount", "improvement_surcharge", "total_amount", "congestion_surcharge", "airport_fee"], inplace = True)
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
df.drop(columns = ["store_and_fwd_flag"] , inplace= True)
df.drop(columns = ["passenger_count"] , inplace= True)

# #o each trip area give rate code
print(df['RatecodeID'].unique())
print(df['VendorID'].unique())

# Load the taxi zone lookup table
lookup_table_url = 'https://s3.amazonaws.com/nyc-tlc/misc/taxi+_zone_lookup.csv'
lookup_table = pd.read_csv(lookup_table_url)
# Assuming your DataFrame is called 'df'
df = df.merge(lookup_table[['LocationID', 'Zone']], left_on='PULocationID', right_on='LocationID', how='left')
df.rename(columns={'Zone': 'PickupLocation'}, inplace=True)
df = df.merge(lookup_table[['LocationID', 'Zone']], left_on='DOLocationID', right_on='LocationID', how='left')
df.rename(columns={'Zone': 'DropoffLocation'}, inplace=True)
df.drop(columns= ['PULocationID', 'DOLocationID' , 'LocationID_x', 'LocationID_y'], inplace= True)
df.drop(columns= 'VendorID' , inplace= True)

df.rename(columns={"tpep_pickup_datetime": "pickUP", "tpep_dropoff_datetime" : "dropOFF", "trip_distance" : "Length",}, inplace=True)
df.isnull().sum()
df.dropna(subset=['PickupLocation', 'DropoffLocation'], inplace=True)
df.dropna(subset=['RatecodeID'], inplace=True)


# #add day hour etc
df['pickUP'] = pd.to_datetime(df['pickUP'])
df['pickupHour'] = df['pickUP'].dt.hour
df['pickupDay'] = df['pickUP'].dt.day
df['pickupMonth'] = df['pickUP'].dt.month
df['pickupDayName'] = df['pickUP'].dt.day_name()

df['dropOFF'] = pd.to_datetime(df['dropOFF'])
df['dropOFFHour'] = df['dropOFF'].dt.hour
df['dropOFFDay'] = df['dropOFF'].dt.day
df['dropOFFMonth'] = df['dropOFF'].dt.month
df['dropOFFDayName'] = df['dropOFF'].dt.day_name()

#TODO:Add new time features
#1:travel in tourist season
tourist_season_start = pd.to_datetime('2023-05-01')
tourist_season_end = pd.to_datetime('2023-09-30')
# add column to indicate if pickup is in tourist season
df['is_tourist_season'] = (df['pickUP'].dt.month >= tourist_season_start.month) & (df['dropOFF'].dt.month <= tourist_season_end.month)
#2: travel in holiday
import holidays
us_holidays = holidays.US()
def is_holiday(date):
    return date in us_holidays
df['is_holiday'] = df['pickUP'].apply(is_holiday)


#3:add if travel in rush hour according to web serach on NY
rush_hour_start1 = pd.Timestamp('07:00:00').time()
rush_hour_end1 = pd.Timestamp('10:00:00').time()
rush_hour_start2 = pd.Timestamp('16:00:00').time()
rush_hour_end2 = pd.Timestamp('19:00:00').time()
# Check if the entire ride duration is during rush hour
df['rush_hour_ride'] = ((df['pickUP'].dt.weekday < 5) &
                        ((df['pickUP'].dt.time >= rush_hour_start1) |(df['pickUP'].dt.time >= rush_hour_start2)))

#5:get season
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'
# Add a new column 'season' based on the 'pickupMonth'
df['season'] = df['pickupMonth'].apply(get_season)

print(df.isnull().sum())
df.to_csv("hacktoncleanedDF.csv")

