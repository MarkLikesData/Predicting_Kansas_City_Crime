### This repo contains information regarding crime rates and statistics for the greater Kansas City Metroplolitan Area. To use these files, clone the repo and ensure all dependencies listed in the requirements.txt file are installed on your device.

#### This is the result of a collaborative machine learning project. We attempted to determine if it is possible to predict whether a victim of crime is more likely to be male or female based on various features found in the historical data.



# Predicting Kansas City Crime Using Machine Learning


```python
### Import Dependancies

import matplotlib
from matplotlib import pyplot as plt
%matplotlib inline
import numpy as np
import os
import pandas as pd
from pandas import datetime
pd.options.display.max_columns = 100
import seaborn as sns
import warnings
warnings.simplefilter('ignore')

# Machine Learning Specific Dependancies:

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
```

### Obtain KCMO crime data (Raw data came from data.kcmo.org)


```python
file_name = os.path.join('Resources', 'CSVs', 'KCPD_Crime_Data_2017.csv')
kc_crime = pd.read_csv(file_name)
kc_crime.reset_index()
kc_crime.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Report_No</th>
      <th>Reported_Date</th>
      <th>Reported_Time</th>
      <th>From_Date</th>
      <th>From_Time</th>
      <th>To_Date</th>
      <th>To_Time</th>
      <th>Offense</th>
      <th>IBRS</th>
      <th>Description</th>
      <th>Beat</th>
      <th>Address</th>
      <th>City</th>
      <th>Zip Code</th>
      <th>Rep_Dist</th>
      <th>Area</th>
      <th>DVFlag</th>
      <th>Invl_No</th>
      <th>Involvement</th>
      <th>Race</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Firearm Used Flag</th>
      <th>Location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100080848</td>
      <td>5/28/2017</td>
      <td>3:44</td>
      <td>5/28/2017</td>
      <td>2:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1352</td>
      <td>280</td>
      <td>Stolen_Property</td>
      <td>133.0</td>
      <td>4000  MILL ST</td>
      <td>KANSAS CITY</td>
      <td>64111</td>
      <td>PJ3255</td>
      <td>CPD</td>
      <td>U</td>
      <td>1</td>
      <td>VIC</td>
      <td>W</td>
      <td>F</td>
      <td>29.0</td>
      <td>N</td>
      <td>4000 MILL ST\nKANSAS CITY 64111\n(39.053635, -...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>120046817</td>
      <td>11/21/2017</td>
      <td>13:30</td>
      <td>11/20/2017</td>
      <td>9:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>101</td>
      <td>09A</td>
      <td>Homicide_Non_Negl</td>
      <td>112.0</td>
      <td>1100  LOCUST ST</td>
      <td>KANSAS CITY</td>
      <td>64105</td>
      <td>PJ1029</td>
      <td>CPD</td>
      <td>U</td>
      <td>1</td>
      <td>SUS</td>
      <td>B</td>
      <td>M</td>
      <td>NaN</td>
      <td>Y</td>
      <td>1100 LOCUST ST\nKANSAS CITY 64105\n(39.10091, ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>120046817</td>
      <td>11/21/2017</td>
      <td>13:30</td>
      <td>11/20/2017</td>
      <td>9:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>101</td>
      <td>09A</td>
      <td>Homicide_Non_Negl</td>
      <td>112.0</td>
      <td>1100  LOCUST ST</td>
      <td>KANSAS CITY</td>
      <td>64105</td>
      <td>PJ1029</td>
      <td>CPD</td>
      <td>N</td>
      <td>1</td>
      <td>VIC</td>
      <td>B</td>
      <td>F</td>
      <td>69.0</td>
      <td>Y</td>
      <td>1100 LOCUST ST\nKANSAS CITY 64105\n(39.10091, ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>120085080</td>
      <td>4/27/2017</td>
      <td>11:12</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>201</td>
      <td>11A</td>
      <td>Rape</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>99999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>U</td>
      <td>1</td>
      <td>VIC</td>
      <td>B</td>
      <td>F</td>
      <td>21.0</td>
      <td>N</td>
      <td>99999\n</td>
    </tr>
    <tr>
      <th>4</th>
      <td>120085080</td>
      <td>4/27/2017</td>
      <td>11:12</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>201</td>
      <td>11A</td>
      <td>Rape</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>99999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>U</td>
      <td>1</td>
      <td>SUS</td>
      <td>B</td>
      <td>M</td>
      <td>52.0</td>
      <td>N</td>
      <td>99999\n</td>
    </tr>
  </tbody>
</table>
</div>



### Features in dataset

* Age
* Date
* Time
* Crime
* Zip
* Firearm Involved
* Geolocation
* Gender


```python
kc_crime.shape
```




    (132131, 24)



### Rename & drop columns


```python
kc_crime.rename(columns={'Zip Code':'zip_code'}, inplace=True)
kc_crime_dropped_columns = kc_crime.drop(['Report_No', 'To_Date', 'From_Date', 'To_Time', 'From_Time',
                          'Offense', 'IBRS', 'Rep_Dist', 'Area', 'Beat', 'DVFlag', 'Address','Invl_No'], axis=1)
```

### Filter for male and female victims age 90 and under in KCMO


```python
kc_crime_clean_zips = kc_crime_dropped_columns[kc_crime_dropped_columns.zip_code != 99999]
only_kc_crime = kc_crime_clean_zips[kc_crime_clean_zips.City.str.contains("KANSAS CITY") == True]

victims = only_kc_crime[only_kc_crime.Involvement.str.contains("VIC") == True]

victims_no_nans = victims[victims.Sex.str.contains("NaN") == False]
male_female_victims_kcmo = victims_no_nans[victims_no_nans.Sex.str.contains("U") == False]

kc_crime_real_ages = male_female_victims_kcmo[male_female_victims_kcmo['Age'] < 91]
```

### Copy kc_crime_real_ages and separate "Location" into 3 columns


```python
kc_crime_real_ages_copy = kc_crime_real_ages.copy()
kc_crime_real_ages_copy.head(1)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reported_Date</th>
      <th>Reported_Time</th>
      <th>Description</th>
      <th>City</th>
      <th>zip_code</th>
      <th>Involvement</th>
      <th>Race</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Firearm Used Flag</th>
      <th>Location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5/28/2017</td>
      <td>3:44</td>
      <td>Stolen_Property</td>
      <td>KANSAS CITY</td>
      <td>64111</td>
      <td>VIC</td>
      <td>W</td>
      <td>F</td>
      <td>29.0</td>
      <td>N</td>
      <td>4000 MILL ST\nKANSAS CITY 64111\n(39.053635, -...</td>
    </tr>
  </tbody>
</table>
</div>



### Split out geo data


```python
location_only = kc_crime_real_ages_copy['Location'].str[0:-1].str.split('\n', expand=True)
location_only.columns = ("address", "city_zip", "geo")
location_only.head(1)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>address</th>
      <th>city_zip</th>
      <th>geo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4000 MILL ST</td>
      <td>KANSAS CITY 64111</td>
      <td>(39.053635, -94.595998</td>
    </tr>
  </tbody>
</table>
</div>



### Parse out latitude and longitude


```python
location_only['geo'] = location_only['geo'].str[1:]
geo_split = location_only['geo'].str[0:].str.split(', ', expand=True)
geo_split.columns = ("Latitude", "Longitude")
geo_split.head(1)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39.053635</td>
      <td>-94.595998</td>
    </tr>
  </tbody>
</table>
</div>




```python
kc_crime_real_ages_copy.count()
```




    Reported_Date        43030
    Reported_Time        43030
    Description          43030
    City                 43030
    zip_code             43030
    Involvement          43030
    Race                 43030
    Sex                  43030
    Age                  43030
    Firearm Used Flag    43030
    Location             43030
    dtype: int64




```python
geo_split.count()
```




    Latitude     32283
    Longitude    32283
    dtype: int64




```python
kcmo_crime_with_nans = pd.concat([kc_crime_real_ages_copy, geo_split], axis=1)
```


```python
kcmo_crime_with_nans.count()
```




    Reported_Date        43030
    Reported_Time        43030
    Description          43030
    City                 43030
    zip_code             43030
    Involvement          43030
    Race                 43030
    Sex                  43030
    Age                  43030
    Firearm Used Flag    43030
    Location             43030
    Latitude             32283
    Longitude            32283
    dtype: int64




```python
kcmo_crime_no_lat_nans = kcmo_crime_with_nans[kcmo_crime_with_nans.Latitude.str.contains("NaN") == False]
kcmo_crime_no_nans = kcmo_crime_no_lat_nans[kcmo_crime_no_lat_nans.Longitude.str.contains("NaN") == False]
kc_crime_for_visualizations = kcmo_crime_no_nans.drop(['City', 'Involvement', 'Location'], axis=1)
kc_crime_for_visualizations.head(1)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reported_Date</th>
      <th>Reported_Time</th>
      <th>Description</th>
      <th>zip_code</th>
      <th>Race</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Firearm Used Flag</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5/28/2017</td>
      <td>3:44</td>
      <td>Stolen_Property</td>
      <td>64111</td>
      <td>W</td>
      <td>F</td>
      <td>29.0</td>
      <td>N</td>
      <td>39.053635</td>
      <td>-94.595998</td>
    </tr>
  </tbody>
</table>
</div>



### Change time to datetime and extract hour


```python
kc_crime_for_visualizations.Reported_Date = pd.to_datetime(kc_crime_for_visualizations.Reported_Date)
kc_crime_for_visualizations.Reported_Time = pd.to_datetime(kc_crime_for_visualizations.Reported_Time)
kc_crime_for_visualizations["Reported_Time"] = kc_crime_for_visualizations["Reported_Time"].dt.floor('h')
kc_crime_for_visualizations['Reported_Time'] = kc_crime_for_visualizations['Reported_Time'].dt.hour
```


```python
kc_crime_for_visualizations.columns = ( "Date", "Hour", "Crime", "Zip", "Race", "Sex", "Age", "Firearm", "Latitude", "Longitude")
kc_crime_for_visualizations = kc_crime_for_visualizations.reset_index(drop=True)
kc_crime_for_visualizations.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Hour</th>
      <th>Crime</th>
      <th>Zip</th>
      <th>Race</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Firearm</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-05-28</td>
      <td>3</td>
      <td>Stolen_Property</td>
      <td>64111</td>
      <td>W</td>
      <td>F</td>
      <td>29.0</td>
      <td>N</td>
      <td>39.053635</td>
      <td>-94.595998</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-11-21</td>
      <td>13</td>
      <td>Homicide_Non_Negl</td>
      <td>64105</td>
      <td>B</td>
      <td>F</td>
      <td>69.0</td>
      <td>Y</td>
      <td>39.10091</td>
      <td>-94.577328</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-08-02</td>
      <td>13</td>
      <td>Auto_Theft</td>
      <td>64119</td>
      <td>W</td>
      <td>F</td>
      <td>31.0</td>
      <td>N</td>
      <td>39.17744</td>
      <td>-94.572069</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-08-24</td>
      <td>16</td>
      <td>Intimidation</td>
      <td>64130</td>
      <td>W</td>
      <td>F</td>
      <td>19.0</td>
      <td>N</td>
      <td>39.033505</td>
      <td>-94.547812</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-03-15</td>
      <td>14</td>
      <td>Auto_Theft</td>
      <td>64157</td>
      <td>W</td>
      <td>F</td>
      <td>62.0</td>
      <td>N</td>
      <td>39.235881</td>
      <td>-94.466171</td>
    </tr>
  </tbody>
</table>
</div>




```python
kc_crime_for_visualizations.dtypes
```




    Date         datetime64[ns]
    Hour                  int64
    Crime                object
    Zip                   int64
    Race                 object
    Sex                  object
    Age                 float64
    Firearm              object
    Latitude             object
    Longitude            object
    dtype: object




```python
kc_crime_for_visualizations['Month'] = kc_crime_for_visualizations.Date.map(lambda x: x.strftime('%m'))
```


```python
kc_crime_for_visualizations.count()
```




    Date         32283
    Hour         32283
    Crime        32283
    Zip          32283
    Race         32283
    Sex          32283
    Age          32283
    Firearm      32283
    Latitude     32283
    Longitude    32283
    Month        32283
    dtype: int64




```python
kc_crime_for_visualizations.to_csv('Resources\CSVs\kc_crime_for_visualizations.csv', index=False)
```


```python
data = kc_crime_for_visualizations
data2 = kc_crime_for_visualizations
```


```python
data2['Male'] = pd.Series(np.where(data2.Sex.values == 'M', 1, 0), data2.index)
```


```python
data2.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Hour</th>
      <th>Crime</th>
      <th>Zip</th>
      <th>Race</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Firearm</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Month</th>
      <th>Male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-05-28</td>
      <td>3</td>
      <td>Stolen_Property</td>
      <td>64111</td>
      <td>W</td>
      <td>F</td>
      <td>29.0</td>
      <td>N</td>
      <td>39.053635</td>
      <td>-94.595998</td>
      <td>05</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-11-21</td>
      <td>13</td>
      <td>Homicide_Non_Negl</td>
      <td>64105</td>
      <td>B</td>
      <td>F</td>
      <td>69.0</td>
      <td>Y</td>
      <td>39.10091</td>
      <td>-94.577328</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-08-02</td>
      <td>13</td>
      <td>Auto_Theft</td>
      <td>64119</td>
      <td>W</td>
      <td>F</td>
      <td>31.0</td>
      <td>N</td>
      <td>39.17744</td>
      <td>-94.572069</td>
      <td>08</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-08-24</td>
      <td>16</td>
      <td>Intimidation</td>
      <td>64130</td>
      <td>W</td>
      <td>F</td>
      <td>19.0</td>
      <td>N</td>
      <td>39.033505</td>
      <td>-94.547812</td>
      <td>08</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-03-15</td>
      <td>14</td>
      <td>Auto_Theft</td>
      <td>64157</td>
      <td>W</td>
      <td>F</td>
      <td>62.0</td>
      <td>N</td>
      <td>39.235881</td>
      <td>-94.466171</td>
      <td>03</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data['Gender'] = pd.Series(np.where(data.Sex.values == 'M', 1, 0), data.index)
data = data.drop(['Sex'], axis=1)
```


```python
data2.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Hour</th>
      <th>Crime</th>
      <th>Zip</th>
      <th>Race</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Firearm</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Month</th>
      <th>Male</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-05-28</td>
      <td>3</td>
      <td>Stolen_Property</td>
      <td>64111</td>
      <td>W</td>
      <td>F</td>
      <td>29.0</td>
      <td>N</td>
      <td>39.053635</td>
      <td>-94.595998</td>
      <td>05</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-11-21</td>
      <td>13</td>
      <td>Homicide_Non_Negl</td>
      <td>64105</td>
      <td>B</td>
      <td>F</td>
      <td>69.0</td>
      <td>Y</td>
      <td>39.10091</td>
      <td>-94.577328</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-08-02</td>
      <td>13</td>
      <td>Auto_Theft</td>
      <td>64119</td>
      <td>W</td>
      <td>F</td>
      <td>31.0</td>
      <td>N</td>
      <td>39.17744</td>
      <td>-94.572069</td>
      <td>08</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-08-24</td>
      <td>16</td>
      <td>Intimidation</td>
      <td>64130</td>
      <td>W</td>
      <td>F</td>
      <td>19.0</td>
      <td>N</td>
      <td>39.033505</td>
      <td>-94.547812</td>
      <td>08</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-03-15</td>
      <td>14</td>
      <td>Auto_Theft</td>
      <td>64157</td>
      <td>W</td>
      <td>F</td>
      <td>62.0</td>
      <td>N</td>
      <td>39.235881</td>
      <td>-94.466171</td>
      <td>03</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# For Visualizations

data2['Female'] = 1 - data2['Male']
data2.groupby('Race').agg('mean')[['Male', 'Female']].plot(kind='bar', stacked=True, color=['b', 'r']);
plt.show()
```


![png](output_33_0.png)



```python
fig = plt.figure(figsize=(25, 12))
sns.violinplot(x='Race', y='Age', 
               hue='Male', data=data2, 
               split=True,
               palette={0: "r", 1: "b"}
              );
plt.show()
```


![png](output_34_0.png)



```python
figure = plt.figure(figsize=(25, 12))
plt.hist([data2[data2['Male'] == 1]['Age'], data2[data2['Male'] == 0]['Age']], 
         stacked=True, color = ['b','r'],
         bins = 50, label = ['Male','Female'])
plt.xlabel('Age')
plt.ylabel('Number of Incidents')
plt.legend();
plt.show()
```


![png](output_35_0.png)



```python
plt.figure(figsize=(25, 12))
ax = plt.subplot()

ax.scatter(data2[data2['Male'] == 1]['Race'], data2[data2['Male'] == 1]['Age'], 
           c='blue', s=data2[data2['Male'] == 1]['Age'])
ax.scatter(data2[data2['Male'] == 0]['Race'], data2[data2['Male'] == 0]['Age'], 
           c='red', s=data2[data2['Male'] == 0]['Age']);
plt.show()
```


![png](output_36_0.png)



```python
ax = plt.subplot()
ax.set_ylabel('Age')
data2.groupby('Race').mean()['Age'].plot(kind='bar', figsize=(10, 7), ax = ax);
plt.show()
```


![png](output_37_0.png)



```python
ax = plt.subplot()
ax.set_ylabel('Age')
data2.groupby('Sex').mean()['Age'].plot(kind='bar', figsize=(10, 7), ax = ax);
plt.show()
```


![png](output_38_0.png)



```python
fig = plt.figure(figsize=(25, 20))
sns.violinplot(x='Race', y='Age', hue='Male', data=data2, split=True, palette={0: "r", 1: "b"});
plt.show()
```


![png](output_39_0.png)



```python
sns.set(style="darkgrid")
fig = plt.figure(figsize=(22,10))
plt.scatter(data2.Longitude, data2.Latitude, c=data2.Age, 
           cmap='viridis_r', s=30)
plt.xlim(-122.33, -77.29)
plt.ylim(26.71, 47.7)
plt.colorbar()
plt.show()
```


![png](output_40_0.png)



```python
sns.set(style="whitegrid")
data2.hist(bins=50, figsize=(20,15))
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x000001C0783B19E8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000001C078322F28>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x000001C078292D68>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000001C078205B38>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x000001C078175A20>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000001C0783D23C8>]], dtype=object)




![png](output_41_1.png)



```python
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Hour</th>
      <th>Crime</th>
      <th>Zip</th>
      <th>Race</th>
      <th>Age</th>
      <th>Firearm</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Month</th>
      <th>Male</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-05-28</td>
      <td>3</td>
      <td>Stolen_Property</td>
      <td>64111</td>
      <td>W</td>
      <td>29.0</td>
      <td>N</td>
      <td>39.053635</td>
      <td>-94.595998</td>
      <td>05</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-11-21</td>
      <td>13</td>
      <td>Homicide_Non_Negl</td>
      <td>64105</td>
      <td>B</td>
      <td>69.0</td>
      <td>Y</td>
      <td>39.10091</td>
      <td>-94.577328</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-08-02</td>
      <td>13</td>
      <td>Auto_Theft</td>
      <td>64119</td>
      <td>W</td>
      <td>31.0</td>
      <td>N</td>
      <td>39.17744</td>
      <td>-94.572069</td>
      <td>08</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-08-24</td>
      <td>16</td>
      <td>Intimidation</td>
      <td>64130</td>
      <td>W</td>
      <td>19.0</td>
      <td>N</td>
      <td>39.033505</td>
      <td>-94.547812</td>
      <td>08</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-03-15</td>
      <td>14</td>
      <td>Auto_Theft</td>
      <td>64157</td>
      <td>W</td>
      <td>62.0</td>
      <td>N</td>
      <td>39.235881</td>
      <td>-94.466171</td>
      <td>03</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = data.drop(['Race', 'Date'], axis=1)
data['Firearm'] = pd.Series(np.where(data.Firearm.values == 'Y', 1, 0), data.index)
```


```python
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hour</th>
      <th>Crime</th>
      <th>Zip</th>
      <th>Age</th>
      <th>Firearm</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Month</th>
      <th>Male</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>Stolen_Property</td>
      <td>64111</td>
      <td>29.0</td>
      <td>0</td>
      <td>39.053635</td>
      <td>-94.595998</td>
      <td>05</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13</td>
      <td>Homicide_Non_Negl</td>
      <td>64105</td>
      <td>69.0</td>
      <td>1</td>
      <td>39.10091</td>
      <td>-94.577328</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13</td>
      <td>Auto_Theft</td>
      <td>64119</td>
      <td>31.0</td>
      <td>0</td>
      <td>39.17744</td>
      <td>-94.572069</td>
      <td>08</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16</td>
      <td>Intimidation</td>
      <td>64130</td>
      <td>19.0</td>
      <td>0</td>
      <td>39.033505</td>
      <td>-94.547812</td>
      <td>08</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14</td>
      <td>Auto_Theft</td>
      <td>64157</td>
      <td>62.0</td>
      <td>0</td>
      <td>39.235881</td>
      <td>-94.466171</td>
      <td>03</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### get_dummies


```python
crime_columns = pd.get_dummies(data['Crime'],prefix = "Crime")
data = pd.concat([data, crime_columns], axis=1)
data.drop('Crime', axis=1, inplace=True)
zip_columns = pd.get_dummies(data['Zip'],prefix = "Zip")
data = pd.concat([data, zip_columns], axis=1)
data.drop('Zip', axis=1, inplace=True)
month_columns = pd.get_dummies(data['Month'],prefix = "Month")
data = pd.concat([data, month_columns], axis=1)
data.drop('Month', axis=1, inplace=True)
age_columns = pd.get_dummies(data['Age'],prefix = "Age")
data = pd.concat([data, age_columns], axis=1)
data.drop('Age', axis=1, inplace=True)
hour_columns = pd.get_dummies(data['Hour'],prefix = "Hour")
data = pd.concat([data, hour_columns], axis=1)
data.drop('Hour', axis=1, inplace=True)
```


```python
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Firearm</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Male</th>
      <th>Gender</th>
      <th>Crime_Agg_Assault_Domest</th>
      <th>Crime_Agg_Assault_Drive</th>
      <th>Crime_Aggravated_Assault</th>
      <th>Crime_Armed_Robbery</th>
      <th>Crime_Arson</th>
      <th>Crime_Attempted_Suicide</th>
      <th>Crime_Auto_Theft</th>
      <th>Crime_Auto_Theft_Outside</th>
      <th>Crime_Bomb_Threat</th>
      <th>Crime_Broken_Window</th>
      <th>Crime_Burglary_Non_Resid</th>
      <th>Crime_Burglary_Residence</th>
      <th>Crime_Casualty</th>
      <th>Crime_Counterfeiting</th>
      <th>Crime_Credit_Debit_ATM_Card</th>
      <th>Crime_DUI</th>
      <th>Crime_Dead_Body</th>
      <th>Crime_Disorderly_Conduct</th>
      <th>Crime_Drunkenness</th>
      <th>Crime_Embezzlement</th>
      <th>Crime_Extortion_Blackmail</th>
      <th>Crime_Failure_to_Return</th>
      <th>Crime_False_Information</th>
      <th>Crime_Family_Disturbance</th>
      <th>Crime_Family_Offense</th>
      <th>Crime_Forcible_Fondling</th>
      <th>Crime_Forcible_Sodomy</th>
      <th>Crime_Forgery</th>
      <th>Crime_Fraud_Confidence_Gamb</th>
      <th>Crime_Hacking_Computer</th>
      <th>Crime_Hit_and_Run_Pers</th>
      <th>Crime_Homicide_Non_Negl</th>
      <th>Crime_Human_Trafficking</th>
      <th>Crime_Identity_Theft</th>
      <th>Crime_Impersonation</th>
      <th>Crime_Interference</th>
      <th>Crime_Intimidation</th>
      <th>Crime_Invasion_of_Privacy</th>
      <th>Crime_Justifiable_Homicide</th>
      <th>Crime_Kidnapping_Abduction</th>
      <th>Crime_Liquor_Law_Violaton</th>
      <th>Crime_Misc_Violation</th>
      <th>Crime_Missing_Runaway_Juvenile</th>
      <th>Crime_Non_Agg_Assault_Dome</th>
      <th>Crime_Non_Aggravated_Assau</th>
      <th>...</th>
      <th>Age_65.0</th>
      <th>Age_66.0</th>
      <th>Age_67.0</th>
      <th>Age_68.0</th>
      <th>Age_69.0</th>
      <th>Age_70.0</th>
      <th>Age_71.0</th>
      <th>Age_72.0</th>
      <th>Age_73.0</th>
      <th>Age_74.0</th>
      <th>Age_75.0</th>
      <th>Age_76.0</th>
      <th>Age_77.0</th>
      <th>Age_78.0</th>
      <th>Age_79.0</th>
      <th>Age_80.0</th>
      <th>Age_81.0</th>
      <th>Age_82.0</th>
      <th>Age_83.0</th>
      <th>Age_84.0</th>
      <th>Age_85.0</th>
      <th>Age_86.0</th>
      <th>Age_87.0</th>
      <th>Age_88.0</th>
      <th>Age_89.0</th>
      <th>Age_90.0</th>
      <th>Hour_0</th>
      <th>Hour_1</th>
      <th>Hour_2</th>
      <th>Hour_3</th>
      <th>Hour_4</th>
      <th>Hour_5</th>
      <th>Hour_6</th>
      <th>Hour_7</th>
      <th>Hour_8</th>
      <th>Hour_9</th>
      <th>Hour_10</th>
      <th>Hour_11</th>
      <th>Hour_12</th>
      <th>Hour_13</th>
      <th>Hour_14</th>
      <th>Hour_15</th>
      <th>Hour_16</th>
      <th>Hour_17</th>
      <th>Hour_18</th>
      <th>Hour_19</th>
      <th>Hour_20</th>
      <th>Hour_21</th>
      <th>Hour_22</th>
      <th>Hour_23</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>39.053635</td>
      <td>-94.595998</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>39.10091</td>
      <td>-94.577328</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>39.17744</td>
      <td>-94.572069</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>39.033505</td>
      <td>-94.547812</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>39.235881</td>
      <td>-94.466171</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 302 columns</p>
</div>




```python
data.shape
```




    (32283, 302)




```python
y = data["Gender"]
X = data.drop('Gender', axis=1)
```


```python
y.head()
```




    0    0
    1    0
    2    0
    3    0
    4    0
    Name: Gender, dtype: int32




```python
X.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Firearm</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Male</th>
      <th>Crime_Agg_Assault_Domest</th>
      <th>Crime_Agg_Assault_Drive</th>
      <th>Crime_Aggravated_Assault</th>
      <th>Crime_Armed_Robbery</th>
      <th>Crime_Arson</th>
      <th>Crime_Attempted_Suicide</th>
      <th>Crime_Auto_Theft</th>
      <th>Crime_Auto_Theft_Outside</th>
      <th>Crime_Bomb_Threat</th>
      <th>Crime_Broken_Window</th>
      <th>Crime_Burglary_Non_Resid</th>
      <th>Crime_Burglary_Residence</th>
      <th>Crime_Casualty</th>
      <th>Crime_Counterfeiting</th>
      <th>Crime_Credit_Debit_ATM_Card</th>
      <th>Crime_DUI</th>
      <th>Crime_Dead_Body</th>
      <th>Crime_Disorderly_Conduct</th>
      <th>Crime_Drunkenness</th>
      <th>Crime_Embezzlement</th>
      <th>Crime_Extortion_Blackmail</th>
      <th>Crime_Failure_to_Return</th>
      <th>Crime_False_Information</th>
      <th>Crime_Family_Disturbance</th>
      <th>Crime_Family_Offense</th>
      <th>Crime_Forcible_Fondling</th>
      <th>Crime_Forcible_Sodomy</th>
      <th>Crime_Forgery</th>
      <th>Crime_Fraud_Confidence_Gamb</th>
      <th>Crime_Hacking_Computer</th>
      <th>Crime_Hit_and_Run_Pers</th>
      <th>Crime_Homicide_Non_Negl</th>
      <th>Crime_Human_Trafficking</th>
      <th>Crime_Identity_Theft</th>
      <th>Crime_Impersonation</th>
      <th>Crime_Interference</th>
      <th>Crime_Intimidation</th>
      <th>Crime_Invasion_of_Privacy</th>
      <th>Crime_Justifiable_Homicide</th>
      <th>Crime_Kidnapping_Abduction</th>
      <th>Crime_Liquor_Law_Violaton</th>
      <th>Crime_Misc_Violation</th>
      <th>Crime_Missing_Runaway_Juvenile</th>
      <th>Crime_Non_Agg_Assault_Dome</th>
      <th>Crime_Non_Aggravated_Assau</th>
      <th>Crime_Peeping_Tom</th>
      <th>...</th>
      <th>Age_65.0</th>
      <th>Age_66.0</th>
      <th>Age_67.0</th>
      <th>Age_68.0</th>
      <th>Age_69.0</th>
      <th>Age_70.0</th>
      <th>Age_71.0</th>
      <th>Age_72.0</th>
      <th>Age_73.0</th>
      <th>Age_74.0</th>
      <th>Age_75.0</th>
      <th>Age_76.0</th>
      <th>Age_77.0</th>
      <th>Age_78.0</th>
      <th>Age_79.0</th>
      <th>Age_80.0</th>
      <th>Age_81.0</th>
      <th>Age_82.0</th>
      <th>Age_83.0</th>
      <th>Age_84.0</th>
      <th>Age_85.0</th>
      <th>Age_86.0</th>
      <th>Age_87.0</th>
      <th>Age_88.0</th>
      <th>Age_89.0</th>
      <th>Age_90.0</th>
      <th>Hour_0</th>
      <th>Hour_1</th>
      <th>Hour_2</th>
      <th>Hour_3</th>
      <th>Hour_4</th>
      <th>Hour_5</th>
      <th>Hour_6</th>
      <th>Hour_7</th>
      <th>Hour_8</th>
      <th>Hour_9</th>
      <th>Hour_10</th>
      <th>Hour_11</th>
      <th>Hour_12</th>
      <th>Hour_13</th>
      <th>Hour_14</th>
      <th>Hour_15</th>
      <th>Hour_16</th>
      <th>Hour_17</th>
      <th>Hour_18</th>
      <th>Hour_19</th>
      <th>Hour_20</th>
      <th>Hour_21</th>
      <th>Hour_22</th>
      <th>Hour_23</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>39.053635</td>
      <td>-94.595998</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>39.10091</td>
      <td>-94.577328</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>39.17744</td>
      <td>-94.572069</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>39.033505</td>
      <td>-94.547812</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>39.235881</td>
      <td>-94.466171</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 301 columns</p>
</div>




```python
data.shape
```




    (32283, 302)




```python
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Firearm</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Male</th>
      <th>Gender</th>
      <th>Crime_Agg_Assault_Domest</th>
      <th>Crime_Agg_Assault_Drive</th>
      <th>Crime_Aggravated_Assault</th>
      <th>Crime_Armed_Robbery</th>
      <th>Crime_Arson</th>
      <th>Crime_Attempted_Suicide</th>
      <th>Crime_Auto_Theft</th>
      <th>Crime_Auto_Theft_Outside</th>
      <th>Crime_Bomb_Threat</th>
      <th>Crime_Broken_Window</th>
      <th>Crime_Burglary_Non_Resid</th>
      <th>Crime_Burglary_Residence</th>
      <th>Crime_Casualty</th>
      <th>Crime_Counterfeiting</th>
      <th>Crime_Credit_Debit_ATM_Card</th>
      <th>Crime_DUI</th>
      <th>Crime_Dead_Body</th>
      <th>Crime_Disorderly_Conduct</th>
      <th>Crime_Drunkenness</th>
      <th>Crime_Embezzlement</th>
      <th>Crime_Extortion_Blackmail</th>
      <th>Crime_Failure_to_Return</th>
      <th>Crime_False_Information</th>
      <th>Crime_Family_Disturbance</th>
      <th>Crime_Family_Offense</th>
      <th>Crime_Forcible_Fondling</th>
      <th>Crime_Forcible_Sodomy</th>
      <th>Crime_Forgery</th>
      <th>Crime_Fraud_Confidence_Gamb</th>
      <th>Crime_Hacking_Computer</th>
      <th>Crime_Hit_and_Run_Pers</th>
      <th>Crime_Homicide_Non_Negl</th>
      <th>Crime_Human_Trafficking</th>
      <th>Crime_Identity_Theft</th>
      <th>Crime_Impersonation</th>
      <th>Crime_Interference</th>
      <th>Crime_Intimidation</th>
      <th>Crime_Invasion_of_Privacy</th>
      <th>Crime_Justifiable_Homicide</th>
      <th>Crime_Kidnapping_Abduction</th>
      <th>Crime_Liquor_Law_Violaton</th>
      <th>Crime_Misc_Violation</th>
      <th>Crime_Missing_Runaway_Juvenile</th>
      <th>Crime_Non_Agg_Assault_Dome</th>
      <th>Crime_Non_Aggravated_Assau</th>
      <th>...</th>
      <th>Age_65.0</th>
      <th>Age_66.0</th>
      <th>Age_67.0</th>
      <th>Age_68.0</th>
      <th>Age_69.0</th>
      <th>Age_70.0</th>
      <th>Age_71.0</th>
      <th>Age_72.0</th>
      <th>Age_73.0</th>
      <th>Age_74.0</th>
      <th>Age_75.0</th>
      <th>Age_76.0</th>
      <th>Age_77.0</th>
      <th>Age_78.0</th>
      <th>Age_79.0</th>
      <th>Age_80.0</th>
      <th>Age_81.0</th>
      <th>Age_82.0</th>
      <th>Age_83.0</th>
      <th>Age_84.0</th>
      <th>Age_85.0</th>
      <th>Age_86.0</th>
      <th>Age_87.0</th>
      <th>Age_88.0</th>
      <th>Age_89.0</th>
      <th>Age_90.0</th>
      <th>Hour_0</th>
      <th>Hour_1</th>
      <th>Hour_2</th>
      <th>Hour_3</th>
      <th>Hour_4</th>
      <th>Hour_5</th>
      <th>Hour_6</th>
      <th>Hour_7</th>
      <th>Hour_8</th>
      <th>Hour_9</th>
      <th>Hour_10</th>
      <th>Hour_11</th>
      <th>Hour_12</th>
      <th>Hour_13</th>
      <th>Hour_14</th>
      <th>Hour_15</th>
      <th>Hour_16</th>
      <th>Hour_17</th>
      <th>Hour_18</th>
      <th>Hour_19</th>
      <th>Hour_20</th>
      <th>Hour_21</th>
      <th>Hour_22</th>
      <th>Hour_23</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>39.053635</td>
      <td>-94.595998</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>39.10091</td>
      <td>-94.577328</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>39.17744</td>
      <td>-94.572069</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>39.033505</td>
      <td>-94.547812</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>39.235881</td>
      <td>-94.466171</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 302 columns</p>
</div>




```python
def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)
```


```python
def recover_train_test_target(data):    
    targets = pd.read_csv('kc_crime_for_visualizations.csv', usecols=['Sex'])['Sex'].values
    data = data.drop('Gender', axis=1)
    train = data.iloc[:32283]
    test = data.iloc[32283:]
    
    return train, test, targets
```


```python
train, test, targets = recover_train_test_target(data)
```


```python
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, targets)
```


```python
features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
```


```python
features.plot(kind='barh', figsize=(25, 225))
plt.show()
```


![png](output_59_0.png)



```python
logreg = LogisticRegression()
logreg_cv = LogisticRegressionCV()
rf = RandomForestClassifier(random_state=42)
gboost = GradientBoostingClassifier(random_state=20)

models = [logreg, logreg_cv, rf, gboost]
```


```python
for model in models:
    print('Cross-validation of : {0}'.format(model.__class__))
    score = compute_score(clf=model, X=train, y=targets, scoring='accuracy')
    print('CV score = {0}'.format(score))
    print('****')
```

    Cross-validation of : <class 'sklearn.linear_model.logistic.LogisticRegression'>
    CV score = 1.0
    ****
    Cross-validation of : <class 'sklearn.linear_model.logistic.LogisticRegressionCV'>
    CV score = 1.0
    ****
    Cross-validation of : <class 'sklearn.ensemble.forest.RandomForestClassifier'>
    CV score = 0.9960661311249964
    ****
    Cross-validation of : <class 'sklearn.ensemble.gradient_boosting.GradientBoostingClassifier'>
    CV score = 1.0
    ****
    


```python
model = GradientBoostingClassifier(verbose=1, random_state=20)
model.fit(train, targets)
```


```python
## turn run_gs to True if you want to run the gridsearch again.

run_gs = False

if run_gs:
    parameter_grid = {
                 'max_depth' : [4, 6, 8, 10, 20, 25],
                 'n_estimators': [50, 10, 20, 100, 200],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [2, 3,4,5,6,7,8,9, 10],
                 'min_samples_leaf': [2,3,4,5,6,7,8],
                 }
    boost = GradientBoostingClassifier()
    cross_validation = StratifiedKFold(n_splits=3)

    grid_search = GridSearchCV(boost,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation,
                               verbose=1,
                               n_jobs=-1
                              )

    grid_search.fit(train, targets)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    
else: 
    parameters = {'min_samples_leaf': 7, 
                  'n_estimators': 200, 'min_samples_split': 6, 
                  'max_features': 'log2', 'max_depth': 4}
    
    model = GradientBoostingClassifier(**parameters)
    model.fit(train, targets)
```

### Setting run_gs to "True" above produced the following below:

![The_40_Hour_Beast.PNG](attachment:The_40_Hour_Beast.PNG)

### 10 Hours in...25,586 fits remaining.  Averaging 14 per minute, it still needed another 30 hours.

### After removing n_estimators of 1000 and reducing # of folds from 5 to 3...

![Capture.PNG](attachment:Capture.PNG)


```python
scores = cross_val_score(model, train, targets, cv = 5, scoring='accuracy')
scores.mean()
```




    0.79778490784234823


