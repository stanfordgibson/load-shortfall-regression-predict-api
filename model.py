"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime as dt

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])
    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    def create_new_features(df):
        #create new features
        # Seasons of the Year
        spring = (3, 4, 5)
        summer = (6, 7, 8)
        autumn = (9, 10, 11)
        winter = (12, 1, 2)
        seasons = [1 if x in spring else 2 if x in summer else 3 if x in autumn else 4 for x in list(df.time.dt.month)]
        
        df['dow'] = df['time'].dt.weekday
        df['lagged_dow'] = df['dow'].shift(1)
        df['woy'] = df['time'].dt.weekofyear
        df['doy'] = df['time'].dt.dayofyear
        df['dom'] = df['time'].dt.day
        df['lagged_dom'] = df['dom'].shift(1)
        df['month'] = df['time'].dt.month
        df['lagged_month'] = df['month'].shift(1)
        df['year'] = df['time'].dt.year
        df['hour_of_day'] = df['time'].dt.hour
        df['lagged_hour'] = df['hour_of_day'].shift(1)
        df['minute'] = df['time'].dt.minute
        df['lagged_minute'] = df['minute'].shift(1)
        df['seasons'] = seasons
        # data['hour_of_year'] = y
        # data['hour_of_week'] = hw
        
        df['lagged_month'] = df['lagged_month'].fillna(df['lagged_month'].median())
        df['lagged_hour'] = df['lagged_hour'].fillna(df['lagged_hour'].median())
        df['lagged_dow'] = df['lagged_dow'].fillna(df['lagged_dow'].median())
        df['lagged_dom'] = df['lagged_dom'].fillna(df['lagged_dom'].median())
        df['lagged_minute'] = df['lagged_minute'].fillna(df['lagged_minute'].median())
        # data['lagged_doy'] = data['lagged_doy'].fillna(data['lagged_doy'].median())
        ints = list(df.select_dtypes(include='int').columns)

        for col in ints:
            if col in ('seasons', 'dow', 'dom', 'month', 'hour_of_day','lagged_minute', 'minute', 'lagged_month', 'lagged_hour', 'lagged_dow', 'lagged_dom'):
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype('float')
        
        df['Valencia_pressure'] = df.Valencia_pressure.fillna(df.Valencia_pressure.median())
        # This column is redundant:
        df = df.drop(['Unnamed: 0'], axis=1)
        
        # engineer existing features
        df = pd.get_dummies(df, drop_first=True, dtype=float)
        df = df.drop(['time'], axis=1)
        
        return df

    # train data
    df1 = pd.read_csv("assets/df_train.csv", parse_dates=['time'])
    df1 = create_new_features(df1)
    
    #----------------------------------------------------------------------------------------
    # test data
    #loading the data
    df_test = pd.read_csv("assets/df_test.csv")
    frames = [df_test, feature_vector_df]
    df_test = pd.concat(frames)
    df_test['time'] = pd.to_datetime(df_test['time'])
    df_test = create_new_features(df_test)
    df_test[['Bilbao_snow_3h', 'Valencia_snow_3h']] = df_test[['Bilbao_snow_3h', 'Valencia_snow_3h']].astype('float')

    # split data
    y = np.array(df1['load_shortfall_3h'])
    X = np.array(df1.drop(['load_shortfall_3h'], axis=1))
    y = y.reshape(-1, 1)

    #scale data
    scale_X = StandardScaler()
    scale_y = StandardScaler()
    X = scale_X.fit_transform(X)
    y = scale_y.fit_transform(y)

    # create targets and features dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    y_test = scale_y.inverse_transform(y_test)

    scale = StandardScaler()
    df_sc = scale.fit_transform(df_test)

    return (df_sc, scale_y)
    # ------------------------------------------------------------------------

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    y_scaled = prep_data[1]
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data[0])
    prediction = prediction.reshape(-1, 1)
    prediction = y_scaled.inverse_transform(prediction)
    # Format as list for output standardisation.
    return prediction[-1].tolist()
