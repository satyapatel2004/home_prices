import pandas as pd 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestRegressor
from tkinter import * 
from tkinter import ttk 
import joblib 
import os

#setting up the files: 
file_path = 'gta_market.csv' 
home_data = pd.read_csv(file_path)
output_file = 'output.csv'

#encode(df, col, enc) will consume a pandas dataframe object, a column string, and a bool 
#    representing whether or not the column should be encoded or decoded, and will encode or decode
#    the column in the dataframe object. 
def encode(df, col:str, enc=True):
    data = df[col]
    label_encoder = LabelEncoder() 
    encoded_col = label_encoder.fit_transform(data) 
    df[col] = encoded_col

    if not enc:
        decoded_col = label_encoder.inverse_transform(encoded_col)
        df[col] = decoded_col

    return df

#train_model(home_datam save_model) will train and save a model depending on the input. 
def train_model(home_data, save_model=True):
    home_data = encode(home_data, 'region')

    #target object is the price (what we are attempting to measure)
    y = home_data.price
    features = ['region','bedrooms','bathrooms']

    #the features (what we are using to predict the target)
    X = home_data[features]

    #where the model will be stored (if it exists): 
    model_file = 'trained_model.joblib'

    if save_model and os.path.exists(model_file):
        #load the pre-existing model: 
        model = joblib.load(model_file)

    else: 
        #splitting the data into validation and training data:
        train_X, val_X, train_y, val_y = train_test_split(X,y, test_size=0.2,random_state=1)

        #specifying model:
        model = RandomForestRegressor(max_leaf_nodes=200, random_state=1)
        model.fit(val_X, val_y)

        if save_model:
            #save the model to a file (using joblib): 
            model_file = 'trained_model.joblib'
            joblib.dump(model, model_file)

    return model 

def price_analysis(model, input_data, actual_price, threshold=0.30):

    input_data = encode(input_data, 'region')

    predicted_price = model.predict(input_data)
    price_difference = abs(predicted_price - actual_price)
    fair_threshold = threshold * actual_price

    if(price_difference <= fair_threshold):

        return True, predicted_price
    else :
        return False, predicted_price

if __name__ == "__main__":
    gta_model = train_model(home_data)

    #load the trained model 
    model_file = 'trained_model.joblib'
    loaded_model = joblib.load(model_file)


    input_features = {'region': 'Toronto, ON', 'bedrooms': 6, 'bathrooms': 2} 
    actual_price_paid = 1000000 

    input_data = pd.DataFrame(input_features, index=[0])
    is_fair, fair_price = price_analysis(loaded_model, input_data, actual_price_paid)

    if is_fair:
        print("The price is fair")
        print("predicted price is: ", fair_price)  

    else:
        print("The price is not fair")
        print("predicted price is: ", fair_price) 



