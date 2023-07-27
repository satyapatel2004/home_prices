import pandas as pd 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestRegressor

file_path = 'gta_market.csv' 
home_data = pd.read_csv(file_path)
output_file = 'output.csv'


#performing string encoding on the region column:
region_data = home_data['region']
label_encoder = LabelEncoder() 
encoded_region = label_encoder.fit_transform(region_data)
home_data['region'] = encoded_region 


#target object is the price (what we are attempting to measure)
y = home_data.price
features = ['region','bedrooms','bathrooms']

#the features (what we are using to predict the target)
X = home_data[features]

decoded_categories = label_encoder.inverse_transform(encoded_region)

#splitting the data into validation and training data:
train_X, val_X, train_y, val_y = train_test_split(X,y, test_size=0.2,random_state=1)


#specifying model:
gta_model = RandomForestRegressor(max_leaf_nodes=200, random_state=1)
gta_model.fit(val_X, val_y)

def is_price_fair(model, input_data, actual_price, threshold=0.10):
    region_data = input_data['region']
    encoded_region = label_encoder.fit_transform(region_data)
    input_data['region'] = encoded_region

    predicted_price = model.predict(input_data)
    price_difference = abs(predicted_price - actual_price)
    fair_threshold = threshold * actual_price

    if(price_difference <= fair_threshold):

        return True, predicted_price
    else :
        return False, predicted_price

if __name__ == "__main__":
    input_features = {'region': 'Toronto, ON', 'bedrooms': 6, 'bathrooms': 2} 
    actual_price_paid = 1000000 

    input_data = pd.DataFrame(input_features, index=[0])
    is_fair, fair_price = is_price_fair(gta_model, input_data, actual_price_paid)

    if is_fair:
        print("The price is fair")
        print("predicted price is: ", fair_price)  

    else:
        print("The price is not fair")
        print("predicted price is: ", fair_price) 



