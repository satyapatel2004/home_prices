import pandas as pd 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestRegressor
import tkinter as tk 
from tkinter import messagebox
from tkinter import ttk 
import joblib 
import os

#setting up the files: 
file_path = 'gta_market.csv' 
home_data = pd.read_csv(file_path)
home_data['region'] = home_data['region'].str.split(',').str[0]

output_file = 'output.csv'

region_data = home_data['region'] 
label_encoder = LabelEncoder() 
encoded_col = label_encoder.fit_transform(region_data) 
home_data['region'] = encoded_col


#train_model(home_datam save_model) will train and save a model depending on the input. 
def train_model(home_data, save_model=True):

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
    predicted_price = model.predict(input_data)
    price_difference = abs(predicted_price - actual_price)
    fair_threshold = threshold * actual_price

    if(price_difference <= fair_threshold):

        return True, predicted_price
    else :
        return False, predicted_price
    
def validate_integer_input(input_in):
    if input_in == "":
        return True 

    try:
        int(input_in)
        return True 
    except ValueError:
        return False 
    

def validate_region(region, le):

    try: 
        encoded_region = le.transform(region)
        return True

    except KeyError or ValueError:
        return False
    


def validate_string_input(input_str):
    if input_str != "" and isinstance(input_str, str):
        return True 
    else :
        return False 

if __name__ == "__main__":
    gta_model = train_model(home_data)

    #load the trained model 
    model_file = 'trained_model.joblib'
    loaded_model = joblib.load(model_file)

    def show_error_message(message:str):
        error_window = tk.Toplevel(root)
        error_window.title("Error")
        error_window.geometry("300x100")

        error_label = tk.Label(error_window, text=message, fg="red", font=("SF Pro", 15))
        error_label.pack(pady=20)

        ok_button = tk.Button(error_window, text="OK", command=error_window.destroy)
        ok_button.pack()

    def show_response(fair_price:int, is_fair:bool, price_paid:int):
        underpaid = False
        overpaid = False 

        response_window = tk.Toplevel(root)
        response_window.title("Result: ")
        response_window.geometry("400x200")

        if int(price_paid) > fair_price and not is_fair:
            overpaid = True 

        if int(price_paid) < fair_price and not is_fair:
            underpaid = True 

        message = "" 
        
        if overpaid:
            message = "You overpaid for your home (according to the model)" 

        elif underpaid:
            message = "You underpaid for your home (according to the model)" 

        else:
            message = "You paid a fair price for your home"

        response_label = tk.Label(response_window, text=message, fg="Green", font=("SF Pro", 15)) 
        response_label.pack(pady=20)

        ok_button = tk.Button(response_window, text="OK", command=response_window.destroy)
        ok_button.pack() 


    def clear_input_fields():
        bedrooms_entry.delete(0, tk.END)
        bathrooms_entry.delete(0, tk.END)
        region_entry.delete(0, tk.END)


    def submit_data():
        bedrooms = bedrooms_entry.get() 
        bathrooms = bathrooms_entry.get() 
        region = region_entry.get() 
        price = price_entry.get() 

        if bedrooms.strip() == "" or bathrooms.strip == "" or region.strip == "":
            show_error_message("All fields are required!")

        else:
            input_features = {'region': region, 'bedrooms': bedrooms, 'bathrooms': bathrooms}
            actual_price_paid = int(price) 

            input_data = pd.DataFrame(input_features, index=[0])
            
            try:
                input_data['region'] = label_encoder.transform(input_data['region'])  

            except :
                show_error_message("Invalid Region!") 


            is_fair, fair_price = price_analysis(loaded_model, input_data, actual_price_paid) 

            show_response(fair_price, is_fair, price) 

    #setting up the window (name is root)
    root = tk.Tk() 
    
    root.title("Home Price Analyser")
    #root.config(bg='f') 

    #setting window size: 
    root.geometry("600x400") 

    screen_width = root.winfo_screenwidth() 
    screen_height = root.winfo_screenheight() 

    # Center the labels and entry fields within the window
    root.columnconfigure(0, weight=1)  # Column 0 will expand to fill any extra space
    root.columnconfigure(1, weight=1)  # Column 1 will expand to fill any extra space
    root.rowconfigure(0, weight=1)     # Row 0 will expand to fill any extra space
    root.rowconfigure(1, weight=1)     # Row 1 will expand to fill any extra space
    root.rowconfigure(2, weight=1)     # so on so forth ... 
    root.rowconfigure(3, weight=1) 
    root.rowconfigure(4, weight=1) 
    root.rowconfigure(5, weight=1) 
    root.rowconfigure(6, weight=1)

    information_box = tk.Text(height = 5, wrap=tk.WORD, width = 65, font=("SF Pro", 15), bg=root.cget("bg"), highlightbackground=root.cget("bg"), highlightcolor=root.cget("bg")) 
    Information = """This application will allow you to input information about your home and determine whether or not you paid a fair price. The model is trained on data from 2022. Note that some towns are recorded as their adjacent towns e.g Kitchener = Waterloo"""
    
    information_box.insert(tk.END, Information) 
    information_box.grid(row = 1, column = 0, columnspan=2, padx=10) 
    information_box.config(state=tk.DISABLED)
    


    # Create and place the labels
    title = tk.Label(root, text="Ontario Home Prices Analyzer", font=('SF Pro','30')) 
    title.grid(row = 0, column=0, columnspan=2, padx=10, pady=5) 


    bedrooms_label = tk.Label(root, text="Bedrooms:", font=('SF Pro', 23))
    bedrooms_label.grid(row=2, column=0, padx=10, pady=5)

    bathrooms_label = tk.Label(root, text="Bathrooms:", font=('SF Pro', 23))
    bathrooms_label.grid(row=3, column=0, padx=10, pady=5)

    region_label = tk.Label(root, text="Region:", font=('SF Pro', 23)) 
    region_label.grid(row=4, column=0, padx=10, pady=5)

    price_label = tk.Label(root, text="Price:", font=('SF Pro', 23)) 
    price_label.grid(row=5, column=0, padx=10, pady=5)
    # Create and place the entry fields

    #registering the validate_integer_input with the Tk object that we have created (root)
    validate_integer = root.register(validate_integer_input)
    bedrooms_entry = tk.Entry(root, validate="key", validatecommand=(validate_integer, "%P"), font=('SF Pro', 23), fg='black', bg='dark gray') 
    bedrooms_entry.grid(row=2, column=1, padx=10, pady=5)

    bathrooms_entry = tk.Entry(root, validate="key", validatecommand=(validate_integer, "%P"), font=('SF Pro', 23), fg='black', bg='dark gray') 
    bathrooms_entry.grid(row=3, column=1, padx=10, pady=5)

    region_entry = tk.Entry(root, font=('SF Pro', 23), fg='black', bg='dark gray')   
    region_entry.grid(row=4, column=1, padx=10, pady=5)

    price_entry = tk.Entry(root, validate="key", validatecommand=(validate_integer, "%P"), font=('SF Pro', 23), fg='black', bg='dark gray')
    price_entry.grid(row=5, column=1, padx=10, pady=5) 

    # Create and place the submit button
    submit_button = tk.Button(root, height=2, width=6, text="Submit", font=('SF Pro', 12), command=submit_data) 
    submit_button.grid(row=6, column=0, columnspan=2, padx=10, pady=10)


    # Start the main event loop
    root.mainloop()






