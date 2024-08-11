import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model and OrdinalEncoder
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

# Define the list of columns that your model requires
data_columns = [
    'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission',
    'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats', 'Age_of_Car', 'Brand'
]

# Category columns
cat_cols = ['Fuel_Type', 'Transmission', 'Owner_Type', 'Brand']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    year = int(request.form['Year'])
    kilometers_driven = float(request.form['Kilometers_Driven'])
    fuel_type = request.form['Fuel_Type']
    transmission = request.form['Transmission']
    owner_type = request.form['Owner_Type']
    mileage = float(request.form['Mileage'])
    engine = float(request.form['Engine'])
    power = float(request.form['Power'])
    seats = float(request.form['Seats'])
    brand = request.form['Brand']
    age_of_car = 2024 - year

    # Create a dataframe with the input data
    input_data = pd.DataFrame({
        'Year': [year],
        'Kilometers_Driven': [kilometers_driven],
        'Fuel_Type': [fuel_type],
        'Transmission': [transmission],
        'Owner_Type': [owner_type],
        'Mileage': [mileage],
        'Engine': [engine],
        'Power': [power],
        'Seats': [seats],
        'AgeofCar': [age_of_car],
        'Brand': [brand]
    })

    # Encode the categorical data
    input_data[cat_cols] = encoder.transform(input_data[cat_cols])

    # Predict the value
    prediction = model.predict(input_data)
    output = prediction[0].round(2)
    return render_template('index.html', prediction_text=f'Predicted Car Price: ${output}')

if __name__ == "__main__":
    app.run(debug=True)
