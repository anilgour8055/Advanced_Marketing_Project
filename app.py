# app.py (version 2)

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model AND the training columns
model = joblib.load('ltv_model.pkl')
training_columns = joblib.load('training_columns.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data
    data = request.get_json()
    
    # Convert incoming json to a pandas DataFrame
    features_df = pd.DataFrame(data, index=[0])
    
    # One-hot encode the categorical features
    features_df = pd.get_dummies(features_df)
    
    # Reindex the dataframe to match the training columns
    # This adds any missing columns (and fills with 0) and ensures the order is correct
    features_df = features_df.reindex(columns=training_columns, fill_value=0)
    
    # Make a prediction
    prediction = model.predict(features_df)
    
    # Return the prediction
    return jsonify({'predicted_ltv': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)