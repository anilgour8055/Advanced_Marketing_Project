# app.py (version 2)

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model AND the training columns
model = joblib.load('ltv_model.pkl')
training_columns = joblib.load('training_columns.pkl')

@app.route('/predict', methods=['POST'])
# Make sure this function has the correct indentation
@app.route('/predict', methods=['POST'])
def predict():
    # This line should have 4 spaces before it
    data = request.get_json()
    
    # This line should have 4 spaces before it
    features_df = pd.DataFrame(data, index=[0])
    
    # This line should have 4 spaces before it
    features_df = pd.get_dummies(features_df)
    
    # This line should have 4 spaces before it
    features_df = features_df.reindex(columns=training_columns, fill_value=0)
    
    # This line should have 4 spaces before it
    prediction = model.predict(features_df)
    
    # This line should have 4 spaces before it
    return jsonify({'predicted_ltv': float(prediction[0])})
if __name__ == '__main__':
    app.run(debug=True)