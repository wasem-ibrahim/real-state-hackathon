import re

import joblib
import numpy as np
import openai
import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, request, session
from flask_session import Session


class RealEstatePricingModel:
    """
    A class that manages the real estate pricing model and related utilities.
    """
    
    def __init__(self):
        self.scaler = joblib.load('scaler.pkl')
        self.kmeans = joblib.load('kmeans.pkl')
        self.feature_means = joblib.load('feature_means.pkl')
        self.model = xgb.XGBRegressor()
        self.model.load_model('XGBoost.json')
        
    def predict(self, preprocessed_data):
        prediction = self.model.predict(preprocessed_data)
        prediction_original = float(np.expm1(prediction)[0])
        return prediction_original

    def predict_data_json(self, data_json):
        preprocessed_data = self.preprocessor.preprocess(data_json)
        prediction = self.predict(preprocessed_data)
        return {"prediction": prediction}



class DataPreprocessor:
    """
    A class that handles the preprocessing of input data for the real estate pricing model.
    """
    
    def __init__(self, model: RealEstatePricingModel):
        self.model = model
        
    def preprocess(self, data):
        df = pd.DataFrame([data])
        df.fillna(self.model.feature_means, inplace=True)
        return self._preprocess_data(df)


    def _preprocess_data(self, df: pd.DataFrame):
        # Log Transformation
        df['median_income'] = np.log1p(df['median_income']) / 1000
        
        # Creating bins for total bedrooms
        bins = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 500, 1000, 5000, 10000]
        labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-10', '10-20', '20-50', '50-100', '100-500', '500-1000', '1000-5000', '5000-10000']
        #@ Total Bedrooms is the total number of bedrooms in the house and not in the block anymore.
        df['bedroom_bins'] = pd.cut(df['total_bedrooms'], bins=bins, labels=labels, right=False)
        
        if 'ocean_proximity' not in df.columns:
            df['ocean_proximity'] = '<1H OCEAN' # as it is the most common value in addition to Near ocean and near bay which are all close to the ocean
        
        # One Hot Encoding
        df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=False)

        # List of columns that the model expects for ocean_proximity
        expected_ocean_columns = [
            'ocean_proximity_<1H OCEAN', 
            'ocean_proximity_INLAND', 
            'ocean_proximity_ISLAND', 
            'ocean_proximity_NEAR BAY', 
            'ocean_proximity_NEAR OCEAN'
        ]

        # Add any missing columns with values set to zero
        for col in expected_ocean_columns:
            if col not in df.columns:
                df[col] = 0

        df = df.rename(columns={'ocean_proximity_<1H OCEAN': 'ocean_proximity_Less_1H OCEAN'})
        
        # Creating New Features
        df['rooms_per_household'] = df['total_rooms']
        df['bedrooms_per_room'] = df['total_bedrooms']
        df['population_per_household'] = df['population']

        # Additional Feature Engineering
        major_cities_coords = [(34.05, -118.25), (37.77, -122.42), (32.71, -117.16)]
        avg_city_coord = np.mean(major_cities_coords, axis=0)
        coast_coord = (35.78, -120.90)
        df['distance_to_major_city'] = np.sqrt((df['latitude'] - avg_city_coord[0])**2 + (df['longitude'] - avg_city_coord[1])**2)
        df['distance_to_coast'] = np.sqrt((df['latitude'] - coast_coord[0])**2 + (df['longitude'] - coast_coord[1])**2)

        # Interaction Features. Divide by 1000 because the original values of the set were divided by 1000
        df['rooms_income_interaction'] = df['total_rooms'] * (df['median_income'] / 1000)

        # Derive Ratios
        df['population_per_bedroom'] = df['population'] / df['total_bedrooms']
        df['households'] = 1
        df['households_per_room'] = 1

        columns_to_scale = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 
                    'population', 'households', 'median_income', 'rooms_per_household', 
                    'bedrooms_per_room', 'population_per_household', 'distance_to_major_city', 
                    'distance_to_coast', 'rooms_income_interaction', 'population_per_bedroom', 'households_per_room']
        
        df[columns_to_scale] = self.model.scaler.transform(df[columns_to_scale])

        # Adjust the list to only scale columns that are present in df
        columns_present = [col for col in columns_to_scale if col in df.columns]

        df[columns_present] = self.model.scaler.transform(df[columns_present])

        # Clustering for Latitude and Longitude
        df['location_cluster'] = self.model.kmeans.predict(df[['longitude', 'latitude']])

        # One Hot Encoding for 'bedroom_bins'
        df = pd.get_dummies(df, columns=['bedroom_bins'], drop_first=True)

        # List of columns that the model expects
        expected_columns = ['bedroom_bins_1-2', 'bedroom_bins_2-3', 'bedroom_bins_3-4', 'bedroom_bins_4-5', 'bedroom_bins_5-10']

        # Drop any extra columns generated from one-hot encoding
        for col in df.columns:
            if "bedroom_bins_" in col and col not in expected_columns:
                df.drop(columns=[col], inplace=True)
        
        # At the end of your preprocess_data function
        expected_column_order = [
            'longitude', 'latitude', 'housing_median_age', 
            'median_income', 'ocean_proximity_Less_1H OCEAN', 'ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 
            'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN', 'rooms_per_household', 'bedrooms_per_room', 
            'population_per_household', 'distance_to_major_city', 'distance_to_coast', 'rooms_income_interaction', 
            'population_per_bedroom', 'location_cluster', 'bedroom_bins_1-2', 'bedroom_bins_2-3', 
            'bedroom_bins_3-4', 'bedroom_bins_4-5', 'bedroom_bins_5-10'
        ]
        
        print(len(expected_column_order))
        print(df.columns)
        # Drop the original columns
        df.drop(columns=['total_rooms', 'total_bedrooms', 'population', 'households_per_room'], inplace=True)
        df = df[expected_column_order]
        
        return df


class ConversationManager:
    """
    A class that manages the conversation flow with the user.
    """
    #! THESE NEED TO BE AJUSTED TO THE NEW MODEL
    REQUIRED_KEYS = ["longitude", "latitude", "housing_median_age", "total_rooms", 
                     "total_bedrooms", "population", "median_income", "ocean_proximity"]
    
    QUESTIONS = {
        "longitude": "Can you share the longitude coordinates of the property? It helps me get a better sense of the location.",
        "latitude": "And what about the latitude coordinates? It's essential for an accurate prediction.",
        "housing_median_age": "How old is the property you're looking at? If you could provide the median age, that'd be great!",
        "total_rooms": "How spacious is the property? Can you tell me the total number of rooms?",
        "total_bedrooms": "How many bedrooms are you looking for in the property? This often affects the price.",
        "population": "What's the population of the people who will be living at the house.",
        "households": "Can you share the number of households in that area? It helps in gauging the community size.",
        "median_income": "Could you please provide me with the average income of the area? This can influence house prices.",
        "ocean_proximity": """
        Is the property close to the ocean or bay? If so, can you specify how close? Keep in mind that your ansers can only be one of the following: 
        <1H OCEAN, INLAND, NEAR OCEAN, NEAR BAY, ISLAND. Please also watch out for typos and spelling errors including capitalization.
        """
    }
    
    def __init__(self):
        openai.api_key = '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'
    
    def start_conversation(self):
        session['user_data'] = {}
        session['messages'] = [{"role": "system", "content": """
                                You are a helpful assistant who works as a real estate pricing predictor.
                                Please assist the user with their query by asking them about the questions that I'll give to you."""}]
        return "How can I assist you with real estate pricing today?"
    
    def continue_conversation(self, user_message_content):
        # Extend the conversation with the user's message
        session['messages'].append({"role": "user", "content": user_message_content})

        # Identify the data point to save
        save_key = next((key for key in self.REQUIRED_KEYS if key not in session['user_data']), None)

        if save_key == "ocean_proximity":
            # Directly save the user's answer if it's about ocean proximity
            session['user_data'][save_key] = user_message_content
        else:
            # Otherwise, use regex to extract a number from the user's message
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", user_message_content)
            print("Numbers: ", numbers)
            
            if numbers:
                session['user_data'][save_key] = float(numbers[0])

        # Identify the next data point that needs to be collected after updating the user_data
        next_key = next((key for key in self.REQUIRED_KEYS if key not in session['user_data']), None)
        print("Next Key: ", next_key)
        print("Session: ", session['user_data'])

        # If we're still collecting data, ask the next question. Otherwise, make a prediction.
        if next_key:
            # Construct an explicit prompt to guide the Chat API
            prompt = f"I need to ask about: {self.QUESTIONS[next_key]}. Please help me phrase this in a conversational manner."
            
            # Use the OpenAI Chat API to generate the question
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=session['messages'] + [{"role": "assistant", "content": prompt}])
            assistant_message_content = response['choices'][0]['message']['content'].strip()
        else:
            # Preprocess the collected data and make a prediction
            prediction_response = self.model.predict_data_json(session['user_data'])
            assistant_message_content = f"Based on your data, the predicted price is: ${prediction_response['prediction']:,.2f}"

        # Extend the conversation with the assistant's response
        session['messages'].append({"role": "assistant", "content": assistant_message_content})

        return assistant_message_content



class FlaskApp:
    """
    A class representing the Flask app with its routes.
    """
    
    def __init__(self):
        self.app = Flask(__name__)
        self.app.secret_key = 'password' 
        self.app.config['SESSION_PERMANENT'] = False  # Sessions will not be permanent
        self.app.config['SESSION_TYPE'] = 'filesystem'  # Use the file system to store session data
        self.app.config['SESSION_FILE_DIR'] = './.flask_session/' 
        Session(self.app)  # Initialize the session extension

        self.model = RealEstatePricingModel()
        self.preprocessor = DataPreprocessor(self.model)
        self.conversation_manager = ConversationManager()

        # Define routes
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/start_conversation', 'start_conversation', self.start_conversation_endpoint, methods=['GET'])
        self.app.add_url_rule('/continue_conversation', 'continue_conversation', self.continue_conversation_endpoint, methods=['POST'])
        self.app.add_url_rule('/predict', 'predict', self.predict_endpoint, methods=['POST'])

    def run(self):
        self.app.run(debug=True, host='127.0.0.1', port=5000)

    def index(self):
        return 'Hello World!'

    def start_conversation_endpoint(self):
        return jsonify({"status": "conversation_started", "message": self.conversation_manager.start_conversation()})

    def continue_conversation_endpoint(self):
        user_message_content = request.json.get('message').strip()
        assistant_message_content = self.conversation_manager.continue_conversation(user_message_content)
        return jsonify({"message": assistant_message_content})


    def predict_endpoint(self):
        data = request.get_json()
        preprocessed_data = self.preprocessor.preprocess(data)
        prediction = self.model.predict(preprocessed_data)
        return jsonify({"prediction": prediction})


if __name__ == '__main__':
    flask_app = FlaskApp()
    flask_app.run()
