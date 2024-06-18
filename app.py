from flask import Flask, request, render_template, redirect, url_for, jsonify
import pickle
import pandas as pd
import os

# Importing the classes from their respective files
from read_pdf_file import ReadPDFFile
from topic_modelling import TopicModelling
from classification import Classification
from prediction import Prediction

app = Flask(__name__)

# Paths to the necessary files
domain_stopwords_path = 'stop_words.ob'
data_file_path = 'trial_diseases_with_description.csv'
# path = "D:\\proj_files\\trial_chapters_1_2_10_12_13.pdf"
# csv_path = "D:\\proj_files\\trial_chapters_1_2_10_12_13.csv"

path = os.path.join(os.path.dirname(__file__), 'disease_files', 'trial_chapters_1_2_10_12_13.pdf')
csv_path = os.path.join(os.path.dirname(__file__), 'disease_files', 'trial_chapters_1_2_10_12_13.csv')

# Initialize the classes
read_pdf_file = ReadPDFFile(path)
topic_modelling = TopicModelling(csv_path, domain_stopwords_path)
classification = Classification(domain_stopwords_path, data_file_path)
prediction = Prediction(domain_stopwords_path, data_file_path)


@app.route('/', methods=['GET'])
def home():
    return render_template('index2.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about2.html')

@app.route('/services', methods=['GET'])
def services():
    return render_template('services2.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact2.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symptoms = data['symptoms']

        # Reinitialize the classes
        classification = Classification(domain_stopwords_path, data_file_path)
        prediction = Prediction(domain_stopwords_path, data_file_path)
        
        # Classification to get the disease class
        predicted_class = classification.classify_new_text(symptoms)
        
        # Prediction to get the specific disease
        predicted_disease = prediction.predict_disease(symptoms)
        
        return jsonify({
            'predicted_class': predicted_class,
            'predicted_disease': predicted_disease
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)
