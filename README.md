Prognify
Introduction
This project is designed to predict diseases based on various inputs using machine learning models. The system leverages natural language processing (NLP) and various classification algorithms to provide predictions.

Project Overview
-Objectives
--To predict disease classes and specific diseases based on symptoms.
--To provide a user-friendly web interface for inputting symptoms and viewing predictions.

-Key Features
--Input symptoms through a web form.
--Predict disease class and specific diseases.
--Display results on a user-friendly interface.

Project Structure
disease_pred_2/
│
├── app2.py
├── classification.py
├── diseases_with_description.csv
├── final_trained_model.sav
├── final_vectorizer.sav
├── prediction.py
├── read_pdf_file.py
├── stop_words.ob
├── text_cleaner.sav
├── topic_modelling.py
├── trained_model.sav
├── trial_diseases_with_description.csv
├── vectorizer.sav
│
├── disease_files/
│   ├── ear_nose_throat_diseases.csv
│   ├── eye_diseases.csv
│   ├── heart_diseases.csv
│   ├── metabolic_diseases.csv
│   ├── respiratory_diseases.csv
│   ├── trial_chapters_1_2_10_12_13.csv
│   └── trial_chapters_1_2_10_12_13.pdf
│
├── static/
│   ├── css/
│   │   ├── styles_about2.css
│   │   ├── styles_contact.css
│   │   ├── styles_services.css
│   │   └── web.css
│   └── images/
│       ├── data_vsls.jpg
│       ├── logo_pred.png
│       ├── man.png
│       ├── predictive_ana.jpg
│       ├── risk_eval.jpg
│       └── woman.png
│
├── templates/
│   ├── about2.html
│   ├── contact2.html
│   ├── index2.html
│   ├── result.html
│   └── services2.html
│
└── __pycache__/
    ├── classification.cpython-311.pyc
    ├── prediction.cpython-311.pyc
    ├── read_pdf_file.cpython-311.pyc
    ├── topic_modelling.cpython-311.pyc

Data
-diseases_with_description.csv: Contains disease descriptions.
-Additional data files are located in the disease_files directory.

Models
The models used in this project are pre-trained and saved in the project directory:
-final_trained_model.sav
-final_vectorizer.sav

Code Explanation
-app.py: Main application script.
-classification.py: Handles disease classification.
-prediction.py: Predictive analysis.
-read_pdf_file.py: Reads and processes PDF files.
-topic_modelling.py: Topic modeling for disease data.
