# Prognify

## Introduction
This project is designed to predict diseases based on various inputs using machine learning models. The system leverages natural language processing (NLP) and various classification algorithms to provide predictions.

## 2. Project Overview

### 2.1. Objectives
- To predict disease classes and specific diseases based on symptoms.
- To provide a user-friendly web interface for inputting symptoms and viewing predictions.

### 2.2. Key Features
- Input symptoms through a web form.
- Predict disease class and specific diseases.
- Display results on a user-friendly interface.

## Project Structure
```plaintext
prognify/
│
├── app.py
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
│   ├── about.html
│   ├── contact.html
│   ├── index.html
│   ├── result.html
│   └── services.html
│
└── __pycache__/
    ├── classification.cpython-311.pyc
    ├── prediction.cpython-311.pyc
    ├── read_pdf_file.cpython-311.pyc
    ├── topic_modelling.cpython-311.pyc
```

## 4. Dependencies
The project relies on several third-party Python libraries. These dependencies are listed in the `requirements.txt` file:


## 5. Installation and Setup

### 5.1. Prerequisites
- Python 3.x
- pip (Python package installer)

### 5.2. Steps to Install and Run the Application

#### Clone the Repository:
Open a terminal and clone the repository using the following command:
```sh
git clone <repository_url>
cd prognify
```
## 6. Create a Virtual Environment (optional but recommended):
### Creating a virtual environment helps manage dependencies and avoid conflicts.
On windows
```sh
python -m venv venv
venv\Scripts\activate
```
On macOS/Linux
```sh
python3 -m venv venv
source venv/bin/activate
```
## 7. Install Dependencies:
### Install the required dependencies from the requirements.txt file:

```sh
pip install -r requirements.txt
```
### Download NLTK Data:
The project uses NLTK, which requires additional data downloads:

```sh
python -m nltk.downloader all
```
### Run the Application:
Start the Flask application by running the main script (app2.py):
```sh
python app2.py
```
### Access the Application:
Open a web browser and navigate to http://127.0.0.1:5000 to access the web application.

