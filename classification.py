#Classification
import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from numpy import dot
from numpy.linalg import norm
from sklearn.linear_model import LogisticRegression

class Classification:
    def __init__(self, domain_stopwords_path, data_file_path):
        self.domain_stopwords_path = domain_stopwords_path
        self.data_file_path = data_file_path
        self.df = pd.read_csv(self.data_file_path)
        self.domain_stop_word = self.load_stop_words()
        self.text_cleaner = self.TextCleaner(self.domain_stop_word)
        self.cv = CountVectorizer(stop_words="english")
        self.df_cv = None
        self.X_train_cv1 = None
        self.y_train = None
        self.cv1 = CountVectorizer()
        self.lr = LogisticRegression()

        self.process_data()
        self.train_model()

    class TextCleaner:
        def __init__(self, stop_words):
            self.stop_words = stop_words

        def clean_text_func(self, text):
            text = str(text)
            text = text.lower()
            text = re.sub(r"[^A-Za-z0-9^,!?.\/'+]", " ", text)
            text = re.sub(r"\+", " ", text)
            text = re.sub(r",", " ", text)
            text = re.sub(r"\.", " ", text)
            text = re.sub(r"!", " ", text)
            text = re.sub(r"\?", " ", text)
            text = re.sub(r"'", " ", text)
            text = re.sub(r":", " : ", text)
            text = re.sub(r"\s{2,}", " ", text)
            text = re.sub(r"[0-9]", " ", text)
            final_text = ""
            for x in text.split():
                if x not in self.stop_words:
                    final_text = final_text + x + " "
            return final_text

    def load_stop_words(self):
        with open(self.domain_stopwords_path, 'rb') as fp:
            return pickle.load(fp)

    def process_data(self):
        self.df['Description'] = self.df['Description'].apply(lambda x: self.text_cleaner.clean_text_func(x))
        X = self.cv.fit_transform(list(self.df.loc[:, 'Description']))
        self.df_cv = pd.DataFrame(X.toarray(), columns=self.cv.get_feature_names_out())
        self.y_train = self.df.D_Name

    def cosine_similarity(self, v1, v2):
        return dot(v1, v2) / (norm(v1) * norm(v2))

    
    def classify_new_text(self, new_text_cl):
        new_text_cv = self.cv.transform([new_text_cl]).toarray()[0]
        for chpter_number in range(int(self.df.shape[0])):
            print(f"This is chapter number: {chpter_number}")
            print(f"Cosine similarity: {self.cosine_similarity(self.df_cv.iloc[chpter_number], new_text_cv)}")
        similarities = []
        for chapter_number in range(int(self.df.shape[0])):
            similarity = self.cosine_similarity(self.df_cv.iloc[chapter_number], new_text_cv)
            similarities.append((chapter_number, similarity))
        
        # Get the chapter with the highest cosine similarity
        most_similar_chapter = max(similarities, key=lambda item: item[1])
        most_similar_chapter_index = most_similar_chapter[0]
        
        # Use the chapter with the highest cosine similarity as the predicted topic
        predicted_topic = self.df.iloc[most_similar_chapter_index].D_Name

        cleaned_text = self.text_cleaner.clean_text_func(new_text_cl)
        X_test_cv3 = self.cv1.transform([cleaned_text])
        y_pred_cv3 = self.lr.predict(X_test_cv3)
        
        print(f"Highest cosine similarity is for chapter number: {most_similar_chapter_index}")
        print(f"Cosine similarity: {most_similar_chapter[1]}")
        print(f"Predicted topic based on cosine similarity: {predicted_topic}")

        return predicted_topic

    def train_model(self):
        X_train = self.df.Description
        self.X_train_cv1 = self.cv1.fit_transform(X_train)
        self.lr.fit(self.X_train_cv1, self.y_train)

    def save_text_cleaner(self, filename='text_cleaner.sav'):
        with open(filename, 'wb') as file:
            pickle.dump(self.text_cleaner, file)

    def load_text_cleaner(self, filename='text_cleaner.sav'):
        with open(filename, 'rb') as file:
            return pickle.load(file)

# Usage example:
# domain_stopwords_path = 'stop_words.ob'
# data_file_path = "D:\\NIVEDITA\\B.TECH\\CDAC\\project_vs_code\\trial_projects\\disease_NLP_trial_02\\code_and_files\\trial_diseases_with_description.csv"
# classifier = Classification(domain_stopwords_path, data_file_path)

# new_text_cl = input("Enter symptoms: ")
# classifier.classify_new_text(new_text_cl)
# classifier.save_text_cleaner()
