#read pdf file
import PyPDF2
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import pickle
import os

class ReadPDFFile:
    def __init__(self, path):
        self.path = path
        self.pdfReader = None
        self.total_pages = 0
        self.list_from_pdf = []
        self.article = ""
        self.df = pd.DataFrame()
        self.All_text = ""
        self.tokens = []
        self.lower_tokens = []
        self.alpha_only = []
        self.no_stops = []
        self.lemmatized = []
        self.adj_list = []
        self.new_stopwords = []
        self.stpwrd = []
        self.corpus = ""
        self.string = ""
        self.df01 = pd.DataFrame()
        self.nlp = spacy.load("en_core_web_sm")

        self.read_pdf()
        self.process_text()
        self.extract_adjectives()
        self.define_stop_words()
        self.remove_stop_words()
        self.save_stop_words()

    def read_pdf(self):
        with open(self.path, "rb") as ff:
            self.pdfReader = PyPDF2.PdfReader(ff)
            self.total_pages = len(self.pdfReader.pages)

            for page_num in range(self.total_pages):
                pageObj = self.pdfReader.pages[page_num]
                extract = pageObj.extract_text().split("\n")
                self.article += " ".join(extract)
                self.list_from_pdf.append(extract)

            self.df = pd.DataFrame([self.article], columns=['string_values'])
            self.All_text = " ".join(self.df.string_values)

    def process_text(self):
        self.tokens = word_tokenize(self.All_text)
        self.lower_tokens = [t.lower() for t in self.tokens]
        self.alpha_only = [t for t in self.lower_tokens if t.isalpha()]
        self.no_stops = [t for t in self.alpha_only if t not in stopwords.words('english')]
        wordnet_lemmatizer = WordNetLemmatizer()
        self.lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in self.no_stops]

    def extract_adjectives(self):
        self.df['spacy_doc'] = list(self.nlp.pipe(self.df.string_values))
        doc_adj = [token.text.lower() for doc in self.df.spacy_doc for token in doc if token.pos_ == 'ADJ']
        self.adj_list = list(set(doc_adj))

    def define_stop_words(self):
        stop_words = [
            "patient", "may", "disease", "cause", "treatment", "also", "symptom", "usually", "sign",
            "diagnosis", "result", "pain", "include", "pressure", "lung", "pulmonary", "respiratory",
            "chest", "fluid", "complication", "change", "blood", "infection", "therapy", "prevent",
            "acute", "care", "child", "level", "air", "use", "severe", "help", "used", "exercise",
            "normal", "incidence", "pneumonia", "tissue", "show", "chronic", "failure", "cast", "increased",
            "monitor", "produce", "increase", "space", "occurs", "alveolar", "heart", "pathophysiology",
            "sputum", "provide", "decreased", "pneumothorax", "test", "special", "tube", "condition", 
            "common", "surgery", "secretion", "fibrosis", "disorder", "pa", "area", "form", "cell", "skin",
            "drainage", "tb", "year", "commonly", "check", "teach", "rest", "watch", "encourage", 
            "underlying", "consideration", "et", "early", "hour", "family", "need", "effusion", "body", 
            "drug", "support", "rate", "syndrome", "requires", "inflammation", "abg", "side", "infant", 
            "however", "upper", "cor", "pulmonale", "ventilator", "mechanical", "breath", "maintain", 
            "foot", "day", "bed", "parent", "especially", "fever", "culture", 'system', 'within', 'factor', 
            'amount', 'death', 'movement', 'progress', 'volume', 'one', 'stage', 'report', 'avoid', 
            'respiration', 'trauma', 'occur', 'atelectasis', 'hand', 'includes', 'weight', 'tendon', 
            'hypertension', 'le', 'time', 'lead', 'damage', 'causing', 'require', 'activity', 'injury', 
            'risk', 'mm', 'measure', 'examination', 'nerve', 'stress', 'make', 'al', 'see', 'decrease', 
            'age', 'hg', 'case', 'month', 'coughing', 'develops', 'formation', 'without', 'site', 'every', 
            'reduce', 'relieve', 'effect', 'percussion', 'ordered', 'develop', 'affect', 'loss', 'flow', 
            'lesion', 'technique', 'exposure', 'gas', 'finding', 'procedure', 'begin', 'wall', 'immediately', 
            'type', 'response', 'position', 'needed', 'administer', 'control', 'ass', 'increasing', 
            'although', 'tell', 'output', 'give', 'analysis', 'history', 'often', 'week', 'home', 'perform',
            'function', 'typically', 'frequently', 'adult', 'indicate', 'administration', 'explain', 'using', 
            'suggest', 'called', 'center', 'head', 'people', 'resulting', 'including', 'period', 'feature', 
            'problem'
        ]
        self.new_stopwords = self.adj_list + stop_words
        self.stpwrd = stopwords.words('english')
        self.stpwrd.extend(self.new_stopwords)

    def remove_stop_words(self):
        self.no_stops01 = [t for t in self.lemmatized if t not in self.stpwrd]
        self.corpus = " ".join(self.no_stops01)
        self.string = self.corpus
        self.df01 = pd.DataFrame([self.string], columns=['string_values'])

    def save_stop_words(self):
        with open('stop_words.ob', 'wb') as fp:
            pickle.dump(self.stpwrd, fp)

# Usage example:
#path = "D:\\NIVEDITA\\B.TECH\\CDAC\\project_vs_code\\trial_projects\\disease_NLP_trial_02\\code_and_files\\data_to_see\\trial_chapters_1_2_10_12_13.pdf"
pdf_path = os.path.join(os.path.dirname(__file__), 'disease_files', 'trial_chapters_1_2_10_12_13.pdf')
pdf_processor = ReadPDFFile(pdf_path)