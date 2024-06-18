#topic modelling
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from gensim import matutils

class TopicModelling:
    def __init__(self, csv_path, stopwords_path):
        self.csv_path = csv_path
        self.stopwords_path = stopwords_path
        self.df02 = pd.read_csv(self.csv_path)
        self.stop_words = self.load_stop_words()
        self.ch_no = ['cardiovascular','respiratory','metabolic nutritional','eye','ear nose throat']
        self.df02['Ch_No'] = self.ch_no
        self.count_vectorizer = CountVectorizer(stop_words=self.stop_words)
        self.doc_word_cv = self.count_vectorizer.fit_transform(self.df02['string_values'])
        self.corpus = matutils.Sparse2Corpus(self.doc_word_cv)
        self.id2word = dict((v, k) for k, v in self.count_vectorizer.vocabulary_.items())
        self.lsa = TruncatedSVD(5)
        self.doc_topic = self.lsa.fit_transform(self.doc_word_cv)
        self.tem_list = []
        self.final_dic = {}
        self.tem_df = None

        self.perform_topic_modelling()
        self.create_topic_dataframe()
        self.save_to_csv()

    def load_stop_words(self):
        with open(self.stopwords_path, 'rb') as fp:
            return pickle.load(fp)

    def perform_topic_modelling(self):
        self.display_topics(self.lsa, self.count_vectorizer.get_feature_names_out(), 20)

    def display_topics(self, model, feature_names, no_top_words, topic_names=None):
        for ix, topic in enumerate(model.components_):
            inner_tem_list = []
            if not topic_names or not topic_names[ix]:
                print("\nTopic ", ix)
            else:
                print("\nTopic: '", topic_names[ix], "'")
            print(", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
            inner_tem_list.append(", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
            self.tem_list.append(inner_tem_list)

    def create_topic_dataframe(self):
        self.final_dic = {
            "Heart": self.tem_list[3],
            "respiratory": self.tem_list[4],
            "metabolism": self.tem_list[1],
            "eye": self.tem_list[2],
            "ear_nose_throat": self.tem_list[0]
        }
        self.tem_df = pd.DataFrame.from_dict(self.final_dic, orient='index')
        self.tem_df = self.tem_df.rename(columns={0: 'Description'})

    def save_to_csv(self):
        self.tem_df.to_csv('diseases_with_description.csv', index=False)

# Usage example:
# csv_path = "D:\\NIVEDITA\\B.TECH\\CDAC\\project_vs_code\\trial_projects\\disease_NLP_trial_02\\all_CSV\\trial_chapters_1_2_10_12_13.csv"
# stopwords_path = 'stop_words.ob'
# topic_modelling = TopicModelling(csv_path, stopwords_path)
