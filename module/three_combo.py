from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack
import lightgbm as lgb
from sklearn.decomposition import LatentDirichletAllocation
from .preprocess import preprocess
from multiprocessing import Pool
from sklearn.decomposition import TruncatedSVD
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


stopwords = set(stopwords.words("english"))
ps = PorterStemmer()


class ThreeComboModel:
    model: lgb.LGBMClassifier
    count_vectorizer: CountVectorizer
    lda: LatentDirichletAllocation
    lsa: TruncatedSVD
    tfidf_vectorizer: TfidfVectorizer

    def __init__(self) -> None:
        self.model = joblib.load("3combo_model.pkl")
        self.count_vectorizer = joblib.load("count_vectorizer.pkl")
        self.tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
        self.lda = joblib.load("lda.pkl")
        self.lsa = joblib.load("lsa.pkl")

    def predict(self, text: str):
        cleaned = preprocess(text, stopwords, ps)
        data = [cleaned]
        tfidf_data = self.tfidf_vectorizer.transform(data)
        tf_data = self.count_vectorizer.transform(data)
        lda_data = self.lda.transform(tf_data)
        lsa_data = self.lsa.transform(tfidf_data)

        for_predict = hstack([tfidf_data, lda_data, lsa_data]).tocsr()
        pred = self.model.predict(for_predict)
        return pred[0]
