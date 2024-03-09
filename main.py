import os
import sys

project_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(project_path))

import nltk

nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import pickle
from flask import Flask, request
from scipy.sparse import hstack
from module import preprocess
from module.three_combo import ThreeComboModel

app = Flask(__name__)
app.tfidf_vectorizer = pickle.load(
    open("data/github_bug_prediction_tfidf_vectorizer.pkl", "rb")
)
app.basic_model = pickle.load(open("data/github_bug_prediction_basic_model.pkl", "rb"))
app.stopword_set = set(stopwords.words())
app.stemmer = PorterStemmer()
app.three_combo = ThreeComboModel()


# api for handon 7 part1
@app.get("/predict-basic")
def predict_basic():
    resposne_object = {"status": "success"}
    args = request.args.to_dict(flat=False)
    title = args["title"][0]
    body = args["body"][0]
    predict = app.basic_model.predict_proba(
        hstack(
            [
                app.tfidf_vectorizer.transform(
                    [
                        preprocess(title + body, app.stopword_set, app.stemmer),
                    ]
                ),
            ]
        ),
    )
    prob = predict[0][1]
    resposne_object["predict_as"] = "bug" if prob > 0.5 else "not bug"
    resposne_object["bug_prob"] = prob
    return resposne_object


@app.get("/predict-combo")
def predict_combo():
    resposne_object = {"status": "success"}
    args = request.args.to_dict(flat=False)
    title = args["title"][0]
    body = args["body"][0]
    predict = app.three_combo.predict(title + body)
    resposne_object["predict_as"] = "bug" if predict > 0.5 else "not bug"
    resposne_object["bug_prob"] = predict
    return resposne_object


if __name__ == "__main__":
    app.run(debug=True)
