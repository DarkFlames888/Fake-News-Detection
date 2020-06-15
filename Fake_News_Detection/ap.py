from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import speech_recognition as s

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    df= pd.read_csv("dataset.csv", encoding="latin-1")

    df['label'] = df['label'].map({'REAL': 1, 'FAKE': 0})
    X = df['text']
    y = df['label']

    x_train,x_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=7)
 


    tfidf_vectorizer=TfidfVectorizer(stop_words='english', ngram_range = (2,2))
    #DataFlair - Fit and transform train set, transform test set ngram_range = (2,2)  max_df=0.7
    tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
    tfidf_test=tfidf_vectorizer.transform(x_test)

    #clf=PassiveAggressiveClassifier(max_iter=50)
    clf=MultinomialNB()
    clf.fit(tfidf_train,y_train)
    #DataFlair - Predict on the test set and calculate accuracy
    y_pred=clf.predict(tfidf_test)

    if request.method == 'POST':
            message = request.form['message']

            data = [message]
            vect = tfidf_vectorizer.transform(data).toarray()
            my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)

@app.route('/speech',methods=['POST'])
def speech():
    r=s.Recognizer()

    with s.Microphone() as source:
    #print("SAY SOMETHING...")
        audio=r.listen(source)
        data=r.recognize_google(audio)
    #print(data)
    return render_template("voiceout.html",fact=data)


@app.route('/keybora')
def keybora():
    return render_template('text.html')

@app.route('/vola')
def vola():
    return render_template('voice.html')
   
if __name__ == '__main__':
    app.run()
