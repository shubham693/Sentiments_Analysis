from flask import Flask, jsonify, request
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import joblib
import numpy as np
import pandas as pd
import pickle


  

import flask
app = Flask(__name__)
# log_reg = joblib.load('model2_artifacts.pkl')
log_reg = pickle.load(open('model2_artifacts.pkl', 'rb'))
# count_vect = joblib.load('vectorizer.pkl')
count_vect = pickle.load(open('vectorizer.pkl', 'rb'))
ps = PorterStemmer()
    
###################################################
def pre_processing(text):
    '''
    This function converts raw text to model's expected format by removing stopwards and lowering the case.
    '''
    all_stopwords=stopwords.words('english')
    r_words=['no','not','off','nor',"didn't","isn't","couldn't","haven't",'or',"should've","aren't",
         "couldn","didn","doesn't",'doesn',"don't",'don','hadn',"hadn't",'hasn',"hasn't",'haven',
         'mightn',"mightn't",'mustn',"mustn't","needn't",'needn',"shouldn","shouldn't",'wasn',
         "wasn't","won't","won","wouldn't","because","same",'wouldn','should']
    for words in r_words:
        all_stopwords.remove(words)

    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in all_stopwords]
    review = ' '.join(review)
    return review
###################################################


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    This function handles the POST predict request. 
    It pre-process raw text.
    Predict the Sentiments on pre-processed text.
    Render the result on screen.
    '''
    to_predict = request.form.to_dict()['review_text'].strip()
    review_text = pre_processing(to_predict)
    final_x = count_vect.transform([review_text]).toarray()
    prob = log_reg.predict(final_x)[0]
    if prob == 1:
        prediction = "Positive"
    else:
        prediction = "Negative"

    return flask.render_template('predict.html', prediction = prediction)


if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='localhost', port=8081)
