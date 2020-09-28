# Importing the Libraries
import logging
import os
# import pickle
import sys
import urllib

import flask
from flask import Flask, request, render_template
from flask_cors import CORS
from textblob import TextBlob

def get_text_sentiment(text):
    '''
    Utility function to classify sentiment of passed text
    using textblob's sentiment method
    '''
    # create TextBlob object of passed app text
    text = text.replace("+", " ")
    analysis = TextBlob(text)

    # set sentiment
    if not analysis.sentiment.polarity < 0:
        return 'POSITIVE'
    # elif analysis.sentiment.polarity == 0:
    #     return 'MIXED'
    else:
        return 'NEGATIVE'


# Loading Flask and assigning the model variable
app = Flask(__name__)
CORS(app)
app = flask.Flask(__name__, template_folder='templates')

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

# with open('pickle_model_senti.pkl', 'rb') as handle:
#     model = pickle.load(handle)


@app.route('/')
def main():
    return render_template('main.html')


# Receiving the input url from the user and using Web Scrapping to extract the news content
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    url = request.get_data(as_text=True)[5:]
    text = urllib.parse.unquote(url)
    outcome = get_text_sentiment(text)

    # LOAD SAVED PICKLE MODEL AND TEST IT WITH SOME RANDOM TEXT
    # from sklearn.feature_extraction.text import CountVectorizer
    # from keras.preprocessing import sequence
    #
    # # mdl = pickle.load(open('pickle_model_senti.pkl', 'rb'))
    # cv = CountVectorizer()
    # text = [text]
    # text_vect = cv.fit_transform(text).toarray()
    #
    # max_words = 500
    # text_vect = sequence.pad_sequences(text_vect, maxlen=max_words)
    # pred = model.predict(text_vect)
    # outcome = "Positive"
    # if pred == 0:
    #     outcome = "Positive"
    # else:
    #     outcome = "Negative"

    # Passing the news article to the model and returing whether it is Fake or Real
    # pred = model.predict([news])
    # return render_template('main.html', prediction_text='The news is "{}"'.format(outcome))
    return render_template('main.html', prediction_text='It is the ' + outcome + ' sentiment')


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)
