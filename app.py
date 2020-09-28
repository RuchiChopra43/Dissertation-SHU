import logging
import os
import sys
import urllib
import flask
from flask import Flask, request, render_template
from flask_cors import CORS

def get_text_sentiment(review):
    '''
    CNN Model
    '''
    from keras.datasets import imdb
    from tensorflow.keras import models
    from keras.preprocessing.text import text_to_word_sequence
    from keras.preprocessing import sequence

    # A dictionary mapping words to an integer index
    word_index = imdb.get_word_index()
    # The first indices are reserved
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3
    # reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    # number of words per review
    max_review_length = 500
    b = models.load_model('model_cnn.h5')

    # review = 'After 30 min I still did not know what the movie is about'
    words = set(text_to_word_sequence(review))
    words = [word_index[w] for w in words]
    words = sequence.pad_sequences([words], maxlen=max_review_length)
    proba = b.predict(words)
    sentiment = 'POSITIVE' if proba * 100 > 0.35 else 'NEGATIVE'
    return sentiment


# Loading Flask and assigning the model variable
app = Flask(__name__)
CORS(app)
app = flask.Flask(__name__, template_folder='templates')
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)


@app.route('/')
def main():
    return render_template('main.html')

# Receiving the input url from the user and using Web Scrapping to extract the news content
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    url = request.get_data(as_text=True)[5:]
    text = urllib.parse.unquote(url)
    outcome = get_text_sentiment(text)

    return render_template('main.html', prediction_text='It is the ' + outcome + ' sentiment')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)


