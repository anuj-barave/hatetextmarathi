from flask import Flask, render_template, request
from nltk.corpus import stopwords
from regex import regex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import re
import nltk
from googletrans import Translator
from textblob import TextBlob

nltk.download('stopwords')
eng_stops = set(stopwords.words("english"))
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

app = Flask(__name__)
tfvect = TfidfVectorizer(stop_words='english')
dataframe = pd.read_csv('../hatespeech2/train_E6oV3lV.csv')


def process_message(review_text):
    # remove all the special characters
    new_review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # convert all letters to lower case
    words = new_review_text.lower().split()
    # remove stop words
    words = [w for w in words if not w in eng_stops]

    words = [lemmatizer.lemmatize(word) for word in words]
    # join all words back to text
    return " ".join(words)


def detect(speech):
    prediction = TextBlob(speech).polarity
    if prediction < 0:
        output = ('Negative')
    else:
        output = ('Neutral')

    return output


dataframe['clean_tweet'] = dataframe['tweet'].apply(lambda x: process_message(x))

from sklearn.utils import resample

train_major = dataframe[dataframe.label == 0]
train_minor = dataframe[dataframe.label == 1]
train_minor_upsampled = resample(train_minor, replace=True, n_samples=len(train_major), random_state=123)
train_unsampled = pd.concat([train_minor_upsampled, train_major])
train_unsampled['label'].value_counts()

x = dataframe['clean_tweet']
y = dataframe['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

tfvect = TfidfVectorizer()
tfid_x_train = tfvect.fit_transform(x_train)
tfid_x_test = tfvect.transform(x_test)

classifier = LogisticRegression()
classifier.fit(tfid_x_train, y_train)


def hate_speech_predictor(speech):
    loaded_model = pickle.load(open('module.pkl', 'rb'))
    input_data = [speech]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    prediction = detect(speech)
    return prediction


# Defining the site route


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        translater = Translator()
        message1 = translater.translate(message, dest="en")
        pred = hate_speech_predictor(message1.text)
        return render_template('index.html', prediction=pred, text=message)


if __name__ == '__main__':
    app.run(debug=True)
