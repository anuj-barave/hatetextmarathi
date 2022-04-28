import re

import nltk as nltk
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)

infile = open('model.pkl', 'rb')
loaded_model = pickle.load(infile)
infile.close()
training_data = pd.read_csv('../hatespeech2/train_E6oV3lV.csv')

nltk.download('stopwords')
eng_stops = set(stopwords.words("english"))

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def process_message(review_text):
    # remove all the special characters
    new_review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # convert all letters to lower case
    words = new_review_text.lower().split()
    # remove stop words
    words = [w for w in words if not w in eng_stops]
    # lemmatizer
    words = [lemmatizer.lemmatize(word) for word in words]
    # join all words back to text
    return " ".join(words)


import nltk
nltk.download('wordnet')

training_data['clean_tweet'] = training_data['tweet'].apply(lambda x: process_message(x))

print(training_data.head())

x = training_data['clean_tweet']
y = training_data['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


