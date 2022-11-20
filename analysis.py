import nltk
import tensorflow as tf
import re
import pandas as pd

DATA_FILENAME = 'processed_data.csv'
STOPWORDS = set(nltk.corpus.stopwords.words('english'))


def preprocess(x):
    try:
        x = re.sub(r'[^A-z\s]', '', x)  # get rid of noise
        x = [w for w in x.split() if w not in STOPWORDS]  # remove stopwords
        return x  # ' '.join(x)  # join the list
    except TypeError:
        return x


data = pd.read_csv(DATA_FILENAME)
X = [x for x in data['A_describe_X_in_three_words'].apply(preprocess).values if type(x) == list and len(x) == 3]
y = data['A_marks_out_of_10']



def main():
    pass
