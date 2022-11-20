import nltk
import tensorflow as tf
import re
import pandas as pd
nltk.download('stopwords')

DATA_FILENAME = 'processed_data.csv'
STOPWORDS = set(nltk.corpus.stopwords.words('english'))


def preprocess(x):
    try:
        x = re.sub(r'[^A-z\s]', '', x)  # get rid of noise
        x = [w.lower() for w in x.split() if w not in STOPWORDS]  # remove stopwords
        return x  # ' '.join(x)  # join the list
    except TypeError:
        return x


def bag_of_words(x):
    """Converts a list of lists of words into a bag of words representation."""
    vocab = list(set([x for l in usable_data['X'] for x in l]))
    vectors = []
    for l in x:
        vector = [0] * len(vocab)
        for word in l:
            vector[vocab.index(word)] += 1
        vectors.append(vector)
    return vectors


raw_data = pd.read_csv(DATA_FILENAME)
selected_data = pd.DataFrame(data={
    'X': pd.concat([raw_data['A_describe_X_in_three_words'].apply(preprocess), raw_data['B_describe_X_in_three_words'].apply(preprocess)], ignore_index=True),
    'y': pd.concat([raw_data['A_marks_out_of_10_float'], raw_data['B_marks_out_of_10_float']], ignore_index=True)
})
usable_data = selected_data[selected_data['X'].notnull() & selected_data['y'].notnull()].reset_index(drop=True)
X = bag_of_words(usable_data['X'])
y = usable_data['y']


def main():
    pass
