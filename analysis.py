import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import nltk
import tensorflow as tf
import re
import pandas as pd
from sklearn.model_selection import train_test_split
nltk.download('stopwords')

DATA_FILENAME = 'processed_data.csv'
STOPWORDS = set(nltk.corpus.stopwords.words('english'))
VALID_MARKS = np.arange(11)


def preprocess(x):
    try:
        x = re.sub(r'[^A-z\s]', '', x)  # get rid of noise
        x = [w.lower() for w in x.split() if w not in STOPWORDS]  # remove stopwords
        return x  # ' '.join(x)  # join the list
    except TypeError:
        return x


def bag_of_words(x):
    """Converts a list of lists of words into a bag of words representation."""
    word_list = list(set([x for l in usable_data['X'] for x in l]))
    vectors = []
    for l in x:
        vector = np.zeros(len(word_list))
        for word in l:
            vector[word_list.index(word)] += 1
        vectors.append(vector)
    return np.asarray(vectors), word_list


raw_data = pd.read_csv(DATA_FILENAME)
selected_data = pd.DataFrame(data={
    'X': pd.concat([raw_data['A_first_impressions'].apply(preprocess), raw_data['B_first_impressions'].apply(preprocess)], ignore_index=True),
    'y': pd.concat([raw_data['A_marks_out_of_10_float'], raw_data['B_marks_out_of_10_float']], ignore_index=True)
})
selected_data['y_valid'] = selected_data['y'].apply(lambda x: x in VALID_MARKS)
usable_data = selected_data[selected_data['X'].notnull() & selected_data['y_valid']].reset_index(drop=True)
X, vocab = bag_of_words(usable_data['X'])
y = np.zeros((len(X), len(VALID_MARKS)))
for i, mark in enumerate(usable_data['y']):
    y[i][int(mark)] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(VALID_MARKS), activation='softmax')
])
model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
model.fit(X_train, y_train, epochs=10)
model.evaluate(X_test, y_test)
