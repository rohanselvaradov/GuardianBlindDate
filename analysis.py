import numpy as np
import nltk
import tensorflow as tf
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

nltk.download('stopwords')

DATA_FILENAME = 'processed_data hand-corrected.xlsx'
STOPWORDS = set(nltk.corpus.stopwords.words('english'))
VALID_MARKS = np.arange(11)


def preprocess(x):
    try:
        x = re.sub(r'[^A-z\s]', '', x)  # get rid of noise
        x = [w.lower() for w in x.split() if w not in STOPWORDS]  # remove stopwords
        return x  # ' '.join(x)  # join the list
    except TypeError:
        return x

def binary_mark(data, threshold=8):
    """Converts a Series of marks into a binary representation."""
    return data.apply(lambda d: 1 if d >= threshold else 0 if d < threshold else pd.NA)

def bag_of_words(x):
    """Converts a list of lists of words into a bag of words representation."""
    word_list = list(set([x for l in usable_data['X'] for x in l]))
    vectors = []
    for l in x:
        vector = [0] * (len(word_list))
        for word in l:
            vector[word_list.index(word)] += 1
        vectors.append(vector)
    return vectors, word_list


raw_data = pd.read_excel(DATA_FILENAME)
selected_data = pd.DataFrame(data={
    'X': pd.concat([raw_data['A_first_impressions'].apply(preprocess), raw_data['B_first_impressions'].apply(preprocess)], ignore_index=True),
    'y': pd.concat([binary_mark(raw_data['A_marks_out_of_10_float']), binary_mark(raw_data['B_marks_out_of_10_float'])], ignore_index=True)
})
# selected_data['y_valid'] = selected_data['y'].apply(lambda x: x in VALID_MARKS)
usable_data = selected_data[selected_data['X'].notnull() & selected_data['y'].notnull()].reset_index(drop=True)
X, vocab = bag_of_words(usable_data['X'])
# y = np.zeros((len(X), len(VALID_MARKS)))
# for i, mark in enumerate(usable_data['y']):
#     y[i][int(mark)] = 1
y = list(usable_data['y'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='softmax')
])
model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')
print("Training model...")
model.fit(X_train, y_train, epochs=30)
print("Evaluating model...")
model.evaluate(X_test, y_test)
