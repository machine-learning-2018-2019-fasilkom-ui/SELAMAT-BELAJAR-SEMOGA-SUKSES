import pandas as pd
import numpy as np
import json
import re
import pickle
from model.naive_bayes import MultilabelMNBTextClassifier
from metric.multilabel import multilabel_accuracy

# given panda dataframe return X(examples) and Y(class)
def data_to_examples(df):
    X = []
    y = []
    for idx, row in df.iterrows():
        categories = json.loads(str(row['categories']).replace("\'", '\"'))
        text = str(row['text']).lower()
        text = re.sub('[^a-z]+', ' ', text)
        x = text.split()

        X.append(x)
        y.append(categories)
    return np.array(X), np.array(y)


if __name__ == '__main__':
    # load data
    train_df = pd.read_csv('data/to_debug.csv')
    cv_df = pd.read_csv('data/cerpen-cross_validation.csv')
    test_df = pd.read_csv('data/cerpen-test.csv')

    # load training data to examples
    print('preprocessing...')
    X, y = data_to_examples(train_df)
    genres = set(genre for y_i in y for genre in y_i)
    print('preprocessing done.')

    clf = MultilabelMNBTextClassifier()
    clf.fit(X, y)
    # with open('multilabel_mnb.classifier', 'rb') as pickle_file:
    #     clf = pickle.load(pickle_file)
    print('done')

    tresholds = np.array([0.01, 0.3, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    tresholds = np.log(tresholds)
    treshold = 0.5
    best_jaccard_similarity = 0
    for t in tresholds:
        y_pred = []
        for x, y_i in list(zip(X, y))[:5]:
            predicted_genres = []
            probas = dict(clf.predict_log_proba_single(x))
            for genre in genres:
                if probas[genre] > t:
                    predicted_genres.append(genre)
            y_pred.append(predicted_genres)
        avg_jaccard = multilabel_accuracy(y, y_pred)
        print(np.exp(t), avg_jaccard)
        if avg_jaccard > best_jaccard_similarity:
            best_jaccard_similarity = avg_jaccard
            treshold = t

    print(np.exp(treshold), best_jaccard_similarity)


