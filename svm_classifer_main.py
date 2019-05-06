import pandas as pd
import numpy as np
import dill
from model.svm import MultilabelSVMClassifier
from metric.multilabel import multilabel_accuracy
from util.preprocessing import data_to_examples
from util.stopword_removal import StopwordRemover
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
import time


def preprocess_for_clf(X_train, X_test):
    sr = StopwordRemover()
    stemmer = StemmerFactory().create_stemmer()
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=1000)
    X_train_str = [sr.remove_stopwords(' '.join(x)) for x in X_train]
    X_test_str = [sr.remove_stopwords(' '.join(x)) for x in X_test]
    X_train_str = [stemmer.stem(x) for x in X_train_str]
    X_test_str = [stemmer.stem(x) for x in X_test_str]

    X_train = vectorizer.fit_transform(X_train_str)
    X_test = vectorizer.transform(X_test_str)
    return X_train.toarray(), X_test.toarray()

if __name__ == '__main__':
    # load data
    # train_df = pd.read_csv('data/to_debug.csv')
    train_df = pd.read_csv('data/cerpen-training.csv')
    test_df = pd.read_csv('data/cerpen-test.csv')

    # load training+test data to examples
    tic = time.time()
    print('preprocessing...')
    X_train, Y_train = data_to_examples(train_df)
    X_test, Y_test = data_to_examples(test_df)
    print('training length:', len(X_train))
    print('test length:', len(X_test))

    X_train, X_test = preprocess_for_clf(X_train, X_test)
    print('preprocessing done in', (time.time() - tic), 'seconds')

    tic = time.time()
    clf = MultilabelSVMClassifier(n_jobs=6)
    clf.fit(X_train, Y_train)
    # print('fitting done, pickling classifier...')
    # with open('multilabel_mnb.classifier', 'wb') as pickle_file:
    #     # clf = dill.load(pickle_file)
    #     dill.dump(clf, pickle_file)
    # print('model fitting done in', (time.time() - tic), 'seconds')

    # thresholds = np.array([0.01, 0.3, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    # thresholds = np.log(thresholds)
    # best_threshold = 0.5
    # best_jaccard_similarity = 0

    # tic = time.time()
    # log_probas_all = clf.predict_log_proba(X_train)
    # print('Predicting training data done in', (time.time() - tic), 'seconds')
    # for t in thresholds:
    #     Y_pred = get_predictions(log_probas_all, t)
    #     avg_jaccard = multilabel_accuracy(Y_train, Y_pred)
    #     print(np.exp(t), avg_jaccard)
    #     if avg_jaccard > best_jaccard_similarity:
    #         best_jaccard_similarity = avg_jaccard
    #         best_threshold = t
    #
    # print('best threshold (in log scale):', np.exp(best_threshold), best_jaccard_similarity)

    tic = time.time()
    Y_test_pred = clf.predict(X_test)
    print('Predicting test data done in', (time.time() - tic), 'seconds')
    test_accuracy = multilabel_accuracy(Y_test, Y_test_pred)
    print('Test accuracy:', test_accuracy)




