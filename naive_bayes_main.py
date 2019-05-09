import pandas as pd
import numpy as np
import os
from model.naive_bayes import MultilabelMNBTextClassifier
from metric.multilabel import multilabel_accuracy
from util.preprocessing import data_to_examples
from util.naive_bayes import get_predictions
import time
from util.pickling import pickle_classifier, load_pickled_classifier

CLF_PATH = os.path.join('pickled_models', 'multilabel_mnb.classifier')
RESULT_PATH = os.path.join('results', 'MultilabelMNB_Result.csv')

# Naive bayes for baseline
# Test accuracy: 0.36252879548108013
if __name__ == '__main__':
    # load data
    train_df = pd.read_csv('data/cerpen-training.csv')
    test_df = pd.read_csv('data/cerpen-test.csv')

    # load training+test data to examples
    tic = time.time()
    print('preprocessing...')
    X_train, Y_train = data_to_examples(train_df)
    X_test, Y_test = data_to_examples(test_df)
    print('preprocessing done in', (time.time() - tic), 'seconds')
    print('training length:', len(X_train))
    print('test length:', len(X_test))

    # Readjust if you want to retrain the model.
    # tic = time.time()
    # clf = MultilabelMNBTextClassifier(n_jobs=6)
    # clf.fit(X_train, Y_train)
    # print('fitting done, pickling classifier...')
    # pickle_classifier(clf, CLF_PATH)
    clf = load_pickled_classifier(CLF_PATH)
    # print('model fitting done in', (time.time() - tic), 'seconds')

    # Uncomment to find best threshold.
    # thresholds = np.array([0.01, 0.3, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    # thresholds = np.log(thresholds)
    # best_threshold = 0.5
    # best_jaccard_similarity = 0
    #
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
    # Best treshold = 0.95

    tic = time.time()
    probas_all_test = clf.predict_log_proba(X_test)
    print('Predicting test data done in', (time.time() - tic), 'seconds')
    Y_test_pred = get_predictions(probas_all_test, np.log(0.95))
    test_accuracy = multilabel_accuracy(Y_test, Y_test_pred)
    print('Test accuracy:', test_accuracy)

    test_result = pd.DataFrame({'id': test_df['id'],
                                'predicted_genre': pd.Series(Y_test_pred)})
    test_result.to_csv(RESULT_PATH,index=False)
    print('Prediction result stored to:', RESULT_PATH)




