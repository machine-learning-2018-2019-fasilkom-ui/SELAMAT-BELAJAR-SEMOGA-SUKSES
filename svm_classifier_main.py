import pandas as pd
import numpy as np
import time
import os
from model.svm import MultilabelSVMClassifier
from metric.multilabel import multilabel_accuracy
from util.preprocessing import data_to_examples
from util.svm import print_clf_reports_per_label
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from util.pickling import pickle_classifier, load_pickled_classifier

DOC2VEC_PATH = os.path.join('pickled_models', 'd2v.model')
CLF_PATH = os.path.join('pickled_models', 'multilabel_svm.classifier')
RESULT_PATH = os.path.join('results', 'MultilabelSVM_Result.csv')

def preprocess_for_clf(X_train, X_test):
    vectorizer = CountVectorizer(ngram_range=(1, 1) , max_features=2000)
    X_train_str = [' '.join(x) for x in X_train]
    X_test_str = [' '.join(x) for x in X_test]

    # Uncomment to train doc2vec model.
    # X_train_tagged = [TaggedDocument(words=x.split(), tags=[idx])
    #                   for idx, x, in enumerate(X_train_str)]
    #
    # max_epochs = 100
    # vec_size = 300
    # alpha = 0.025
    # model = Doc2Vec(size=vec_size,
    #                 alpha=alpha,
    #                 min_alpha=0.00025,
    #                 min_count=1,
    #                 dm=1,
    #                 workers=6)
    # model.build_vocab(X_train_tagged)
    #
    # for epoch in range(max_epochs):
    #     print('iteration {0}'.format(epoch))
    #     model.train(X_train_tagged,
    #                 total_examples=model.corpus_count,
    #                 epochs=model.iter)
    #     # decrease the learning rate
    #     model.alpha -= 0.0002
    #     # fix the learning rate, no decay
    #     model.min_alpha = model.alpha
    #
    # model.save(DOC2VEC_PATH)
    # print("model saved")

    model = Doc2Vec.load(DOC2VEC_PATH)

    X_train_d2v = np.array([model.infer_vector(x.split()) for x in X_train_str])
    X_test_d2v = np.array([model.infer_vector(x.split()) for x in X_test_str])
    X_train_cnt = vectorizer.fit_transform(X_train_str).toarray()
    X_test_cnt = vectorizer.transform(X_test_str).toarray()

    X_train = np.hstack((X_train_d2v, X_train_cnt))
    X_test = np.hstack((X_test_d2v, X_test_cnt))

    return X_train, X_test


def get_best_C(X_train, Y_train, X_val, Y_val):
    best_accuracy = 0
    cs = np.linspace(0.1, 1, 10)  # 0, 0.1, 0.2, ..., 1
    c_select = -1
    c_accuracies = []
    for c in cs:
        clf = MultilabelSVMClassifier(C=c, kernel='linear')
        clf.fit(X_train, Y_train)
        Y_val_pred = clf.predict(X_val)
        val_accuracy = multilabel_accuracy(Y_val, Y_val_pred)
        c_accuracies.append(val_accuracy)
        print("Trying C:", c)
        print("Validation accuracy:", val_accuracy)
        if val_accuracy > best_accuracy:
            c_select = c
            best_accuracy = val_accuracy

    print(cs)
    print(c_accuracies)
    print("Best C (according to validation accuracy):", c_select)

    return c_select


if __name__ == '__main__':
    # load data
    train_df = pd.read_csv('data/cerpen-training.csv')
    test_df = pd.read_csv('data/cerpen-test.csv')

    # load training+test data to examples
    tic = time.time()
    print('preprocessing...')
    X_train_full, Y_train_full = data_to_examples(train_df)
    X_test, Y_test = data_to_examples(test_df)
    print('Training size:', len(X_train_full))
    print('Test size:', len(X_test))

    X_train_full, X_test = preprocess_for_clf(X_train_full, X_test)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, Y_train_full, test_size=0.2, random_state=322)
    print('preprocessing done in', (time.time() - tic), 'seconds')

    # Uncomment to get best C
    # c = get_best_C(X_train, Y_train, X_val, Y_val) best C = 0.3
    c = 0.3

    # Readjust if you want to retrain the model.
    # clf = MultilabelSVMClassifier(C=c, kernel='linear')
    # clf.fit(X_train_full, Y_train_full)
    # pickle_classifier(clf, CLF_PATH)
    clf = load_pickled_classifier(CLF_PATH)

    tic = time.time()
    Y_test_pred = clf.predict(X_test)
    print('Predicting test data done in', (time.time() - tic), 'seconds')
    test_accuracy = multilabel_accuracy(Y_test, Y_test_pred)
    print_clf_reports_per_label(clf, X_test, Y_test)
    print('Test accuracy:', test_accuracy)

    test_result = pd.DataFrame({'id': test_df['id'],
                                'predicted_genre': pd.Series(Y_test_pred)})
    test_result.to_csv(RESULT_PATH, index=False)
    print('Prediction result stored to:', RESULT_PATH)




