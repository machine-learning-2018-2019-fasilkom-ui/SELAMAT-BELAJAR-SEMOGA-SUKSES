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
from gensim.models.doc2vec import TaggedDocument, Doc2Vec


def preprocess_for_clf(X_train, Y_train, X_test):
    sr = StopwordRemover()
    stemmer = StemmerFactory().create_stemmer()
    vectorizer = TfidfVectorizer(ngram_range=(1, 3) , max_features=5000)
    X_train_str = [sr.remove_stopwords(' '.join(x)) for x in X_train]
    X_test_str = [sr.remove_stopwords(' '.join(x)) for x in X_test]
    X_train_str = [stemmer.stem(x) for x in X_train_str]
    X_test_str = [stemmer.stem(x) for x in X_test_str]

    # X_train_tagged = [TaggedDocument(words=x.split(), tags=y)
    #                   for x, y in zip(X_train_str, Y_train)]
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
    # model.save('d2v.model')
    # print("model saved")

    model = Doc2Vec.load('d2v.model')

    X_train = np.array([model.infer_vector(x.split()) for x in X_train_str])
    X_test = np.array([model.infer_vector(x.split()) for x in X_test_str])
    # X_train = vectorizer.fit_transform(X_train_str)
    # X_test = vectorizer.transform(X_test_str)
    # return X_train.toarray(), X_test.toarray()
    return X_train, X_test

# Best with doc2vec vec_size 300 Test accuracy: 0.32416832339297535
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

    X_train, X_test = preprocess_for_clf(X_train, Y_train, X_test)
    print('preprocessing done in', (time.time() - tic), 'seconds')

    tic = time.time()
    clf = MultilabelSVMClassifier(n_jobs=6, C=0.2, kernel='poly', show_progress=True)
    clf.fit(X_train, Y_train)

    tic = time.time()
    Y_test_pred = clf.predict(X_test)
    print('Predicting test data done in', (time.time() - tic), 'seconds')
    test_accuracy = multilabel_accuracy(Y_test, Y_test_pred)
    print('Test accuracy:', test_accuracy)




