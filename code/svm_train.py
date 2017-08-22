from sklearn import svm
import jieba
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import random
from sklearn.decomposition import  PCA



def svm_train_predict(cate_name):
    # percentage of training set in dataset
    validation_split = 0.8
    # random seed
    seed_random = 7

    # load data
    pos_lines = open('./%s_pos.txt' % cate_name).readlines()
    neu_lines = open('./%s_neu.txt' % cate_name).readlines()
    neg_lines = open('./%s_neg.txt' % cate_name).readlines()

    # build the dataset
    corpus = []
    y = []
    pos_corpus = []
    neu_corpus = []
    neg_corpus = []
    for line in pos_lines:
        pos_corpus.append(' '.join(jieba.cut(line.replace('\n', ''))))
        y.append(1)
    for i in range(3):
        for line in neu_lines:
            neu_corpus.append(' '.join(jieba.cut(line.replace('\n', ''))))
            y.append(2)
    for i in range(12):
        for line in neg_lines:
            neg_corpus.append(' '.join(jieba.cut(line.replace('\n', ''))))
            y.append(3)

    corpus.extend(pos_corpus)
    corpus.extend(neu_corpus)
    corpus.extend(neg_corpus)

    num_pos = len(pos_corpus)
    num_neu = len(neu_corpus)
    num_neg = len(neg_corpus)
    num_all = len(corpus)
    print('num_pos: %s' % num_pos)
    print('num_neu: %s' % num_neu)
    print('num_neg: %s' % num_neg)
    print('num_all: %s' % num_all)
    print('all_percentage_pos: %s' % str(num_pos / num_all))
    print('all_percentage_neu: %s' % str(num_neu / num_all))
    print('all_percentage_neg: %s' % str(num_neg / num_all))

    random.Random(seed_random).shuffle(corpus)
    random.Random(seed_random).shuffle(y)

    # transform the dataset to the bag_of_word format
    vectorizer = CountVectorizer(min_df=1)
    x = vectorizer.fit_transform(corpus).toarray()
    y = np.array(y)
    x = np.array(x)

    print('----> PCA <----')
    pca = PCA(n_components=50)
    x = pca.fit_transform(x)

    print('----> Shape of data set: <----')
    print(x.shape)

    # build training set and test set
    x_train = x[:int(num_all * validation_split)]
    y_train = y[:int(num_all * validation_split)]

    num_train_pos = len(list(filter(lambda n: n == 1, y_train)))
    num_train_neu = len(list(filter(lambda n: n == 2, y_train)))
    num_train_neg = len(list(filter(lambda n: n == 3, y_train)))

    print('num_train_pos', num_train_pos)
    print('num_train_neu', num_train_neu)
    print('num_train_neg', num_train_neg)

    x_test = x[int(num_all * validation_split):]
    y_test = y[int(num_all * validation_split):]
    print('Num of training set: %s' % len(y_train))
    print('Num of test set: %s' % len(y_test))

    num_test_pos = len(list(filter(lambda n: n == 1, y_test)))
    num_test_neu = len(list(filter(lambda n: n == 2, y_test)))
    num_test_neg = len(list(filter(lambda n: n == 3, y_test)))

    print('num_test_pos', num_test_pos)
    print('num_test_neu', num_test_neu)
    print('num_test_neg', num_test_neg)
    print('test_percentage_pos: %s' % str(num_test_pos / len(y_test)))
    print('test_percentage_neu: %s' % str(num_test_neu / len(y_test)))
    print('test_percentage_neg: %s' % str(num_test_neg / len(y_test)))

    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)

    num_predict_pos = len(list(filter(lambda n: n == 1, y_predict)))
    num_predict_neu = len(list(filter(lambda n: n == 2, y_predict)))
    num_predict_neg = len(list(filter(lambda n: n == 3, y_predict)))
    print('num_predict_pos', num_predict_pos)
    print('num_predict_neu', num_predict_neu)
    print('num_predict_neg', num_predict_neg)

    # calculate the accuracy
    y_predict = np.array(y_predict)
    y_test = np.array(y_test)
    diff = y_predict - y_test
    predict_true = list(filter(lambda n: n == 0, diff))
    accu = len(predict_true) / len(y_predict)

    print('Num of predict:')
    print(len(y_predict))
    print('Num of true predict:')
    print(len(predict_true))
    print('Accuracy:')
    print(accu)


if __name__ == '__main__':
    svm_train_predict('food')



