from sklearn import svm
import jieba
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import random

def svm_train_predict(cate_name):
    # percentage of training set in dataset
    validation_split = 0.8
    # random seed
    seed_random = 7

    # load data
    pos_lines = open('./%s_pos.txt' % cate_name).readlines()
    num_pos = len(pos_lines)
    neu_lines = open('./%s_neu.txt' % cate_name).readlines()
    num_neu = len(neu_lines)
    neg_lines = open('./%s_neg.txt' % cate_name).readlines()
    num_neg = len(neg_lines)
    num_all = num_pos + num_neu + num_neg
    print('num_pos: %s' % num_pos)
    print('num_neu: %s' % num_neu)
    print('num_neg: %s' % num_neg)
    print('num_all: %s' % num_all)
    print('percentage_pos: %s' % str(num_pos / num_all))
    print('percentage_neu: %s' % str(num_neu / num_all))
    print('percentage_neg: %s' % str(num_neg / num_all))
    train_point_pos = int(num_pos * validation_split)
    train_point_neu = int(num_pos + num_neu * validation_split)
    train_point_neg = int(num_pos + num_neu + num_neg * validation_split)

    # build the dataset
    corpus = []
    y = []
    for line in pos_lines:
        corpus.append(' '.join(jieba.cut(line.replace('\n', ''))))
        y.append(1)
    for line in neu_lines:
        corpus.append(' '.join(jieba.cut(line.replace('\n', ''))))
        y.append(2)
    for line in neg_lines:
        corpus.append(' '.join(jieba.cut(line.replace('\n', ''))))
        y.append(3)

    # transform the dataset to the bag_of_word format
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus).toarray()
    y = np.array(y)
    X = np.array(X)

    print('----> Shape of data set: <----')
    print(X.shape)
    print('----> Shape of label <----')
    print(y.shape)

    # build training set and test set
    x_train = []
    y_train = []
    x_train.extend(X[0:train_point_pos])
    x_train.extend(X[num_pos:train_point_neu])
    x_train.extend(X[num_pos + num_neu:train_point_neg])

    y_train.extend(y[0:train_point_pos])
    y_train.extend(y[num_pos:train_point_neu])
    y_train.extend(y[num_pos + num_neu:train_point_neg])

    random.Random(seed_random).shuffle(x_train)
    random.Random(seed_random).shuffle(y_train)

    X_test = []
    y_test = []
    X_test.extend(X[train_point_pos:num_pos])
    X_test.extend(X[train_point_neu:num_pos + num_neu])
    X_test.extend(X[train_point_neg:num_pos + num_neu + num_neg])

    y_test.extend(y[train_point_pos:num_pos])
    y_test.extend(y[train_point_neu:num_pos + num_neu])
    y_test.extend(y[train_point_neg:num_pos + num_neu + num_neg])

    print('Num of training set: %s' % len(x_train))

    print('Num of test set: %s' % len(X_test))

    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_predict = clf.predict(X_test)

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
    svm_train_predict('price')



