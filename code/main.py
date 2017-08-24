import numpy as np
import random
import jieba
import time
import utils

# ======== parameters part start ========
# k_fold value
k_fold = 5

# random seed
rand_seed = 7

# Input parameters
max_features = 5000
max_len = 200
embedding_size = 300
border_mode = 'same'
dropout = 0.8
l2_regularization = 0.05

# RNN parameters
output_size = 50
rnn_activation = 'tanh'
recurrent_activation = 'hard_sigmoid'

# Compile parameters
loss = 'categorical_crossentropy'
optimizer = 'rmsprop'

# Training parameters
batch_size = 128
num_epoch = 10
validation_split = 0
shuffle = True
# ======== parameter part end ========


def build_data_cv(f_pos, f_neu, f_neg, cv=10):
    """
    Loads the data and split into k folds.
    """
    docs = []
    with open(f_pos) as f:
        for line in f:
            line = line.replace('\n', '')
            doc = {
                'y': 1,
                'text': line,
                'split': np.random.randint(0, cv)}
            docs.append(doc)
    with open(f_neu) as f:
        for line in f:
            line = line.replace('\n', '')
            doc = {
                'y': 2,
                'text': line,
                'split': np.random.randint(0, cv)
            }
            docs.append(doc)
    with open(f_neg) as f:
        for line in f:
            line = line.replace('\n', '')
            doc = {
                'y': 3,
                'text': line,
                'split': np.random.randint(0, cv)
            }
            docs.append(doc)
    return docs


def trans_label(i):
    if i == 1:
        return [1, 0, 0]
    elif i == 2:
        return [0, 1, 0]
    elif i == 3:
        return [0, 0, 1]
    else:
        raise Exception('No this label!')


def lstm_training_predict(set_traning, label_training, set_test, label_test):
    import gensim
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras import regularizers
    from keras.layers import Embedding
    from keras.layers import LSTM
    from keras.preprocessing import text, sequence

    # cut sentences to words
    x_training = np.array(list(map(lambda n: ' '.join(jieba.cut(n.replace('\n', ''))), set_traning)))
    y_training = np.array(label_training)
    x_test = np.array(list(map(lambda n: ' '.join(jieba.cut(n.replace('\n', ''))), set_test)))
    y_test = np.array(label_test)

    # Build vocabulary & sequences
    tk = text.Tokenizer(num_words=max_features)
    tk.fit_on_texts(x_training)
    x_training = tk.texts_to_sequences(x_training)
    word_index = tk.word_index
    x_training = sequence.pad_sequences(x_training, maxlen=max_len)

    # Build pre-trained embedding layer
    w2v = gensim.models.Word2Vec.load(utils.get_w2v_model_path('dianping/dianping.model'))
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_size))
    for word, i in word_index.items():
        if word in w2v.wv.vocab:
            embedding_matrix[i] = w2v[word]
    embedding_layer = Embedding(len(word_index) + 1, embedding_size, weights=[embedding_matrix], input_length=max_len)
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(dropout))
    model.add(LSTM(output_dim=output_size, activation=rnn_activation, recurrent_activation=recurrent_activation))
    model.add(Dropout(dropout))
    model.add(Dense(3, kernel_regularizer=regularizers.l2(l2_regularization)))
    model.add(Activation('softmax'))
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print('============ LSTM w2v model training begin ===============')
    model.fit(x_training, y_training, batch_size=batch_size, epochs=num_epoch, validation_split=validation_split, shuffle=shuffle)
    print('============ LSTM w2v model training finish ==============')

    x_test = tk.texts_to_sequences(x_test)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    y_predict = model.predict(x_test)
    y_predict = np.array(list(map(lambda n: n.argmax() + 1, y_predict)))
    diff = y_predict - y_test
    predict_true = list(filter(lambda n: n == 0, diff))
    accu = len(predict_true) / len(y_predict)
    print('accuracy is', accu)
    return accu

    # model.save(filepath=utils.get_model_path(model_name))
    # print('model was saved to ' + utils.get_model_path(model_name))


def svm_training_predict(training_set, label_training, set_test, label_test, pca_flag=False):
    from sklearn import svm
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import PCA

    x_training = list(map(lambda n: ' '.join(jieba.cut(n.replace('\n', ''))), training_set))
    y_training = np.array(label_training)
    y_test = np.array(label_test)
    x_test = list(map(lambda n: ' '.join(jieba.cut(n.replace('\n', ''))), set_test))

    # transform the dataset to the bag_of_word format
    vectorizer = CountVectorizer(min_df=1)
    x_training = vectorizer.fit_transform(x_training).toarray()
    x_test = vectorizer.transform(x_test).toarray()
    if pca_flag:
        print('==== PCA ====')
        pca  = PCA(n_components=50)
        x_training = pca.fit_transform(x_training)
        clf = svm.SVC(decision_function_shape='ovo')
        clf.fit(x_training, y_training)
        x_test = pca.transform(x_test)
        y_predict = clf.predict(x_test)
        y_predict = np.array(y_predict)
        diff = y_predict - y_test
        predict_ture = list(filter(lambda n: n == 0, diff))
        accu = len(predict_ture) / len(y_predict)
        print('accuracy is', accu)
        return accu
    else:
        clf = svm.SVC(decision_function_shape='ovo')
        clf.fit(x_training, y_training)
        y_predict = clf.predict(x_test)
        y_predict = np.array(y_predict)
        diff = y_predict - y_test
        predict_ture = list(filter(lambda n: n == 0, diff))
        accu = len(predict_ture) / len(y_predict)
        print('accuracy is', accu)
        return accu


def train_test(f_pos, f_neu, f_neg, model_name, pca_flag):
    data_cv = build_data_cv(f_pos, f_neu, f_neg, cv=k_fold)
    random.Random(rand_seed).shuffle(data_cv)  # shuffle
    accu_list = []
    if model_name == 'lstm':
        # k fold cross validation in lstm w2v
        for i in range(k_fold):
            print('fold %s' % i)
            x_training = []
            y_training = []
            x_test = []
            y_test = []
            for doc in data_cv:
                if doc['split'] == i:
                    x_test.append(doc['text'])
                    y_test.append(doc['y'])
                else:
                    x_training.append(doc['text'])
                    y_training.append(doc['y'])
            print("Train/Test split: {:d}/{:d}".format(len(y_training), len(y_test)))
            y_training = list(map(trans_label, y_training))
            # train and test
            accu_list.append(lstm_training_predict(x_training, y_training, x_test, y_test))
        print('Final average accuracy: %s in %s fold cross validation' % (str(sum(accu_list) / len(accu_list)), k_fold))

    elif model_name == 'svm':
        # k fold cross validation in svm
        for i in range(k_fold):
            print('fold %s' % i)
            x_training = []
            y_training = []
            x_test = []
            y_test = []
            for doc in data_cv:
                if doc['split'] == i:
                    x_test.append(doc['text'])
                    y_test.append(doc['y'])
                else:
                    x_training.append(doc['text'])
                    y_training.append(doc['y'])
            print("Train/Test split: {:d}/{:d}".format(len(y_training), len(y_test)))
            # train and test
            accu_list.append(svm_training_predict(x_training, y_training, x_test, y_test, pca_flag=pca_flag))
        print('Final average accuracy: %s in %s fold cross validation' % (str(sum(accu_list) / len(accu_list)), k_fold))

    else:
        print('not this model')


if __name__ == '__main__':
    train_test('./data/food_pos.txt', './data/food_neu.txt', './data/food_neg.txt', model_name='svm', pca_flag=True)