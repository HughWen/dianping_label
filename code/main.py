import numpy as np
import random
import time
import jieba
import gensim
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import regularizers
from keras.layers import Embedding
from keras.layers import LSTM
from keras.preprocessing import text, sequence
import code.utils as utils

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


def build_data_cv(pos_file, neu_file, neg_file, cv=10):
    """
    Loads the data and split into k folds.
    """
    docs = []
    with open(pos_file) as f:
        for line in f:
            line = line.replace('\n', '')
            doc = {
                'y': 1,
                'text': line,
                'split': np.random.randint(0, cv)}
            docs.append(doc)
    with open(neu_file) as f:
        for line in f:
            line = line.replace('\n', '')
            doc = {
                'y': 2,
                'text': line,
                'split': np.random.randint(0, cv)
            }
            docs.append(doc)
    with open(neg_file) as f:
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


def lstm_training_predict(training_set, training_label, test_set, test_label):
    # cut sentences to words
    training_x = np.array(list(map(lambda n: ' '.join(jieba.cut(n.replace('\n', ''))), training_set)))
    training_y = np.array(training_label)
    test_x = np.array(list(map(lambda n: ' '.join(jieba.cut(n.replace('\n', ''))), test_set)))
    test_y = np.array(test_label)

    # Build vocabulary & sequences
    tk = text.Tokenizer(num_words=max_features)
    tk.fit_on_texts(training_x)
    training_x = tk.texts_to_sequences(training_x)
    word_index = tk.word_index
    training_x = sequence.pad_sequences(training_x, maxlen=max_len)

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
    model.fit(training_x, training_y, batch_size=batch_size, epochs=num_epoch, validation_split=validation_split, shuffle=shuffle)
    print('============ LSTM w2v model training finish ==============')

    test_x = tk.texts_to_sequences(test_x)
    test_x = sequence.pad_sequences(test_x, maxlen=max_len)
    predict_y = model.predict(test_x)
    predict_y = np.array(list(map(lambda n: n.argmax() + 1, predict_y)))
    diff = predict_y - test_y
    predict_true = list(filter(lambda n: n == 0, diff))
    accu = len(predict_true) / len(predict_y)
    print(accu)
    return accu

    # model.save(filepath=utils.get_model_path(model_name))
    # print('model was saved to ' + utils.get_model_path(model_name))


def train_test():
    data_cv = build_data_cv('./food_pos.txt', './food_neu.txt', './food_neg.txt', cv=k_fold)
    random.Random(rand_seed).shuffle(data_cv)  # shuffle
    # k time training and test
    for i in range(k_fold):
        print('fold %s' % i)
        accu_list = []
        training_set = []
        training_y = []
        test_set = []
        test_y = []
        for doc in data_cv:
            if doc['split'] == i:
                test_set.append(doc['text'])
                test_y.append(doc['y'])
            else:
                training_set.append(doc['text'])
                training_y.append(doc['y'])
        print("Train/Test split: {:d}/{:d}".format(len(training_y), len(test_y)))
        training_y = list(map(trans_label, training_y))
        # train and test
        accu_list.append(lstm_training_predict(training_set, training_y, test_set, test_y))
    print('Final average accuracy: %s in %s fold cross validation' % (str(sum(accu_list) / len(accu_list)), k_fold))



if __name__ == '__main__':
    train_test()