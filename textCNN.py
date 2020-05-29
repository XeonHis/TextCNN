import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
import keras
from gensim.models import word2vec
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import *
from keras.models import Model
import numpy as np
from keras.models import load_model

stop_list = stopwords.words('english')


# get the stop words of NLTK and elimite
def stopwords(input_sent):
    return [word for word in nltk.word_tokenize(input_sent) if word not in stop_list]


# elimiate puctuation of plain text
def clean_str(string):
    string = re.sub("[^0-9A-Za-z\u4e00-\u9fa5]", ' ', string)
    return string.strip().lower()


# load data, reture list(job_requirements, label)
def load():
    data = pd.read_csv('res/simplyhired.csv', header=None)
    requirement = data[6].apply(lambda x: x.replace('\n', ' '))
    category = data[8]

    x_text = [stopwords(clean_str(sent)) for sent in requirement]
    # web developer: [1, 0]
    # software engineer: [0, 1]
    wb_tag = []
    se_tag = []
    for _ in category:
        if _ == 'web developer':
            wb_tag.append([1, 0])
        else:
            se_tag.append([0, 1])
    label = wb_tag + se_tag

    return [x_text, label]


# vectorize
def vectorize(text):
    model = word2vec.Word2Vec(text, workers=8, size=300, min_count=3, window=4)
    model.init_sims(replace=True)
    model.save('cnn_w2v_model.model')
    model.wv.save_word2vec_format('cnn_vector.vector')
    return model


# customized test, using new crawled data, just for test
def my_test():
    data = pd.read_csv('res/mytest.csv', header=None)
    req = data[0].apply(lambda x: x.replace('\n', ' '))
    cat = data[1]
    x_text = [stopwords(clean_str(sent)) for sent in req]
    return [x_text, [1, 0]]


if __name__ == '__main__':
    data = load()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data[0])
    vocab = tokenizer.word_index
    w2v_model = word2vec.Word2Vec.load('cnn_w2v_model.model')

    # print(len(vocab))
    requirement = pad_sequences(tokenizer.texts_to_sequences(data[0]), maxlen=300)

    train_x, test_x, train_y, test_y = train_test_split(
        requirement, data[1], test_size=0.2, random_state=0
    )

    # print(train_x.shape,train_y.shape)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    # print(train_x.shape, train_y.shape)

    embedding_martix = np.zeros((len(vocab) + 1, 300))
    for word, i in vocab.items():
        try:
            embedding_vector = w2v_model[str(word)]
            embedding_martix[i] = embedding_vector
        except KeyError:
            continue

    main_input = keras.Input(shape=(300,), dtype='float64')
    embedder = Embedding(len(vocab) + 1, 300, input_length=300, weights=[embedding_martix], trainable=True)
    model = Sequential()
    model.add(embedder)
    model.add(Conv1D(256, 5, padding='same', activation='relu'))
    model.add(MaxPool1D(256, 5, padding='same'))
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    history = model.fit(train_x, train_y, batch_size=256, epochs=10, validation_split=0.2)
    model.save('text_cnn.mdl')

    # predict
    mainModel = load_model('text_cnn.mdl')
    # result = mainModel.predict(test_x)
    # print(result)
    # print(np.argmax(result, axis=1))
    score = mainModel.evaluate(test_x, test_y, batch_size=32)
    print(score)

    # data = my_test()
    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(data[0])
    # w2v_model = word2vec.Word2Vec.load('cnn_w2v_model.model')
    # test_data = np.array(pad_sequences(tokenizer.texts_to_sequences(data[0]), maxlen=300))
    # print(test_data.shape)
    # test_y = np.array(data[1])
    # print(test_y.shape)
    # # print(test_data)
    # main_model = load_model('text_cnn.mdl')
    # result = main_model.predict(test_data)
    # print(result)
