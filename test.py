import pandas as pd
from textCNN import clean_str
from nltk.corpus import stopwords
import nltk
from keras.preprocessing.text import Tokenizer
from gensim.models import word2vec
from keras.models import load_model

# stop_list = stopwords.words('english')
#
#
# def stopwords(input_sent):
#     return [word for word in nltk.word_tokenize(input_sent) if word not in stop_list]
#
#
# data = pd.read_csv('res/simplyhired.csv', header=None)
# # print(data)
# requirement = data[6].apply(lambda x: x.replace('\n', ' ').strip())
# category = data[8].apply(lambda x: x.strip())
#
# # temp = nltk.word_tokenize(requirement[0])
# # print(temp)
# # print([word for word in temp if word not in stop_list])
#
# x_text = [stopwords(clean_str(sent)) for sent in requirement]
# # print(x_text[0])
# wb_tag = []
# se_tag = []
# for _ in category:
#     if _ == 'web developer':
#         wb_tag.append([1, 0])
#     else:
#         se_tag.append([0, 1])
# y = wb_tag + se_tag
#
# conc = [x_text, y]
# # print(conc[0][0], '\n', conc[1][0])
#
#
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(conc[0])
# vocab = tokenizer.word_index
# print(vocab)


# w2v_model = word2vec.Word2Vec.load('cnn_w2v_model.vector')
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(w2v_model)
# test = [[1, 3, 5], [2, 3, 5], [4, 7, 9]]
# print(test[0])


