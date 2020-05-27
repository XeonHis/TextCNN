import pandas as pd
from data_modification import clean_str
from nltk.corpus import stopwords
import nltk

stop_list = stopwords.words('english')


def stopwords(input_sent):
    return [word for word in nltk.word_tokenize(input_sent) if word not in stop_list]


data = pd.read_csv('res/simplyhired.csv', header=None)
# print(data)
requirement = data[6].apply(lambda x: x.replace('\n', ' ').strip())
category = data[8].apply(lambda x: x.strip())

# temp = nltk.word_tokenize(requirement[0])
# print(temp)
# print([word for word in temp if word not in stop_list])

x_text = [stopwords(clean_str(sent)) for sent in requirement]
# print(x_text[0])
wb_tag = []
se_tag = []
for _ in category:
    if _ == 'web developer':
        wb_tag.append([1, 0])
    else:
        se_tag.append([0, 1])
y = wb_tag + se_tag

conc = [x_text, y]
print(conc[0][0], '\n', conc[1][0])
# print(nltk.word_tokenize(conc[0][0]))
