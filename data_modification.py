import pandas as pd
import re
from nltk.corpus import stopwords
import nltk

stop_list = stopwords.words('english')


def stopwords(input_sent):
    return [word for word in nltk.word_tokenize(input_sent) if word not in stop_list]


def clean_str(string):
    string = re.sub("[^0-9A-Za-z\u4e00-\u9fa5]", ' ', string)
    return string.strip().lower()


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


if __name__ == '__main__':
    data = load()
    print(data[0][2])
    print(data[1][2])
