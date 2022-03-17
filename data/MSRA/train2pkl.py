# coding:utf-8
import codecs
import re

import numpy as np
import pandas as pd
import collections
import pickle as pkl


def wordtag():
    input_data = codecs.open('train1.txt', 'r', 'utf-8')
    output_data = codecs.open('wordtag.txt', 'w', 'utf-8')
    for line in input_data.readlines():
        # line=re.split('[，。；！：？、‘’“”]/[o]'.decode('utf-8'),line.strip())
        line = line.strip().split()

        if len(line) == 0:
            continue
        for word in line:
            word = word.split('/')
            if word[1] != 'o':
                if len(word[0]) == 1:
                    output_data.write(word[0] + "/B_" + word[1] + " ")
                elif len(word[0]) == 2:
                    output_data.write(word[0][0] + "/B_" + word[1] + " ")
                    output_data.write(word[0][1] + "/E_" + word[1] + " ")
                else:
                    output_data.write(word[0][0] + "/B_" + word[1] + " ")
                    for j in word[0][1:len(word[0]) - 1]:
                        output_data.write(j + "/M_" + word[1] + " ")
                    output_data.write(word[0][-1] + "/E_" + word[1] + " ")
            else:
                for j in word[0]:
                    output_data.write(j + "/o" + " ")
        output_data.write('\n')

    input_data.close()
    output_data.close()


wordtag()
datas = list()
labels = list()
linedata = list()
linelabel = list()


idx = list(range(11))
val = ['', 'B_ns', 'B_nr', 'B_nt', 'M_ns', 'M_nr', 'M_nt', 'E_ns', 'E_nr', 'E_nt', 'o']

tag2id = pd.Series(idx, index=val)
id2tag = pd.Series(val, index=idx)

input_data = codecs.open('wordtag.txt', 'r', 'utf-8')
for line in input_data.readlines():
    line = re.split('[，。；！：？、‘’“”]/[o]', line.strip())
    for sen in line:
        sen = sen.strip().split()
        if len(sen) == 0:
            continue
        linedata = []
        linelabel = []
        num_not_o = 0
        for word in sen:
            word = word.split('/')
            linedata.append(word[0])
            linelabel.append(tag2id[word[1]])

            if word[1] != 'o':
                num_not_o += 1
        if num_not_o != 0:
            datas.append(linedata)
            labels.append(linelabel)

input_data.close()
print(len(datas))
print(len(labels))


def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


all_words = flatten(datas)
sr_allwords = pd.Series(all_words)
sr_allwords = sr_allwords.value_counts()
set_words = sr_allwords.index
set_ids = range(1, len(set_words) + 1)
word2id = pd.Series(set_ids, index=set_words)
id2word = pd.Series(set_words, index=set_ids)

word2id["unknow"] = len(word2id) + 1

max_len = 50


def X_padding(words):
    """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
    ids = list(word2id[words])
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0] * (max_len - len(ids)))  # 短则补全
    return ids


def y_padding(ids):
    """把 tags 转为 id 形式， 并自动补全位 max_len 长度。"""
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0] * (max_len - len(ids)))  # 短则补全
    return ids


df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
df_data['x'] = df_data['words'].apply(X_padding)
df_data['y'] = df_data['tags'].apply(y_padding)
x = np.asarray(list(df_data['x'].values))
y = np.asarray(list(df_data['y'].values))

print('Finished creating the data generator.')


with open('../dataMSRA.pkl', 'wb') as outp:
    pkl.dump(word2id, outp)
    pkl.dump(id2word, outp)
    pkl.dump(tag2id, outp)
    pkl.dump(id2tag, outp)
    pkl.dump(x, outp)
    pkl.dump(y, outp)
print('** Finished saving the data.')
