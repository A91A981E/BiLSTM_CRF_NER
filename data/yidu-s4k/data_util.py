#!/usr/bin/python
# -*- coding: UTF-8 -*-

import codecs
import collections
import re

import numpy as np
import pandas as pd


def data2pkl():
    datas = list()
    labels = list()
    tags = set()

    input_data = codecs.open('./wordtagsplit.txt', 'r', 'utf-8')
    for line in input_data.readlines():
        line = line.split()
        linedata = []
        linelabel = []
        numNotO = 0
        for word in line:
            word = word.split('@')
            linedata.append(word[0])
            linelabel.append(word[1])
            tags.add(word[1])
            if word[1] != 'O':
                numNotO += 1
        if numNotO != 0:
            datas.append(linedata)
            labels.append(linelabel)

    input_data.close()
    print(len(datas), tags)
    print(len(labels))
    #    from compiler.ast import flatten
    all_words = flatten(datas)
    sr_allwords = pd.Series(all_words)
    sr_allwords = sr_allwords.value_counts()
    set_words = sr_allwords.index
    set_ids = list(range(1, len(set_words) + 1))

    tags = [i for i in tags]
    tag_ids = list(range(len(tags)))
    word2id = pd.Series(set_ids, index=set_words)
    id2word = pd.Series(set_words, index=set_ids)
    tag2id = pd.Series(tag_ids, index=tags)
    id2tag = pd.Series(tags, index=tag_ids)

    word2id["unknow"] = len(word2id) + 1
    print(word2id)
    max_len = 50

    def X_padding(words):
        ids = list(word2id[words])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0] * (max_len - len(ids)))
        return ids

    def y_padding(tags):
        ids = list(tag2id[tags])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0] * (max_len - len(ids)))
        return ids

    df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=list(range(len(datas))))
    df_data['x'] = df_data['words'].apply(X_padding)
    df_data['y'] = df_data['tags'].apply(y_padding)
    x = np.asarray(list(df_data['x'].values))
    y = np.asarray(list(df_data['y'].values))

    import pickle
    with open('../yidu-s4k.pkl', 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(x, outp)
        pickle.dump(y, outp)
    print('** Finished saving the data.')
    print(tag2id)


def tagsplit():
    with open('./wordtag.txt', 'rb') as inp:
        texts = inp.read().decode('utf-8')
    sentences = re.split('[，。！？、‘’“”（）]@[O]', texts)
    output_data = codecs.open('./wordtagsplit.txt', 'w', 'utf-8')
    for sentence in sentences:
        if sentence not in set(list([' ', ''])):
            output_data.write(sentence.strip() + '\n')
    output_data.close()


def gen_dataset():
    with open('train_data.txt', 'r', encoding='utf-8') as f_train, \
            open('test_data.txt', 'r', encoding='utf-8') as f_test, \
            open('wordtag.txt', 'w', encoding='utf-8') as f_out:
        text = (f_train.read() + f_test.read()).split('\n')
        _i = 0
        assert len(text) >= 50, "dataset too small"
        for i in range(50, len(text), 50):
            line = ""
            while text[i].strip() == '\n' or not text[i].strip() or text[i][-1] != 'O':
                i -= 1
            for item in text[_i: i + 1]:
                if item.strip() != '':
                    line += item.replace(' ', '@') + ' '
            f_out.write(line + '\n')
            _i = i + 1


def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


gen_dataset()
tagsplit()
data2pkl()
