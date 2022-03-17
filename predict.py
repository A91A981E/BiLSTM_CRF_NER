# coding=utf-8
import argparse
import pickle as pkl

import numpy as np
import pandas as pd
import torch

from utils.str2bool import str2bool


def main(args):
    if torch.cuda.is_available() and args.gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    input_data = []
    model = torch.load(args.model)
    model.to(device)

    with open(args.file, 'r', encoding='UTF-8') as fin:
        data = fin.readline()
        while data != '':
            input_data.append(list(data.replace('\n', '')))
            data = fin.readline()

    with open(f'data/{args.dataset}.pkl', 'rb') as inp:
        word2id = pkl.load(inp)
        id2word = pkl.load(inp)
        tag2id = pkl.load(inp)
        id2tag = pkl.load(inp)

    max_len = 50

    def X_padding(words):
        ids = list(word2id[words])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0] * (max_len - len(ids)))
        return ids

    data_df = pd.DataFrame({'data': input_data}, index=list(range(len(input_data))))
    data_df['x'] = data_df['data'].apply(X_padding)
    Xs = np.asarray(list(data_df['x'].values))
    for idx, item in enumerate(Xs):
        X = torch.tensor(item, dtype=torch.long, device=device)
        score, tag = model(X)
        res = []
        for i in tag:
            res.append(id2tag[i])
        pd.DataFrame({'word': input_data[idx], 'tag': res[:len(input_data[idx])]}) \
            .to_csv(f'./res/sentence_{idx}.csv', encoding='utf-8', index=False)
        print(score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='model/model152148yidu-s4k5.pkl')
    parser.add_argument('-f', '--file', type=str, default='predict_input_data.txt')
    parser.add_argument('-d', '--dataset', type=str, default='yidu-s4k')
    parser.add_argument('--gpu', type=str2bool, default=True)
    main(parser.parse_args())
