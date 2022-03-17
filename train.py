# coding=utf-8
import argparse
import logging
import os
import pickle as pkl
import time

import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset, DataLoader

from utils.BiLSTM_CRF import BiLSTM_CRF
from utils.resultCal import calculate
from utils.str2bool import str2bool

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Data(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return X, y

    def __len__(self):
        return len(self.X)


def save(model, path):
    logging.debug(f"Saving model to {path}...")
    torch.save(model, path)
    logging.debug(f"Model saved.")


def train(args, device, model, train_data, test_data, id2word, id2tag):
    lr_ad = len(train_data) // 4
    for epoch in range(args.epoch):
        lr = args.lr
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)
        losses = []
        for idx, (data, target) in enumerate(train_data):
            data, target = data.to(device)[0], target.to(device)[0]
            model.zero_grad()
            loss = model.neg_log_likelihood(data, target)
            losses.append(float(loss))
            loss.backward()
            optimizer.step()
            if idx % lr_ad == 0 and idx != 0:
                lr /= 5
                optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)
                logging.debug(f"Change learning rate to {lr}")
            if (idx + 1) % 50 == 0:
                logging.debug(f"Epoch {epoch + 1}/{args.epoch}: item {idx + 1}/{len(train_data)}, loss {float(loss)}")
        if args.save:
            time_now = str(time.localtime().tm_mday) + \
                       str(time.localtime().tm_hour) + str(time.localtime().tm_min)
            path = "./model/model" + time_now + args.dataset + str(epoch + 1) + ".pkl"
            save(model, path)
            pd.DataFrame({"loss": losses}).to_csv(f"res/{args.dataset}_{time_now}_loss_{epoch}.csv", encoding='utf-8')
        logging.debug(f"Epoch {epoch} done, evaluating...")
        entityres = []
        entityall = []
        for idx, (data, target) in enumerate(test_data):
            data, target = data.to(device)[0], target.to(device)[0]
            score, predict = model(data)
            entityres = calculate(data, predict, id2word, id2tag, entityres)
            entityall = calculate(data, target, id2word, id2tag, entityall)
            if (idx + 1) % 50 == 0:
                logging.debug(f"Epoch {epoch + 1}/{args.epoch}: item {idx + 1}/{len(test_data)}")
        inter = [i for i in entityres if i in entityall]
        if len(inter) != 0:
            precise = float(len(inter)) / len(entityres)
            recall = float(len(inter)) / len(entityall)
            logging.debug(f"test:")
            logging.debug(f"P (TP / (TP + FP)): {precise}")
            logging.debug(f"R (TP / (TP + FN)): {recall}")
            logging.debug(f"F1: {(2 * precise * recall) / (precise + recall)}")
        else:
            logging.debug(f"P (TP / (TP + FP)): 0")


def form_data(xtrain_data, xtest_data, ytrain_data, ytest_data):
    train_data = Data(xtrain_data, ytrain_data)
    test_data = Data(xtest_data, ytest_data)

    return DataLoader(train_data, shuffle=True, drop_last=False), \
        DataLoader(test_data, shuffle=True, drop_last=False)

def main(args):
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)-s - "
                                                    "%(filename)-8s : %(lineno)s line - %(message)s ",
                        datefmt="%Y-%m-%d %H:%M:%S")
    if torch.cuda.is_available() and args.gpu:
        logging.debug(f"Using device: {torch.cuda.get_device_name()}")
    else:
        logging.debug("Using device: CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(rf'data/{args.dataset}.pkl', 'rb') as inp:
        word2id = pkl.load(inp)
        id2word = pkl.load(inp)
        tag2id = pkl.load(inp)
        id2tag = pkl.load(inp)
        X = pkl.load(inp)
        y = pkl.load(inp)

    logging.debug(f"Using selected dataset {args.dataset}, dataset length {len(X)}")

    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    EMBEDDING_DIM = 100
    HIDDEN_SIZE = 200
    NUM_LAYERS = 1
    EPOCHS = args.epoch

    tag2id[START_TAG] = len(tag2id)
    tag2id[STOP_TAG] = len(tag2id)

    if not args.model:
        model = BiLSTM_CRF(len(id2word) + 1, tag2id, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS)
    else:
        model = torch.load(args.model)
        logging.debug(f"Loaded model {args.model}")
    model.to(device)

    logging.debug(f"Parameters: dataset {args.dataset}, learning rate: {args.lr}, epoch {args.epoch}")
    logging.debug(f"            train-test-split {args.train_test_split}, {args.k} fold cross validation")
    logging.debug(f"Start training...")
    data = Data(X, y)
    if not args.k:
        train_data, test_data = \
            train_test_split(data, train_size=args.train_test_split, random_state=43, shuffle=True)
        train_data, test_data = DataLoader(train_data, shuffle=True, drop_last=False), \
                                                        DataLoader(test_data, shuffle=True, drop_last=False)
        train(args, device, model, train_data, test_data, id2word, id2tag)
    else:
        kf = KFold(n_splits=args.k)
        for train_data, test_data in kf.split(data):
            train_data, test_data = DataLoader(train_data, shuffle=True, drop_last=False), \
                                                        DataLoader(test_data, shuffle=True, drop_last=False)
            train(args, device, model, train_data, test_data, id2word, id2tag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BiLSTM_CRF trainer")
    parser.add_argument('-d', '--dataset', type=str, default="yidu-s4k",
                        choices=["Bosondata", "dataMSRA", "renmindata", "yidu-s4k"])
    parser.add_argument('-t', '--train-test-split', type=float, default=0.8)
    parser.add_argument('-e', '--epoch', type=int, default=5)
    parser.add_argument('-s', '--save', type=str2bool, default=True)
    parser.add_argument('-m', '--model', type=str, default=None,
                        help="Use significant model if not null. Train a new one otherwise.\n"
                             "Such as model/model271749dataMSRA2.pkl")
    parser.add_argument('-l', '--lr', type=float, default=1e-2)
    parser.add_argument('-g', '--gpu', type=str2bool, default=True)
    parser.add_argument('-k', type=int, default=None, help="K Fold cross validation if not none.")
    main(parser.parse_args())
