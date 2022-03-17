import torch
import torch.nn as nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.manual_seed(1)
START_TAG = "<START>"
STOP_TAG = "<STOP>"


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, num_layers=1):
        """
        使用BiLSTM+条件随机场进行中文的命名实体识别
        :param vocab_size: id2word映射表的长度 + 1
        :param tag_to_ix: tag2id映射表({"B_location" -> 1 ...})
        :param embedding_dim: LSTM输入数据的特征维数，通常为嵌入向量的大小
        :param hidden_dim: LSTM中隐藏层的维度
        :param num_layers: LSTM中循环神经网络的隐藏层数，default=1
        """
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tag_set_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=num_layers, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tag_set_size)
        self.num_layers = num_layers
        self.transitions = nn.Parameter(torch.randn(self.tag_set_size, self.tag_set_size))

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        """
        初始化BiLSTM隐藏层参数
        :return:
        """
        return (torch.randn(2, 1, self.hidden_dim // 2, device=device),
                torch.randn(2, 1, self.hidden_dim // 2, device=device))

    def _forward_alg(self, feats):
        """
        前向传播算法，计算由模型得出的序列的正向概率，经过线性映射作为模型标注得分
        :param feats: BiLSTM输出特征
        :return: 前向概率alpha
        """
        init_alphas = torch.full((1, self.tag_set_size), -10000., device=device)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        forward_var = init_alphas

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tag_set_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tag_set_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        """
        正向传播，得到BiLSTM的输出特征
        :param sentence: 中文语句
        :return: 返回BiLSTM特征
        """
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        """
        对tag进行打分，这个打分与上面的前向传播算法类似，是最佳得分
        :param feats: BiLSTM输出特征
        :param tags: 标签
        :return: 最佳得分
        """
        score = torch.zeros(1, device=device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long, device=device), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        """
        维特比算法解码，参照李航《统计学习方法》（清华大学出版社）
        :param feats: BiLSTM输出特征
        :return: 标注序列
        """
        backpointers = []

        init_vvars = torch.full((1, self.tag_set_size), -10000., device=device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbi_vars_t = []

            for next_tag in range(self.tag_set_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbi_vars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbi_vars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)

        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
