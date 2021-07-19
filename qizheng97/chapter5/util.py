import re
import collections

def read():
    with open('timemachine.txt', 'r',encoding='UTF-8') as f:
        lines = [re.sub('[^a-z]+', ' ', line.strip().lower()) for line in f]
    return lines

def tokenize(sentences):
     return [sentence.split(' ') for sentence in sentences]


class Vocab(object):
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        counter = count_corpus(tokens)

        self.token_freqs = sorted(counter.items(),key=lambda item:item[1], reverse=True)
        print(self.token_freqs)
        self.idx_to_token = []
        if use_special_tokens:
            # padding, begin of sentence, end of sentence, unknown
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            self.idx_to_token += ['', '', '', '']
        else:
            self.unk = 0
            self.idx_to_token += ['']
        self.idx_to_token += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in self.idx_to_token]
        self.token_to_idx = dict()
        for idx, token in enumerate(self.idx_to_token):
            self.token_to_idx[token] = idx

    def __len__(self):   #返回词典大小
        return len(self.idx_to_token)

    def __getitem__(self, tokens):    #定义了Vocab类的索引，参数token可以是1.列表或元组，2.字符串。   从词到索引的映射
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)  #找到token则返回token，否则返回self.unk
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):   #从索引到词的映射
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def count_corpus(sentences):  #统计词频的函数，sentences是前面的tokens，是一个二维列表
    tokens = [tk for st in sentences for tk in st]  #将sentences展平得到一个一维列表
    return collections.Counter(tokens)  # 返回一个字典，记录每个词的出现次数


def load_data():
    with open('timemachine.txt', 'r',encoding='UTF-8') as f:
        corpus_chars = f.read()
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


