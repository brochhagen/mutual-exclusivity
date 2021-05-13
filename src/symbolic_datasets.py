
import torch
import numpy as np

from vocabulary import get_vocab
from datasets import AbstractWordDataset

import os
import random

import pandas as pd
os.environ['PYTHONHASHSEED']=str(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

class SymbolicDataPoint(object):

    def __init__(self, words: str, objects: str):
        self.ref_names = words.split(" ")
        self.objects = objects.split(" ")
        self.nwords = len(self.ref_names)


def inverse_freqs(vocab: list, all_items: list):
    freqs = torch.zeros(len(vocab))
    for i in range(len(vocab)):
        count = all_items.count(vocab[i])
        if count > 0:
            freqs[i] = 1 / count
    return freqs


class SymbolicWordObjectDataset(AbstractWordDataset):

    def __init__(self, words, objects, daxes,
                       word2idx: dict, idx2word: list,
                       obj2idx: dict, idx2obj: list,
                       char2idx: dict, idx2char: list,
                       level: str, shuffle: bool):
        super(SymbolicWordObjectDataset, self).__init__(word2idx, idx2word, char2idx, idx2char, level)

        self.words = words   # a list of words per scene (list of strings, where words are separated by space)
        self.objects = objects  # a list of objects per scene
        self.novel_words = daxes    # a list of novel words

        self.word2idx, self.idx2word = word2idx, idx2word
        self.obj2idx, self.idx2obj = obj2idx, idx2obj
        self.char2idx, self.idx2char = char2idx, idx2char

        self.datapoints = []
        self.level = level
        self.total_words = 0

        for ws, os in zip(self.words, self.objects):
            example = SymbolicDataPoint(ws, os)
            self.datapoints.append(example)
            self.total_words += example.nwords

        # to balance out prediction losses according to word frequency
        # frequent words should have smaller loss than infrequent words
        self.word_freqs = inverse_freqs(self.idx2word, " ".join(self.words).split(" "))
        self.obj_freqs = inverse_freqs(self.idx2obj, " ".join(self.objects).split(" "))

        if shuffle:
            random.shuffle(self.datapoints)

    def __getitem__(self, i):
        example = self.datapoints[i]
        visual_input = self.get_visual_tensor(example.objects)

        if self.level == "word":
            text_input = self.get_text_tensor(example.ref_names)
        elif self.level == "char":
            text_input = self.get_char_tensor(example.ref_names)

        return example, visual_input, text_input

    def get_text_tensor(self, ref_names):
        return torch.LongTensor([[self.word2idx[name]] for name in ref_names]).to(device)

    def get_char_tensor(self, ref_names, output_new_order=False):
        ws = [torch.LongTensor([self.char2idx[c] for c in name + "#"]) for name in ref_names]

        idx_sorted, ws_sorted = zip(*sorted(enumerate(ws), key=lambda x: len(x[1]), reverse=True))
        ws = torch.nn.utils.rnn.pad_sequence(ws_sorted, batch_first=True)
        if output_new_order:
            return ws.to(device), idx_sorted
        else:
            return ws.to(device)

    def get_visual_tensor(self, objects):
        return torch.LongTensor([self.obj2idx[obj] for obj in objects]).view(-1, 1).to(device)

    def __len__(self):
        return len(self.datapoints)

    def idx2text(self, indices):
        return [self.idx2word[idx] for idx in indices]

    def object_word_match(self, datapoint, visual_input, object_idx, text_input, word_idx):
        # works both for word embeddings and char-level encodings
        return self.idx2obj[visual_input[object_idx]] == datapoint.ref_names[word_idx]

    def is_novel_word(self, word):
        return word in self.novel_words


def add_novel_vocabulary_items(word2idx, idx2word, daxes):
    for dax in daxes:
        word2idx[dax] = len(idx2word)
        idx2word.append(dax)
    return word2idx, idx2word


def load_symbolic_dataset_and_vocabularies(datapath, level, shuffle, daxes, char2idx=None, idx2char=None):
    """ to run experiments with different number of daxes without rewritting vocab every time,
        we store only core vocab and add dax novel words afterwards
        crucially, the dax 'generation' should be always the same, to have training and eval vocab aligned """
    rows = []
    with open(datapath + "/symbolic_dataset.txt", 'rt') as f:
        for line in f:
            if len(line.strip().split("\t")) != 2:
                continue
            ws, os = line.strip().split("\t")
            ws = [w for w in ws.strip().split(" ") if w != ""]
            rows.append((" ".join(ws), os))

    words, objects = list(zip(*rows))
    vocab_words = " ".join(words).split(" ")
    vocab_objects = " ".join(objects).split(" ")

    word2idx, idx2word = get_vocab(datapath + "/symbolic_vocab_words.txt", vocab_words)
    obj2idx, idx2obj = get_vocab(datapath + "/symbolic_vocab_objects.txt", vocab_objects)

    # adding novel dax items to the word and object vocabulary
    # (words and object daxes are the same strings)
    word2idx, idx2word = add_novel_vocabulary_items(word2idx, idx2word, daxes=daxes)
    obj2idx, idx2obj = add_novel_vocabulary_items(obj2idx, idx2obj, daxes=daxes)

    if not char2idx:
        char2idx, idx2char = get_vocab(datapath + "/symbolic_vocab_chars.txt", " ".join(idx2word) + "#")

    return SymbolicWordObjectDataset(words, objects, [],
                                     word2idx, idx2word,
                                     obj2idx, idx2obj,
                                     char2idx, idx2char,
                                     level=level, shuffle=shuffle)


def create_novel_words_symbolic_dataset(train_dataset: SymbolicWordObjectDataset, daxes: list,
                                        level: str, shuffle=True, nperdax=10):

    observed_objs = [o for o in train_dataset.idx2obj[1:] if not o in daxes]

    words = []
    objects = []
    for dax in daxes:
        for j in range(nperdax):
            nobjs = 2
            objs = list(np.random.choice(observed_objs, nobjs, replace=False))
            words.append(dax)
            objects.append(" ".join([dax] + objs))     # assumes word and obj daxes are the same strings

    return SymbolicWordObjectDataset(words, objects, daxes,
                                     train_dataset.word2idx, train_dataset.idx2word,
                                     train_dataset.obj2idx, train_dataset.idx2obj,
                                     train_dataset.char2idx, train_dataset.idx2char,
                                     level=level, shuffle=shuffle)

def create_novel_words_symbolic_sim_biased_dataset(train_dataset: SymbolicWordObjectDataset, daxes: list,
                                        level: str, shuffle=True, nperdax=10):

    observed_objs = [o for o in train_dataset.idx2obj[1:] if not o in daxes]

    words = []
    objects = []

    df = pd.read_csv('../data/daxes_clean.csv')
    for dax in daxes:
        confusable_obj = df.loc[df['dax'] == dax]['known'].iloc[0]
        for j in range(nperdax):
            nobjs = 1
            objs = list(np.random.choice(observed_objs, nobjs, replace=False))
            objs.append(confusable_obj)
            words.append(dax)
            objects.append(" ".join([dax] + objs))     # assumes word and obj daxes are the same strings
    return SymbolicWordObjectDataset(words, objects, daxes,
                                     train_dataset.word2idx, train_dataset.idx2word,
                                     train_dataset.obj2idx, train_dataset.idx2obj,
                                     train_dataset.char2idx, train_dataset.idx2char,
                                     level=level, shuffle=shuffle)

