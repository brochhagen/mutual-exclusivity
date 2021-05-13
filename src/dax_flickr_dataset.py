import torch
import numpy as np
import random

from visual_preprocessing import get_image_representation
from symbolic_datasets import add_novel_vocabulary_items
from datasets import AbstractWordDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


class DaxDataPoint(object):
    """ Data point = scene = tuple (ref_names, visual_vectors) for N objects in the scene """

    def __init__(self, ref_names, vs):
        self.ref_names = ref_names
        self.nwords = len(self.ref_names)
        self.vs = vs


class DaxDataset(AbstractWordDataset):

    def __init__(self, dax_names, dax_vs, datapoints, word2idx, idx2word, char2idx, idx2char, level):

        self.word2idx, self.idx2word, self.char2idx, self.idx2char = word2idx, idx2word, char2idx, idx2char
        self.level = level

        self.datapoints = datapoints
        self.dax_vs = dax_vs
        self.novel_words = dax_names
        self.ndaxes = len(dax_names)

    def __getitem__(self, i):
        example = self.datapoints[i]

        if self.level == "word":
            text_input = self.get_text_tensor(example.ref_names)
        elif self.level == "char":
            text_input = self.get_char_tensor(example.ref_names)

        visual_input = example.vs.to(device)

        return example, visual_input, text_input     

    def __len__(self):
        return len(self.datapoints)

    def idx2text(self, indices):
        return [self.idx2word[idx] for idx in indices]

    def object_word_match(self, datapoint, visual_input, object_idx, text_input, word_idx):
        # relies on the fact that in flickr dataset we always have aligned data [objects] - [words]
        return object_idx == word_idx

    def print_dax_mappings(self, model):
        production_probs = model.get_production_probs(torch.stack(self.dax_vs)).squeeze(1).cpu().numpy()
        indices = np.argmax(production_probs, axis=1)
        max_probs = np.exp(np.max(production_probs, axis=1))
        for d, i, prob in zip(self.novel_words, indices, max_probs):
            print(d, self.idx2word[i], prob)

    def is_novel_word(self, word):
        return word in self.novel_words


def sample_objects(dataset, n):
    r = np.arange(0, len(dataset.datapoints))
    idx = np.random.choice(r, 10 * n)  
    vs = []
    ref_names = []
    for i in idx:
        example, vs_, ts_ = dataset.__getitem__(i)
        j = np.random.randint(len(vs_))

        if not example.is_novel(j):
            vs.append(vs_[j])
            ref_names.append(example.ref_names[j])

        if len(ref_names) == n:
            break

    return vs, ref_names


daxes = ["dax" + 'abcdefgijklmno'[i] for i in range(10)]

def create_novel_objects_flickr_dataset(daxpath, train_dataset, novel_words,
                                        level, shuffle=True, nperdax=50):
    datapoints = []
    dax_vs = []

    for i in range(len(novel_words)):
        novel_word = novel_words[i]
        nv = get_image_representation(daxpath + "/" + novel_word + ".jpg",
                                      train_dataset.visual_model, train_dataset.vgg_layer)
        dax_vs.append(nv)

        for j in range(nperdax):

            nobjs = 2
            vs, ref_names = sample_objects(train_dataset, nobjs)

            ref_names.append(novel_word)
            vs.append(nv)

            vs = torch.stack(vs)

            dp = DaxDataPoint(ref_names, vs)
            datapoints.append(dp)

    if shuffle:
        random.shuffle(datapoints)

    word2idx, idx2word = add_novel_vocabulary_items(train_dataset.word2idx, train_dataset.idx2word, daxes=novel_words)

    return DaxDataset(novel_words, dax_vs, datapoints, word2idx, idx2word,
                      train_dataset.char2idx, train_dataset.idx2char, level)
