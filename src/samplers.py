
import numpy as np
import torch
import random
from collections import defaultdict

from dax_flickr_dataset import DaxDataset
from symbolic_datasets import SymbolicWordObjectDataset


class AbstractSamplerUniformWords(object):
    """ samples words *uniformly* from the vocabulary """

    def __init__(self, dataset, include_novel: bool, level: str, ndaxes=5):
        self.dataset = dataset
        self.include_novel = include_novel
        self.ndaxes = ndaxes  # a bit of a workaround to remove daxes which are always the last n words in vocab
        self.level = level
        # get and store all words encoded as char indices
        if level == "char":
            self.char_encodings = self.dataset.get_char_tensor(self.dataset.idx2word)

    def random_objects(self, n):
        pass

    def random_words(self, n):
        if self.include_novel:
            r = np.arange(1, len(self.dataset.word2idx))
        else:
            r = np.arange(1, len(self.dataset.word2idx) - self.ndaxes)

        indices = np.random.choice(r, n)
        if self.level == "word":
            return torch.tensor(indices)
        else:  # char level
            return self.char_encodings[indices]


class FlickrSampler(object):

    def __init__(self, dataset, include_novel):
        self.dataset = dataset
        self.include_novel = include_novel

    def random_objects(self, n):
        r = np.arange(0, len(self.dataset.datapoints))
        idx = np.random.choice(r, 4*n)   
        os = []
        for i in idx:
            example, objs, _ = self.dataset[i]
            j = np.random.randint(len(objs))

            if example.is_novel(j):
                if self.include_novel:
                    os.append(objs[j])
            else:
                os.append(objs[j])

            if len(os) == n:
                break
        return torch.stack(os)

    def random_words(self, n):
        r = np.arange(0, len(self.dataset.datapoints))
        idx = np.random.choice(r, 4*n)   
        ws = []
        for i in idx:
            example, _, words = self.dataset[i]
            j = np.random.randint(len(words))

            if example.is_novel(j):
                if self.include_novel:
                    ws.append(words[j])
            else:
                ws.append(words[j])

            if len(ws) == n:
                break
        return torch.nn.utils.rnn.pad_sequence(sorted(ws, key=len, reverse=True), batch_first=True)


class FlickrSamplerUniformVocabulary(AbstractSamplerUniformWords):

    def __init__(self, dataset, include_novel, level: str, novel_dataset: DaxDataset = None):
        super(FlickrSamplerUniformVocabulary, self).__init__(dataset, include_novel, level)
        self.novel_dataset = novel_dataset
        if novel_dataset:
            self.ndaxes = novel_dataset.ndaxes
        else:
            self.ndaxes = len(dataset.novel_words)

        self.w2obj_indices = None

    def create_index(self):
        self.w2obj_indices = defaultdict(list)
        # training dataset
        for i, (example, visual, text) in enumerate(self.dataset):
            for j, w in enumerate(example.ref_names):
                self.w2obj_indices[w].append((i, j))
        # dax words
        if self.novel_dataset:
            for w, vs in zip(self.novel_dataset.novel_words, self.novel_dataset.dax_vs):
                self.w2obj_indices[w].append(vs)

    def random_objects(self, n):
        if self.w2obj_indices is None:
            self.create_index()

        # first sample words
        if self.include_novel:
            r = np.arange(1, len(self.dataset.word2idx))
        else:
            r = np.arange(1, len(self.dataset.word2idx) - self.ndaxes)

        words = [self.dataset.idx2word[i] for i in np.random.choice(r, n)]

        objs = []
        for w in words:
            # sample one visual representation per word
            obj_indices = random.choice(self.w2obj_indices[w])
            # it's a bit ugly but for the moment better this than duplicating code
            if isinstance(obj_indices, tuple):
                obj_representation = self.dataset[obj_indices[0]][1][obj_indices[1],:]
            else:
                obj_representation = obj_indices
            objs.append(obj_representation)

        return torch.stack(objs)



class SymbolicSampler(AbstractSamplerUniformWords):
    """ samples objects and words *uniformly* from the vocabulary """

    def __init__(self, dataset: SymbolicWordObjectDataset, include_novel: bool, level: str, ndaxes=5):
        super().__init__(dataset, include_novel, level, ndaxes)
        self.dataset = dataset
        self.include_novel = include_novel
        self.ndaxes = ndaxes  # a bit of a workaround to remove daxes which are always the last n words in vocab
        self.level = level
        # get and store all words encoded as char indices
        if level == "char":
            self.char_encodings = self.dataset.get_char_tensor(self.dataset.idx2word)

    def random_objects(self, n):
        if self.include_novel:
            # since our vocabs now contain padding symbol as 0
            r = np.arange(1, len(self.dataset.obj2idx))
        else:
            r = np.arange(1, len(self.dataset.obj2idx) - self.ndaxes)
        return torch.tensor(np.random.choice(r, n))

