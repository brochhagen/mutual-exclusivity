import random
from collections import defaultdict

import numpy as np
import torch

from dax_flickr_dataset import DaxDataset


class MilestoneIterator(object): # no batching for the moment
    """ The idea is that we want milestone to correspond to some meaningful period for language acquisition,
        e.g. a month, computed as number of processed = encountered words """

    def __init__(self, dataset, nepochs, milestone, skip_novel, sampling=None):
        self.dataset = dataset

        self.milestone = milestone    # words per day approx
        self.epoch = 0            # one epoch = one day in processing #milestone words
        self.nepochs = nepochs

        self.nwords_processed = 0
        self.nimages_processed = 0

        self.index = 0

        self.sampling = sampling
        self.skip_novel = skip_novel

        if sampling == "zipfian_images":
            x = np.random.zipf(1.1, len(dataset) * 4)
            self.sampled_idx = x[x < len(dataset)]
            #print(len(x))
            assert len(x) >= len(dataset)   

        if sampling == "zipfian_nrefs":
            x = np.random.zipf(1.4, len(dataset) * 100)
            self.sampled_idx = x[x <= 5]
            assert len(x) >= len(dataset)

    def __iter__(self):
        return self

    def __next__(self):
        if self.epoch >= self.nepochs:
            raise StopIteration

        if self.index >= len(self.dataset):  
            self.index = 0

        if self.sampling == "zipfian_images":
            index = self.sampled_idx[self.index]
        elif self.sampling == "zipfian_nrefs":
            index = self.index
            while len(self.dataset[index][0].ref_expressions) != self.sampled_idx[self.index]:
                index += 1
                if index >= len(self.dataset):
                    index = 0
        else:
            index = self.index

        example, visual_input, text_input = self.dataset[index]

        self.index += 1

        self.nwords_processed += example.nwords
        if self.skip_novel:
            output = []
            for i, (v, t) in enumerate(zip(visual_input, text_input)):  
                if not example.is_novel(i):
                    output.append((v, t))
                    continue

            if len(output) > 0:
                visual_input, text_input = zip(*output)
                visual_input, text_input = torch.stack(visual_input), torch.stack(text_input)
            else:
                return self.__next__()

        self.nimages_processed += 1

        self.epoch_flag = False

        if self.nwords_processed >= self.milestone * (self.epoch + 1):
            self.epoch += 1
            self.epoch_flag = True

        return visual_input, text_input

    def is_milestone(self):
        return self.epoch_flag