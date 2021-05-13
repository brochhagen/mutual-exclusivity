
import torch.nn as nn
import torch
import numpy as np


class CutoffModule(nn.Module):

    def __init__(self, size=200):
        super(CutoffModule, self).__init__()
        self.size = size

    def forward(self, input):
        return input[:self.size, :]


class SimilarityModelA(nn.Module):

    def __init__(self, vocab_size, hidden_size, visual_features_extractor, init_range=None):
        super(SimilarityModelA, self).__init__()
        #print("Angeliki's model")
        self.hidden_size = hidden_size   # should be 200
        self.visual_features_extractor = visual_features_extractor   # should be embeddings for symbolic dataset
        self.nobjects = self.visual_features_extractor.nobjects
        self.nwords = vocab_size
        self.text_encoder = nn.Embedding(vocab_size, hidden_size)
        if init_range:
            self.text_encoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, visual_input, text_input):
        v_ = self.visual_features_extractor(visual_input)
        v_ = v_.squeeze(1)
        t_ = self.text_encoder(text_input)
        t_ = t_.squeeze(1)
        v_ = v_ / torch.norm(v_, dim=1, keepdim=True)
        t_ = t_ / torch.norm(t_, dim=1, keepdim=True)

        sims = torch.matmul(v_, t_.t())
        return sims

    def get_production_probs(self, visual_input):
        similarities = self.forward(visual_input, torch.arange(self.nwords).long())
        #weights = weights / torch.norm(weights)
        # weight size bsz x |V|
        return torch.nn.LogSoftmax(dim=1)(similarities)  # over words

    def get_comprehension_probs(self, vs, t):
        similarities = self.forward(vs, t)
        return torch.nn.LogSoftmax(dim=0)(similarities)   # over objects

    def cross_situational_loss(self, vs, ts):
        loss = 0
        for v in vs:
            for t in ts:
                loss += self.supervised_loss(v, t)
        return loss / ts.shape[0]

    def supervised_loss(self, v, t):
        m = 0.5
        positive = self.forward(v, t)

        r = list(range(0, self.nobjects))
        r = r[:v.item()] + r[v.item() + 1:]   # works only for symbolic dataset
        
        neg_vs = torch.tensor(np.random.choice(r, 5)).long()
        #loss = 0
        negative = self.forward(neg_vs, t)
        loss = torch.mean(torch.max(torch.zeros(1), m - positive + negative))

        return loss

    def get_embeddings(self):   # only for symbolic encoding of objects
        vis_embs = self.visual_features_extractor(torch.arange(0, self.nobjects).long()).detach().numpy()
        word_embs = self.text_encoder.weight.detach().numpy()
        return vis_embs, word_embs


