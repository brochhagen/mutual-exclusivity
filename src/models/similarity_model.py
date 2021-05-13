
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderModel(nn.Module):

    def __init__(self, hidden_size, visual_encoder, text_encoder,
                 obj_weights=None, word_weights=None):
        super(EncoderModel, self).__init__()
        self.hidden_size = hidden_size
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder

        self.obj_weight = obj_weights
        self.word_weight = word_weights
        if obj_weights is not None:
            self.obj_weight = obj_weights.to(device)

        if word_weights is not None:
            self.word_weight = word_weights.to(device)

    def forward(self, visual_input, text_input):
        """ computes cosine similarity between visual and textual inputs
            visual_input: a list of object, tensor of shape n_objects x visual_dim
            text_input: a list of words, tensor of shape n_words x 1 (since words input are indices) """
        v = self.visual_encoder(visual_input)
        v = v.squeeze(1)    # reduce batch dimension
        t = self.text_encoder(text_input)
        t = t.squeeze(1)    # reduce batch dimension
        t = t / torch.norm(t, dim=1, keepdim=True)
        v = v / torch.norm(v, dim=1, keepdim=True)
        return torch.matmul(v, t.t())    # pairwise cosine similarities

    def cross_situational_loss(self, vs, ts, loss_type=None, sampler=None):
        if loss_type == "maxmargin_words":
            loss = self.maxmargin_loss_over_words(vs, ts, sampler=sampler.random_words)
        elif loss_type == "maxmargin_objects":
            loss = self.maxmargin_loss_over_objects(vs, ts, sampler=sampler.random_objects)
        elif loss_type == "softmax_words":
            loss = self.softmax_loss_over_words(vs, ts)
        elif loss_type == "softmax_objects":
            loss = self.softmax_loss_over_objects(vs, ts)
        elif loss_type == "softmax_joint":
            loss = self.softmax_loss_over_objects(vs, ts) \
                   + self.softmax_loss_over_words(vs, ts)
        elif loss_type == "maxmargin_joint":
            loss = self.maxmargin_loss_over_objects(vs, ts, sampler=sampler.random_objects) + \
                   self.maxmargin_loss_over_words(vs, ts, sampler=sampler.random_words)
        else:
            loss = self.softmax_loss_over_words(vs, ts)
        return loss

    def get_comprehension_probs(self, vs, t):
        """ for a word t and objects vs get probabilities P(o|w) """
        # TODO: this should work for multiple words t -> ts, check
        similarities = self.forward(vs, t)
        return torch.nn.LogSoftmax(dim=0)(similarities)

    def softmax_loss_over_words(self, vs, ts):
        """ computes cross-situational loss = loss for all pairwise mappings between objects (vs) and words/names (ts)
            over words = mappings are v -> t and the loss is NLL loss = -logP(t)
            which induces competition between all words in vocabulary
            vs : n_objects x visual_dim
            ts : n_words x 1 """

        logprobs = self.get_production_probs(vs) # dim 0 is number of objects, dim 1 is vocab size
        # we "copy" logprobs [x, y] as [x, y, x, y] (x, y are rows with probabilities over vocab)
        # the number of times is equal to the number of names/words ts.shape[0]
        nobj = vs.shape[0]
        nnames = ts.shape[0]
        logprobs = logprobs.unsqueeze(0).expand(nnames, nobj, logprobs.shape[1]). \
            contiguous().view(nnames * nobj, logprobs.shape[1])

        # we "copy" text idx [i, j] as [i, i, j, j] nobj number of times so that probs * ids give pairwise nlls
        ts = ts.expand(nnames, nobj).contiguous().view(1, -1)

        # we can choose whether to sum the losses over all pairs or average them
        # this affects the importance of the update across scenes (e.g. (2 obj, 2 words) scene vs (4,5) scene)
        loss = torch.sum(torch.nn.NLLLoss(reduce=False, weight=self.word_weight)(logprobs, ts.view(ts.shape[1])))
        # TODO: I'm not sure this normalization makes more sense than normalization over nnames, or plain averaging
        return loss / nobj   # sum of nnames losses

    def softmax_loss_over_objects(self, vs, ts):
        """ the same as over words but in other direction """
        similarities = self.forward(torch.arange(self.visual_encoder.nobjects).long().to(device), ts)

        logprobs = torch.nn.LogSoftmax(dim=0)(similarities).t()
        #print(logprobs.shape)
        nobj = vs.shape[0]
        nnames = ts.shape[0]
        logprobs = logprobs.unsqueeze(0).expand(nobj, nnames, logprobs.shape[1]). \
            contiguous().view(nnames * nobj, logprobs.shape[1])
        vs = vs.expand(nobj, nnames).contiguous().view(1, -1)
        loss = torch.sum(torch.nn.NLLLoss(reduce=False, weight=self.obj_weight)(logprobs, vs.view(vs.shape[1])))
        return loss / nnames   # sum of nobj losses

    def maxmargin_loss_over_words(self, vs, ts, nsample=5, sampler=None):
        """ the same as softmax_over_words but with maxmargin loss instead of NLL loss """
        m = 1.0    # margin is fixed to 1.0
        loss = 0
        # TODO: this iteration is not efficient
        for t in ts:
            positive = self.forward(vs, t.unsqueeze(0)).view(1, -1)
            #print(positive.shape)
            neg_ts = sampler(nsample * vs.shape[0]).to(device)#.unsqueeze(1)  #TODO why need unsqueeze()?

            # TODO sampler should return sequences, not word index
            negative = self.forward(vs, neg_ts).view(1, -1)
            #print(negative.shape)
            positive = positive.expand(nsample * vs.shape[0], positive.shape[1]).contiguous().view(1,-1)
            #print("Loss", torch.max(torch.zeros(1).to(device), m - positive + negative))

            loss_t = torch.sum(torch.max(torch.zeros(1).to(device), m - positive + negative)) / nsample

            if self.word_weight is not None:
                loss_t = loss_t * self.word_weight[t]
            loss += loss_t

        return loss / vs.shape[0]   # normalized by nobj <=> sum of nnames losses

    def maxmargin_loss_over_objects(self, vs, ts, nsample=5, sampler=None):
        m = 1.0   # margin is fixed to 1.0
        loss = 0
        for v in vs:
            positive = self.forward(v.unsqueeze(0), ts).view(1, -1)

            neg_vs = sampler(nsample * ts.shape[0]).to(device)#.unsqueeze(1)

            negative = self.forward(neg_vs, ts).view(1, -1)
            #print("N", negative)
            positive = positive.expand(nsample * ts.shape[0], positive.shape[1]).contiguous().view(1,-1)
            #print("P", positive)
            #print("Loss", torch.max(torch.zeros(1).to(device), m - positive + negative))


            loss_v = torch.sum(torch.max(torch.zeros(1).to(device), m - positive + negative)) / nsample
            if self.obj_weight is not None:
                loss_v = loss_v * self.obj_weight[v]
            loss += loss_v

        return loss / ts.shape[0]    # normalized by nnames


class SimilarityModel(EncoderModel):
    """ the model itself is just cosine similarity between two encoded representations (visual and text)
        most of the functionality in the class is for computing various loss functions based on these similarities """

    def __init__(self, vocab_size, hidden_size, visual_features_extractor, init_range=None,
                 obj_weights=None, word_weights=None):
        self.hidden_size = hidden_size
        self.nwords = vocab_size

        text_encoder = nn.Embedding(vocab_size, hidden_size)
        if init_range:
            text_encoder.weight.data.uniform_(-init_range, init_range)

        super(SimilarityModel, self).__init__(hidden_size, visual_features_extractor, text_encoder,
                                              obj_weights, word_weights)

    def supervised_loss(self, vs, ts):
        """ supervised loss = only for correct pairs (v,t)
            assumes that order of objects in vs is the same as the order of corresponding names in ts
            => used as a baseline for flickr dataset
            the loss here is NLL over words """
        # TODO: rename, move to EncoderModel?
        logprobs = self.get_production_probs(vs)
        loss = torch.sum(torch.nn.NLLLoss(reduce=False)(logprobs, ts.view(ts.shape[0])))
        return loss    # sum of n pairs (n = nobj = nnames)

    def get_embeddings(self):   # only for symbolic encoding of objects
        vis_embs = self.visual_features_extractor(torch.arange(0, self.visual_features_extractor.nobjects).long().to(device)).detach().cpu().numpy()
        word_embs = self.text_encoder.weight.detach().cpu().numpy()
        return vis_embs, word_embs

    def get_production_probs(self, visual_input):
        """ for each object in visual_input compute the log-probabilities P(w|o) over all words in vocabulary """
        similarities = self.forward(visual_input, torch.arange(self.nwords).long().to(device))
        return torch.nn.LogSoftmax(dim=1)(similarities)   # probs are normalised using softmax


