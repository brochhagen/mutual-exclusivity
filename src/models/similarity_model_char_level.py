
import torch.nn as nn
import torch
from models.char_modules import CharEncoder, CharDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimilarityModelCharEncoder(nn.Module):
    """ old model to be removed """

    def __init__(self, vocab_size, hidden_size, visual_features_extractor, init_range=None,
                 obj_weights=None, word_weights=None):
        super(SimilarityModelCharEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.visual_features_extractor = visual_features_extractor
        #self.visual_encoder = nn.Linear(visual_dim, hidden_size)

        self.embeddings = nn.Embedding(vocab_size, 50)
        self.lstm1 = nn.LSTM(input_size=50, hidden_size=hidden_size, num_layers=1)
        self.text_encoder = CharEncoder(self.embeddings, self.lstm1, hidden_size=hidden_size)

        self.linear_after_encoder = nn.Linear(hidden_size, hidden_size)

        self.hidden_size = hidden_size
        self.visual_features_extractor = visual_features_extractor

        self.obj_weight = obj_weights
        self.word_weight = word_weights
        if obj_weights is not None:
            self.obj_weight = obj_weights.to(device)

        if word_weights is not None:
            self.word_weight = word_weights.to(device)

    def forward(self, visual_input, text_input):
        v = self.visual_features_extractor(visual_input)
        v = v.squeeze(1)
        t = self.text_encoder(text_input)
        t = t.squeeze(1)
        t = t / torch.norm(t, dim=1, keepdim=True)
        v = v / torch.norm(v, dim=1, keepdim=True)
        return torch.matmul(v, t.t())    # pairwise cosine similarities

    def cross_situational_loss(self, vs, ts, loss_type=None, sampler=None):
        if loss_type == "maxmargin_words":
            loss = self.maxmargin_loss_over_words(vs, ts, sampler=sampler.random_words)
        elif loss_type == "maxmargin_objects":
            loss = self.maxmargin_loss_over_objects(vs, ts, sampler=sampler.random_objects)
        elif loss_type == "softmax_words":
            raise Exception("Wrong loss for SimilarityModelCharEncoder model")
        elif loss_type == "softmax_objects":
            loss = self.softmax_loss_over_objects(vs, ts)
        elif loss_type == "softmax_joint":
            raise Exception("Wrong loss for SimilarityModelCharEncoder model")
        elif loss_type == "maxmargin_joint":
            loss = self.maxmargin_loss_over_objects(vs, ts, sampler=sampler.random_objects) + \
                   self.maxmargin_loss_over_words(vs, ts, sampler=sampler.random_words)
        else:
            loss = self.maxmargin_loss_over_words(vs, ts, sampler=sampler.random_words)
        return loss

    def get_production_probs(self, visual_input, text_input):
        similarities = self.forward(visual_input, torch.arange(self.nwords).long().to(device))
        #similarities = similarities / torch.norm(similarities)
        # weight size bsz x |V|
        #return torch.log(similarities / torch.sum(similarities, dim=1, keepdim=True))
        return torch.nn.LogSoftmax(dim=1)(similarities)

    def get_comprehension_probs(self, vs, t):
        similarities = self.forward(vs, t)
        return torch.nn.LogSoftmax(dim=0)(similarities)

    def softmax_loss_over_objects(self, vs, ts):
        similarities = self.forward(torch.arange(self.visual_features_extractor.nobjects).long().to(device), ts)

        logprobs = torch.nn.LogSoftmax(dim=0)(similarities).t()
        #print(logprobs.shape)
        nobj = vs.shape[0]
        nnames = ts.shape[0]
        logprobs = logprobs.unsqueeze(0).expand(nobj, nnames, logprobs.shape[1]). \
            contiguous().view(nnames * nobj, logprobs.shape[1])
        vs = vs.expand(nobj, nnames).contiguous().view(1, -1)
        loss = torch.sum(torch.nn.NLLLoss(reduce=False, weight=self.obj_weight)(logprobs, vs.view(vs.shape[1])))
        return loss / nnames

    def maxmargin_loss_over_objects(self, vs, ts, nsample=5, sampler=None):
        m = 1.0
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

        return loss / ts.shape[0]

    def maxmargin_loss_over_words(self, vs, ts, nsample=5, sampler=None):
        m = 1.0
        loss = 0
        #print(ts.shape)
        for i in range(ts.shape[0]):
            t = ts[i].unsqueeze(0)  # i-th row - word
            positive = self.forward(vs, t).view(1, -1)
            #print(positive.shape)
            neg_ts = sampler(nsample * vs.shape[0]).to(device)#.unsqueeze(1)  #TODO why need unsqueeze()?

            negative = self.forward(vs, neg_ts).view(1, -1)
            #print(negative.shape)
            positive = positive.expand(nsample * vs.shape[0], positive.shape[1]).contiguous().view(1,-1)
            #print("Loss", torch.max(torch.zeros(1).to(device), m - positive + negative))

            loss_t = torch.sum(torch.max(torch.zeros(1).to(device), m - positive + negative)) / nsample

            if self.word_weight is not None:
                loss_t = loss_t * self.word_weight[t]
            loss += loss_t

        return loss / vs.shape[0]

    def supervised_loss(self, vs, ts):
        logprobs = self.get_production_probs(vs)
        loss = torch.sum(torch.nn.NLLLoss(reduce=False)(logprobs, ts.view(ts.shape[0])))  # ts.shape ?
        return loss

    def get_embeddings(self):   # only for symbolic encoding of objects
        vis_embs = self.visual_features_extractor(torch.arange(0, self.visual_features_extractor.nobjects).long().to(device)).detach().cpu().numpy()
        word_embs = self.text_encoder.weight.detach().cpu().numpy()
        return vis_embs, word_embs



class SimilarityModelCharDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size, visual_features_extractor, init_range=None,
                 obj_weights=None, word_weights=None):
        super(SimilarityModelCharDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.visual_features_extractor = visual_features_extractor
        #self.visual_encoder = nn.Linear(visual_dim, hidden_size)

        self.embeddings = nn.Embedding(vocab_size, 50)

        #self.linear_after_encoder = nn.Linear(hidden_size, hidden_size)

        self.lstm = nn.LSTM(input_size=50, hidden_size=hidden_size, num_layers=1)
        self.text_decoder = CharDecoder(self.embeddings, self.lstm, hidden_size=hidden_size)

        self.hidden_size = hidden_size
        self.visual_features_extractor = visual_features_extractor

        self.obj_weight = obj_weights
        self.word_weight = word_weights
        if obj_weights is not None:
            self.obj_weight = obj_weights.to(device)

        if word_weights is not None:
            self.word_weight = word_weights.to(device)

    def forward(self, vs, ts):
        nobj = vs.shape[0]
        nnames = ts.shape[0]
        #print(vs.shape)
        #print(ts.shape)

        ts = ts.unsqueeze(0).expand(nobj, nnames, ts.shape[1]). \
            contiguous().view(nnames * nobj, ts.shape[1])
        vs = vs.expand(nobj, nnames).contiguous().view(-1, vs.shape[1])

        #print(ts)
        #print(vs)

        v = self.visual_features_extractor(vs)
        v = v.squeeze(1)
        #v = v / torch.norm(v, dim=1, keepdim=True)

        loss = self.text_decoder(v, ts).view(nobj, nnames)
        #print(loss)
        return loss

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
            raise Exception("Wrong loss for SimilarityModelCharEncoder model")
        elif loss_type == "maxmargin_joint":
            loss = self.maxmargin_loss_over_objects(vs, ts, sampler=sampler.random_objects) + \
                   self.maxmargin_loss_over_words(vs, ts, sampler=sampler.random_words)
        else:
            loss = self.maxmargin_loss_over_words(vs, ts, sampler=sampler.random_words)
        return loss

    def get_production_probs(self, visual_input, text_input):
        similarities = self.forward(visual_input, torch.arange(self.nwords).long().to(device))
        #similarities = similarities / torch.norm(similarities)
        # weight size bsz x |V|
        #return torch.log(similarities / torch.sum(similarities, dim=1, keepdim=True))
        return torch.nn.LogSoftmax(dim=1)(similarities)

    def get_comprehension_probs(self, vs, t):
        similarities = self.forward(vs, t)
        return torch.nn.LogSoftmax(dim=0)(similarities)

    def softmax_loss_over_words(self, vs, ts):
        #print(logprobs.shape)
        # nobj = vs.shape[0]
        # nnames = ts.shape[0]
        # ts = ts.unsqueeze(0).expand(nobj, nnames, ts.shape[1]). \
        #     contiguous().view(nnames * nobj, ts.shape[1])
        # print(ts)
        # vs = vs.expand(nobj, nnames).contiguous().view(-1, vs.shape[1])
        # print(vs)
        nnames = ts.shape[0]
        loss = torch.sum(self.forward(vs, ts))
        return loss / nnames

    def maxmargin_loss_over_objects(self, vs, ts, nsample=5, sampler=None):
        m = 1.0
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

        return loss / ts.shape[0]

    def maxmargin_loss_over_words(self, vs, ts, nsample=5, sampler=None):
        m = 1.0
        loss = 0
        #print(ts.shape)
        for i in range(ts.shape[0]):
            t = ts[i].unsqueeze(0)  # i-th row - word
            positive = self.forward(vs, t).view(1, -1)
            #print(positive.shape)
            neg_ts = sampler(nsample * vs.shape[0]).to(device)#.unsqueeze(1)  #TODO why need unsqueeze()?

            negative = self.forward(vs, neg_ts).view(1, -1)
            #print(negative.shape)
            positive = positive.expand(nsample * vs.shape[0], positive.shape[1]).contiguous().view(1,-1)
            #print("Loss", torch.max(torch.zeros(1).to(device), m - positive + negative))

            loss_t = torch.sum(torch.max(torch.zeros(1).to(device), m - positive + negative)) / nsample

            if self.word_weight is not None:
                loss_t = loss_t * self.word_weight[t]
            loss += loss_t

        return loss / vs.shape[0]

    def supervised_loss(self, vs, ts):
        logprobs = self.get_production_probs(vs)
        loss = torch.sum(torch.nn.NLLLoss(reduce=False)(logprobs, ts.view(ts.shape[0])))  # ts.shape ?
        return loss

    def get_embeddings(self):   # only for symbolic encoding of objects
        vis_embs = self.visual_features_extractor(torch.arange(0, self.visual_features_extractor.nobjects).long().to(device)).detach().cpu().numpy()
        word_embs = self.text_encoder.weight.detach().cpu().numpy()
        return vis_embs, word_embs