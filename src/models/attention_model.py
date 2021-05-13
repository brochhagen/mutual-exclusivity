
import torch.nn as nn
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionModel(nn.Module):

    def __init__(self, vocab_size, hidden_size, visual_features_extractor, init_range=0.01, tied=True, temperature=1,
                 obj_weights=None, word_weights=None):
        super(AttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.visual_features_extractor = visual_features_extractor

        self.text_encoder = nn.Embedding(vocab_size, hidden_size)
        if init_range:
            self.text_encoder.weight.data.uniform_(-init_range, init_range)
        #self.text_encoder = CharEncoder(vocab_size, emb_size=50, hidden_size=hidden_size)
        self.text_decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        #self.text_decoder = CharDecoder(vocab_size, emb_size=50, hidden_size=hidden_size)
        if tied:
            self.text_decoder.weight = self.text_encoder.weight
        self.temperature = temperature
        self.nwords = vocab_size

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

    def softmax_loss(self, h, ps, target, over):
        h = h / torch.norm(h, dim=1, keepdim=True)
        ps = ps / torch.norm(ps, dim=1, keepdim=True)
        weights = torch.matmul(h, ps.t())
        logprobs = torch.nn.LogSoftmax(dim=1)(weights)
        if over == "words":
            loss = torch.sum(torch.nn.NLLLoss(reduce=False, weight=self.word_weight)(logprobs, target))
        elif over == "objects":
            loss = torch.sum(torch.nn.NLLLoss(reduce=False, weight=self.obj_weight)(logprobs, target))
        return loss

    def cosine(self, x, y):
        x = x.squeeze(1)
        y = y.squeeze(1)
        x = x / torch.norm(x, dim=1, keepdim=True)
        y = y / torch.norm(y, dim=1, keepdim=True)
        return torch.matmul(x, y.t())

    def maxmargin_loss_over_objects(self, vs, hs, nsample=5, sampler=None):
        m = 1.0
        loss = 0
        # predicting objects, vs
        # hs are attention representation of words, for each vs
        for v, h in zip(vs, hs):
            v = v.unsqueeze(0)
            h = h.unsqueeze(0)
            v_ = self.visual_features_extractor(v)
            positive = self.cosine(v_, h).view(1, -1)

            neg_vs = sampler(nsample).to(device)#.unsqueeze(1)
            neg_vs = self.visual_features_extractor(neg_vs)

            negative = self.cosine(neg_vs, h).view(1, -1)
            #print("N", negative)
            positive = positive.expand(nsample, positive.shape[1]).contiguous().view(1,-1)
            #print("P", positive)
            #print("Loss", torch.max(torch.zeros(1).to(device), m - positive + negative).shape)


            loss_v = torch.sum(torch.max(torch.zeros(1).to(device), m - positive + negative)) / nsample
            if self.obj_weight is not None:
                loss_v = loss_v * self.obj_weight[v]

            loss += loss_v

        return loss

    def maxmargin_loss_over_words(self, hs, ts, nsample=5, sampler=None):
        m = 1.0
        loss = 0
        # predicting words, ts
        # hs are attention representation of objects, for each ts
        for t, h in zip(ts, hs):
            t = t.unsqueeze(0)
            h = h.unsqueeze(0)
            t_ = self.text_decoder.weight[t]
            positive = self.cosine(h, t_).view(1, -1)
            #print(positive.shape)
            neg_ts = sampler(nsample).to(device)#.unsqueeze(1)  #TODO why need unsqueeze()?
            neg_ts = self.text_decoder.weight[neg_ts]

            negative = self.cosine(h, neg_ts).view(1, -1)
            #print(negative.shape)
            positive = positive.expand(nsample, positive.shape[1]).contiguous().view(1,-1)
            #print("P", positive)
            #print("Loss", torch.max(torch.zeros(1).to(device), m - positive + negative))

            loss_t = torch.sum(torch.max(torch.zeros(1).to(device), m - positive + negative)) / nsample

            if self.word_weight is not None:
                loss_t = loss_t * self.word_weight[t]
            loss += loss_t

        return loss

    def maxmargin_loss(self, h, targets, over, nsample=5, sampler=None):
        # TODO for the moment only global
        m = 1.0
        loss = 0
        print(h.shape)
        for t in targets:
            #print(positive)

            neg_ps = sampler(nsample).to(device)
            print(neg_ps)
            if over == "words":
                # NB: here decoder, not encoder, but because it's Linear from h to vocab, we can't do ()
                neg_ps = self.text_decoder.weight[neg_ps]
                p = self.text_decoder.weight[t]

            elif over == "objects":
                neg_ps = self.visual_features_extractor(neg_ps)
                p = self.visual_features_extractor(t)

            p = p.unsqueeze(0)
            positive = self.cosine(p, h)

            #loss = 0
            negative = self.cosine(neg_ps, h)
            print("N", negative)
            loss_p = torch.mean(torch.max(torch.zeros(1).to(device), m - positive + negative))

            if over == "words" and self.word_weight is not None:
                loss_p = loss_p * self.word_weight[t]
            if over == "objects" and self.obj_weight is not None:
                loss_p = loss_p * self.obj_weight[t]

            print(t, loss_p)

            loss += loss_p
        return loss



    def attention_over_objects(self, visual_input, text_input, loss_type, sampler=None, prediction="global"):
        sims = self.forward(visual_input, text_input)
        #print(sims.shape)   # len(v) x len(t)
        # attention over object for each name (text input)

        attention = torch.nn.Softmax(dim=0)(sims / self.temperature)   # len(v) x len(t)
        #print(attention)

        v = self.visual_features_extractor(visual_input)
        v = v.squeeze(1)
        h = torch.matmul(attention.t(), v)   # len(t) x hidden_size

        if prediction == "global":
            ts = self.text_decoder.weight
            #ts = torch.arange(self.nwords).long().to(device)
            targets = text_input.view(text_input.shape[0])
        elif prediction == "local":
            ts = self.text_decoder(text_input)
            ts = ts.squeeze(1)
            targets = torch.arange(text_input.shape[0]).long().to(device)
            #ts = text_input

        if loss_type == "maxmargin":
            loss = self.maxmargin_loss_over_words(h, text_input, sampler=sampler)
        elif loss_type == "softmax":
            loss = self.softmax_loss(h, ts, targets, over="words")


        #print(weights.shape)

        return loss

    def attention_over_words(self, visual_input, text_input, loss_type, sampler=None, prediction="global"):
        sims = self.forward(visual_input, text_input)
        #print(sims.shape)   # len(v) x len(t)
        # attention over object for each name (text input)

        attention = torch.nn.Softmax(dim=1)(sims / self.temperature)   # len(v) x len(t)
        #print("Attention", attention.shape)

        t = self.text_encoder(text_input)
        t = t.squeeze(1)
        h = torch.matmul(attention, t)   # len(t) x hidden_size
        #print(h.shape)

        # if prediction == "global":
        #     vs = self.visual_features_extractor.visual_encoder.weight
        #     #ts = torch.arange(self.nwords).long().to(device)
        #     targets = visual_input.view(visual_input.shape[0])
        #     #print(target)
        # elif prediction == "local":
        #     vs = self.visual_features_extractor.visual_encoder.weight[visual_input]
        #     vs = vs.squeeze(1)
        #     targets = torch.arange(visual_input.shape[0]).long().to(device)
        #     #ts = text_input

        if loss_type == "maxmargin":
            loss = self.maxmargin_loss_over_objects(visual_input, h, sampler=sampler)
        elif loss_type == "softmax":
            # NB only for symbolic objects
            vs = self.visual_features_extractor.visual_encoder.weight
            targets = visual_input.view(visual_input.shape[0])
            loss = self.softmax_loss(h, vs, targets, over="objects")

        #print(weights.shape)

        return loss

    def get_prob(self, visual_input, text_input):
        logprobs = self.get_production_probs(visual_input)
        # bsz x |V|
        return logprobs[:, text_input]

    def cross_situational_loss(self, vs, ts, loss_type=None, sampler=None):

        if loss_type == "maxmargin_words":
            loss = self.attention_over_objects(vs, ts, loss_type="maxmargin", sampler=sampler.random_words)
        elif loss_type == "maxmargin_objects":
            loss = self.attention_over_words(vs, ts, loss_type="maxmargin", sampler=sampler.random_objects)
        elif loss_type == "softmax_words":
            loss = self.attention_over_objects(vs, ts, loss_type="softmax", prediction="global")
        elif loss_type == "softmax_objects":
            loss = self.attention_over_words(vs, ts, loss_type="softmax", prediction="global")
        elif loss_type == "softmax_joint":
            loss = self.attention_over_objects(vs, ts, loss_type="softmax", prediction="global")\
                   + self.attention_over_words(vs, ts, loss_type="softmax", prediction="global")
        elif loss_type == "maxmargin_joint":
            loss =  self.attention_over_objects(vs, ts, loss_type="maxmargin", sampler=sampler.random_words) \
                  + self.attention_over_words(vs, ts, loss_type="maxmargin", sampler=sampler.random_objects)

        else:
            loss = self.attention_over_objects(vs, ts, prediction="global")
        return loss

    def supervised_loss(self, visual_input, text_input):
        # is the same as for similarity model
        v = self.visual_features_extractor(visual_input)
        #print(v.shape)
        t = self.text_encoder(text_input)
        #print(t)

        # basically one-to-one pairs are given by diagonal attention
        # attention = torch.eye(t.shape[0])
        #print(attention)
        # h = torch.matmul(attention.t(), v)   # len(t) x hidden_size
        h = v
        #print(h)
        weights = self.text_decoder(h)
        logprobs = torch.nn.LogSoftmax(dim=1)(weights)
        #print(weights.shape)
        loss = torch.sum(torch.nn.NLLLoss(reduce=False)(logprobs, text_input.view(text_input.shape[0])))
        return loss

    def get_production_probs(self, visual_input):
        # get the whole distribution P(.|o) for all w
        v = self.visual_features_extractor(visual_input)

        weights = torch.matmul(v, self.text_decoder.weight.t()).squeeze(1)

        # weight size bsz x |V|
        #print("logmax shape", torch.nn.LogSoftmax(dim=1)(weights).shape)
        return torch.nn.LogSoftmax(dim=1)(weights)

    def get_comprehension_probs(self, vs, t):
        v = self.visual_features_extractor(vs)
        #print(v.shape)
        #t = self.text_encoder(t)    # could also use text_decoder in principle??
        t = self.text_decoder.weight[t]

        similarities = torch.matmul(v, t.view(t.shape[0], t.shape[2]).t())
        return torch.nn.LogSoftmax(dim=0)(similarities)

    def get_embeddings(self):   # only for symbolic encoding of objects
        vis_embs = self.visual_features_extractor(torch.arange(0, self.visual_features_extractor.nobjects).long().to(device)).detach().cpu().numpy()
        word_embs = self.text_decoder.weight.detach().cpu().numpy()
        return vis_embs, word_embs




