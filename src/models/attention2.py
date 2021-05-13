
import torch.nn as nn
import torch


class AttentionModel2(nn.Module):
    # RSA-like model, draft

    def __init__(self, vocab_size, hidden_size, visual_features_extractor, visual_dim=4096):
        super(AttentionModel2, self).__init__()
        self.hidden_size = hidden_size
        self.visual_features_extractor = visual_features_extractor
        self.visual_encoder = nn.Linear(visual_dim, hidden_size)

        self.text_encoder = nn.Embedding(vocab_size, hidden_size)
        self.text_decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        #self.text_decoder.weight = self.text_encoder.weight

    def forward(self, visual_input, text_input):
        v = self.visual_features_extractor(visual_input)
        v = self.visual_encoder(v)
        #print(v.shape)
        t = self.text_encoder(text_input)
        v = v.squeeze(1)
        #print(t)

        probs = torch.matmul(v, self.text_encoder.weight.t())
        #print(probs.shape)   # len(v) x len(t)
        probs = probs[:,text_input].squeeze(2)
        #print(probs.shape)   # len(v) x len(t)
        # attention over object for each name (text input)
        attention = torch.nn.Softmax(dim=0)(probs)   # len(v) x len(t)
        #print(attention)
        h = torch.matmul(attention.t(), v)   # len(t) x hidden_size
        #print(h)
        weights = self.text_decoder(h)
        #print(weights.shape)

        return weights

    def get_prob(self, visual_input, text_input):
        logprobs = self.get_probs(visual_input)
        # bsz x |V|
        return logprobs[:, text_input]

    def cross_situational_loss(self, visual_input, text_input):
        logprobs = torch.nn.LogSoftmax(dim=1)(self.forward(visual_input, text_input))
        loss = torch.sum(torch.nn.NLLLoss(reduce=False)(logprobs, text_input.view(text_input.shape[0])))
        return loss

    def supervised_loss(self, visual_input, text_input):
        # is the same as for similarity model
        v = self.visual_features_extractor(visual_input)
        v = self.visual_encoder(v)
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
        v = self.visual_encoder(v)
        weights = torch.matmul(v, self.text_encoder.weight.t())
        # weight size bsz x |V|
        return torch.nn.LogSoftmax(dim=1)(weights)

    def get_comprehension_probs(self, vs, t):
        v = self.visual_features_extractor(vs)
        v = self.visual_encoder(v)
        #print(v.shape)
        #t = self.text_encoder(t)    # could also use text_decoder in principle??
        #t = self.text_decoder.weight[t]
        t = self.text_encoder.weight[t]

        similarities = torch.matmul(v, t.view(t.shape[0], t.shape[2]).t())
        return torch.nn.LogSoftmax(dim=0)(similarities)


