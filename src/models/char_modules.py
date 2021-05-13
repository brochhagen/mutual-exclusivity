
import torch.nn as nn
import torch


import torch
from torch.nn import Parameter
from functools import wraps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (°° 
        return

    def _setup(self):

                # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda: mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class CharEncoder(nn.Module):
    """
        takes as input a seq of characters and outputs a vector = its hidden representations (encoded using an lstm)
    """

    def __init__(self, embeddings, lstm, hidden_size):
        super(CharEncoder, self).__init__()

        self.embeddings = embeddings
        self.lstm = lstm
        # todo this should be tied to lstm
        self.nlayers = 1
        self.hidden_size = hidden_size

    def forward(self, char_input):
        batch_size = char_input.shape[0]    # or the number of words/utterances etc
        input_lengths = char_input.ne(0).sum(1)
        # batch first to batch second packing of sequences
        char_input = char_input.t()
        # seq_len x batch_size x 1

        # initial value for hidden values at the beginning of each char sequences
        hidden = (torch.Tensor(self.nlayers, batch_size, self.hidden_size).fill_(0.1).to(device),
                  torch.Tensor(self.nlayers, batch_size, self.hidden_size).fill_(0.1).to(device))
        embs = self.embeddings(char_input)
        #print("Embs", embs.shape)
        output, hidden = self.lstm(embs, hidden)
        #print(output[-1].shape)
        return output[input_lengths - 1, torch.arange(batch_size).long()]


class PretrainedCharEncoder(nn.Module):

    def __init__(self, pretrained_encoder: CharEncoder, output_hidden_size: int):
        super(PretrainedCharEncoder, self).__init__()

        self.pretrained_encoder = pretrained_encoder
        # freeze pre-trained weights
        self.linear = nn.Linear(self.pretrained_encoder.hidden_size, output_hidden_size)
        for p in pretrained_encoder.parameters():
            p.requires_grad = False

    def forward(self, char_input):
        fixed_encoding = self.pretrained_encoder(char_input)
        return self.linear(fixed_encoding)


class CharDecoder(nn.Module):

    def __init__(self, embeddings, lstm, hidden_size):
        super(CharDecoder, self).__init__()

        self.embeddings = embeddings
        self.lstm = lstm
        self.dropout = nn.Dropout(0.1)
        self.output_linear = nn.Linear(hidden_size, embeddings.weight.shape[0],bias=False)
        self.nlayers = 1
        self.hidden_size = hidden_size

    def forward(self, encoded_input, target):
        # TODO rename to get_prob() - it's a (conditional) probability of a sequence
        batch_size = target.shape[0]
        word_length = target.shape[1]
        target = target.t()

        hidden = (encoded_input.expand(self.nlayers, -1, -1), encoded_input.expand(self.nlayers, -1, -1))
        #print(target.shape)
        loss = 0
        zero_input = torch.zeros((1, batch_size)).long().to(device)
        char = zero_input.view(1, batch_size)
        #print("Target", target)
        for i in range(target.shape[0]):
            #print(i)
            char = self.embeddings(char)
            output, hidden = self.lstm(char, hidden)
            #print("output", output[0].shape)
            o = self.dropout(output[0])
            target_prediction = self.output_linear(o)
            logprobs = nn.LogSoftmax(dim=1)(target_prediction)
            #print(logprobs)
            loss += torch.nn.NLLLoss(reduce=False)(logprobs, target[i])
            char = target[i].view(1, batch_size)
            #print(torch.argmax(logprobs,dim=1))
        return loss / word_length #TODO normalise or not??


class AttentionModelCharLevel(nn.Module):

    def __init__(self, vocab_size, hidden_size, visual_features_extractor):
        super(AttentionModelCharLevel, self).__init__()
        self.hidden_size = hidden_size
        self.visual_features_extractor = visual_features_extractor
        #self.visual_encoder = nn.Linear(visual_dim, hidden_size)

        self.embeddings = nn.Embedding(vocab_size, 50)
        self.lstm1 = nn.LSTM(input_size=50, hidden_size=hidden_size, num_layers=1)
        self.text_encoder = CharEncoder(self.embeddings, self.lstm1, hidden_size=hidden_size)

        self.linear_after_encoder = nn.Linear(hidden_size, hidden_size)

        self.lstm2 = nn.LSTM(input_size=50, hidden_size=hidden_size, num_layers=1)
        self.text_decoder = CharDecoder(self.embeddings, self.lstm1, hidden_size=hidden_size)

        #self.text_decoder.lstm = self.text_encoder.lstm   # sharing lstms ?

        #self.lstm = nn.LSTM(input_size=50, hidden_size=hidden_size, num_layers=1)
        #self.lstm_wd = WeightDrop(self.lstm1, ['weight_hh_l0'], dropout=0.2)
        #self.text_decoder.lstm = self.lstm_wd
        #self.text_encoder.lstm = self.lstm_wd
        #self.text_decoder.weight = self.text_encoder.weight

    def forward(self, visual_input, text_input):
        v = self.visual_features_extractor(visual_input)
        #v = self.visual_encoder(v)
        #print(v.shape)
        t = self.text_encoder(text_input)
        t = self.linear_after_encoder(t)
        t = torch.nn.ReLU()(t)
        #print("Encoded t", t.shape)
        v = v.squeeze(1)
        #print("Encoded v", v.shape)
        #t = t.view(t.shape[0], t.shape[2])

        v = v / torch.norm(v, dim=1, keepdim=True)
        t = t / torch.norm(t, dim=1, keepdim=True)

        sims = torch.matmul(v, t.t())
        #print(sims.shape)   # len(v) x len(t)
        # attention over object for each name (text input)
        attention = torch.nn.Softmax(dim=0)(sims)   # len(v) x len(t)
        #print(attention)
        h = torch.matmul(attention.t(), v)   # len(t) x hidden_size
        #print(h)
        h = h / torch.norm(h, dim=1, keepdim=True)

        loss = self.text_decoder(h, text_input)
        return loss

    def get_prob(self, visual_input, text_input):
        return self.get_production_prob_char(visual_input, text_input)

    def cross_situational_loss(self, visual_input, text_input):
        loss = torch.sum(self.forward(visual_input, text_input))
        #print("Loss", loss)
        return loss

    def supervised_loss(self, visual_input, text_input):
        # is the same as for similarity model
        v = self.visual_features_extractor(visual_input)
        #v = self.visual_encoder(v)
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

    def get_production_probs(self, objects, words):
        #print(len(words))
        #print(len(objects))
        m = torch.zeros(len(words), len(objects))
        #print(m.shape)
        for i, w in enumerate(words):
            #print(w.shape)
            for j, o in enumerate(objects):

                m[i,j] = torch.exp(-self.get_production_prob_char(o.unsqueeze(0), w.view(1,-1)))
        return m
        # get the whole distribution P(.|o) for all w
        v = self.visual_features_extractor(visual_input)
        #v = self.visual_encoder(v)

        weights = torch.matmul(v, self.text_decoder.weight.t()).squeeze(1)

        # weight size bsz x |V|
        #print("logmax shape", torch.nn.LogSoftmax(dim=1)(weights).shape)
        return torch.nn.LogSoftmax(dim=1)(weights)

    def get_production_prob_char(self, visual_input, text_input):
        #print(visual_input, text_input)
        v = self.visual_features_extractor(visual_input)
        #v = self.visual_encoder(v)

        loss = self.text_decoder(v, text_input)

        return loss

    def get_similarity_matrix(self, all_words, all_objects):
        m = torch.zeros(len(all_words), len(all_objects))
        for i, w in enumerate(all_words):
            for j, o in enumerate(all_objects):
                m[i,j] = torch.exp(-self.get_production_prob_char(o, w))
        return m

    def get_comprehension_probs(self, vs, t):
        v = self.visual_features_extractor(vs)
        #v = self.visual_encoder(v)
        #print(v.shape)
        t = self.text_encoder(t)
        t = self.linear_after_encoder(t)
        t = torch.nn.ReLU()(t)
        #print("Encoded t", t.shape)
        v = v.squeeze(1)
        #print("Encoded v", v.shape)
        #t = t.view(t.shape[0], t.shape[2])

        v = v / torch.norm(v, dim=1, keepdim=True)
        t = t / torch.norm(t, dim=1, keepdim=True)

        sims = torch.matmul(v, t.t())
        #print(sims.shape)   # len(v) x len(t)
        # attention over object for each name (text input)
        attention = torch.nn.Softmax(dim=0)(sims)
        return attention
