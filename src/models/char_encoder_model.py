

import torch.nn as nn
import torch
from models.char_modules import CharEncoder
from models.char_modules import PretrainedCharEncoder
from models.similarity_model import EncoderModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimilarityModelCharEncoder(EncoderModel):
    """ should be almost completely analogous to standard similarity model """

    def __init__(self, dataset, hidden_size, visual_features_extractor, text_encoder=None,
                 obj_weights=None,
                 word_weights=None):
        """ padding symbol is always 0 """

        if not text_encoder:
            embeddings = nn.Embedding(len(dataset.char2idx), 50, padding_idx=0)
            lstm = nn.LSTM(input_size=50, hidden_size=hidden_size, num_layers=1)
            text_encoder = CharEncoder(embeddings, lstm, hidden_size=hidden_size)
        # pre-trained encoder is available
        else:
            text_encoder = PretrainedCharEncoder(text_encoder, hidden_size)

        super(SimilarityModelCharEncoder, self).__init__(hidden_size, visual_features_extractor, text_encoder,
                                                         obj_weights, word_weights)

        # since 0 is padding, we want to compute probabilities only over words starting from index 1
        self.all_words_char_matrix = dataset.get_char_tensor(dataset.idx2word[1:])

    def get_production_probs(self, visual_input):
        """ for each object in visual_input compute the log-probabilities P(w|o) over all words in vocabulary """
        similarities = self.forward(visual_input, self.all_words_char_matrix)
        # compute softmax over words in the vocabulary *excluding* special padding word
        # we the add the 0 column corresponding to the padding word to be able to index prob probs matrix with
        # word indices starting from 1
        prod_probs = torch.zeros(similarities.shape[0], similarities.shape[1] + 1).fill_(-100)
        prod_probs[:,1:] = torch.nn.LogSoftmax(dim=1)(similarities)  # probs are normalised using softmax
        return prod_probs
