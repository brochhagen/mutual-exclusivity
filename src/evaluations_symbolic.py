
import numpy as np
import torch
import pandas as pd

from symbolic_datasets import SymbolicWordObjectDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bestF(score_matrix, gold_lex):
    """ Measure from Frank et al. 2009, implemented based on their code and paper """
    np.seterr(divide='ignore', invalid='ignore')  # get nans instead of division by zero
    best_f = 0
    best_th = 0

    for th in np.linspace(0, 1, 101):   # threshold
        lex = (score_matrix >= th)
        p = np.sum(lex * (lex == gold_lex)) / np.sum(lex)
        r = np.sum(lex * (lex == gold_lex)) / np.sum(gold_lex)
        f = 2 * (p * r) / (p + r)

        if f > best_f:
            best_f = f
            best_th = th

    return best_f, best_th


def read_gold_lexicon(datapath):
    gold_words = {}
    with open(datapath + "/gold_mapping.txt") as f:
        for l in f:
            k, v = l.strip().split(" = ")
            gold_words[k] = v
    words = list(sorted(gold_words.keys()))
    objects = list(sorted(set(gold_words.values())))
    gold_lex = np.zeros((len(words), len(objects)))
    for w in gold_words:
        gold_lex[words.index(w), objects.index(gold_words[w])] = 1
    return gold_lex, words, objects


def normalise(a):
    norm = np.sqrt(np.sum(a * a, axis=1, keepdims=True))
    return a / norm


def similarity_matrix(model, dataset, level, words: list, objects: list):
    object_ids = dataset.get_visual_tensor(objects)

    if level == "word":
        word_ids = dataset.get_text_tensor(words)
        sim_matrix = model(object_ids, word_ids).t().cpu().numpy()   # shape nwords x nobjs

    elif level == "char":
        char_words, idx_sorted = dataset.get_char_tensor(words, output_new_order=True)
        inv_idx = np.array([idx_sorted.index(i) for i in range(len(idx_sorted))])

        sim_matrix = model(object_ids, char_words).t().cpu().numpy()
        sim_matrix = sim_matrix[inv_idx,:]

    return sim_matrix


def object_object_similarity_matrix(model, dataset, objects: list):
    object_ids = dataset.get_visual_tensor(objects)
    vs = model.visual_encoder(object_ids)
    vs = vs.squeeze(1)
    vs = vs / torch.norm(vs, dim=1, keepdim=True)
    result = torch.matmul(vs, vs.t())
    return result.cpu().numpy()


def word_word_similarity_matrix(model, dataset, level, words: list):

    if level == "word":
        word_ids = dataset.get_text_tensor(words)
        ts = model.text_encoder(word_ids)   # shape nwords x nobjs

    elif level == "char":
        char_words, idx_sorted = dataset.get_char_tensor(words, output_new_order=True)
        inv_idx = np.array([idx_sorted.index(i) for i in range(len(idx_sorted))])

        ts = model.text_encoder(char_words)
        ts = ts[inv_idx]

    ts = ts.squeeze(1)
    ts = ts / torch.norm(ts, dim=1, keepdim=True)
    result = torch.matmul(ts, ts.t())
    return result.cpu().numpy()


def evaluate_lexicon(dataset: SymbolicWordObjectDataset, model, datapath, level, print_bestf=False):
    gold_lex, words, objects = read_gold_lexicon(datapath)

    sim_matrix = similarity_matrix(model, dataset, level, words, objects)


    sim_matrix = torch.nn.Softmax(dim=1)(torch.tensor(sim_matrix))
    sim_matrix = sim_matrix.numpy()

    max_sim_matrix = np.zeros_like(sim_matrix)
    max_idx = np.argmax(sim_matrix, axis=1)
    max_sim_matrix[np.arange(sim_matrix.shape[0]), max_idx] = 1

    bestf, th = bestF(sim_matrix, gold_lex)
    bestf_max, _ = bestF(max_sim_matrix, gold_lex)
    if print_bestf:
        print("Best F {:.2f}".format(bestf))
        print("Threshold {:.2f}".format(th), "Best F max {:.2f}".format(bestf_max))
    return bestf


def print_results(dataset, model):
    names = ["cow", "baby", "pig", "book", "mirror", "bird", "hat"]
    objs = ["cow", "baby", "pig", "book", "mirror", "bird", "hat"]

    widx = [dataset.word2idx[w] for w in names]
    charidx = [[dataset.char2idx[ch] for ch in w ] for w in names ]
    oidx = [dataset.obj2idx[o] for o in objs]

    for i, w in enumerate(charidx):  # widx
        print(names[i])
        for j, o in enumerate(oidx):
            print(objs[j], model.get_prob(torch.tensor([[o,]]),
                                                 torch.tensor([w])).item())  #[[w]] for words

    names = ["cow", "bird", "pig", "book", "mirror"]
    widx = [dataset.word2idx[w] for w in names]
    oidx = [dataset.obj2idx[o] for o in names]

def get_accuracy_words(dataset, model):
    vis_emb = model.visual_encoder(model.visual_features_extractor(torch.arange(0, len(dataset.obj2idx)).long()))
    acc_input = 0
    acc_output = 0
    for w in dataset.word2idx:


        match_output = torch.argmax(torch.matmul(model.text_decoder.weight[dataset.word2idx[w]], vis_emb.t()))
        match_input = torch.argmax(torch.matmul(model.text_encoder.weight[dataset.word2idx[w]], vis_emb.t()))
        print(w, dataset.idx2obj[match_output])
        if w == dataset.idx2obj[match_output]:
            acc_output += 1
        if w == dataset.idx2obj[match_input]:
            acc_input += 1
    return acc_output/len(dataset.word2idx), acc_input/len(dataset.word2idx)


def get_accuracy_objects(dataset, model):
    acc_output = 0
    acc_input = 0
    print('Objects')
    for o in dataset.obj2idx:
        obj_idx = dataset.obj2idx[o]

        match_output = torch.argmax(torch.matmul(model.text_decoder.weight,
                           model.visual_encoder(model.visual_features_extractor.weight[obj_idx])))
        match_input = torch.argmax(torch.matmul(model.text_encoder.weight,
                                                 model.visual_encoder(
                                                     model.visual_features_extractor.weight[obj_idx])))
        if o == dataset.idx2word[match_output]:
            acc_output += 1
        if o == dataset.idx2word[match_input]:
            acc_input += 1
    return acc_output / len(dataset.obj2idx), acc_input / len(dataset.obj2idx)