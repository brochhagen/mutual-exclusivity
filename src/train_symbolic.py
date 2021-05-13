import argparse
import csv
import os
import random
import time

import pandas as pd
import numpy as np
import torch

os.environ['PYTHONHASHSEED']=str(1234)

from iterators import MilestoneIterator
from symbolic_datasets import load_symbolic_dataset_and_vocabularies, create_novel_words_symbolic_dataset
from samplers import SymbolicSampler
from models.attention_model import AttentionModel
from models.similarity_model import SimilarityModel
from models.angelikis_model import SimilarityModelA
from models.char_modules import AttentionModelCharLevel
from models.char_encoder_model import SimilarityModelCharEncoder
from evaluations_symbolic import evaluate_lexicon, similarity_matrix, object_object_similarity_matrix, word_word_similarity_matrix
from evaluations import evaluate_comprehension
from vocabulary import load_vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

from models.visual_models import SymbolicVisualExtractor


def print_similarity_matrices(dataset, model, level):
    with torch.no_grad():
        df1 = pd.DataFrame(word_word_similarity_matrix(model, dataset, level, dataset.idx2word),
                           index=dataset.idx2word, columns=dataset.idx2word)
        df2 = pd.DataFrame(similarity_matrix(model, dataset, level, dataset.idx2word, dataset.idx2obj),
                           index=dataset.idx2word, columns=dataset.idx2obj)
        print("Word-object\n", df2)


def main():

    parser = argparse.ArgumentParser(description="Training word reference similarity model on symbolic data")
    parser.add_argument("--level", type=str, choices=['word', 'char'], default="word")
    parser.add_argument("--loss", type=str, default='softmax_words', choices=[
                                                                    "maxmargin_words", "maxmargin_objects",
                                                                    "softmax_words", "softmax_objects",
                                                                    "maxmargin_joint", "softmax_joint"],
                        help="the type of loss function")
    parser.add_argument("--save", type=str, help="model name prefix to save checkpoints", default="model.pt")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--sampling", type=str, choices=["shuffle", "zipfian_images", "zipfian_nrefs"],
                                      help="sample images (scenes) according to a strategy, default: none")
    parser.add_argument("--nepochs", type=int, default=10)
    parser.add_argument("--optimiser", type=str, choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", default=0.0, type=float) 
    parser.add_argument("--hidden_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epoch_interval", type=int, default=30, help="how often to save the model")
    parser.add_argument("--milestone", type=int, default=-1,
                        help='default: all datapoints in the dataset, 10000 could be number of words per month')
    parser.add_argument("--model_type", type=str, default="similarity",
                        choices=["similarity","attention","similarityA"])
    parser.add_argument("--fixed_visual", action='store_true')
    parser.add_argument("--pretrained_text_encoder", type=str)

    parser.add_argument("--daxes", nargs='+', type=str, default=[])
    parser.add_argument("--irw", type=float, default=0.05)
    parser.add_argument("--iro", type=float, default=1.0)
    parser.add_argument("--include_novel", action='store_true',
                        help="whether to include novel words/objects in the negative samples (for maxmargin loss)")
    parser.add_argument("--save_results", type=bool, default=False)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("***** Arguments *****")
    print(args)

    start = time.time()

    print("***** Dataset *****")
    print("Loading symbolic dataset")

    if args.pretrained_text_encoder:
        char2idx, idx2char = load_vocab(args.pretrained_text_encoder + "/vocab.txt")

        dataset = load_symbolic_dataset_and_vocabularies(args.data, level=args.level,
                                                         shuffle=args.sampling == "shuffle",
                                                         daxes=args.daxes,
                                                         char2idx=char2idx,
                                                         idx2char=idx2char)
    else:
        dataset = load_symbolic_dataset_and_vocabularies(args.data, level=args.level,
                                                         shuffle=args.sampling == "shuffle",
                                                         daxes=args.daxes)

    if args.milestone == -1:
        args.milestone = dataset.total_words     # default milestone for the moment
    iterator = MilestoneIterator(dataset, args.nepochs,
                                 skip_novel=False,
                                 sampling=args.sampling,
                                 milestone=args.milestone)   # number of words per day

    print("Dataset size", len(dataset))
    print("Vocab size", len(dataset.word2idx))

    novel_dataset = None
    if args.daxes:
        novel_dataset = create_novel_words_symbolic_dataset(dataset, level=args.level, daxes=args.daxes, nperdax=100)

    print("***** Model *****")
    visual_model = SymbolicVisualExtractor(len(dataset.obj2idx), args.hidden_size,
                                           fixed_weights=args.fixed_visual, init_range=args.iro)

    if args.level == "word":
        if args.model_type == "attention":
            model = AttentionModel(len(dataset.word2idx), args.hidden_size, visual_model,
                                   obj_weights=dataset.obj_freqs, word_weights=dataset.word_freqs).to(device)

        elif args.model_type == "similarity":
            model = SimilarityModel(len(dataset.word2idx), args.hidden_size, visual_model,
                                    init_range=args.irw, obj_weights=dataset.obj_freqs, word_weights=dataset.word_freqs).to(device)
        elif args.model_type == "similarityA":
            model = SimilarityModelA(len(dataset.word2idx), args.hidden_size, visual_model).to(device)
    elif args.level == "char":
        if args.model_type == "attention":
            model = AttentionModelCharLevel(len(dataset.char2idx), args.hidden_size, visual_model).to(device)
        elif args.model_type == "similarity":
            pretrained_encoder = None
            if args.pretrained_text_encoder:
                pretrained_encoder = torch.load(args.pretrained_text_encoder + "/best_model.pt")
            model = SimilarityModelCharEncoder(dataset, args.hidden_size, visual_model,
                                               text_encoder=pretrained_encoder).to(device)

    print(model)

    # sampler for maxmargin loss negative examples
    negative_sampler = SymbolicSampler(dataset, args.include_novel, level=args.level, ndaxes=len(args.daxes))

    if args.optimiser == "sgd":
        optimiser = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.lr, weight_decay=args.weight_decay)
    average_loss = 0
    nexamples = 0

    nrefs = []


    bestfs = []

    accs = [] #save accuracies per epoch
    lsss = [] #save avg loss per epoch

    for visual_input, text_input in iterator:
        model.train()
        model.zero_grad()
        nrefs.append(len(text_input))

        vs = visual_input.to(device)
        ts = text_input.to(device)

        loss = model.cross_situational_loss(vs, ts, loss_type=args.loss, sampler=negative_sampler)

        loss.backward()
        optimiser.step()

        average_loss += loss.cpu().item()
        nexamples += len(visual_input)

        if iterator.is_milestone():
            print("Processed {:5d} images ({:5d} words), {:5d} objects, {:5d} types, "
                  "average nrefs {:.2f} in {:5.2f}s".format(iterator.nimages_processed,
                                                            iterator.nwords_processed,
                                                            nexamples,
                                                            iterator.nwords_processed,
                                                            np.mean(nrefs),
                                                            time.time() - start))
            print("Epoch", iterator.epoch, "Training loss {:5.3f}".format(average_loss / nexamples))

            lsss.append(average_loss / nexamples)

            with torch.no_grad():
                model.eval()
                print("==== Lexicon Best-F evaluation ====")
                bestfs.append(evaluate_lexicon(dataset, model, args.data, args.level, print_bestf=True))

                print("==== Familiar word comprehension (training data) ====")
                acc, random_acc, n, acc_per_word, n_per_word, acc_by_nobj, n_by_nobj = \
                    evaluate_comprehension(dataset, model, False)
                acc_me, random_acc, n, acc_per_word_me, n_per_word, acc_by_nobj, n_by_nobj = \
                    evaluate_comprehension(dataset, model, True)
                print("Acc familiar", "Random {:.2f}".format(random_acc["seen"] / n["seen"]), "P(o|w)",
                      "{:.2f}".format(acc["seen"] / n["seen"]), "P(w|o)",
                      "{:.2f}".format(acc_me["seen"] / n["seen"]))

            # saving checkpoints
            if iterator.epoch % args.epoch_interval == 0 and iterator.epoch != 0:
                torch.save(model, args.save + str(iterator.epoch))

            average_loss = 0
            nexamples = 0
            start = time.time()

    # saving final epoch model
    if args:
        torch.save(model, args.save)

    if args.save_results:
        in_path = os.path.basename(os.path.dirname(args.data))
        outfile_path = 'mutual-exclusivity/results/local/symbolic_'+ in_path + '_' + args.model_type + '_' + args.loss
        if not os.path.isfile(outfile_path + '.csv') :
            with open(outfile_path + '.csv', mode = 'w') as outfile:
                outwriter = csv.writer(outfile, delimiter=',')
                outwriter.writerow(["lr", "hs", "wd", "irw", "iro", "fv", "inn", "ne", 'seed', "bestEpoch", 'bestF', 'accNRandom', 'accNMatch', 'accNME','daxwords'])
        with open(outfile_path + '.csv', mode='a') as outfile:
            bestF_idx = lsss.index(min(lsss))#bestfs.index(max(bestfs))
            bestF = bestfs[bestF_idx]
            accuracies = accs[bestF_idx]

            outwriter = csv.writer(outfile, delimiter=',')
            outwriter.writerow([args.lr, args.hidden_size, args.weight_decay, args.irw, args.iro,
                                            args.fixed_visual, args.include_novel, args.nepochs, args.seed,
                                             bestF_idx, bestF, accuracies[0], accuracies[1], accuracies[2],args.daxes])

        with torch.no_grad():
            pd.DataFrame(similarity_matrix(model, dataset, args.level,
                                             dataset.idx2word, dataset.idx2obj),
                               index=dataset.idx2word, columns=dataset.idx2obj)\
                .to_csv(outfile_path + '_word_object_similarity_matrix_' + str(args.include_novel) + '_' + args.daxes[0] + '.csv', sep="\t")
        with torch.no_grad():
            pd.DataFrame(word_word_similarity_matrix(model, dataset, args.level,
                                             dataset.idx2word),
                               index=dataset.idx2word, columns=dataset.idx2word)\
                .to_csv(outfile_path + '_word_word_similarity_matrix_' + str(args.include_novel) + '_' + args.daxes[0] + '.csv', sep="\t")


if __name__ == "__main__":
    main()
