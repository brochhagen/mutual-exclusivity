
import argparse
import random
import os
import csv

import numpy as np
import pandas as pd
import torch

from symbolic_datasets import load_symbolic_dataset_and_vocabularies, create_novel_words_symbolic_dataset,create_novel_words_symbolic_sim_biased_dataset
from evaluations_symbolic import evaluate_lexicon, similarity_matrix, object_object_similarity_matrix, word_word_similarity_matrix
from evaluations import evaluate_comprehension
from vocabulary import load_vocab


def main():

    parser = argparse.ArgumentParser(description="Training word reference similarity model on symbolic data")
    parser.add_argument("--level", type=str, choices=['word', 'char'], default="word")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--pretrained_text_encoder", type=str)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--name", type=str, help='To make saved evaluations identifiable', default='test')

    parser.add_argument("--daxes", nargs='+', type=str, required=True)
    parser.add_argument("--save_results", type=bool, default=False)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("***** Arguments *****")
    print(args)

    print("***** Dataset *****")
    print("Loading symbolic dataset")

    if args.pretrained_text_encoder:
        char2idx, idx2char = load_vocab(args.pretrained_text_encoder + "/vocab.txt")

        dataset = load_symbolic_dataset_and_vocabularies(args.data, level=args.level,
                                                         daxes=args.daxes,
                                                         char2idx=char2idx,
                                                         idx2char=idx2char, shuffle=False)
    else:
        dataset = load_symbolic_dataset_and_vocabularies(args.data, level=args.level,
                                                         daxes=args.daxes, shuffle=False)

    mean_random, mean_match_familiar, mean_bayes_familiar, mean_match_dax, mean_bayes_dax, mean_match_confusable, mean_bayes_confusable = 0,0,0,0,0,0,0
    dax_accs = [[0,0,0,0] for _ in range(len(args.daxes))]
    
    print("Dataset size", len(dataset))
    print("Vocab size", len(dataset.word2idx))

    print("***** Model *****")
    model = torch.load(args.model)
    print(model)
    with torch.no_grad():
        model.eval()

        for dax in args.daxes:
            novel_dataset = create_novel_words_symbolic_dataset(dataset, level=args.level, daxes=[dax], nperdax=100)
            sim_biased_novel_dataset = create_novel_words_symbolic_sim_biased_dataset(dataset,level=args.level, daxes=[dax], nperdax=100)


            print('============ Dax: %s ============' % dax)
            print("==== Familiar word comprehension (training data) ====")
            acc, random_acc, n, acc_per_word, n_per_word, acc_by_nobj, n_by_nobj = \
                evaluate_comprehension(dataset, model, False)
            acc_me, random_acc, n, acc_per_word_me, n_per_word, acc_by_nobj, n_by_nobj = \
                evaluate_comprehension(dataset, model, True)
            print("Acc familiar", "Random {:.2f}".format(random_acc["seen"] / n["seen"]), "P(o|w)",
                  "{:.2f}".format(acc["seen"] / n["seen"]), "P(w|o)",
                  "{:.2f}".format(acc_me["seen"] / n["seen"]))
    
            mean_random += random_acc["seen"] / n["seen"]
            mean_match_familiar += acc["seen"] / n["seen"]
            mean_bayes_familiar += acc_me["seen"] / n["seen"]
    
            print("==== Novel word comprehension evaluation ====")
            acc, random_acc, n, acc_per_word, n_per_word, acc_by_nobj, n_by_nobj = \
                evaluate_comprehension(novel_dataset, model, False)
            acc_me, random_acc, n, acc_per_word_me, n_per_word, acc_by_nobj, n_by_nobj = \
                evaluate_comprehension(novel_dataset, model, True)
            print("Acc novel", "Random {:.2f}".format(random_acc["unseen"] / n["unseen"]), "P(o|w)",
                  "{:.2f}".format(acc["unseen"] / n["unseen"]), "P(w|o)",
                  "{:.2f}".format(acc_me["unseen"] / n["unseen"]))

    
            mean_match = acc["unseen"] / n["unseen"]
            mean_bayes = acc_me['unseen'] / n['unseen']

            mean_match_dax += mean_match
            mean_bayes_dax += mean_bayes
            dax_accs[args.daxes.index(dax)][0] += mean_match
            dax_accs[args.daxes.index(dax)][1] += mean_bayes

         
            print("==== Novel sim-biased word comprehension evaluation ====")
            acc, random_acc, n, acc_per_word, n_per_word, acc_by_nobj, n_by_nobj = \
                evaluate_comprehension(sim_biased_novel_dataset, model, False)
            acc_me, random_acc, n, acc_per_word_me, n_per_word, acc_by_nobj, n_by_nobj = \
                evaluate_comprehension(sim_biased_novel_dataset, model, True)
            print("Acc novel", "Random {:.2f}".format(random_acc["unseen"] / n["unseen"]), "P(o|w)",
                  "{:.2f}".format(acc["unseen"] / n["unseen"]), "P(w|o)",
                  "{:.2f}".format(acc_me["unseen"] / n["unseen"]))
    
            mean_match = acc["unseen"] / n["unseen"]
            mean_bayes = acc_me['unseen'] / n['unseen']

            mean_match_confusable += mean_match
            mean_bayes_confusable += mean_bayes
            dax_accs[args.daxes.index(dax)][2] += mean_match
            dax_accs[args.daxes.index(dax)][3] += mean_bayes


    #Normalizing accuracies
    ndaxes = len(args.daxes)
    mean_random = mean_random / ndaxes
    mean_match_familiar = mean_match_familiar / ndaxes
    mean_bayes_familiar = mean_bayes_familiar / ndaxes
    mean_match_dax = mean_match_dax / ndaxes
    mean_bayes_dax = mean_bayes_dax / ndaxes
    mean_match_confusable = mean_match_confusable / ndaxes
    mean_bayes_confusable = mean_bayes_confusable / ndaxes

    if args.save_results:
        in_path = os.path.basename(os.path.dirname(args.data))
        outfile_path = '../results/local/symbolic_'+ in_path + '_' + args.level + '_' + args.name
        if not os.path.isfile(outfile_path + '.csv') :
            with open(outfile_path + '.csv', mode = 'w') as outfile:
                outwriter = csv.writer(outfile, delimiter=',')
                outwriter.writerow(['seed', 'ndaxes', 'random', 'familiar_match', 'familiar_bayes','novel_match_dax', 'novel_bayes_dax', 'novel_match_confusable', 'novel_bayes_confusable'])
        with open(outfile_path + '.csv', mode='a') as outfile:
            outwriter = csv.writer(outfile, delimiter=',')
            outwriter.writerow([args.seed, len(args.daxes), mean_random, mean_match_familiar, mean_bayes_familiar,
                                            mean_match_dax, mean_bayes_dax, mean_match_confusable, mean_bayes_confusable])
        
        with open('../results/local/daxes_symbolic_' + in_path + '_' + args.level + '_' + args.name + '.csv', mode='a') as f:
            outwriter = csv.writer(f, delimiter=',')
            for d in range(len(args.daxes)):
                outwriter.writerow([args.seed, args.daxes[d], dax_accs[d][0], dax_accs[d][1], dax_accs[d][2], dax_accs[d][3]])

if __name__ == "__main__":
    main()
