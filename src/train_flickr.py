import argparse
import time

import numpy as np
import torch
import random
import sys
import os

from datasets import load_flickr_dataset, dog_names
from dax_flickr_dataset import create_novel_objects_flickr_dataset
from iterators import MilestoneIterator
from samplers import FlickrSampler, FlickrSamplerUniformVocabulary
from models.attention_model import AttentionModel
from models.similarity_model import SimilarityModel
from models.angelikis_model import SimilarityModelA
from models.char_modules import AttentionModelCharLevel
from models.char_encoder_model import SimilarityModelCharEncoder
from evaluations import evaluate_comprehension

os.environ['PYTHONHASHSEED']=str(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models.visual_models import VGGVisualExtractor, PreprocessedExtractor

def word_nns(dataset, model, top=5):
    """ nearest neighbours for chosen (novel) words """
    dax_names = ["daxa"]
    for w in dax_names:
        v = model.text_encoder.weight[dataset.word2idx[w]]
        v = v / torch.norm(v)
        ts = model.text_encoder.weight / torch.norm(model.text_encoder.weight, dim=1, keepdim=True)
        sims = torch.matmul(v, ts.t()).detach().cpu().numpy()
        top_nns = np.argpartition(-sims, top)[:top]
        print(w, [dataset.idx2word[i] for i in top_nns])


def store_vectors(dataset, model, suffix):
    """ to visualize in the notebook """
    names = []
    visual_vectors = torch.Tensor()
    text_vectors = torch.Tensor()
    # to store vectors for each example
    for example, vs, ts in dataset:
        model.eval()
        names.extend(example.ref_names)
        vs = vs.to(device)
        ts = ts.to(device)
        visual_vectors = torch.cat([visual_vectors, model.visual_features_extractor(vs).squeeze(1)])
        text_vectors = torch.cat([text_vectors, model.text_encoder(ts).squeeze(1)])

    np.savetxt("../results/local/visual_vectors_" + suffix + ".txt", visual_vectors.detach().cpu().numpy())
    np.savetxt("../results/local/text_vectors_" + suffix + ".txt", text_vectors.detach().cpu().numpy())
    open("../results/local/names_" + suffix + ".txt", "w").write("\n".join(names))


def print_evaluation(dataset, model, novel_dataset, nseed):
    with torch.no_grad():
        model.eval()
        acc, random_acc, n, acc_per_word, n_per_word, acc_by_nobj, n_by_nobj = \
            evaluate_comprehension(novel_dataset, model, False)

        acc_me, random_acc, n, acc_per_word_me, n_per_word, acc_by_nobj, n_by_nobj = \
            evaluate_comprehension(novel_dataset, model, True)
        print("Training acc:", "N", n["seen"], "Rand", "{:.2f}".format(random_acc["seen"] / n["seen"]),
              "P(o|w)", "{:.2f}".format(acc["seen"] / n["seen"]), "P(w|o)",
              "{:.2f}".format(acc_me["seen"] / n["seen"]))

        novels_in_sample = [key for key in n_per_word if key in novel_dataset.novel_words]

        if len(novels_in_sample) > 0:
            print("Novel ws acc:", "N", n["unseen"], "Rand",
                "{:.2f}".format(random_acc["unseen"] / n["unseen"]), "P(o|w)",
                "{:.2f}".format(acc["unseen"] / n["unseen"]), "P(w|o)",
                "{:.2f}".format(acc_me["unseen"] / n["unseen"]))

            print('\n')
            for w in novels_in_sample:
                print('\t', w, acc_per_word[w] / n_per_word[w], acc_per_word_me[w] / n_per_word[w])

            return([random_acc["unseen"] / n["unseen"],
                acc["unseen"] / n["unseen"],
                acc_me["unseen"] / n["unseen"],
                random_acc["seen"] / n["seen"],
                acc['seen'] / n['seen'],
                acc_me['seen'] / n['seen'], len(novels_in_sample)])
        else: 
            print("No daxes")




def main():

    parser = argparse.ArgumentParser(description="Training word reference model on Flickr data")
    parser.add_argument("--level", type=str, choices=['word', 'char'], default="word")
    parser.add_argument("--supervision", type=str, default='cs', choices=['cs', 'o2o'],
                        help="Cross-situational 'cs' or one-to-one object-word supervision 'o2o'")
    parser.add_argument("--loss", type=str, default='softmax_words', choices=['nll', 'energy',
                                                                    "maxmargin_words", "maxmargin_objects",
                                                                    "softmax_words", "softmax_objects",
                                                                    "maxmargin_joint", "softmax_joint"],
                        help="Energy loss 'energy' / NLL for p(w|o) loss 'nll' / ...")
    parser.add_argument("--debug", action='store_true', help="Only use 100 images")
    parser.add_argument("--save", type=str, help="Model name prefix to save checkpoints", default="model")
    parser.add_argument("--data", type=str, default="../data/flickr/")
    parser.add_argument("--novel_set", type=str, default="set1", choices=["set1", "set2", "dogs"])
    parser.add_argument("--daxes", nargs='+', type=str)

    parser.add_argument("--sampling", type=str, choices=["shuffle", "zipfian_images", "zipfian_nrefs"],
                                      help="Sample images according to a strategy")
    parser.add_argument("--nepochs", type=int, default=10)
    parser.add_argument("--optimiser", type=str, choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--hidden_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--nimgs", type=int, default=-1)
    parser.add_argument("--epoch_interval", type=int, default=30)
    parser.add_argument("--milestone", type=int, default=-1) 
    parser.add_argument("--model_type", type=str, default="similarity", choices=["similarity","attention","similarityA"])
    parser.add_argument("--fixed_visual", action='store_true')
    parser.add_argument("--vgg_layer", type=str, default="lastlayer", choices=["lastlayer","fc7","fc6"])

    parser.add_argument("--include_novel", action='store_true',
                        help="Whether include novel words/objects in the negative samples")
    parser.add_argument("--save_results", type=bool, default=False, 
                        help='Write best epoch (lowest loss) evaluation to ~/results/local/')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("***** Arguments *****")
    print(args)

    start = time.time()

    img_ids = np.loadtxt(args.data + "/split/imagenames_train.txt").astype(dtype=np.int64)

    if args.debug:
        img_ids = img_ids[:100]
    elif args.nimgs == -1: #take all
        img_ids = img_ids[:args.nimgs]
    else: #take random subset
        img_ids = np.random.choice(img_ids,args.nimgs, replace=False)


    print("Number of images", len(img_ids))

    if args.novel_set == "dogs":
        novel_words = dog_names
    else:
        if not args.daxes:
            print("Please specify dax words using '--daxes'")
            return
        novel_words = args.daxes

    dataset = load_flickr_dataset(args.data, args.level, args.vgg_layer, img_ids, novel_words=novel_words)
    if args.milestone == -1:
        args.milestone = dataset.total_words     # default milestone
    iterator = MilestoneIterator(dataset, args.nepochs,
                                 skip_novel=True,
                                 sampling=args.sampling,
                                 milestone=args.milestone)   # number of words per day
    print("Dataset size", len(dataset))
    print("Vocab size", len(dataset.word2idx))
    print("Milestone", args.milestone, "words (in sentences)")

    if args.novel_set == "dogs":
        # use objects named as dogs from the flickr images as novel objects
        novel_dataset = load_flickr_dataset(args.data, args.level, args.vgg_layer, img_ids, novel_words=novel_words)
        novel_dataset.datapoints = novel_dataset.datapoints[:]
        # sampler for maxmargin loss negative examples
        negative_sampler = FlickrSamplerUniformVocabulary(dataset, args.include_novel, level=args.level)

    else:
        # use external images of (strange) objects as novel objects with 'dax' names
        novel_dataset = create_novel_objects_flickr_dataset(args.data + "/daxes/" + args.novel_set,
                                                            dataset, novel_words=novel_words,
                                                            level=args.level)
        negative_sampler = FlickrSamplerUniformVocabulary(dataset, args.include_novel,
                                                          level=args.level,
                                                          novel_dataset=novel_dataset)
                                                           # sampler for maxmargin loss negative examples

    visual_model = PreprocessedExtractor(args.hidden_size, fixed_weights=args.fixed_visual, vgg_layer=args.vgg_layer)

    if args.level == "word":
        if args.model_type == "attention":
            model = AttentionModel(len(dataset.word2idx), args.hidden_size, visual_model,
                                   obj_weights=dataset.obj_freqs, word_weights=dataset.word_freqs).to(device)

        elif args.model_type == "similarity":
            model = SimilarityModel(len(dataset.word2idx), args.hidden_size, visual_model,
                                    init_range=0.1, obj_weights=dataset.obj_freqs, word_weights=dataset.word_freqs).to(device)
        elif args.model_type == "similarityA":
            model = SimilarityModelA(len(dataset.word2idx), args.hidden_size, visual_model).to(device)
    elif args.level == "char":
        if args.model_type == "attention":
            model = AttentionModelCharLevel(len(dataset.char2idx), args.hidden_size, visual_model).to(device)
        elif args.model_type == "similarity":
            model = SimilarityModelCharEncoder(dataset, args.hidden_size, visual_model).to(device)
    print(model)


    if args.optimiser == "sgd":
        optimiser = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.lr, weight_decay=args.weight_decay)

    average_loss = 0
    nexamples = 0
    nrefs = []

    accs = []
    lsss = []

    for visual_input, text_input in iterator:
        model.train()
        model.zero_grad()
        nrefs.append(len(text_input))

        vs = visual_input.to(device)
        ts = text_input.to(device)

        if args.supervision == "o2o": # one-to-one = supervised loss for v_i t_i pairs
            loss = model.supervised_loss(vs, ts)
        elif args.supervision == "cs":   # cross-situational all v_i, t_j pairs
            loss = model.cross_situational_loss(vs, ts, loss_type=args.loss, sampler=negative_sampler)

        loss.backward()
        optimiser.step()

        average_loss += loss.cpu().item()
        nexamples += len(visual_input)

        if iterator.is_milestone():
            print("\nProcessed {:5d} datapoints (total {:5d} words in sentences), "
                  "{:5d} objects, "
                  "average nrefs {:.2f} in {:5.2f}s".format(iterator.nimages_processed,
                                                            iterator.nwords_processed,
                                                            nexamples,
                                                            np.mean(nrefs),
                                                            time.time() - start))
            print("Epoch", iterator.epoch, "Training loss {:5.3f}".format(average_loss / nexamples))


            accs.append( print_evaluation(dataset, model, novel_dataset, args.seed) )
            lsss.append( average_loss / nexamples )
            print(args.epoch_interval)

            if iterator.epoch % args.epoch_interval == 0 and iterator.epoch != 0:
                with open(args.save + str(iterator.epoch) + ".pt", 'wb') as f:
                    torch.save(model, f)

            average_loss = 0
            nexamples = 0
            start = time.time()


    if args.save_results:
        import os
        import csv
        outfile_path = '../results/local/flickr_' + args.model_type + '_' + args.loss + '.csv'
        if not(  os.path.isfile(outfile_path) ):
            with open(outfile_path, mode = 'w') as outfile:
                outwriter = csv.writer(outfile, delimiter=',')
                outwriter.writerow(["lr", "hs", "wd", "fv", "inn", "nepochs", "nimgs", 'ndaxes', 'seed', "novelSet", "minLossEpoch", 'accNRandom', 'accNMatch', 'accNME','accSRandom', 'accSMatch', 'accSME'])
        with open(outfile_path, mode='a') as outfile:
            minLoss_idx = lsss.index(min(lsss))
            minLoss = lsss[minLoss_idx]
            accuracies = accs[minLoss_idx]

            outwriter = csv.writer(outfile, delimiter=',')
            outwriter.writerow([args.lr, args.hidden_size, args.weight_decay,
                                            args.fixed_visual, args.include_novel, args.nepochs, args.nimgs, accs[minLoss_idx][6], args.seed, args.novel_set,
                                            minLoss_idx+1, accs[minLoss_idx][0], accs[minLoss_idx][1], accs[minLoss_idx][2],
                                            accs[minLoss_idx][3], accs[minLoss_idx][4], accs[minLoss_idx][5]])


if __name__ == "__main__":
    main()
