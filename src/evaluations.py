import torch
import argparse
from datasets import load_flickr_dataset
from symbolic_datasets import load_symbolic_dataset_and_vocabularies
import numpy as np
from collections import defaultdict
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE", device)
dogs = ["dog", "dogs", "puppy", "retriever", "shepherd"]

np.set_printoptions(precision=3)

def get_prob(logprobs, text_input):
    return logprobs[:, text_input]


def evaluate_comprehension(dataset, model, do_mutual_exclusivity):
    # known words, overall
    # one image input
    acc = {"seen": 0, "unseen": 0}
    random_acc = {"seen": 0, "unseen": 0}
    n = {"seen": 0, "unseen": 0}

    acc_by_nobj = defaultdict(int)
    n_by_nobj = defaultdict(int)

    acc_per_word = defaultdict(int)
    n_per_word = defaultdict(int)

    for nimg, (datapoint, vs, ts) in enumerate(dataset):
        vs = vs.to(device)
        ts = ts.to(device)
        if len(vs) == 1:
            continue

        if do_mutual_exclusivity:
            production_probs = model.get_production_probs(vs).squeeze(1).cpu().numpy()
        else:
            if dataset.level == "char":
                ts, idx_sorted = dataset.get_char_tensor(datapoint.ref_names, output_new_order=True)
                inv_idx = np.array([idx_sorted.index(i) for i in range(len(idx_sorted))])

            similarities = model(vs, ts).t().cpu()   #forward pass

            if dataset.level == "char":
                similarities = similarities[inv_idx, :]

        for i, t in enumerate(ts):

            if dataset.is_novel_word(datapoint.ref_names[i]):
                cat = "unseen"
            else:
                cat = "seen"

            # choose one of the objects
            if not do_mutual_exclusivity:

                object = np.argmax(similarities[i].cpu().numpy())
            else:
                if dataset.level == "char":
                    word_probs = production_probs[:, dataset.word2idx[datapoint.ref_names[i]]]
                else:
                    word_probs = get_prob(production_probs, t)   # P(w|o)
                object = np.argmax(word_probs)
                
                max_sim = word_probs[object]



            if dataset.object_word_match(datapoint, vs, object, ts, i):


                acc[cat] += 1
                acc_per_word[datapoint.ref_names[i]] += 1
                acc_by_nobj[len(vs)] += 1


            random_acc[cat] += 1/len(vs)

            n[cat] += 1
            n_per_word[datapoint.ref_names[i]] += 1
            n_by_nobj[len(vs)] += 1

    return acc, random_acc, n, acc_per_word, n_per_word, acc_by_nobj, n_by_nobj


def evaluate_production():
    pass


def evaluate_one_model(model_name, dataset):
    model = torch.load(model_name, map_location=lambda storage, loc: storage).to(device)
    model.eval()
    model.visual_features_extractor.eval()

    with torch.no_grad():
        acc, random_acc, n, acc_per_word, n_per_word, acc_by_nobj, n_by_nobj = evaluate_comprehension(dataset, model,
                                                                                                      do_mutual_exclusivity=False)
        me_acc, random_acc, n, me_acc_per_word, _, me_acc_by_nobj, _ = evaluate_comprehension(dataset, model, do_mutual_exclusivity=True)

    print("** Already observed words **")
    print("Random acc", random_acc["seen"] / n["seen"])
    print("P(o|w) evaluation", acc["seen"] / n["seen"])
    print("ME evaluation", me_acc["seen"] / n["seen"])
    if n["unseen"] > 0:
        print("** Novel words **")
        print("Random acc", random_acc["unseen"] / n["unseen"])
        print("P(o|w) evaluation", acc["unseen"] / n["unseen"])
        print("ME evaluation", me_acc["unseen"] / n["unseen"])

    print("Acc by number of objects")
    for i in range(1, 6):
        if i in acc_by_nobj:
            print("{:5d}\t{:.2f}\t{:.2f}\t{:5d}".format(i,
                                                        acc_by_nobj[i] / n_by_nobj[i],
                                                        me_acc_by_nobj[i] / n_by_nobj[i],
                                                        n_by_nobj[i]))

    acquired_words = []
    acquired_words_count = 0
    acquired_words_me = 0
    for w in acc_per_word:
        if acc_per_word[w] / n_per_word[w] > .8 and n_per_word[w] > 4:
            acquired_words_count += 1
            acquired_words.append(w)
        if me_acc_per_word[w] / n_per_word[w] > .8 and n_per_word[w] > 4:
            acquired_words_me += 1

    for w, count in sorted(n_per_word.items(), key=lambda x: x[1]):
        print(w, count, w in acquired_words)

    print("Words comprehended above threshold 80%", acquired_words_count, acquired_words_me, sep="\t")


def evaluate_models_across_time(model_name, dataset):
    results = []

    for i in range(1, 390, 1):
        print(i)

        try:
            model = torch.load(model_name + str(i) + ".pt", map_location=lambda storage, loc: storage).to(device)
        except:
            break
        model.eval()
        model.visual_features_extractor.eval()

        with torch.no_grad():
            acc, random_acc, n, acc_per_word, n_per_word, acc_by_nobj, n_by_nobj = \
                evaluate_comprehension(dataset, model, do_mutual_exclusivity=False)
            me_acc, random_acc, n, me_acc_per_word, n_per_word, me_acc_by_nobj, n_by_obj = \
                evaluate_comprehension(dataset, model, do_mutual_exclusivity=True)

        row = [i, random_acc["seen"] / n["seen"], acc["seen"] / n["seen"], me_acc["seen"] / n["seen"]]

        if n["unseen"] > 0:
            row = row + [random_acc["unseen"] / n["unseen"],
                         acc["unseen"] / n["unseen"],
                         me_acc["unseen"] / n["unseen"]]
        else:
            row = row + [np.nan, np.nan, np.nan]

        for j in range(1, 6):
            if j in acc_by_nobj:
                row = row + [acc_by_nobj[j] / n_by_nobj[j],
                             me_acc_by_nobj[j] / n_by_nobj[j],
                             n_by_nobj[j]]
            else:
                row = row + [np.nan, np.nan, np.nan]

        acquired_words = 0
        acquired_words_me = 0
        for w in acc_per_word:
            if acc_per_word[w] / n_per_word[w] > .8 and n_per_word[w] > 4:
                acquired_words += 1
            if me_acc_per_word[w] / n_per_word[w] > .8 and n_per_word[w] > 4:
                acquired_words_me += 1

        row = row + [acquired_words, acquired_words_me]
        results.append(row)

    by_obj_cols = [["acc_" + str(i) + "obj", "acc_" + str(i) + "obj_me", "n_" + str(i) + "obj"]
                                        for i in range(1,6) ]
    by_obj_cols = [l for sublist in by_obj_cols for l in sublist]
    df = pd.DataFrame(results, columns=["day",
                                        "acc_seen_random", "acc_seen", "acc_seen_me",
                                        "acc_unseen_random", "acc_unseen", "acc_unseen_me"]
                                         + by_obj_cols + ["acq_words", "acq_words_me"])
    print(df)
    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluating word reference models")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, default="../data/")
    parser.add_argument("--data_type", type=str, default="flickr", choices=["flickr", "symbolic"])
    parser.add_argument("--level", type=str, choices=['word', 'char'], default="word")
    parser.add_argument("--nimgs", type=int, default=500)

    args = parser.parse_args()

    if args.data_type == "flickr":
        # just retrospectively computing observed words
        img_ids = np.loadtxt(args.data + "/split/imagenames_train.txt").astype(dtype=np.int64)[:args.nimgs]

        dataset = load_flickr_dataset(args.data, args.level, img_ids)

        print("Number of images", len(img_ids))

    elif args.data_type == "symbolic":
        dataset = load_symbolic_dataset_and_vocabularies(args.data, args.level, shuffle=False)

    if args.model.endswith(".pt"):
        evaluate_one_model(args.model, dataset)
    else:
        df = evaluate_models_across_time(args.model, dataset)
        model_name = args.model.split("/")[-1]
        df.to_csv(args.data + "results/" + model_name)


if __name__ == "__main__":
    main()
