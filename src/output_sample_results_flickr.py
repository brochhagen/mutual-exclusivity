
import scipy.spatial.distance
import torch
import numpy as np
import argparse
from datasets import load_flickr_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_evaluation_sample(dataset, model, nsample=100):
    n = 0

    for img_id, (datapoint, visual_input, text_input) in enumerate(dataset):
        if (img_id + 2) % 5 != 0:
            continue

        # less than two objects in the image
        if len(text_input) < 2:
            continue

        if n > nsample:
            break

        contains_novel = False
        for i, t in enumerate(text_input):
            if datapoint.is_novel(i):
                contains_novel = True

        if not contains_novel:
            continue

        n += 1
        print(visual_input.shape)
        print("Cosine",  - scipy.spatial.distance.cosine(visual_input[0], visual_input[1]) + 1)

        print("Example", n, "Image ID", datapoint.image_id, "#objects", len(text_input), sep="\t")

        obj_ps = model.get_production_probs(visual_input.to(device))   # production probs
        obj_predicted = np.argmax(obj_ps, axis=1)  # for each object, the highest prob

        print("Production:")
        for i in range(len(text_input)):
            # object_id, object_bbox, object_name, object_predicted_name
            print(i, datapoint.bboxes[i], dataset.idx2word[text_input[i]], dataset.idx2word[obj_predicted[i]], sep="\t")
            print(obj_ps[i][text_input].numpy())

        print("Comprehension:")
        for i, t in enumerate(text_input):    # correct name
            if datapoint.is_novel(i):
                cat = "unseen"
            else:
                cat = "seen"

            # choose one of the objects

            obj_sims = model(visual_input.to(device), t.to(device).view(1, *t.shape))

            object_standard_eval = np.argmax(obj_sims)

            object_me_eval = np.argmax(obj_ps[:, t])  # out of objects present in the scene

            correct_std = object_standard_eval == i
            correct_me = object_me_eval == i

            print(i, dataset.idx2word[t], object_standard_eval.item(), object_me_eval.item(), sep="\t")
            print(obj_sims.numpy())

def main():
    parser = argparse.ArgumentParser(description="Evaluating word reference models")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, default="../data/")
    parser.add_argument("--data_type", type=str, default="flickr", choices=["flickr", "symbolic"])
    parser.add_argument("--level", type=str, choices=['word', 'char'], default="word")
    parser.add_argument("--nimgs", type=int, default=500)
    parser.add_argument("--nexamples", type=int, default=300)

    args = parser.parse_args()

    img_ids = np.loadtxt(args.data + "/split/imagenames_train.txt").astype(dtype=np.int64)[:args.nimgs]

    dataset = load_flickr_dataset(args.data, args.level, img_ids)

    if args.print_examples:
        model = torch.load(args.model, map_location=lambda storage, loc: storage).to(device)
        print(model)
        model.eval()

        with torch.no_grad():
            print_evaluation_sample(dataset, model, nsample=args.nexamples)
        return