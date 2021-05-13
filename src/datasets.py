import os

import pandas as pd
import torch

from visual_preprocessing import load_visual_model, get_bbox_representation
from vocabulary import get_vocab, create_vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def intersection(bbox1, bbox2):
    # adopted from iou computation, but computes instead a relative intersection to understand how much overlap there is
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = w1 * h1
    boxBArea = w2 * h2

    #iou = interArea / float(boxAArea + boxBArea - interArea)

    relative_intersection = interArea / min(boxAArea, boxBArea)
    return relative_intersection


stop_words = ["one", "other", "someone", "something", "object",
              "person", "man", "woman", "child", "kid", "girl", "boy", "guy", "lady"]
stop_words = []

class DataPoint(object):

    def __init__(self, sent, visual_model, vgg_layer, image_path=None, bboxes=None, ref_expressions=None,
                 level="word", novel_words=None):
        self.original_sent = sent     # original sentence, complete   - transformed / not transformed? list of words / string
        # at the moment it's string
        self.nwords = len(sent.split(" ")) - 2 # for <eos> and punctuation
        self.input_sent = sent    

        self.image_path = image_path
        self.image_id = image_path.split("/")[-1].split(".")[0]
        self.vector_path = "/".join(image_path.split("/")[:-2]) \
                           + "/preprocessed_vectors/" + self.image_id + "_" + vgg_layer

        self.novel_words = novel_words

        self.bboxes = None
        self.ref_expressions = None
        self.regions = None
        self.bboxes_preprocessed = None
        self.novel = None

        self.visual_model = visual_model

        output = self._get_bbox_RE_pairs(bboxes, ref_expressions)
        if output is not None:
            self.ref_expressions, regions, bboxes = list(zip(*output))
            # re is a string e.g. 'a man'
            ref_names = [re.split(" ")[-1] for re in self.ref_expressions]
            filtered_tuples, self.novel = \
                self.filter_regions(ref_names, regions, bboxes)
            if len(filtered_tuples) == 3:
                self.ref_names, self.regions, self.bboxes = filtered_tuples

            if level == "word":
                self.input_sent = self.input_sent.split(" ")
                self.ref_expressions = [re.split(" ") for re in self.ref_expressions]
            elif level == "char":
                pass

    def _get_bbox_RE_pairs(self, bboxes, ref_expressions):
        regions, _ = bboxes
        # one region can be made of several bounding boxes, e.g. for plural nouns ('rocks')
        # we keep only objects that have single box regions
        _, sent_regions = ref_expressions
        objects = [r for r in regions if regions.count(r) == 1 and r in sent_regions]
        if len(objects) == 0:
            return None
        regions_to_bbs = {r: bb for r, bb in zip(*bboxes) if r in objects}

        return [(re, region, regions_to_bbs[region]) for re, region in zip(*ref_expressions) if region in objects]

    def filter_regions(self, words, regions, bboxes):
        novel = []
        excluded_regions = []
        tuples = zip(words, regions, bboxes)
        for i, (name, region, bbox) in enumerate(tuples):

            if name in self.novel_words:
                novel.append(True)
            else:
                novel.append(False)

            if name in stop_words:
                excluded_regions.append(region)

            if region not in excluded_regions:
                for j in list(range(0, i)) + list(range(i + 1, len(bboxes))):
                    iou = intersection(bbox, bboxes[j])
                    if iou > 1:   # exclude only objects with 100% overlap (e.g. one object is part of the other)
                        excluded_regions.append(regions[j])

        novel_filtered = []
        filtered = []
        for i, (name, region, bbox) in enumerate(zip(words, regions, bboxes)):
            if region not in excluded_regions:
                filtered.append((name, region, bbox))
                novel_filtered.append(novel[i])
        return list(zip(*filtered)), novel_filtered

    def get_bbox_features(self):

        if self.bboxes_preprocessed is None:
            vectors = []
            for bbox, region_id in zip(self.bboxes, self.regions):
                bbox_path = self.vector_path + "_" + str(region_id)
                #print(bbox_path)
                vector = get_bbox_representation(self.image_path, self.visual_model, bbox, bbox_path)
                vectors.append(vector)

            self.bboxes_preprocessed = vectors
        return torch.stack(self.bboxes_preprocessed)

    def is_novel(self, i):
        return self.novel[i]


class AbstractWordDataset(object):

    def __init__(self, word2idx: dict, idx2word: list,
                       char2idx: dict, idx2char: list,
                       level: str):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.level = level

    def get_text_tensor(self, ref_names):
        return torch.LongTensor([[self.word2idx[name]] for name in ref_names]).to(device)

    def get_char_tensor(self, ref_names, output_new_order=False):
        ws = [torch.LongTensor([self.char2idx[c] for c in name + "#"]) for name in ref_names]

        idx_sorted, ws_sorted = zip(*sorted(enumerate(ws), key=lambda x: len(x[1]), reverse=True))
        ws = torch.nn.utils.rnn.pad_sequence(ws_sorted, batch_first=True)
        if output_new_order:
            return ws.to(device), idx_sorted
        else:
            return ws.to(device)

    def idx2text(self, indices):
        return [self.idx2word[idx] for idx in indices]


class WordObjectTuplesDataset(AbstractWordDataset):

    def __init__(self, image_folder_path, img_ids_to_sents, refs_to_regions, bbox_df,
                 word2idx, idx2word, char2idx, idx2char, level, vgg_layer, img_ids=None, novel_words=None):
        super(WordObjectTuplesDataset, self).__init__(word2idx, idx2word, char2idx, idx2char, level)

        self.refs = refs_to_regions       # pandas DF with REs and their region id
        self.bboxes = bbox_df   # DF with bounding boxes for regions

        if img_ids is None:
            img_ids = [x.split("_")[0] for x in refs_to_regions.keys() if "_0" in x]
        self.img_ids = img_ids

        self.image_folder_path = image_folder_path
        self.level = level

        self.novel_words = novel_words

        self.datapoints = []
        self.total_words = 0
        self.visual_model = load_visual_model(vgg_layer)
        self.vgg_layer = vgg_layer

        used_words_train = set()
        used_words_novel = set()

        for img_id in img_ids:
            image_path = os.path.join(self.image_folder_path, str(img_id) + ".jpg")

            for sent_id in range(5):
                extended_id = str(img_id) + "_" + str(sent_id)
                sent = img_ids_to_sents[extended_id].lower()

                if extended_id not in refs_to_regions:
                    print(extended_id)
                    continue

                res = refs_to_regions[extended_id]

                df = bbox_df[bbox_df.image_id == int(img_id)]
                bboxes = (list(df.region_id), list(df.bb))

                example = DataPoint(sent, self.visual_model, self.vgg_layer, image_path,
                                    bboxes, res, novel_words=novel_words)
                if example.bboxes is not None :
                    self.datapoints.append(example)
                    self.total_words += example.nwords
                    for j, w in enumerate(example.ref_names):
                        if example.is_novel(j):
                            used_words_novel.add(w)
                        else:
                            used_words_train.add(w)

        print("Full vocab size", len(word2idx))
        self.word2idx, self.idx2word = create_vocab(list(used_words_train) + list(used_words_novel))
        print("Effective vocab size", len(self.word2idx))
        print("Novel words", self.idx2word[-len(used_words_novel):] if len(used_words_novel) > 0 else [])
        self.novel_words = used_words_novel

        self.char2idx = char2idx
        self.idx2char = idx2char

        self.word_freqs = torch.ones(len(self.word2idx))   # uniform frequency
        self.obj_freqs = None

    def __getitem__(self, i):
        # load image
        example = self.datapoints[i]
        visual_input = example.get_bbox_features().to(device)

        if self.level == "word":
            text_input = self.get_text_tensor(example.ref_names)
        elif self.level == "char":
            text_input = self.get_char_tensor(example.ref_names)

        return example, visual_input, text_input     

    def __len__(self):
        return len(self.datapoints)

    def object_word_match(self, datapoint, visual_input, object_idx, text_input, word_idx):
        # relies on the fact that in flickr dataset we always have aligned data [objects] - [words]
        return object_idx == word_idx

    def is_novel_word(self, word):
        return word in self.novel_words


dog_names = ["dog", "dogs", "puppy", "retriever", "shepherd", "corgi", "korgi", "collie", "pug", "spaniel"]


def load_flickr_dataset(datapath, level, vgg_layer, img_ids=None, novel_words:list=None):
    image_folder_path = datapath + "/local/flickr30k_images/"
    sents_df = pd.read_csv(datapath + "/all_sentences_df.tab", sep="\t")
    sents_df = sents_df[sents_df.image_id.isin(img_ids)]
    img_ids_to_sents = dict(zip(sents_df.image_id.map(str) + "_" + sents_df.sent_id.map(str), sents_df.sent))

    re_df = pd.read_json(datapath + "/images_to_refs2regions.json.gz", compression="gzip", orient='split',
                         dtype={"image_ext_id": str, "refexps2regions": list})
    refs_to_regions = dict(zip(re_df.image_ext_id, re_df.refexps2regions))

    names = [re.split(" ")[-1] for image_ext_id, (res, _) in refs_to_regions.items() for re in res
             if int(image_ext_id.split("_")[0]) in img_ids]

    vocab_path = datapath + "/flickr_vocab." + str(len(img_ids)) + ".txt"
    names = names + novel_words
    word2idx, idx2word = get_vocab(vocab_path, names)
    char2idx, idx2char = get_vocab(datapath + "/flickr_vocab.char" + str(len(img_ids)) + ".txt", "".join(names) + "#")

    bbdf = pd.read_json(datapath + "/flickr30k_bbdf.json.gz", compression='gzip', orient='split')
    dataset = WordObjectTuplesDataset(image_folder_path, img_ids_to_sents, refs_to_regions, bbdf,
                                      word2idx, idx2word, char2idx, idx2char, level, vgg_layer,
                                      img_ids=img_ids, novel_words=novel_words)
    return dataset


