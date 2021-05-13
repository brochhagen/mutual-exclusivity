

def create_vocab(text):
    word2idx = {}
    idx2word = []
    # reserve 0 index for padding
    idx2word.append("@")
    word2idx["@"] = 0

    for word in text:
        word = word.lower()
        if word not in word2idx:
            idx2word.append(word)
            word2idx[word] = len(idx2word) - 1
    return word2idx, idx2word


def load_vocab(path):
    word2idx = {}
    idx2word = []
    with open(path) as f:
        for line in f:
            word2idx[line.strip()] = len(idx2word)
            idx2word.append(line.strip())
    return word2idx, idx2word


def get_vocab_symbolic(vocab_path, words):
    try:
        word2idx, idx2word = load_vocab(vocab_path)
    except FileNotFoundError:
        word2idx, idx2word = create_vocab(words)
        print("Creating vocabulary")
        with open(vocab_path, "wt") as f:
            f.writelines([w + '\n' for w in idx2word])
    return word2idx, idx2word


def get_vocab(vocab_path, words):
    word2idx, idx2word = create_vocab(words)
    print("Creating vocabulary from scratch")
    return word2idx, idx2word    