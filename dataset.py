#encoding

def build_dict():
    files = [
            "video_music_book_datasets/data/train.txt",
            "video_music_book_datasets/data/test.txt"
            ]
    words = set()
    labels = set()
    words.add('<unk>')
    for f in files:
        for line in open(f):
            line = line.strip("\n")
            if len(line) <= 1 or line == "<end>":
                pass
            else:
                data = line.split(" ")
                word = " ".join(data[0:-1]).lstrip(" ")
                label = data[-1]
                words.add(word)
                labels.add(label)

    word_idx = dict(list(zip(words, range(len(words)))))
    label_idx = dict(list(zip(labels, range(len(labels)))))

    idx_word = dict(map(reversed, word_idx.items()))
    idx_label = dict(map(reversed, label_idx.items()))
    return word_idx, label_idx, idx_word, idx_label


def train():
    word_idx, label_idx, _, _ = build_dict()
    UNK = word_idx['<unk>']
    INS = []
    tokens = []
    labels = []
    for line in open("video_music_book_datasets/data/train.txt"):
        line = line.strip("\n")
        if len(line) <= 1:
            INS.append((tokens, labels))
            tokens = []
            labels = []
        else:
            data = line.split(" ")
            word = " ".join(data[0:-1]).lstrip(" ")
            label = data[-1]
            tokens.append(word_idx.get(word, UNK))
            labels.append(label_idx.get(label))

    print(len(INS))
    def reader():
        for doc, label in INS:
            yield doc, label
    return reader    


if __name__ == "__main__":
    #result = train()().next()
    #print result
    word_idx, label_idx, idx_word, idx_label = build_dict()
    print label_idx
    print idx_label
