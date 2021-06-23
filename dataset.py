import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


class Embedding:
    def __init__(self):
        self.word_embeddings = {}
        self.load_word_embedding()

    def load_word_embedding(self):
        vocab, embeddings = [], []

        f = open('datasets/glove.6B.300d.txt', encoding='utf8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            vocab.append(word)
            embeddings.append(coefs)
        f.close()

        vocab = np.array(vocab)
        embeddings = np.array(embeddings)

        # insert '<pad>' and '<unk>' tokens at start of vocab_npa.
        vocab = np.insert(vocab, 0, '<pad>')
        vocab = np.insert(vocab, 1, '<unk>')

        pad_emb_npa = np.zeros((1, embeddings.shape[1]), dtype='float32')  # embedding for '<pad>' token.
        unk_emb_npa = np.mean(embeddings, axis=0, keepdims=True)  # embedding for '<unk>' token.

        # insert embeddings for pad and unk tokens at top of embs_npa.
        embeddings = np.vstack((pad_emb_npa, unk_emb_npa, embeddings))

        embeddings_dict = {}
        for idx, (word, embed) in enumerate(zip(vocab, embeddings)):
            embeddings_dict[word] = embed

        word_list = load_str_list("datasets/vocab_questions_train.txt")
        self.word_embeddings = {}
        for w in word_list:
            if w in embeddings_dict:
                self.word_embeddings[w] = embeddings_dict[w]
            else:
                self.word_embeddings[w] = embeddings_dict['<unk>']

    def get_word_embedding(self):
        return self.word_embeddings


def load_str_list(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines


def tokenize(sentence):
    import re
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    tokens = SENTENCE_SPLIT_REGEX.split(sentence.lower())
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens


def preprocess(image_dir, question, max_qst_length=14):
    image = Image.open(image_dir).convert('RGB')
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    image = transform(image)

    question = tokenize(question)
    question_length = len(question)
    if question_length > max_qst_length:
        question_expand = np.array(question[:max_qst_length])
    else:
        for _ in range(max_qst_length - question_length):
            question.append('<pad>')
        question_expand = np.array(question)

    word_list = load_str_list("datasets/vocab_questions_train.txt")
    word_idx_dict = {w: n_w for n_w, w in enumerate(word_list)}
    question_embeddings = []
    for q in question_expand:
        if q in word_idx_dict:
            question_embeddings.append(word_idx_dict[q])
        else:
            question_embeddings.append(word_idx_dict['<unk>'])

    return image.unsqueeze(0), torch.tensor(question_embeddings).unsqueeze(0)


if __name__ == "__main__":
    e = Embedding()
    embed = e.get_word_embedding()
    np.save("word_embedding_train.npy", np.array(embed))
