import numpy as np

from utils.text_helper import load_str_list


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


if __name__ == "__main__":
    e = Embedding()
    embed = e.get_word_embedding()
    np.save("word_embedding_train.npy", np.array(embed))