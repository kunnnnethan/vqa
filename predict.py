import torch

from dataset import preprocess
from model.vqa import VqaModel


def predict(image_dir, question, word_embeddings):
    embed_size = 1024
    word_embed_size = 300
    num_layers = 1
    hidden_size = 512
    ans_vocab_size = 1000

    model = VqaModel(
        embed_size=embed_size,
        ans_vocab_size=ans_vocab_size,
        word_embed_size=word_embed_size,
        num_layers=num_layers,
        hidden_size=hidden_size,
        word_embeddings=word_embeddings,
        embeddings_size=len(word_embeddings)
    )

    model.load_state_dict(torch.load("weights/glove_MRN9.pth", map_location=torch.device('cpu')))

    image, question_embeddings = preprocess(image_dir, question)
    print(question_embeddings)

    model.eval()
    with torch.no_grad():
        output = model(image, question_embeddings)
        prob = torch.sigmoid(output).squeeze()
    pred = torch.argsort(prob)

    return pred[-5:].numpy()[::-1], prob[pred][-5:].numpy()[::-1]


if __name__ == "__main__":
    import numpy as np
    #embeddings = Embedding()
    #word_embeddings = embeddings.get_word_embedding()
    #np.save("word_embedding.npy", word_embeddings)
    word_embeddings = np.load("word_embedding_train.npy", allow_pickle=True).item()
    word_embeddings = np.array(list(word_embeddings.values()))
    pred = predict("abstract_v002_train2015_000000011779.png", "Who looks happier?", word_embeddings)
    print(pred)
