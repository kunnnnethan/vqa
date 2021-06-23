import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from utils.text_helper import load_str_list, tokenize


def preprocess(image_dir, question, word_list_path, max_qst_length=14):
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

    word_list = load_str_list(word_list_path)
    word_idx_dict = {w: n_w for n_w, w in enumerate(word_list)}
    question_embeddings = []
    for q in question_expand:
        if q in word_idx_dict:
            question_embeddings.append(word_idx_dict[q])
        else:
            question_embeddings.append(word_idx_dict['<unk>'])

    return image.unsqueeze(0), torch.tensor(question_embeddings).unsqueeze(0)
