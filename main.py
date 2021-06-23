from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
import numpy as np
import glob
import argparse
import torch

from ui import Ui_MainWindow
from model.vqa import VqaModel
from utils.preprocess import preprocess
from utils.text_helper import load_str_list

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default="image/", help="path to image")
parser.add_argument("--output_folder", type=str, default="results/", help="path to outputs")
parser.add_argument("--weights_path", type=str, default="weights/glove_MRN9.pth", help="path to weights file")
parser.add_argument("--word_embeddings_path", type=str, default="data/word_embedding_train.npy",
                    help="path to pre-defined word embeddings received from GloVe")
parser.add_argument("--question_path", type=str, default="data/vocab_questions_train.txt",
                    help="path to pre-defined vocabulary collected from datasets")
parser.add_argument("--answer_path", type=str, default="data/vocab_answers_train.txt",
                    help="path to pre-defined 1000 candidate answers")
args = parser.parse_args()
print(args)


class AppWindow(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.show()

        self.image_dir = None
        self.image_list = sorted(glob.glob(args.image_folder + "*png"))
        word_embeddings = np.load(args.word_embeddings_path, allow_pickle=True).item()
        self.word_embeddings = np.array(list(word_embeddings.values()))
        self.answer_list = load_str_list(args.answer_path)

        self.ui.pushButton.clicked.connect(self.predict_button)
        self.ui.pushButton_2.clicked.connect(self.select_button)

        # ======= Build Model========
        embed_size = 1024
        word_embed_size = 300
        num_layers = 1
        hidden_size = 512
        ans_vocab_size = 1000

        self.model = VqaModel(
            embed_size=embed_size,
            ans_vocab_size=ans_vocab_size,
            word_embed_size=word_embed_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            word_embeddings=self.word_embeddings,
            embeddings_size=len(self.word_embeddings)
        )
        self.model.load_state_dict(torch.load("weights/glove_MRN9.pth", map_location=torch.device('cpu')))
        self.model.eval()

    def predict(self, image_dir, question):
        image, question_embeddings = preprocess(image_dir, question, args.question_path)
        print(question_embeddings)

        with torch.no_grad():
            output = self.model(image, question_embeddings)
            prob = torch.sigmoid(output).squeeze()
        pred = torch.argsort(prob)

        return pred[-5:].numpy()[::-1], prob[pred][-5:].numpy()[::-1]

    def predict_button(self):
        if self.image_dir is None:
            self.ui.result_display.setText(
                "<html><head/><body><p><span style=\" font-size:18pt;\">You haven't selected image yet!</span></p></body></html>")
            return
        elif len(self.ui.textEdit.toPlainText().split()) <= 2:
            self.ui.result_display.setText(
                "<html><head/><body><p><span style=\" font-size:18pt;\">Your question is not valid!</span></p></body></html>")
            return
        label, prob = self.predict(self.image_dir, self.ui.textEdit.toPlainText())
        answer = []
        for l in label:
            answer.append(self.answer_list[l])

        plt.clf()
        plt.barh(answer, prob)
        plt.savefig(args.output_folder + "result.png", dpi=2000)
        image = QImage(args.output_folder + "result.png")
        image = image.scaled(570, 360)
        self.ui.result_display.setPixmap(QPixmap(image))

    def select_button(self):
        self.image_dir = self.image_list[int(self.ui.comboBox.currentText()) - 1]
        image = QImage(self.image_dir)
        image = image.scaled(560, 320)
        self.ui.image_display.setPixmap(QPixmap(image))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = AppWindow()
    #pred = w.predict("test/abstract_v002_train2015_000000011779.png", "Who is happier?")
    #print(pred)
    #exit(1)
    w.show()
    sys.exit(app.exec_())
