from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
import numpy as np
import glob

from ui import Ui_MainWindow
from predict import predict
from dataset import load_str_list


class AppWindow(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.show()

        self.image_dir = None
        self.image_list = sorted(glob.glob("image/*png"))
        word_embeddings = np.load("word_embedding_train.npy", allow_pickle=True).item()
        self.word_embeddings = np.array(list(word_embeddings.values()))
        self.answer_list = load_str_list("datasets/vocab_answers_train.txt")

        self.ui.pushButton.clicked.connect(self.predict_button)
        self.ui.pushButton_2.clicked.connect(self.select_button)

    def select_button(self):
        self.image_dir = self.image_list[int(self.ui.comboBox.currentText()) - 1]
        image = QImage(self.image_dir)
        image = image.scaled(560, 320)
        self.ui.image_display.setPixmap(QPixmap(image))

    def predict_button(self):
        if self.image_dir is None:
            self.ui.result_display.setText("<html><head/><body><p><span style=\" font-size:18pt;\">You haven't selected image yet!</span></p></body></html>")
            return
        elif len(self.ui.textEdit.toPlainText().split()) <= 2:
            self.ui.result_display.setText("<html><head/><body><p><span style=\" font-size:18pt;\">Your question is not valid!</span></p></body></html>")
            return
        label, prob = predict(self.image_dir, self.ui.textEdit.toPlainText(), self.word_embeddings)
        answer = []
        for l in label:
            answer.append(self.answer_list[l])

        plt.clf()
        plt.barh(answer, prob)
        plt.savefig("results/result.png", dpi=2000)
        image = QImage("results/result.png")
        image = image.scaled(570, 360)
        self.ui.result_display.setPixmap(QPixmap(image))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = AppWindow()
    w.show()
    sys.exit(app.exec_())
