import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap
from torchvision import transforms
from PIL import Image

class ZodiacClassifier(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.label = QLabel(self)
        self.btn = QPushButton('上傳圖片', self)
        self.btn.clicked.connect(self.upload_image)
        
        self.result = QLabel(self)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.btn)
        layout.addWidget(self.result)
        
        self.setLayout(layout)
        self.setWindowTitle('十二生肖分類器')
        self.show()

    def upload_image(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "選擇圖片", "", "Images (*.png *.xpm *.jpg)", options=options)
        if fileName:
            pixmap = QPixmap(fileName)
            self.label.setPixmap(pixmap)
            self.classify_image(fileName)

    def classify_image(self, file_path):
        model = models.resnet18()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 12)
        model.load_state_dict(torch.load('zodiac_model.pth'))
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        image = Image.open(file_path)
        image = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            zodiac_sign = ['鼠', '牛', '虎', '兔', '龍', '蛇', '馬', '羊', '猴', '雞', '狗', '豬']
            result_text = f'這張圖片是：{zodiac_sign[predicted.item()]}'
            self.result.setText(result_text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ZodiacClassifier()
    sys.exit(app.exec_())
