import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QListWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

def find_images(root_dir):
    image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    image_paths = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_exts:
                image_paths.append(os.path.join(subdir, file))
    return image_paths

class ImageViewer(QWidget):
    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths
        self.idx = 0
        self.init_ui()
        self.show_image()

    def init_ui(self):
        self.setWindowTitle('Scene Image Viewer')
        self.resize(900, 600)
        main_layout = QHBoxLayout()

        # 图片显示区
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumWidth(600)
        main_layout.addWidget(self.image_label)

        # 右侧列表和输入
        right_layout = QVBoxLayout()
        self.list_widget = QListWidget()
        for i, path in enumerate(self.image_paths):
            self.list_widget.addItem(f"{i+1}: {os.path.basename(path)}")
        self.list_widget.setCurrentRow(self.idx)
        self.list_widget.currentRowChanged.connect(self.on_list_change)
        right_layout.addWidget(self.list_widget)

        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText('输入编号跳转')
        right_layout.addWidget(self.input_line)

        self.jump_btn = QPushButton('跳转')
        self.jump_btn.clicked.connect(self.on_jump)
        right_layout.addWidget(self.jump_btn)

        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

    def show_image(self):
        path = self.image_paths[self.idx]
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.image_label.setText(f"无法加载图片: {path}")
        else:
            scaled = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled)
        self.list_widget.setCurrentRow(self.idx)

    def on_list_change(self, row):
        self.idx = row
        self.show_image()

    def on_jump(self):
        try:
            num = int(self.input_line.text())
            if 1 <= num <= len(self.image_paths):
                self.idx = num - 1
                self.show_image()
            else:
                QMessageBox.warning(self, '跳转失败', '编号超出范围')
        except:
            QMessageBox.warning(self, '跳转失败', '请输入有效编号')

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Up:
            self.idx = (self.idx - 1) % len(self.image_paths)
            self.show_image()
        elif event.key() == Qt.Key_Down:
            self.idx = (self.idx + 1) % len(self.image_paths)
            self.show_image()
        elif event.key() == Qt.Key_Escape:
            self.close()

if __name__ == '__main__':
    root_dir = os.path.join(os.path.dirname(__file__), 'logs', 'scene_check_images')
    image_paths = find_images(root_dir)
    if not image_paths:
        print('没有找到图片。')
        sys.exit(1)
    app = QApplication(sys.argv)
    viewer = ImageViewer(image_paths)
    viewer.show()
    sys.exit(app.exec_())
