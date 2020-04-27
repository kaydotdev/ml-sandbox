import sys

from PyQt5.QtWidgets import QApplication
from ui.window import Window

app = QApplication(sys.argv)
w = Window()
sys.exit(app.exec_())
