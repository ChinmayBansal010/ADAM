from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import pyqtSignal, Qt
import pyrebase
import logging


class RegisterWindow(QWidget):
    registration_successful_signal = pyqtSignal(dict)
    show_login_signal = pyqtSignal()

    def __init__(self, auth, db):
        super().__init__()
        self.auth = auth
        self.db = db
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Jarvis - Register")
        self.setFixedSize(400, 450)
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: white;
                font-family: 'Segoe UI';
                font-size: 14px;
                border-radius: 15px;
            }

            QLineEdit {
                background-color: #1E1E1E;
                border: 1px solid #3A3A3A;
                border-radius: 10px;
                padding: 10px;
                color: white;
            }

            QLineEdit:focus {
                border: 1px solid #2D87F0;
            }

            QPushButton {
                background-color: #2D87F0;
                border: none;
                border-radius: 10px;
                padding: 12px;
                font-weight: bold;
                color: white;
            }

            QPushButton:hover {
                background-color: #1565C0;
            }

            QLabel {
                color: #ffffff;
            }

            #linkButton {
                background-color: transparent;
                color: #90CAF9;
                border: 1px solid #2D87F0;
                border-radius: 10px;
                padding: 10px;
            }

            #linkButton:hover {
                background-color: rgba(45, 135, 240, 0.1);
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("Create your Jarvis Account")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Your Name")
        layout.addWidget(self.name_input)

        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Email")
        layout.addWidget(self.email_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password_input)

        register_button = QPushButton("Register")
        register_button.clicked.connect(self.register_user)
        layout.addWidget(register_button)

        login_button = QPushButton("Already have an account? Login")
        login_button.setObjectName("linkButton")
        login_button.clicked.connect(self.show_login_signal.emit)
        layout.addWidget(login_button)

        self.setLayout(layout)

    def register_user(self):
        name = self.name_input.text().strip()
        email = self.email_input.text().strip()
        password = self.password_input.text()

        if not name or not email or not password:
            QMessageBox.warning(self, "Input Error", "Please fill all fields.")
            return

        try:
            user = self.auth.create_user_with_email_and_password(email, password)
            user_id = user['localId']

            user_data = {
                "name": name,
                "email": email,
                "preferences": {
                    "voice": "en-IN-PrabhatNeural",
                    "wake_word_enabled": True
                }
            }
            self.db.child("users").child(user_id).set(user_data)

            QMessageBox.information(self, "Success", "Registration successful!")

            user_info = {
                "email": user['email'],
                "localId": user['localId'],
                "displayName": name,
                "preferences": user_data['preferences']
            }
            self.registration_successful_signal.emit(user_info)
        except Exception as e:
            logging.error("Registration Error: %s", str(e))
            QMessageBox.critical(self, "Registration Failed", str(e))
