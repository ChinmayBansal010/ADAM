from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QMessageBox, QCheckBox, QGraphicsDropShadowEffect
)
from PyQt5.QtCore import pyqtSignal, Qt, QSize, QPropertyAnimation
from PyQt5.QtGui import QColor, QIcon
import json
import os


class LoginWindow(QWidget):
    login_successful_signal = pyqtSignal(dict)
    auto_login_successful_signal = pyqtSignal(dict)
    show_register_signal = pyqtSignal()

    def __init__(self, auth, db):
        super().__init__()
        self.auth = auth
        self.db = db
        self.init_ui()
        self.apply_styles()
        self.fade_in()
        self.try_auto_login()

    def init_ui(self):
        self.setWindowTitle("Jarvis Login")
        self.setFixedSize(440, 520)
        self.setStyleSheet("background-color: #121212;")

        container = QVBoxLayout()
        container.setContentsMargins(30, 30, 30, 30)
        container.setAlignment(Qt.AlignCenter)

        frame = QWidget()
        frame_layout = QVBoxLayout()
        frame_layout.setSpacing(20)
        frame.setObjectName("GlassFrame")
        frame.setLayout(frame_layout)
        
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 160))

        frame.setGraphicsEffect(shadow)

        self.title_label = QLabel("Welcome to Jarvis")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setObjectName("TitleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(self.title_label)

        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Email")
        self.email_input.setObjectName("InputField")
        frame_layout.addWidget(self.email_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setObjectName("InputField")

        self.show_password = QPushButton()
        self.show_password.setCheckable(True)
        self.show_password.setFixedWidth(40)
        self.show_password.setIcon(QIcon("assets/icons/eye-off.png"))
        self.show_password.setStyleSheet("border: none;")
        self.show_password.setObjectName("ShowPasswordButton")
        self.show_password.setIconSize(QSize(24, 24))
        self.show_password.clicked.connect(self.toggle_password)


        password_layout = QHBoxLayout()
        password_layout.addWidget(self.password_input)
        password_layout.addWidget(self.show_password)
        frame_layout.addLayout(password_layout)

        self.remember_checkbox = QCheckBox("Remember Me")
        self.remember_checkbox.setObjectName("RememberCheckbox")
        frame_layout.addWidget(self.remember_checkbox)

        login_button = QPushButton("Login")
        login_button.setObjectName("LoginButton")
        login_button.clicked.connect(self.login)
        frame_layout.addWidget(login_button)

        register_button = QPushButton("Don't have an account? Register")
        register_button.setObjectName("RegisterButton")
        register_button.clicked.connect(self.show_register_signal.emit)
        frame_layout.addWidget(register_button)

        container.addWidget(frame)
        self.setLayout(container)
    
    def fade_in(self):
        self.setWindowOpacity(0)
        self.show()
        animation = QPropertyAnimation(self, b"windowOpacity")
        animation.setDuration(600)
        animation.setStartValue(0)
        animation.setEndValue(1)
        animation.start()
        
    def toggle_password(self):
        if self.show_password.isChecked():
            self.password_input.setEchoMode(QLineEdit.Normal)
            self.show_password.setIcon(QIcon("assets/icons/eye.png"))
        else:
            self.password_input.setEchoMode(QLineEdit.Password)
            self.show_password.setIcon(QIcon("assets/icons/eye-off.png"))

    def login(self):
        email = self.email_input.text()
        password = self.password_input.text()

        if not email or not password:
            msg = QMessageBox(self)
            msg.setWindowTitle("Input Error")
            msg.setText("Please enter both email and password.")
            msg.setIcon(QMessageBox.Warning)
            msg.exec_()
            return

        try:
            user = self.auth.sign_in_with_email_and_password(email, password)
            self.handle_login_success(user, remember=self.remember_checkbox.isChecked())
        except Exception as e:
            error_message = self.extract_firebase_error(e)
            msg = QMessageBox(self)
            msg.setWindowTitle("Login Failed")
            msg.setText(error_message)
            msg.setIcon(QMessageBox.Critical)
            msg.exec_()

    def handle_login_success(self, user, remember=False):
        user_id = user['localId']
        user_data = self.db.child("users").child(user_id).get().val() or {}

        user_info = {
            "email": user['email'],
            "localId": user['localId'],
            "displayName": user_data.get("name", "User"),
            "preferences": user_data.get("preferences", {})
        }

        if remember:
            with open("token.json", "w") as f:
                json.dump({"refreshToken": user['refreshToken']}, f)

        self.login_successful_signal.emit(user_info)

    def try_auto_login(self):
        if os.path.exists("token.json"):
            try:
                with open("token.json", "r") as f:
                    saved = json.load(f)
                refresh_token = saved.get("refreshToken")
                if refresh_token:
                    user = self.auth.refresh(refresh_token)
                    self.handle_login_success(user, remember=True)
                    self.auto_login_successful_signal.emit({
                        "email": user['userId'],
                        "localId": user['userId'],
                        "displayName": "AutoUser",
                        "preferences": {}
                    })
            except Exception as e:
                print("Auto login failed:", e)

    def extract_firebase_error(self, error):
        try:
            error_json = json.loads(error.args[1])
            message = error_json['error']['message'].replace('_', ' ').capitalize()
            return message
        except:
            return "An unexpected error occurred. Please try again."

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
                font-size: 15px;
                color: #e0e0e0;
            }

            #GlassFrame {
                background-color: rgba(30, 30, 30, 0.75);
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                padding: 30px;
            }

            #TitleLabel {
                font-size: 30px;
                font-weight: 600;
                color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #66aaff, stop:1 #3399ff);
            }

            #InputField {
                background-color: rgba(40, 40, 40, 0.95);
                border: 1px solid #555;
                border-radius: 14px;
                padding: 14px;
                font-size: 16px;
            }

            #InputField:focus {
                border: 1px solid #80b3ff;
                background-color: #2f2f2f;
            }
            QPushButton#ShowPasswordButton {
                background-color: #3c5f90;
                border-radius: 6px;
                padding: 4px;
            }
            QPushButton#ShowPasswordButton:hover {
                background-color: #80b3ff;
            }
            QPushButton#ShowPasswordButton:pressed {
                background-color: #2a3a4d;
            }

            QPushButton#LoginButton {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                stop:0 #80b3ff, stop:1 #66aaff);
                border: none;
                border-radius: 14px;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
                color: #121212;
            }

            QPushButton#LoginButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                stop:0 #99c2ff, stop:1 #80b3ff);
            }

            QPushButton#LoginButton:pressed {
                background-color: #4d88cc;
            }

            QPushButton#RegisterButton {
                background-color: transparent;
                border: 1px solid #80b3ff;
                border-radius: 12px;
                color: #80b3ff;
                font-size: 14px;
                padding: 10px;
            }

            QPushButton#RegisterButton:hover {
                background-color: rgba(128, 179, 255, 0.1);
            }

            #RememberCheckbox {
                font-size: 14px;
                color: #ccc;
            }

            #RememberCheckbox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                background-color: #444;
                border: 1px solid #888;
            }

            #RememberCheckbox::indicator:checked {
                background-color: #80b3ff;
                border: 1px solid #80b3ff;
            }

            QMessageBox {
                background-color: #1e1e1e;
                color: #e0e0e0;
                font-size: 14px;
                border: 1px solid #444;
            }

            QMessageBox QLabel {
                color: #e0e0e0;
                font-size: 15px;
                padding: 10px;
            }

            QMessageBox QPushButton {
                background-color: #80b3ff;
                border-radius: 10px;
                padding: 6px 12px;
                font-weight: bold;
                color: #121212;
                min-width: 80px;
            }

            QMessageBox QPushButton:hover {
                background-color: #99c2ff;
            }

            QMessageBox QPushButton:pressed {
                background-color: #4d88cc;
            }
        """)

        
