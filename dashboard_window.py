from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFormLayout,
    QLineEdit, QCheckBox, QMessageBox
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont
import pyrebase


class DashboardWindow(QWidget):
    logout_signal = pyqtSignal()
    start_jarvis_signal = pyqtSignal(dict)

    def __init__(self, auth, db):
        super().__init__()
        self.auth = auth
        self.db = db
        self.user_info = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Jarvis Dashboard")
        self.setMinimumSize(600, 500)

        # Main container layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)

        # Glass panel container
        glass_panel = QWidget()
        glass_panel_layout = QVBoxLayout()
        glass_panel_layout.setSpacing(20)
        glass_panel_layout.setContentsMargins(30, 30, 30, 30)
        glass_panel.setLayout(glass_panel_layout)

        # Welcome text
        self.welcome_label = QLabel(f"Welcome!")
        self.welcome_label.setAlignment(Qt.AlignCenter)
        self.welcome_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #ffffff;")
        glass_panel_layout.addWidget(self.welcome_label)

        # Form fields
        form_layout = QFormLayout()
        form_layout.setSpacing(15)

        self.name_value = QLineEdit()
        self.name_value.setReadOnly(True)

        self.email_value = QLineEdit()
        self.email_value.setReadOnly(True)

        self.voice_input = QLineEdit()

        self.wakeword_checkbox = QCheckBox("Enable Wake Word")

        form_layout.addRow("Name:", self.name_value)
        form_layout.addRow("Email:", self.email_value)
        form_layout.addRow("Preferred Voice:", self.voice_input)
        form_layout.addRow("", self.wakeword_checkbox)

        glass_panel_layout.addLayout(form_layout)

        # Buttons
        button_layout = QVBoxLayout()
        button_layout.setSpacing(15)

        save_preferences_button = QPushButton("💾 Save Preferences")
        save_preferences_button.clicked.connect(self.save_preferences)

        start_jarvis_button = QPushButton("🎤 Start Jarvis (Listen Mode)")
        start_jarvis_button.clicked.connect(self.start_jarvis)

        logout_button = QPushButton("🔒 Logout")
        logout_button.clicked.connect(self.logout_signal.emit)

        for btn in [save_preferences_button, start_jarvis_button, logout_button]:
            btn.setMinimumHeight(40)
            button_layout.addWidget(btn)

        glass_panel_layout.addLayout(button_layout)
        main_layout.addWidget(glass_panel)

        # Apply glassmorphism stylesheet
        self.setStyleSheet("""
            QWidget {
                background-color: #0f0f0f;
                font-family: 'Segoe UI', sans-serif;
                color: #ffffff;
            }

            QLabel {
                font-size: 14px;
            }

            QLineEdit {
                background-color: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
                padding: 10px;
                color: #ffffff;
            }

            QCheckBox {
                padding: 10px;
                font-size: 14px;
            }

            QPushButton {
                background: rgba(59, 130, 246, 0.8);
                border: none;
                border-radius: 12px;
                padding: 10px;
                font-size: 15px;
                color: white;
                font-weight: bold;
            }

            QPushButton:hover {
                background: rgba(37, 99, 235, 0.9);
            }

            QPushButton:pressed {
                background: rgba(29, 78, 216, 1);
            }

            QWidget#glass_panel {
                background-color: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 20px;
            }
        """)

        # Set object name for special glass panel styling
        glass_panel.setObjectName("glass_panel")


    def set_user_data(self, user_info):
        self.user_info = user_info

        try:
            user_id = user_info.get("localId")
            user_data = self.db.child("users").child(user_id).get().val()

            name = user_data.get("name") if user_data else None
            if not name:
                name = user_info.get("displayName") or user_info.get("name", "User")

            email = user_info.get("email", "N/A")

            self.welcome_label.setText(f"Welcome, {name}")
            self.name_value.setText(name)
            self.email_value.setText(email)

            preferences = user_data.get("preferences", {}) if user_data else {}
            self.voice_input.setText(preferences.get("voice", "en-IN-PrabhatNeural"))
            self.wakeword_checkbox.setChecked(preferences.get("wake_word_enabled", True))

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Profile",
                f"Something went wrong while fetching your data.\n\nDetails:\n{str(e)}"
            )
            self.welcome_label.setText("Welcome")

    def show_message(self, title, message, icon_type="info"):
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(message)

        if icon_type == "info":
            msg.setIcon(QMessageBox.Information)
        elif icon_type == "warning":
            msg.setIcon(QMessageBox.Warning)
        elif icon_type == "error":
            msg.setIcon(QMessageBox.Critical)

        msg.setStyleSheet("""
            QMessageBox {
                background-color: #1e1e1e;
                color: #e0e0e0;
                font-size: 14px;
            }
            QMessageBox QPushButton {
                background-color: #80b3ff;
                border-radius: 6px;
                padding: 6px 12px;
                color: #121212;
                font-weight: bold;
            }
            QMessageBox QPushButton:hover {
                background-color: #99c2ff;
            }
            QMessageBox QPushButton:pressed {
                background-color: #4d88cc;
            }
        """)
        msg.exec_()

    def save_preferences(self):
        if not self.user_info:
            self.show_message("Error", "No user logged in to save preferences.", "warning")
            return

        user_id = self.user_info['localId']
        new_preferences = {
            "voice": self.voice_input.text(),
            "wake_word_enabled": self.wakeword_checkbox.isChecked()
        }

        try:
            self.db.child("users").child(user_id).child("preferences").update(new_preferences)
            self.user_info['preferences'] = new_preferences
            self.show_message("Success", "Preferences saved!", "info")
        except Exception as e:
            self.show_message("Save Error", str(e), "error")

    def start_jarvis(self):
        if self.user_info:
            self.start_jarvis_signal.emit(self.user_info)
        else:
            self.show_message("Error", "No user logged in to start Jarvis.", "warning")
