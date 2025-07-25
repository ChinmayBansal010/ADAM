import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox, QSizePolicy
import pyrebase

from login_window import LoginWindow
from register_window import RegisterWindow
from dashboard_window import DashboardWindow
from firebase_config import FIREBASE_CONFIG
from jarvis_core import load_all_models_and_intents, start_jarvis_listening, stop_jarvis_listening


class JarvisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jarvis Personal Assistant")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QStackedWidget()
        self.central_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCentralWidget(self.central_widget)

        self.firebase_app = pyrebase.initialize_app(FIREBASE_CONFIG)
        self.auth = self.firebase_app.auth()
        self.db = self.firebase_app.database()

        self.model = None
        self.tags = None
        self.all_words = None
        self.device = None
        self.intents = None
        self.whisper_model = None

        self.init_ui()
        self.load_jarvis_core_components()

    def init_ui(self):
        self.login_window = LoginWindow(self.auth, self.db)
        self.register_window = RegisterWindow(self.auth, self.db)
        self.dashboard_window = DashboardWindow(self.auth, self.db)

        self.central_widget.addWidget(self.login_window)
        self.central_widget.addWidget(self.register_window)
        self.central_widget.addWidget(self.dashboard_window)

        self.login_window.show_register_signal.connect(self.show_register)
        self.register_window.show_login_signal.connect(self.show_login)

        self.login_window.login_successful_signal.connect(self.handle_login_success)
        self.login_window.auto_login_successful_signal.connect(self.handle_login_success)

        self.register_window.registration_successful_signal.connect(self.handle_registration_success)
        self.dashboard_window.logout_signal.connect(self.handle_logout)
        self.dashboard_window.start_jarvis_signal.connect(self.start_jarvis_from_dashboard)

        self.show_login()

    def load_jarvis_core_components(self):
        self.model, self.tags, self.all_words, self.device, self.intents, self.whisper_model = load_all_models_and_intents()
        if None in [self.model, self.tags, self.all_words, self.device, self.intents, self.whisper_model]:
            QMessageBox.critical(self, "Startup Error", "Failed to load Jarvis core components. Please check jarvis.log for details.")
            sys.exit(1)

    def show_login(self):
        self.setFixedSize(440, 500)  # smaller window for login
        self.central_widget.setCurrentWidget(self.login_window)
        self.center_window()

    def show_register(self):
        self.setFixedSize(410, 490)  # slightly larger for register
        self.central_widget.setCurrentWidget(self.register_window)
        self.center_window()

    def handle_login_success(self, user_info):
        QMessageBox.information(self, "Login Success", "Logged in as " + user_info['email'])
        self.dashboard_window.set_user_data(user_info)
        self.resize(1000, 700) 
        self.central_widget.setCurrentWidget(self.dashboard_window)
        self.center_window()

    def center_window(self):
        qr = self.frameGeometry()
        screen = QApplication.primaryScreen()
        cp = screen.availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def handle_registration_success(self, user_info):
        QMessageBox.information(self, "Registration Success", "Account created for " + user_info['email'])
        self.dashboard_window.set_user_data(user_info)
        self.central_widget.setCurrentWidget(self.dashboard_window)

    def start_jarvis_from_dashboard(self, user_info):
        if self.model and self.tags and self.all_words and self.device and self.intents and self.whisper_model:
            start_jarvis_listening(user_info, self.model, self.tags, self.all_words, self.device, self.intents, self.whisper_model)
            QMessageBox.information(self, "Jarvis Status", f"Jarvis is now active for {user_info.get('displayName', 'you')}!")
        else:
            QMessageBox.critical(self, "Jarvis Error", "Jarvis core components are not loaded. Cannot start listening.")

    def handle_logout(self):
        try:
            stop_jarvis_listening()
            self.auth.current_user = None
            # Clear saved token
            if os.path.exists("token.json"):
                os.remove("token.json")
            QMessageBox.information(self, "Logout", "Logged out successfully.")
            self.show_login()
        except Exception as e:
            QMessageBox.critical(self, "Logout Error", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setApplicationName("Jarvis")
    app.setOrganizationName("YourCompany")

    app.setStyleSheet("""
        QMainWindow, QStackedWidget {
            background-color: #1e1e1e;
            color: #e0e0e0;
            font-family: 'Arial', sans-serif;
            font-size: 15px;
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

    window = JarvisApp()
    window.show()
    sys.exit(app.exec_())
