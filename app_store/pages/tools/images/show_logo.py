import os
from streamlit_extras.app_logo import add_logo


def show_logo():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    image_path = os.path.join(project_root, "resources", "atomcomposer.png")
    add_logo(logo_url=image_path, height=230)
