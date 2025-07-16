
import os
from PIL import Image
import streamlit as st
from streamlit_extras.app_logo import add_logo


def show_logo():

    image_name = 'H8C7N2O2_0-pixel_rot.png'
    
    file_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(file_dir, image_name)

    #st.write(image_path)
    #st.image(Image.open(image_path), use_column_width=False)

    add_logo(logo_url=image_path, height=230)
    # st.logo(image_path, size='large')