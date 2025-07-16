import pickle, glob, os, time
from typing import Optional

import numpy as np
import pandas as pd
import torch

from ase import Atoms
from ase.visualize import view

import streamlit as st
import streamlit_nested_layout

import altair as alt

from app_store.pages.tools.images.show_logo import show_logo
from app_store.pages.tools.app_utils import configure_canvas, initialize_session_state
from app_store.pages.tools.agent_selector import load_from_file, display_loaded_agents
from app_store.pages.tools.env_selector import configure_env, display_loaded_envs
from app_store.pages.tools.generate import generate_buttons
from app_store.pages.tools.analysis import display_results_tabs


def main():
    show_logo()
    configure_canvas()
    initialize_session_state()
    
    

    agent_col, dead_col1, mid_col, dead_col2, env_col = st.columns([12, 1, 20, 2, 14])


    with agent_col:
        agent_header_cols = st.columns([5, 10])
        with agent_header_cols[1]:
            st.header("Agents 🤖")
        load_from_file()
        with st.expander(r"$\textsf{\Large All agents}$"):
            display_loaded_agents()
        # st.write(f'session_state: {st.session_state}')

    with env_col:
        env_header_cols = st.columns([3, 10])
        with env_header_cols[1]:
            st.header("Environment 🌍")
        with st.expander(r"$\textsf{\Large Env Builder}$"):
            configure_env()
        with st.expander(r"$\textsf{\Large All envs}$"):
            display_loaded_envs()


    with mid_col:
        generator_header_cols = st.columns([5, 10])
        with generator_header_cols[1]:
            st.header("Generator 🚀")
        generate_buttons()
        st.session_state.pm.playgrounds_GUI()


    st.markdown("---") #############################################################################################
    display_results_tabs()
    # TODO: Idea: print results on the fly
      



    latent_space_viz = False # st.checkbox('Explore latent space')

    if latent_space_viz:
        df = pd.DataFrame(
            np.random.randn(200, 3),
            columns=['a', 'b', 'c'])

        c = alt.Chart(df).mark_circle().encode(
            x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])

        st.write(c)


if __name__ == "__main__":
    main()