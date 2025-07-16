import time
from typing import Callable, List, Tuple, Dict

import pandas as pd
import streamlit as st

from app_store.pages.tools.playground import PlaygroundManager
from src.performance.metrics import MoleculeProcessor


def configure_canvas():
    st.write("""
    <style>
    .custom-remove-btn {
        height: 20px;
        line-height: 20px;
        font-size: 12px;
        padding: 0 8px;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    if 'loaded_models' not in st.session_state:
        st.session_state.loaded_models = {}
    if "loaded_envs" not in st.session_state:
        st.session_state.loaded_envs = {}
    if "pm" not in st.session_state:
        st.session_state.pm = PlaygroundManager()
    if "mols" not in st.session_state:
        st.session_state.mols = {}
    if "index_in_explore_mols" not in st.session_state:
        st.session_state.index_in_explore_mols = None
    if 'loaded_mols' not in st.session_state:
        st.session_state.loaded_mols = pd.DataFrame()
    if 'datasets' not in st.session_state:
        st.session_state.datasets = {}
    if 'mol_processor' not in st.session_state:
        st.session_state.mol_processor = MoleculeProcessor()


def make_multi_button_columns(
    df: pd.DataFrame,
    button_configs: List[Tuple[str, Callable, Dict]],
    unique_key: str,
):
    """Creates multiple button columns in a dataframe that call functions when clicked."""
    # TODO: Investigate why the viewer pops up again when we rerun the app
    
    button_dict_key = f'{unique_key}_button_dict'

    if button_dict_key not in st.session_state:
        setattr(st.session_state, button_dict_key, {})

    button_dict = getattr(st.session_state, button_dict_key)

    # Create buttons or reload existing button values
    for (button_name, func, _) in button_configs:
        df[button_name] = False
        if button_name not in button_dict.keys():
            button_dict[button_name] = {
                'all_indices': set(),
                'new_index': None,
                'old_indices': set()
            }
        #else:
        #    st.write(f"button_dict[button_name]['all_indices']: {button_dict[button_name]['all_indices']}")
        #    df[button_name].iloc[list(button_dict[button_name]['all_indices'])] = True


    # Create data_editor
    editable_cols = [button_name for (button_name, _, _) in button_configs]
    non_editable_cols = [col for col in df.columns if col not in editable_cols]
    df = st.data_editor(df, key=unique_key, disabled=non_editable_cols)


    # Run functions upon click and ipdate button values in session state 
    do_rerun = False
    for (button_name, func, args_dict) in button_configs:
        button_info = button_dict[button_name]

        button_info['all_indices'] = set(df[df[button_name] == True].index)
        new_index = button_info['all_indices'] - button_info['old_indices']
        newly_removed = button_info['old_indices'] - button_info['all_indices']

        if len(newly_removed) > 0 or len(new_index) > 0:
            do_rerun = True

        if len(new_index) == 1:
            new_index = list(new_index)[0]
            func(new_index, **args_dict)

        st.session_state[button_dict_key][button_name]['old_indices'] = button_info['all_indices'].copy()



    #if do_rerun:
    #    st.rerun() # Attempting to avoid ASE viewer unintendedly popping up again

    # for (button_name, func, args_dict) in button_configs:
    #     button_info = button_dict[button_name]
    #     st.write(f"{button_name}: {button_info['all_indices']}")


def is_hashable(obj):
    try:
        hash(obj)
        return True
    except TypeError:
        return False


def check_hashability(obj, path=None):
    if path is None:
        path = []

    if is_hashable(obj):
        print(f"{'.'.join(path)} is hashable")
    else:
        print(f"{'.'.join(path)} is not hashable")

    if isinstance(obj, dict):
        for key, value in obj.items():
            check_hashability(value, path + [str(key)])
    elif isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            check_hashability(item, path + [str(index)])
    elif hasattr(obj, '__dict__'):
        for attr_name, attr_value in obj.__dict__.items():
            check_hashability(attr_value, path + [attr_name])
