# Web App

A Streamlit-based interface for interactive molecule generation with trained RL agents.

## Launch

Run from the project root (with `rl-env` activated):

```bash
streamlit run app_store/About.py
```

## Pages

### About
The landing page. Displays the paper citation and an overview of generated molecule examples.

### Generator
The main inference page. Laid out in three columns:

- **Agents 🤖** (left): Load and manage agent checkpoints. Five agents from the paper (A, AV, F, FV, AFV) are pre-loaded from `model_objects/` on startup (seed 0).
- **Generator 🚀** (center): Create playgrounds, deploy agent–environment pairs, and run molecule generation.
- **Environment 🌍** (right): Configure the chemical formula (bag) to generate molecules for. Five formulas from the paper are pre-loaded: C3H8O, C4H7N, C3H5NO3, C7H10O2, C7H8N2O2.

**Typical workflow:**
1. In the center column, click **New Playground** to create a new playground in edit mode.
2. In the left column, find the desired agent under **All agents** and click **To Playgrounds**.
3. In the right column, build or select an environment under **Env Builder** / **All envs** and assign it to the playground.
4. Back in the center column, click **Deploy** to lock in the agent–environment pair.
5. Click **Generate** (stochastic or argmax) and set the number of samples. Start small — generation runs on CPU by default.
6. Results (SMILES, energies, validity) appear in the tabs below.

**Loading custom checkpoints:**
Use the **Agent Loader** expander in the left column. Either upload a file or paste a file path directly (e.g. `runs/Agent-A/seed_1/_steps-50000.model`). Select the device (`cpu` or `cuda`) and give the agent a name before clicking **Load agent**. Additional seeds are available under `model_objects/`.

### Explore Datasets
Browse the QM7, QM9, or tmQM reference datasets. Filter molecules by atom count, element presence, or chemical formula. For any selected bag, view individual conformers, inspect energetics, and explore the atom-placement decomposition schemes used during training.
