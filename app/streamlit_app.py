# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 10:42:21 2025

@author: Erfan
"""

# app/streamlit_app.py
import pandas as pd
import streamlit as st
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Company Peer Detection", layout="wide")
st.title("📊 Company Peer Detection")


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "outputs"


qna_csv = OUT_DIR / "similarity_matrix_qna.csv"
prep_csv = OUT_DIR / "similarity_matrix_prepared.csv"
csv_path = qna_csv if qna_csv.exists() else prep_csv

if not csv_path.exists():
    st.error(
        "No similarity matrix found.\n\n"
        "Please run the pipeline first:\n"
        "```bash\npython main.py\n```"
    )
    st.stop()

@st.cache_data(show_spinner=False)
def load_similarity(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    # Ensure numeric and symmetric (defensive)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0.0)
    # Clip range to [-1, 1] for cosine-like values
    df = df.clip(lower=-1.0, upper=1.0)
    return df

sim = load_similarity(csv_path)
companies = sim.index.tolist()

# ---------- Peer table ----------
left, right = st.columns([1, 2])

with left:
    sel = st.selectbox("Select a company", companies)
    top_k = st.slider("Number of peers to show", 1, min(15, len(companies) - 1), 5)
    peers = (
        sim.loc[sel]
        .sort_values(ascending=False)
        .drop(labels=[sel], errors="ignore")
        .head(top_k)
    )
    st.subheader(f"Top {top_k} peers for **{sel}**")
    st.dataframe(peers.to_frame("Similarity"))

with right:
    st.caption(f"Loaded matrix: `{csv_path.relative_to(REPO_ROOT)}`")
    st.write("")


st.markdown("---")
st.subheader("🔥 Similarity Heatmap")

st.caption(
    "Tip: If you have many companies, select a subset below for readability."
)


default_n = min(25, len(companies))
n_show = st.slider("Number of companies to display", 5, len(companies), default_n)


row_strength = sim.sum(axis=1).sort_values(ascending=False)
subset_names = row_strength.head(n_show).index.tolist()
sub = sim.loc[subset_names, subset_names]


try:
    u, s, vt = np.linalg.svd(sub.values - np.mean(sub.values))
    order = np.argsort(u[:, 0])
    sub = sub.iloc[order, :].iloc[:, order]
except Exception:
    pass  

# Plot 
fig, ax = plt.subplots(figsize=(min(12, 0.45 * n_show + 4), min(10, 0.45 * n_show + 3)), dpi=120)
im = ax.imshow(sub.values, aspect="auto")
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Cosine Similarity", rotation=270, labelpad=12)

ax.set_xticks(range(len(sub.columns)))
ax.set_yticks(range(len(sub.index)))
ax.set_xticklabels(sub.columns, rotation=90, fontsize=8)
ax.set_yticklabels(sub.index, fontsize=8)

ax.set_xlabel("Peer")
ax.set_ylabel("Company")
ax.set_title("Company–Company Similarity (subset)")

st.pyplot(fig, clear_figure=True)

