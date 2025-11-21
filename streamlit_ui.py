import os
import streamlit as st
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from embedding_index import (
    load_clap,
    load_index,
    build_index,
    search_index,
    safe_load_audio,
    AUDIO_FOLDER
)
from zs_clap import generate_zero_shot_labels


def main():
    st.set_page_config(page_title="CLAP + FAISS Search", layout="wide")
    st.title("🎧 Multimodal Audio Search (CLAP + FAISS)")
    st.markdown("Search loops using **Text**, **Audio**, and **Zero-Shot Prediction**")

    processor, model, device = load_clap()

    # Sidebar index
    st.sidebar.subheader("Index Controls")
    if st.sidebar.button("📌 Rebuild Index"):
        with st.spinner("Building FAISS Index..."):
            build_index()
        st.sidebar.success("Index Created! 🎉")

    index, meta = load_index()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Text → Audio", "🎵 Audio → Audio", "🧠 Zero Shot"])

    # TAB 1 — Text Search
    with tab1:
        query = st.text_input("Text Query:", placeholder="drum beats loop")
        if st.button("Search"):
            if index is None:
                st.error("⚠️ Build index first!")
                st.stop()

            inputs = processor(text=[query], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                q_emb = model.get_text_features(**inputs)
                q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=1).cpu().numpy()[0]

            ids, scores = search_index(index, q_emb)

            for i, s in zip(ids, scores):
                row = meta.iloc[i]
                st.write(f"🎧 {row['filename']} — score {s:.3f}")
                st.audio(os.path.join(AUDIO_FOLDER, row['filename']))

    # TAB 2 — Audio Similarity
    with tab2:
        up = st.file_uploader("Upload Audio (.wav)", type=["wav"], key="audio_sim")
        if up and st.button("Find Similar"):
            if index is None:
                st.error("⚠️ Build index first!")
                st.stop()

            y, sr = safe_load_audio(up)
            inputs = processor(audios=[y], sampling_rate=sr, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                emb = model.get_audio_features(**inputs)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1).cpu().numpy()[0]

            ids, scores = search_index(index, emb, top_k=5)

            for i, s in zip(ids, scores):
                row = meta.iloc[i]
                st.write(f"🎧 {row['filename']} — score {s:.3f}")
                st.audio(os.path.join(AUDIO_FOLDER, row['filename']))

    # TAB 3 — Zero-Shot Labeling
    with tab3:
        up2 = st.file_uploader("Upload Audio for Labeling (.wav)", type=["wav"], key="zshot")
        if up2 and st.button("Predict"):
            st.audio(up2)
            label = generate_zero_shot_labels(up2)
            st.success(f"🎯 Predicted Label: **{label}**")

    st.caption("❤️ Powered by CLAP + FAISS + Streamlit")


if __name__ == "__main__":
    main()
