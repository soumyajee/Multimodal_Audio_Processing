🎧 Multimodal Audio Retrieval using CLAP + FAISS + Streamlit

A Text-to-Audio Retrieval System using Contrastive Language-Audio Pretraining (CLAP) that indexes audio samples (Drums, Keys) and enables prompt-based search. The embeddings are stored in a FAISS vector database, making similarity-based retrieval fast and scalable.
📌 Objective

Build a system that:

✔ Extracts audio embeddings using CLAP
✔ Supports Text → Audio and Audio → Audio similarity search
✔ Stores embeddings in FAISS Vector DB
✔ Generates labels using Zero-Shot Text prompts
✔ Provides a Streamlit UI for retrieval
✔ Evaluates performance using Confusion Matrix + Classification Report
📂 Dataset

Source: loperman.com
Classes:

🥁 Drums

🎹 Keys

Class	Samples Count
Drums	20
Keys	20
Data/
   drums/
   keys/
metadata_simple.csv

| Model          | CLAP (laion/clap-htsat-unfused)          |
| -------------- | ---------------------------------------- |
| Embedding Type | Joint Text & Audio Embeddings            |
| Trained For    | Audio classification, retrieval, tagging |
| Frameworks     | PyTorch + Transformers                   |

     Audio Files  ──► CLAP Audio Encoder ──► Embeddings ┐
                                                         │
                                                         ▼
            Text Query ─► CLAP Text Encoder ─► Vector Similarity ─► Top-K Audio Results
                                                         │
                                                         ▼
                                                  FAISS Vector DB

| Feature                          | Status |
| -------------------------------- | ------ |
| Audio Embeddings + FAISS Index   | ✅      |
| Streamlit UI for Retrieval       | ✅      |
| Zero-shot Labeling using CLAP    | ✅      |
| Audio→Audio Similarity Search    | ✅      |
| Evaluation with Confusion Matrix | ✅      |
| Save CM + Report as PNG/TXT      | ✅      |

git clone <your_repo_url>
cd multimodal-clap-faiss
pip install -r requirements.txt

▶️ Usage
🏗 Build embeddings & FAISS index
output/
 ├ faiss_index.bin
 ├ metadata.pkl
 ├ confusion_matrix.png
 ├ classification_report.txt

🖥 Run Streamlit Application
streamlit run streamlit_ui.py
User can now:

✔ Upload audio
✔ Enter text prompts e.g. "play drums"
✔ Retrieve top-K relevant samples
📊 Evaluation

Confusion matrix stored at:
output/confusion_matrix.png

Example screenshot 👇
(Insert your generated CM image here)

Also saved:

classification_report.txt

🔍 Zero-Shot Prompt Examples

| Prompt                   | Expected Retrieval |
| ------------------------ | ------------------ |
| “drum beats”             | Drums samples      |
| “beautiful piano chords” | Keys samples       |

📌 Directory Structure
.
├ Data/
├ output/
├ metadata_simple.csv
├ embedding_index.py
├ streamlit_ui.py
├ zs_clap.py
├ requirements.txt
└ README.md

📌 Future Enhancements

Add more instrument classes 🎻 🎷 🎸

Apply quantization for large FAISS DB

Deploy Streamlit to cloud (Render / Hugging Face Spaces)

🏁 Conclusion

This project demonstrates:

✔ Multimodal learning
✔ FAISS-based fast retrieval
✔ Prompt-based semantic search
✔ Zero-shot classification capability of CLAP