# 📊 Company Peer Detection via Conference Call Transcripts

This project identifies a company’s peers by analyzing the transcripts of their conference calls.  
The underlying idea is that peer firms often discuss similar themes, which can be measured using **cosine similarity** applied to transcript embeddings.  

The implementation leverages both **OpenAI models** and **open-source alternatives from Hugging Face** for embedding generation.  

---

## 🚀 Features
- Extracts text from **PDF conference call transcripts**  
- Focuses on the **Q&A section** for higher relevance  
- Generates embeddings using `text-embedding-3-large` (OpenAI)  
- Computes **pairwise company similarity** via Top-K Symmetric MaxSim  
- Outputs each company’s **closest peer** based on transcript similarity  

---

## 📂 Project Structure
- **main.py** → core script (PDF parsing, embeddings, similarity computation)  
- **requirements.txt** → dependencies list  
- **.env** → stores API key (not included in repo; add it locally)  

---

## 🛠️ Requirements
Install dependencies with pip:

```bash
pip install -r requirements.txt
