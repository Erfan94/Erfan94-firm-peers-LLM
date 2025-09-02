import os
from tkinter import Tk, filedialog
import PyPDF2
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
import re

# ─── Load environment variables ──────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ─── Select PDF files ────────────────────────────────────────────────────────
print("Select Conference Transcripts")
Tk().withdraw()
file_paths = filedialog.askopenfilenames(
    title='Select PDF Transcripts',
    filetypes=[('PDF files', '*.pdf')]
)
if not file_paths:
    print('No files selected!')
    exit()

# ─── Extract text from PDFs ──────────────────────────────────────────────────
company_transcripts = {}
for path in file_paths:
    company_name = os.path.splitext(os.path.basename(path))[0]
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        full_text = "\n".join(text_parts).strip()

        # Keep only Q&A section
        match = re.search(r"(Q&A|question[- ]and[- ]answer)", full_text, re.IGNORECASE)
        if match:
            qna_start = match.start()
            full_text = full_text[qna_start:]
        else:
            print(f"⚠️ No Q&A section found in {company_name}, keeping full transcript.")
        company_transcripts[company_name] = full_text

# ─── Tokenize and chunk using tiktoken ───────────────────────────────────────
encoding = tiktoken.encoding_for_model("text-embedding-3-large")
max_tokens = 8192  # model token limit per embedding call

def chunk_text(text, max_tokens):
    # Simple token-budgeted chunker
    words = text.split()
    chunks, current_chunk, current_tokens = [], [], 0
    for word in words:
        token_count = len(encoding.encode(word))
        if current_tokens + token_count > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_tokens = token_count
        else:
            current_chunk.append(word)
            current_tokens += token_count
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# ─── Generate embeddings for each chunk ──────────────────────────────────────
model_id = "text-embedding-3-large"
company_chunk_embeddings = {}
company_names = list(company_transcripts.keys())

for name in tqdm(company_names, desc="Generating chunk embeddings"):
    text = company_transcripts[name]
    chunks = chunk_text(text, max_tokens=max_tokens)
    chunk_vectors = []
    for chunk in chunks:
        response = client.embeddings.create(
            model=model_id,
            input=chunk
        )
        chunk_vectors.append(response.data[0].embedding)
    company_chunk_embeddings[name] = np.array(chunk_vectors, dtype=np.float32)

# ─── Top-K MaxSim (Solution 1) helper ────────────────────────────────────────
def topk_maxsim_symmetric(chunks_a: np.ndarray, chunks_b: np.ndarray, k: int = 10) -> float:
    if chunks_a.size == 0 or chunks_b.size == 0:
        return 0.0
    S = cosine_similarity(chunks_a, chunks_b)
    a2b = S.max(axis=1)
    b2a = S.max(axis=0)
    K = max(1, min(k, len(a2b), len(b2a)))
    a2b_topk_mean = np.sort(a2b)[-K:].mean()
    b2a_topk_mean = np.sort(b2a)[-K:].mean()
    return float((a2b_topk_mean + b2a_topk_mean) / 2.0)

# ─── Compute similarity matrix ───────────────────────────────────────────────
K_TOP = 10
similarity_matrix = pd.DataFrame(index=company_names, columns=company_names, dtype=float)

for i, name_a in enumerate(company_names):
    for j, name_b in enumerate(company_names):
        if i == j:
            similarity_matrix.at[name_a, name_b] = 1.0
            continue
        if pd.notnull(similarity_matrix.at[name_a, name_b]):
            continue
        chunks_a = company_chunk_embeddings[name_a]
        chunks_b = company_chunk_embeddings[name_b]
        score = topk_maxsim_symmetric(chunks_a, chunks_b, k=K_TOP)
        similarity_matrix.at[name_a, name_b] = score
        similarity_matrix.at[name_b, name_a] = score

# ─── Print Closest Peers ─────────────────────────────────────────────────────
print("\nClosest Peers (Top-K Symmetric MaxSim, Q&A only):\n")
for name in company_names:
    sims = similarity_matrix.loc[name].sort_values(ascending=False)
    if len(sims) > 1:
        top_peer = sims.index[1]
        print(f"{name} → {top_peer} (Similarity: {sims.iloc[1]:.4f})")
    else:
        print(f"{name} → (no peers)")
