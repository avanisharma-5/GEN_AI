# -*- coding: utf-8 -*-
"""Backend for Multimodal Insurance: RAG, Gemini narrative, Stable Diffusion visual."""

import io
import os
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import torch
from diffusers import StableDiffusionPipeline

# Lazy-loaded singletons
_embed_model = None
_sd_pipeline = None


def sd_diagnostics():
    info = {
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
    }
    if info["cuda_available"]:
        try:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            info["cuda_device_name"] = None
    else:
        info["cuda_device_name"] = None
    return info


def get_embedding_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def get_sd_pipeline():
    global _sd_pipeline
    if _sd_pipeline is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        if device == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass

        _sd_pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)

        # Memory/perf tweaks (especially useful on 4–6GB VRAM GPUs like RTX 3050)
        try:
            _sd_pipeline.enable_attention_slicing()
        except Exception:
            pass
        try:
            _sd_pipeline.enable_vae_slicing()
        except Exception:
            pass
    return _sd_pipeline


def chunk_text(text, chunk_size=800):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i : i + chunk_size])
    return chunks


def build_vector_db(pdf_file):
    """Build FAISS index from uploaded PDF (file-like or path). Returns (index, documents)."""
    if hasattr(pdf_file, "read"):
        reader = PdfReader(io.BytesIO(pdf_file.read()))
    else:
        reader = PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            full_text += t + "\n"
    if not full_text.strip():
        return None, []
    documents = chunk_text(full_text)
    embed_model = get_embedding_model()
    doc_embeddings = embed_model.encode(documents)
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(doc_embeddings).astype("float32"))
    return index, documents


def retrieve_context(query, index, documents, embed_model=None, top_k=3):
    if embed_model is None:
        embed_model = get_embedding_model()
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype("float32"), top_k)
    return [documents[i] for i in indices[0]]


def is_insurance_query(query):
    insurance_keywords = [
        "insurance", "premium", "claim", "coverage",
        "policy", "risk", "flood", "health",
        "motor", "life", "accident", "IRDAI",
    ]
    q = query.lower()
    return any(kw.lower() in q for kw in insurance_keywords)


def _get_gemini_api_key():
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    # Load from config_local.py in same folder as this file (works regardless of cwd)
    _dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(_dir, "config_local.py")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY") and "=" in line:
                    val = line.split("=", 1)[1].strip().strip('"\'')
                    if val:
                        return val
    return None


def generate_insurance_narrative(user_query, context_text=""):
    api_key = _get_gemini_api_key()
    if not api_key:
        return "[Gemini API key not set. Add GEMINI_API_KEY in config_local.py or set env GEMINI_API_KEY.]"
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""You are an insurance expert.

Context:
{context_text or "Use general insurance knowledge."}

User Question:
{user_query}

Provide structured explanation including:
- Risks
- Coverage
- Cost Impact
- Prevention
"""
    try:
        response = model.generate_content(prompt)
        if response.candidates and getattr(response.candidates[0], "content", None) and response.candidates[0].content.parts:
            return response.text
        reason = getattr(response.candidates[0], "finish_reason", None) if response.candidates else None
        return f"[Response blocked or empty. finish_reason={reason}]"
    except Exception as e:
        return f"[Gemini API error: {e}]"


def generate_insurance_visual(user_query, pipe=None):
    if pipe is None:
        pipe = get_sd_pipeline()
    # Stable Diffusion struggles with readable text. Prefer icon-only infographics.
    image_prompt = f"""
    flat vector infographic, icon-only, clean layout, crisp edges, high contrast,
    simple insurance icons, minimal clutter, modern UI illustration,
    topic: {user_query},
    no words, no letters, no sentences
    """.strip()

    negative_prompt = """
    text, words, letters, watermark, logo, signature,
    blurry, lowres, noisy, jpeg artifacts, cluttered, illegible
    """.strip()

    image = pipe(
        prompt=image_prompt,
        negative_prompt=negative_prompt,
        height=512,
        width=768,
        num_inference_steps=20,
        guidance_scale=6.5,
    ).images[0]
    return image
