# -*- coding: utf-8 -*-


pip install -q sentence-transformers faiss-cpu pypdf google-generativeai torch

import faiss
import numpy as np
import matplotlib.pyplot as plt
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from google import genai as genai_new
import torch
from google.colab import files
from diffusers import StableDiffusionPipeline

from google.colab import files
uploaded = files.upload()

pdf_path = list(uploaded.keys())[0]
reader = PdfReader(pdf_path)

full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

print("PDF Loaded Successfully ✅")

def chunk_text(text, chunk_size=800):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

documents = chunk_text(full_text)
print("Total Chunks:", len(documents))

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = embed_model.encode(documents)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

print("Vector DB Ready ✅")

def retrieve_context(query, top_k=3):
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [documents[i] for i in indices[0]]

api_key = input("Enter your Gemini API Key: ").strip()

# Text model setup
genai.configure(api_key=api_key)
text_model = genai.GenerativeModel("gemini-2.5-flash")

print("Loading Stable Diffusion v1.5...")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

print("Stable Diffusion Loaded ✅")

def is_insurance_query(query):
    insurance_keywords = [
        "insurance", "premium", "claim", "coverage",
        "policy", "risk", "flood", "health",
        "motor", "life", "accident", "IRDAI"
    ]

    for word in insurance_keywords:
        if word.lower() in query.lower():
            return True
    return False

def generate_insurance_visual(user_query):

    image_prompt = f"""
    insurance relevant,
    flat vector illustration,
    clean layout,
    minimal readable text,
    topic: {user_query}
    """

    image = pipe(
        image_prompt,
        height=512,
        width=768,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    return image

while True:
    user_query = input("\nEnter your query (type 'exit' to stop): ")

    if user_query.lower() == "exit":
        break

    if not is_insurance_query(user_query):
        print("\nPlease ask only queries related to insurance.")
        continue

    # Retrieve Context
    context = retrieve_context(user_query)
    context_text = "\n".join(context)

    # Generate Text Response
    prompt = f"""
    You are an insurance expert.

    Context:
    {context_text}

    If context is insufficient, use general insurance knowledge.

    User Question:
    {user_query}

    Provide structured explanation including:
    - Risks
    - Coverage
    - Cost Impact
    - Prevention
    """

    response = text_model.generate_content(prompt)

    print("\n=========== INSURANCE RESPONSE ===========\n")
    print(response.text)

    # Generate Image
    print("\nGenerating Visual...\n")
    image = generate_insurance_visual(user_query)

    if image:
       plt.figure(figsize=(8,6))
       plt.imshow(image)
       plt.axis("off")
       plt.show()

       image.save("insurance_output.png")
       print("Image Saved as insurance_output.png ✅")
    else:
        print("Image generation failed.")