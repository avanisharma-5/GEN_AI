"""
Streamlit UI for Multimodal Insurance Concept Visualizer.
Uses insurance_backend for RAG, Gemini narrative, and Stable Diffusion visual.
"""

import streamlit as st
from insurance_backend import (
    build_vector_db,
    retrieve_context,
    generate_insurance_narrative,
    generate_insurance_visual,
    is_insurance_query,
    get_embedding_model,
    get_sd_pipeline,
    sd_diagnostics,
)

st.set_page_config(page_title="Multimodal Insurance Creator", layout="wide")
st.title("🛡️ Multimodal Insurance Concept Visualizer")

# ---------------- System status ----------------
with st.sidebar:
    st.subheader("⚙️ System")
    sd_info = sd_diagnostics()
    if sd_info.get("cuda_available"):
        st.success(f"CUDA detected: {sd_info.get('cuda_device_name')}")
    else:
        st.warning("CUDA not detected. Image generation will be very slow (CPU).")
    st.caption(f"Torch: {sd_info.get('torch_version')}")

# ---------------- Inputs ----------------
uploaded_pdf = st.file_uploader("Upload Insurance PDF (optional – for RAG context)", type=["pdf"])
user_query = st.text_input("Enter Insurance Query", placeholder="e.g. What are the risks in health insurance?")

# ---------------- Generate Button ----------------
if st.button("Generate Analysis & Visual"):
    if not user_query or not user_query.strip():
        st.error("Please enter a query.")
    elif not is_insurance_query(user_query):
        st.warning("Please ask insurance-related queries only (e.g. insurance, premium, claim, coverage, risk, health, motor, life).")
    else:
        context_text = ""
        try:
            if uploaded_pdf:
                with st.spinner("Building knowledge base from PDF..."):
                    embed_model = get_embedding_model()
                    index, documents = build_vector_db(uploaded_pdf)
                    if documents:
                        with st.spinner("Retrieving context..."):
                            context_list = retrieve_context(user_query, index, documents)
                            context_text = "\n".join(context_list)
                    else:
                        st.info("PDF had no extractable text. Using general insurance knowledge.")
            else:
                st.info("No PDF uploaded. Using general insurance knowledge.")
        except Exception as e:
            st.error(f"PDF / RAG error: {e}")
            context_text = ""

        if context_text or not uploaded_pdf:
            with st.spinner("Generating insurance analysis..."):
                try:
                    text_output = generate_insurance_narrative(user_query, context_text)
                except Exception as e:
                    text_output = f"[Error: {e}]"

            st.subheader("📊 Insurance Analysis")
            st.write(text_output)

            with st.spinner("Generating visual infographic..."):
                try:
                    pipe = get_sd_pipeline()
                    image = generate_insurance_visual(user_query, pipe)
                    st.subheader("🖼️ Generated Insurance Visual")
                    st.image(image, use_container_width=True)
                except Exception as e:
                    st.error(f"Image generation failed: {e}")
                    st.caption("Ensure you have a GPU and enough VRAM for Stable Diffusion, or run on a machine with CUDA.")
