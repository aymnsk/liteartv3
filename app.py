import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import streamlit as st
from PIL import Image
from style_transfer import run_style_transfer
from utils import apply_dreamy_effects

st.set_page_config(page_title="LiteArt Studio", layout="centered")

st.title("🎨 LiteArt Studio - Neural Style Transfer")

content_img = st.file_uploader("📤 Upload Content Image", type=["jpg", "jpeg", "png"])
style_img = st.file_uploader("🎨 Upload Style Image", type=["jpg", "jpeg", "png"])
effect = st.selectbox("✨ Choose Final Effect Filter", ["none", "lucid", "lsd", "dreamy"])

if content_img and style_img:
    st.image(content_img, caption="🖼 Content Image", width=300)
    st.image(style_img, caption="🎨 Style Image", width=300)

    if st.button("🚀 Stylize Image"):
        with st.spinner("Working... Please wait..."):
            output_path = run_style_transfer(content_img, style_img, effect_mode=effect)
            st.success("✅ Stylization Complete!")
            st.image(output_path, caption="🌈 Final Stylized Image", use_column_width=True)
            with open(output_path, "rb") as f:
                st.download_button("📥 Download Stylized Image", f, "final_output.png")
