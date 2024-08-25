import io
import streamlit as st
from PIL import Image

from src.resnet import ResNet
from src.vit import VisionTransformer, CustomViTForImageClassification
from src.download_model import setup
setup()

st.set_page_config(page_title="Attack Detection", page_icon="🧮")

st.markdown("# Attack Detection")
st.sidebar.header("Attack Detection")


image_types = ["png", "jpg", "heif"]

with st.sidebar:
    with st.form(key="latex_form"):
        model = st.selectbox(label="Model", options=['ResNet', 'Vision Transformer'])
        if st.session_state.get("model",  None) is None or st.session_state.model != model:
            st.session_state.model = ResNet() if model =="ResNet" else VisionTransformer()
            
        file_uploader = st.file_uploader(label="Upload image", type=image_types)
        
        if st.form_submit_button(label="Submit"):
            
            if not file_uploader:
                st.error("File not found", icon="🚨")
            else:
                image = Image.open(io.BytesIO(file_uploader.read()))
                category = st.session_state.model.predict(image=image)
                
                st.session_state.category=category



if "category" in st.session_state:
    st.write("**🫴 Origin Image:** ")
    st.image(file_uploader)
    
    st.write("**🫴 Result:** ")
    if st.session_state.category == "normal":
        st.write(f":green[{st.session_state.category.title()}]")
    else:
        st.write(f":red[{st.session_state.category.title()}]")
