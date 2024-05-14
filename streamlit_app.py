import streamlit as st
from PIL import Image

def load_image(image_file):
    img = Image.open(image_file)
    return img

def main():
    st.title("Image Loader")
    st.write("Carregue uma imagem para exibir:")
    
    image_file = st.file_uploader("Selecione uma imagem", type=["jpg", "jpeg", "png"])
    
    if image_file is not None:
        img = load_image(image_file)
        st.write("Imagem carregada:")
        st.image(img)

if __name__ == "__main__":
    main()
    