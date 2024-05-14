import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Carregar o modelo pré-treinado
model = tf.keras.models.load_model('model.h5')

# Função para carregar e pré-processar a imagem
def load_and_prep_image(image, img_shape=224):
    image = np.array(image)
    image = tf.image.resize(image, [img_shape, img_shape])
    image = image / 255.0  # Normalizar a imagem
    return image

# Configurar a interface do usuário
st.title("Detecção de Pneumonia em Raios X")
st.write("Envie uma imagem de raio X para determinar se o paciente tem pneumonia ou não.")

uploaded_file = st.file_uploader("Escolha uma imagem...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagem carregada.', use_column_width=True)
    
    st.write("")
    st.write("Classificando...")
    
    img = load_and_prep_image(image)
    img = np.expand_dims(img, axis=0)  # Adicionar batch dimension

    prediction = model.predict(img)
    
    if prediction[0] > 0.5:
        st.write("O modelo prevê que o paciente tem pneumonia.")
    else:
        st.write("O modelo prevê que o paciente não tem pneumonia.")
