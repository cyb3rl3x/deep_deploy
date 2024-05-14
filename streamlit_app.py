import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Carregar o modelo pré-treinado
model = tf.keras.models.load_model('model.h5')

def load_and_prep_image(image, img_shape=224):
    # Certifique-se de que a imagem está em formato Pillow Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Converta a imagem para RGB, se necessário
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Converta a imagem para um array numpy
    image = np.array(image)
    
    # Adicione uma dimensão extra se necessário (grayscale para RGB)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        image = np.concatenate([image, image, image], axis=-1)
    
    # Redimensionar a imagem usando ImageDataGenerator
    image = datagen.apply_transform(image, {'tx': 0, 'ty': 0, 'theta': 0, 'shear': 0, 'zx': 1, 'zy': 1, 'flip_horizontal': False, 'flip_vertical': False})
    image = datagen.standardize(image)
    
    # Adicionar a dimensão do batch
    image = np.expand_dims(image, axis=0)
    
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
