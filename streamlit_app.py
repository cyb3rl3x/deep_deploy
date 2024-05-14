import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Carregar o modelo pré-treinado
model = tf.keras.models.load_model('model.h5')
datagen = ImageDataGenerator(rescale=1./255)
def load_and_prep_image(image, img_shape=224):
    # Certifique-se de que a imagem está em formato Pillow Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        print('IMAGEM EM MODO ARRAY')
    
    # Converta a imagem para RGB, se necessário
    if image.mode != 'RGB':
        image = image.convert('RGB')
        print('IMAGEM EM MODO RGB')
    
    # Redimensione a imagem
    image = image.resize((img_shape, img_shape))
    print('SHAPE DA IMAGEM DEPOIS DO RESIZE', image.shape)
    # Converta a imagem para um array numpy e normalize
    image = np.array(image) / 255.0
    
    # Adicione a dimensão do batch
    image = np.expand_dims(image, axis=0)
    print('SHAPE DA IMAGEM COM BATCH', image.shape)

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
