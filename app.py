import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

st.title("Dog & Cat Image Classifier")

@st.cache_resource
def load_model():
    try:
        base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, pooling='avg', weights='imagenet')
        base_model.trainable = False
        output = Dense(2, activation='softmax')(base_model.output)
        model = Model(inputs=base_model.input, outputs=output)
        
        # Load only Dense layer weights (last layer)
        import h5py
        with h5py.File('model_weights.h5', 'r') as f:
            # Get the dense layer weights from saved file
            keys = list(f.keys())
            dense_key = keys[-1]
            weight_names = list(f[dense_key][dense_key].keys())
            kernel = f[dense_key][dense_key][weight_names[0]][:]
            bias = f[dense_key][dense_key][weight_names[1]][:]
            model.layers[-1].set_weights([kernel, bias])
        
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model = load_model()

if model is not None:
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        img_array = np.array(image)
        img_array = cv2.resize(img_array, (224, 224))
        img_array = img_array.astype(np.float32) / 255.0
        img_array = img_array.reshape(1, 224, 224, 3)
        
        prediction = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        if confidence < 0.70:
            st.warning("⚠️ Unable to classify. This might not be a dog or cat image.")
            st.write(f"Low confidence: {confidence * 100:.2f}%")
        else:
            if predicted_class == 1:
                st.success("🐕 The image is a Dog")
            else:
                st.success("🐱 The image is a Cat")
            st.write(f"Confidence: {confidence * 100:.2f}%")
