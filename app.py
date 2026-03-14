import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import tensorflow_hub as hub
import tf_keras
import numpy as np
from PIL import Image
import cv2

st.title("Dog & Cat Image Classifier")

# Model Loading
@st.cache_resource
def load_model():
    try:
        mobilenet_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
        feature_extractor = hub.KerasLayer(mobilenet_url, input_shape=(224, 224, 3), trainable=False)
        
        model = tf_keras.Sequential([
            feature_extractor,
            tf_keras.layers.Dense(2, activation='softmax')  # 2 units for 2 classes
        ])
        
        model.build((None, 224, 224, 3))
        
        # Load your trained weights
        model.load_weights('model_weights.h5')
        
        
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
        
        # Keep RGB format (don't convert to BGR)
        img_array = cv2.resize(img_array, (224, 224))
        img_array = img_array.astype(np.float32) / 255.0
        img_array = img_array.reshape(1, 224, 224, 3)
        
        prediction = model.predict(img_array, verbose=0)
        
        # Get the predicted class (0=Cat, 1=Dog)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        # Set confidence threshold
        confidence_threshold = 0.70
        
        if confidence < confidence_threshold:
            st.warning("⚠️ Unable to classify with confidence. This might not be a dog or cat image.")
            st.write(f"Prediction: {'Dog' if predicted_class == 1 else 'Cat'} (Low confidence: {confidence * 100:.2f}%)")
        else:
            if predicted_class == 1:
                st.success("🐕 The image is a Dog")
            else:
                st.success("🐱 The image is a Cat")
            
            st.write(f"Confidence: {confidence * 100:.2f}%")

st.markdown("---")
