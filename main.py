import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Skin Disease Classifier",
    page_icon="🔬",
    layout="centered"
)

# Title and description
st.title("🔬 Skin Disease Classification")
st.write("Upload an image to detect skin conditions")

# Load TFLite model
@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="skin_lesion_model.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load class labels
@st.cache_data
def load_labels():
    try:
        with open("labels.txt", "r") as f:
            labels = [line.strip() for line in f.readlines()]
        return labels
    except FileNotFoundError:
        # Default labels if file doesn't exist
        return ["Class 0", "Class 1", "Class 2", "Class 3"]

# Preprocess image
def preprocess_image(image, input_shape):
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize to model input size
    image = image.resize((input_shape[1], input_shape[2]))
    
    # Convert to array and normalize
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Make prediction
def predict(interpreter, image, labels):
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get input shape
    input_shape = input_details[0]['shape']
    
    # Preprocess image
    processed_img = preprocess_image(image, input_shape)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_img)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predictions = output_data[0]
    
    # Get top prediction
    predicted_class_idx = np.argmax(predictions)
    confidence = predictions[predicted_class_idx]
    
    return predicted_class_idx, confidence, predictions

# Load model and labels
interpreter = load_model()
labels = load_labels()

if interpreter is None:
    st.error("Failed to load the model. Please check if 'skin_lesion_model.tflite' exists in the repository.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of the affected skin area"
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        # Make prediction
        with st.spinner("Analyzing image..."):
            try:
                predicted_idx, confidence, all_predictions = predict(interpreter, image, labels)
                
                # Display results
                st.success("Analysis Complete!")
                st.metric(
                    label="Predicted Condition",
                    value=labels[predicted_idx]
                )
                st.metric(
                    label="Confidence",
                    value=f"{confidence * 100:.2f}%"
                )
                
                # Show all predictions
                st.write("### All Predictions:")
                for i, (label, prob) in enumerate(zip(labels, all_predictions)):
                    st.progress(float(prob), text=f"{label}: {prob * 100:.2f}%")
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

# Add disclaimer
st.markdown("---")
st.warning("""
⚠️ **Medical Disclaimer**: This tool is for educational purposes only. 
It should not be used as a substitute for professional medical advice, 
diagnosis, or treatment. Always consult a qualified healthcare provider 
for any skin conditions.
""")

# Add information in sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This application uses a TensorFlow Lite model to classify 
    skin conditions from images.
    
    **How to use:**
    1. Upload a clear image of the skin area
    2. Wait for the analysis
    3. Review the prediction and confidence score
    
    **Tips for best results:**
    - Use good lighting
    - Keep the camera steady
    - Fill the frame with the affected area
    - Avoid blurry images
    """)
    
    st.header("📊 Model Info")
    if interpreter:
        input_details = interpreter.get_input_details()
        st.write(f"**Input Shape:** {input_details[0]['shape']}")
        st.write(f"**Classes:** {len(labels)}")