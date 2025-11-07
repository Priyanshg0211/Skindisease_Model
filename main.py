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
        # Try the actual model file name first
        model_path = "sagalyze_skin_model.tflite"
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except FileNotFoundError:
        # Fallback to alternative name
        try:
            model_path = "skin_lesion_model.tflite"
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
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
def preprocess_image(image, input_shape, input_details):
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Get input dimensions (handle both NHWC and NCHW formats)
    # Input shape is typically [1, height, width, channels] for NHWC
    if len(input_shape) == 4:
        # NHWC format: [batch, height, width, channels]
        height, width = input_shape[1], input_shape[2]
    elif len(input_shape) == 3:
        # No batch dimension: [height, width, channels]
        height, width = input_shape[0], input_shape[1]
    else:
        height, width = input_shape[0], input_shape[1]
    
    # Resize to model input size (use high-quality resampling)
    try:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    except AttributeError:
        # Fallback for older PIL versions
        image = image.resize((width, height), Image.LANCZOS)
    
    # Convert to array (keep as float32 initially for processing)
    img_array = np.array(image, dtype=np.float32)
    
    # Check input type and normalize accordingly
    input_type = input_details[0]['dtype']
    
    if input_type == np.uint8:
        # For uint8 input, keep values in [0, 255] range
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    else:
        # For float32 input, normalize to [0, 1]
        img_array = img_array / 255.0
        img_array = img_array.astype(np.float32)
    
    # Handle input shape format (NHWC vs NCHW)
    if len(input_shape) == 4:
        # Add batch dimension if not present
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
    elif len(input_shape) == 3:
        # No batch dimension needed
        pass
    else:
        # Handle 2D input shapes (grayscale)
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    
    # Ensure the shape matches exactly (handle any mismatches)
    expected_shape = tuple(input_shape)
    if img_array.shape != expected_shape:
        try:
            img_array = img_array.reshape(expected_shape)
        except ValueError:
            # If reshape fails, try to fix common issues
            if len(expected_shape) == 4 and len(img_array.shape) == 3:
                img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.reshape(expected_shape)
    
    return img_array

# Make prediction
def predict(interpreter, image, labels):
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get input shape
    input_shape = input_details[0]['shape']
    
    # Preprocess image
    processed_img = preprocess_image(image, input_shape, input_details)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_img)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Handle different output shapes
    if len(output_data.shape) > 1:
        predictions = output_data[0]
    else:
        predictions = output_data
    
    # Apply softmax to convert logits to probabilities (if needed)
    # Check if predictions are already probabilities (sum close to 1) or logits
    pred_sum = np.sum(predictions)
    if pred_sum > 1.1 or pred_sum < 0.9 or np.any(predictions < 0):
        # Likely logits, apply softmax
        exp_predictions = np.exp(predictions - np.max(predictions))  # Numerical stability
        predictions = exp_predictions / np.sum(exp_predictions)
    
    # Ensure predictions are within valid range
    predictions = np.clip(predictions, 0.0, 1.0)
    predictions = predictions / np.sum(predictions)  # Normalize to sum to 1
    
    # Get top prediction
    predicted_class_idx = np.argmax(predictions)
    confidence = float(predictions[predicted_class_idx])
    
    # Ensure predicted index is within valid range
    if predicted_class_idx >= len(labels):
        predicted_class_idx = 0
    
    return predicted_class_idx, confidence, predictions

# Load model and labels
interpreter = load_model()
labels = load_labels()

if interpreter is None:
    st.error("Failed to load the model. Please check if 'sagalyze_skin_model.tflite' or 'skin_lesion_model.tflite' exists in the repository.")
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
                
                # Show all predictions (sorted by confidence)
                st.write("### All Predictions:")
                # Sort predictions by confidence
                sorted_indices = np.argsort(all_predictions)[::-1]
                for idx in sorted_indices:
                    if idx < len(labels):
                        label = labels[idx]
                        prob = float(all_predictions[idx])
                        st.progress(prob, text=f"{label}: {prob * 100:.2f}%")
                
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