
import streamlit as st
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw
import numpy as np

# --- Model Definition and Loading ---

def create_model(num_classes):
    """
    Creates a Faster R-CNN model with a ResNet-50 backbone.
    
    Args:
        num_classes (int): The number of classes including the background.
    
    Returns:
        A PyTorch model.
    """
    # Load a pre-trained model but without pre-trained weights
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

@st.cache_resource
def load_model(model_path, device):
    """
    Loads the trained model from a file. The model is cached.
    
    Args:
        model_path (str): The path to the saved model state dictionary.
        device (torch.device): The device to load the model onto.
        
    Returns:
        A trained PyTorch model.
    """
    try:
        # Number of classes is 2 (1 for wheat + 1 for background)
        num_classes = 2
        model = create_model(num_classes)
        
        # Load the trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        model.to(device)
        model.eval() # Set the model to evaluation mode
        st.success("Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at: {model_path}. Please update the path.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# --- Prediction and Visualization ---

def make_prediction(model, img_tensor, device):
    """
    Makes a prediction on a single image.
    
    Args:
        model: The trained PyTorch model.
        img_tensor: The image converted to a tensor.
        device: The device the model is on.
        
    Returns:
        A dictionary containing the predicted boxes, labels, and scores.
    """
    with torch.no_grad():
        prediction = model([img_tensor.to(device)])
    return prediction

def draw_boxes(image, prediction, threshold):
    """
    Draws bounding boxes on an image.
    
    Args:
        image (PIL.Image): The original image.
        prediction (dict): The model's prediction.
        threshold (float): The confidence score threshold for displaying boxes.
        
    Returns:
        A PIL.Image with bounding boxes drawn on it.
    """
    draw = ImageDraw.Draw(image)
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    # Filter boxes based on the score threshold
    boxes_to_draw = boxes[scores >= threshold]
    
    if len(boxes_to_draw) == 0:
        st.warning("No wheat heads detected with the current confidence threshold.")
        return image

    for box in boxes_to_draw:
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
        
    return image

# --- Streamlit App UI ---

st.set_page_config(page_title="Global Wheat Detection", layout="wide")
st.title("🌾 Global Wheat Detection")
st.write("Upload an image to detect wheat heads using a trained Faster R-CNN model.")

# --- IMPORTANT: Please update this path ---
MODEL_PATH = "modified_fasterrcnn_resnet50_fpn.pth" 
# For example: "C:/Users/YourUser/Downloads/modified_fasterrcnn_resnet50_fpn.pth"
# Or place the model file in the same directory as this app.py file.

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = load_model(MODEL_PATH, device)

if model:
    st.sidebar.header("Configuration")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Open the image
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)

            # Convert image to tensor
            img_tensor = torchvision.transforms.functional.to_tensor(image)

            # Make prediction
            prediction = make_prediction(model, img_tensor, device)

            # Draw boxes on the image
            image_with_boxes = draw_boxes(image.copy(), prediction, confidence_threshold)
            
            with col2:
                st.subheader("Prediction Result")
                st.image(image_with_boxes, use_column_width=True)

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

st.markdown("---")
st.markdown("Developed by Gemini")

