
import streamlit as st
import torch
import torch.nn as nn
import json

from PIL import Image
from torchvision import models, transforms


# -----------------------------
# Configuration
# -----------------------------

st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="wide"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "ecosort_efficientnet_b0.pth"
CLASS_PATH = "class_names.json"


# -----------------------------
# Load model
# -----------------------------

@st.cache_resource
def load_model():

    with open(CLASS_PATH, "r") as f:
        class_names = json.load(f)

    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        len(class_names)
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, class_names


model, CLASS_NAMES = load_model()


# -----------------------------
# Image preprocessing
# -----------------------------

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Prediction
# -----------------------------

def predict(image):

    image = image.convert("RGB")

    tensor = transform(image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(tensor)

        probabilities = torch.softmax(
            outputs, dim=1
        )[0]

    top_probs, top_indices = torch.topk(
        probabilities, k=3
    )

    results = []

    for probability, index in zip(
        top_probs, top_indices
    ):
        results.append({
            "class": CLASS_NAMES[index.item()],
            "confidence": probability.item() * 100
        })

    return results


# -----------------------------
# UI
# -----------------------------

st.title("EcoSort AI")
st.subheader("AI-powered Waste Classification")

st.write(
    "Upload an image to identify its waste category "
    "using a fine-tuned EfficientNet-B0 model."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:

        if st.button("Classify Image", type="primary"):

            with st.spinner("Analyzing image..."):

                results = predict(image)

            best = results[0]

            st.success("Classification completed!")

            st.metric(
                "Predicted Category",
                best["class"].replace("-", " ").title()
            )

            st.metric(
                "Confidence",
                f"{best['confidence']:.2f}%"
            )

            st.subheader("Top 3 Predictions")

            for result in results:

                st.write(
                    f"**{result['class'].title()}**"
                )

                st.progress(
                    min(result["confidence"] / 100, 1.0)
                )

                st.caption(
                    f"{result['confidence']:.2f}%"
                )

st.divider()

st.caption(
    "EcoSort AI | Transfer Learning with EfficientNet-B0 | "
    "12 Waste Categories"
)
