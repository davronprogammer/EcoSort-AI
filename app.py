
import streamlit as st
import torch
import torch.nn as nn
import json

from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "ecosort_efficientnet_b0.pth"
CLASS_PATH = "class_names.json"

IMAGE_SIZE = 224
TOP_K = 3


# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #0e1117;
}

/* Main content */
.block-container {
    max-width: 1250px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

/* Main heading */
.main-title {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -2px;
    color: #ffffff;
    margin-bottom: 0;
}

.subtitle {
    font-size: 1.25rem;
    font-weight: 600;
    color: #16a34a;
    margin-top: 8px;
}

.description {
    color: #a1a1aa;
    font-size: 1rem;
    margin-top: 12px;
}

/* Drag and drop uploader */
[data-testid="stFileUploader"] {
    width: 100%;
    border: 2px dashed #16a34a;
    border-radius: 16px;
    padding: 18px;
    background: rgba(22, 163, 74, 0.05);
    transition: all 0.2s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #22c55e;
    background: rgba(22, 163, 74, 0.10);
}

/* Buttons */
.stButton > button {
    width: 100%;
    height: 48px;
    border-radius: 10px;
    font-weight: 700;
    border: none;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
}

/* Result panel */
.result-panel {
    background: #18251e;
    border: 1px solid #245c3c;
    padding: 22px;
    border-radius: 14px;
    margin-top: 12px;
}

.result-label {
    color: #a1a1aa;
    font-size: 0.9rem;
}

.result-value {
    color: #4ade80;
    font-size: 2rem;
    font-weight: 750;
    margin-top: 5px;
}

/* Footer */
.footer {
    text-align: center;
    color: #71717a;
    font-size: 0.85rem;
    padding-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    with open(CLASS_PATH, "r") as file:
        class_names = json.load(file)

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


try:
    model, CLASS_NAMES = load_model()

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


# =========================================================
# 4. IMAGE PREPROCESSING
# =========================================================

transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(IMAGE_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# =========================================================
# 5. PREDICTION FUNCTION
# =========================================================

def predict(image):

    image = image.convert("RGB")

    input_tensor = transform(image)

    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    top_probs, top_indices = torch.topk(
        probabilities,
        k=min(TOP_K, len(CLASS_NAMES))
    )

    results = []

    for probability, index in zip(
        top_probs,
        top_indices
    ):

        results.append({

            "class": CLASS_NAMES[index.item()],

            "confidence": probability.item() * 100

        })

    return results


# =========================================================
# 6. HEADER
# =========================================================

st.markdown(
    '<div class="main-title">EcoSort AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Waste Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="description">
        Upload an image and let Artificial Intelligence
        identify its waste category using EfficientNet-B0.
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")
st.divider()


# =========================================================
# 7. IMAGE UPLOAD
# =========================================================

st.subheader("Upload your image")

st.caption(
    "Drag and drop an image below or browse files from your device."
)

uploaded_file = st.file_uploader(

    "Drop your waste image here",

    type=["jpg", "jpeg", "png", "webp"],

    accept_multiple_files=False,

    help="Supported formats: JPG, JPEG, PNG, WEBP",

    key="waste_uploader"

)


# =========================================================
# 8. IMAGE PREVIEW AND CLASSIFICATION
# =========================================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        st.write("")

        col1, col2 = st.columns(
            [1.1, 1],
            gap="large"
        )

        # -----------------------------
        # LEFT: IMAGE PREVIEW
        # -----------------------------

        with col1:

            st.subheader("Uploaded Image")

            st.image(
                image,
                use_container_width=True
            )

            st.caption(
                f"Image dimensions: {image.width} × {image.height}"
            )

            st.caption(
                f"File size: {uploaded_file.size / 1024:.2f} KB"
            )

        # -----------------------------
        # RIGHT: PREDICTION
        # -----------------------------

        with col2:

            st.subheader("Classification")

            classify_button = st.button(
                "Classify Image",
                type="primary",
                use_container_width=True
            )

            if classify_button:

                with st.spinner("Analyzing image..."):

                    results = predict(image)

                    st.session_state["prediction_results"] = results

                    st.session_state["prediction_filename"] = (
                        uploaded_file.name
                    )

            results = st.session_state.get(
                "prediction_results"
            )

            filename = st.session_state.get(
                "prediction_filename"
            )

            if results and filename == uploaded_file.name:

                best = results[0]

                st.success("Classification completed!")

                st.markdown(
                    f"""
                    <div class="result-panel">

                        <div class="result-label">
                            Predicted Category
                        </div>

                        <div class="result-value">
                            {best["class"].replace("-", " ").title()}
                        </div>

                        <br>

                        <div class="result-label">
                            Prediction Confidence
                        </div>

                        <div class="result-value">
                            {best["confidence"]:.2f}%
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")

                st.subheader("Top 3 Predictions")

                for rank, result in enumerate(results, start=1):

                    class_name = result["class"].replace(
                        "-", " "
                    ).title()

                    confidence = result["confidence"]

                    st.markdown(
                        f"**{rank}. {class_name}**"
                    )

                    st.progress(
                        min(confidence / 100, 1.0)
                    )

                    st.caption(
                        f"Confidence: {confidence:.2f}%"
                    )

    except (UnidentifiedImageError, OSError):

        st.error(
            "Invalid image. Please upload a valid JPG, PNG or WEBP file."
        )

    except Exception as error:

        st.error(f"Prediction failed: {error}")


else:

    st.info(
        "Upload an image to start waste classification."
    )


# =========================================================
# 9. SUPPORTED CATEGORIES
# =========================================================

st.write("")
st.divider()

st.subheader("Supported Waste Categories")

category_cols = st.columns(4)

for index, class_name in enumerate(CLASS_NAMES):

    with category_cols[index % 4]:

        st.markdown(
            f"- {class_name.replace('-', ' ').title()}"
        )


# =========================================================
# 10. FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer">

        <b>EcoSort AI</b><br>

        Intelligent Waste Classification System<br>

        Powered by PyTorch and EfficientNet-B0<br><br>

        12 Categories · Transfer Learning · Deep Learning

    </div>
    """,
    unsafe_allow_html=True
)
