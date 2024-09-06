import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from torchvision.models import resnet50, ResNet50_Weights

from captum.attr import IntegratedGradients
from captum.attr import visualization as viz


preprocess_func = ResNet50_Weights.IMAGENET1K_V2.transforms()
categories = np.array(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])

@st.cache_resource
def load_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    return model

def make_prediction(model, processed_img):
    probs = model(processed_img.unsqueeze(0))
    probs = probs.softmax(dim=1)
    probs = probs[0].detach().numpy()

    prob, idxs = probs[probs.argsort()[-5:][::-1]], probs.argsort()[-5:][::-1]
    return prob, idxs

def interpret_prediction(model, processed_img, target):
    interpretation_algo = IntegratedGradients(model)
    feature_imp = interpretation_algo.attribute(processed_img.unsqueeze(0), target=int(target))
    feature_imp = feature_imp[0].numpy()
    feature_imp = feature_imp.transpose(1, 2, 0)

    return feature_imp


st.title("ResNet-50 Image Classifier :tea: :coffee:")
upload = st.file_uploader(label="Upload Image:", type=["png", "jpg", "jpeg"])

if upload:
    img = Image.open(upload)

    model = load_model()
    preprocessed_img = preprocess_func(img)
    probs, idxs = make_prediction(model, preprocessed_img)
    feature_imp = interpret_prediction(model, preprocessed_img, idxs[0])


    fig, axs = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 2]})


    axs[0, 0].barh(y=categories[idxs][::-1], width=probs[::-1], color=["dodgerblue"]*4 + ["tomato"])
    axs[0, 0].set_title("Top 5 Probabilities", fontsize=15)
    axs[0, 0].invert_yaxis() 
    axs[0, 1].axis('off')


    axs[1, 0].imshow(img)
    axs[1, 0].set_title("Original Image", fontsize=15)
    axs[1, 0].axis('off')  # Remove axis ticks


    viz.visualize_image_attr(feature_imp, method='heat_map', show_colorbar=True, use_pyplot=True, plt_fig_axis=(fig, axs[1, 1]))
    axs[1, 1].set_title("Feature Attribution (Integrated Gradients)", fontsize=15)


    plt.tight_layout()


    st.pyplot(fig, use_container_width=True)
