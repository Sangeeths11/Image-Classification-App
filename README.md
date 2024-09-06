# 🖼️ Image Classification Dashboard using ResNet-50

This project is an interactive web app designed to classify images using a pre-trained **ResNet-50** model from **PyTorch**. It allows users to upload images, view top predictions, and understand the classification results with insightful visualizations.

### Key Features:
1. **Upload and Classify**: Users can upload any image, and the ResNet-50 model will classify it in real-time. The app leverages the power of deep learning, offering high-accuracy predictions for a wide range of categories.
   
2. **Top 5 Predictions**: The app returns the top 5 predicted categories for each uploaded image, along with their corresponding probabilities, giving users a clear understanding of the model’s confidence in each prediction.
   
3. **Saliency Map Visualization**: A unique feature of the app is its ability to highlight key pixels in the image that contributed to the top prediction. This "saliency map" provides a visual explanation of how the model interprets the image and identifies the important features influencing its decision.

4. **User-friendly Dashboard**: Built with a clean, responsive design, the dashboard is easy to use and navigate. Users can quickly upload images, view results, and interact with the model's predictions in real-time.

### Who This Project Is For:
This project is ideal for individuals learning how to build machine learning-powered web applications. It demonstrates how to integrate a pre-trained model like ResNet-50 into an interactive web interface and offers a great starting point for anyone interested in deploying AI models for real-world applications.

Whether you’re new to machine learning or an experienced developer, this project provides valuable insights into building and deploying image classification systems.

### Results:
<img width="100%" alt="Screenshot 2024-09-06 195513" src="https://github.com/user-attachments/assets/672e9e6e-9671-4379-a2fd-491af51334db">

### Setup Porject

1. **Install Dependencies**: Run the following to install required packages:
   ```bash
   pip install -r requirements.txt
2. **Run the App**: Launch the app by running:
   ```bash
   streamlit run app.py
