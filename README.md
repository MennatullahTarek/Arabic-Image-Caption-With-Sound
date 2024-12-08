# Image Caption Generator with CNN & LSTM

Have you ever looked at an image and easily identified what it represents? Now, imagine a computer doing the same. With advancements in deep learning, large datasets, and powerful computation, it's now possible for models to generate captions for images!

In this project, we implement an **Image Caption Generator** using **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** networks to automatically generate captions for images.

## Objective

The objective of this project is to demonstrate how a combination of **CNN** and **LSTM** can be used to generate captions for images. The CNN will be used to extract features from the images, and the LSTM will then generate descriptive captions based on those features.

---

## Project Overview

### What is an Image Caption Generator?

An Image Caption Generator is a model that uses **computer vision** and **natural language processing (NLP)** to recognize the content of an image and describe it in a natural language, such as English. The model learns from both visual features (via CNN) and textual data (via LSTM) to generate appropriate captions for new images.

---

## Techniques Used in This Project

1. **Convolutional Neural Networks (CNN)**:
   - Used to extract features from images. We use the **Xception model**, a CNN pre-trained on the **ImageNet** dataset, to extract meaningful visual features from the images.

2. **Long Short-Term Memory (LSTM)**:
   - Used for sequence modeling and generating captions from the extracted image features. LSTM is a type of **Recurrent Neural Network (RNN)** that is effective in handling sequential data like text.

---

## Dataset

We use the **Flickr8K dataset** for this project. This dataset contains 8,000 images, each with five captions describing its content. The smaller size of the **Flickr8K** dataset makes it ideal for this project, as training on larger datasets like **Flickr30K** or **MSCOCO** could take weeks to complete.

You can download the **Flickr8K dataset** from **Kaggle** here:  
[Flickr8K Dataset on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)  
(Size: 1GB)

---

## Project Workflow

1. **Data Preprocessing**:
   - Load and preprocess the images.
   - Extract features from the images using the **Xception model**.

2. **Caption Generation**:
   - Use the **LSTM** model to generate captions for the images based on the extracted features.

3. **Model Training**:
   - Train the CNN-LSTM model on the preprocessed image data and corresponding captions.

4. **Evaluation**:
   - Evaluate the performance of the model by generating captions for test images.

---

## Requirements

The following Python libraries are required for this project:

- TensorFlow/Keras (for deep learning models)
- NumPy (for numerical operations)
- Matplotlib (for visualization)
- Pillow (for image processing)
- Kaggle (for downloading the dataset)
- h5py (for saving the trained model)

You can install the necessary libraries by running the following:

```bash
pip install tensorflow numpy matplotlib pillow kaggle h5py
