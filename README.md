# xray-image-classification
Image classification machine learning model

## Project Overview
This project leverages TensorFlow and Keras to develop a machine learning model for classifying chest X-ray images into two categories: normal and pneumonia. The dataset used for training and testing the model is sourced from Kaggle and can be found [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data). The project includes a Jupyter Notebook that details the entire process, from data preprocessing to model training and evaluation. The trained model is saved in the models folder.

## Features
* Data Preprocessing: Techniques for cleaning and preparing chest X-ray images.
* Model Development: Using TensorFlow and Keras to build and train a convolutional neural network (CNN) for image classification.
* Evaluation: Assessing the model's performance on a test dataset.
* Model Storage: Saving the trained model for future use.

## Requirements
* Python 3.8 or higher
* Jupyter Notebook
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Scikit-learn

## Installation
1. Clone the repository
2. Download the dataset from this repo or from Kaggle:
   ```
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   ```

## Usage
1. Launch Jupyter Notebook.
2. Open and run the `pneumonia_classification.ipynb` notebook to see the full process of data preprocessing, model training, and evaluation.

## Data Preprocessing
The dataset is preprocessed to prepare the images for training the model. This includes:
Resizing images to a uniform size.
Normalizing pixel values.

## Model Development
The model is built using a Convolutional Neural Network (CNN) architecture, implemented with TensorFlow and Keras. The Jupyter Notebook details the following steps:
* Model Architecture: Defining the layers of the CNN.
* Compilation: Setting the optimizer, loss function, and evaluation metrics.
* Training: Training the model on the training dataset and validating it on the validation dataset.
* Evaluation: Evaluating the model's performance on the test dataset. The model's performance is evaluated using various metrics such as accuracy, precision, and recall.

## Acknowledgements
The dataset is provided by Kaggle and includes chest X-ray images for normal and pneumonia cases.
