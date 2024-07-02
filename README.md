# fire_smoke_
# Fire CNN

## Project Description
This project aims to develop a Convolutional Neural Network (CNN) model to detect fire in images. The purpose of this project is to provide an efficient tool for early detection of fire, which can be used in various applications such as surveillance systems, forest monitoring, and safety systems.

## Problem Statement
Early detection of fire is crucial for preventing disasters and saving lives. Traditional methods of fire detection, such as smoke detectors and human observation, can be slow and ineffective in certain scenarios. This project leverages deep learning techniques to develop a CNN model that can automatically detect fire , smoke and non fire in images, providing a faster and more reliable solution.

## Data Sources
The dataset used for this project consists of images labeled as 'fire' ,'smoke' and 'non fire'.

## Methodology
1. **Data Extraction**:
   - **Zip File Handling**: The zip file containing the dataset is extracted to a specified directory using the `zipfile` module.
   - **Directory Structure**: The dataset is organized into train and test directories, simplifying the data loading process.

2. **Data Augmentation**:
   - **ImageDataGenerator**: The `ImageDataGenerator` class from `tensorflow.keras.preprocessing.image` is used for data augmentation, applying random transformations like rotation, width/height shifts, shear, zoom, and horizontal flip to the training set to enhance model generalization.

3. **Model Architecture**:
   - **Pre-trained Model**: A pre-trained MobileNetV2 model (with ImageNet weights) is used as the base model, leveraging transfer learning.
   - **Custom Layers**: The model includes a `GlobalAveragePooling2D` layer, a `Dense` layer with 128 units and ReLU activation, and a final `Dense` layer with 3 units and softmax activation for classification.

4. **Learning Rate Scheduler**:
   - **LearningRateScheduler Callback**: A custom learning rate scheduler is implemented to decrease the learning rate after a certain number of epochs, helping in fine-tuning the model and preventing overfitting.

## Model Training and Evaluation
- **Model Compilation**: The model is compiled with the Adam optimizer, categorical cross-entropy loss, and accuracy metric.
- **Training**: The model is trained for several epochs for demonstration purposes. In practice, more epochs would likely be needed for better performance.
- **Evaluation**: The model's performance is evaluated on the test set, providing metrics like loss and accuracy.

## Data Preprocessing
- **Resizing**: All images are resized to a consistent shape to feed into the CNN.
- **Normalization**: Pixel values are normalized to a range of 0 to 1.
- **Data Augmentation**: Techniques such as rotation, flipping, and zooming are applied to increase the diversity of the training dataset.

## Saving the Model
The trained model is saved to disk, allowing it to be reused or deployed without retraining.

## Prediction and Visualization
- **Image Preprocessing**: User-provided images are loaded, resized, and pre-processed before being fed into the model for prediction.
- **Result Display**: The predicted class labels are displayed along with the input images using `matplotlib`.
- **Confusion Matrix and Accuracy**: A confusion matrix is generated to visualize the model's performance across different classes, and the overall accuracy score is calculated and printed.

## Dependencies
The project requires the following dependencies:
- Python 3.x
- Google Collab
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- scikit-learn

