# Face Recognition with CNNs: Exploring Dataset Size and Lighting Conditions

## Overview
This Google Colab notebook implements a face recognition system using a Convolutional Neural Network (CNN) with TensorFlow/Keras. The primary goal is to investigate how the size of the training dataset and various lighting conditions impact the performance of the CNN model.

## Features
-   **Google Drive Integration:** Seamlessly connects to Google Drive for dataset access and saving results/models.
-   **Library Management:** Installs and imports all necessary libraries (OpenCV, TensorFlow, Keras, scikit-learn, matplotlib, seaborn).
-   **Dataset Verification:** Includes a utility to check the completeness and structure of the custom face dataset.
-   **Data Loading & Preprocessing:** Functions to load grayscale images, resize them, normalize pixel values, and split data into training, validation, and test sets with categorical labels.
-   **CNN Model Architecture:** A sequential CNN model featuring `Conv2D`, `BatchNormalization`, `MaxPooling2D`, `Dropout`, and `GlobalAveragePooling2D` layers for efficient feature extraction and classification.
-   **Image Augmentation:** Uses `ImageDataGenerator` during training to enhance robustness and prevent overfitting.
-   **Experimentation Framework:** Structured functions to run multiple training experiments with varying numbers of images per person.
-   **Model Evaluation:** Calculates and reports standard classification metrics (accuracy, precision, recall, F1-score).
-   **Visualization:** Generates plots for:
    -   Sample images from the dataset.
    -   Training and validation accuracy/loss curves.
    -   Confusion matrices for each experiment.
    -   A final comparison bar chart of metrics across all experiments.
    -   Model accuracy under simulated normal, dim, low, and very dark lighting conditions.
-   **Results Storage:** Saves trained models (`.h5` files) and experiment results (plots and JSON summaries) to Google Drive.

## Setup and Usage

1.  **Google Drive:** Ensure your Google Drive is mounted and authenticated as performed in Cell 1.
2.  **Dataset Preparation:**
    -   Create a folder named `face_dataset` in your Google Drive's root directory .
    -   Inside `face_dataset`, create subfolders for each person (e.g., `sayani`, `Rohit`).
    -   Within each person's folder, create subfolders for different lighting conditions: `normal_light`, `dim_light`, `low_light`, `motion`.
    -   Place a minimum of 20 `.jpg`, `.jpeg`, or `.png` images in each condition subfolder for each person. *Note: The notebook identifies some data missing for 'spandan' and 'sumouli', indicating some folders had more or fewer than 20 images.* Make sure to adjust `NUM_PHOTOS` in Cell 3 or provide sufficient images.
3.  **Run Cells:** Execute the notebook cells sequentially.
    -   **Cell 1:** Authenticates and mounts Google Drive.
    -   **Cell 2:** Installs libraries and checks GPU availability.
    -   **Cell 3:** Sets up paths and global settings (`IMAGE_SIZE`, `NUM_PHOTOS`). You might need to adjust `DATASET_PATH` if your dataset is in a different location.
    -   **Cell 4:** Verifies the dataset structure and counts images.
    -   **Cell 5:** Displays sample images from your dataset.
    -   **Cell 6:** Defines data loading and splitting functions.
    -   **Cell 7:** Defines the CNN model architecture.
    -   **Cell 8:** Defines the `train_experiment` function.
    -   **Cell 9:** Defines all plotting functions.
    -   **Cells 10, 11, 12:** Execute the three main experiments with 10, 15, and 20 images per person, respectively.
    -   **Cell 13:** Generates the final comparison table and plot.
    -   **Cell 14:** Tests the best model against simulated lighting conditions.
    -   **Cell 15:** Saves a comprehensive JSON summary and confirms all saved files.

## Key Findings from Experiments
-   **Impact of Dataset Size:** The experiments (`Exp1_10images`, `Exp2_15images`, `Exp3_20images`) show that increasing the number of training images per person generally leads to improved model performance (e.g., accuracy, precision, recall, F1-score). For instance, `Exp2_15images` achieved higher accuracy (44.44%) than `Exp1_10images` (33.33%). *However, `Exp3_20images` showed a lower accuracy (16.67%), which might indicate issues with data quality for 'spandan' and 'sumouli' or overfitting if the validation split was too small or imbalanced. This suggests further investigation or a more robust dataset is needed for the 20-image experiment.*
-   **Lighting Conditions:** The analysis on simulated lighting conditions (Normal, Dim, Low, Very Dark) reveals that CNN accuracy can significantly drop as lighting deteriorates. This highlights a critical challenge addressed by advanced techniques like BiLSTMs, as mentioned in the notebook's conclusion.

## Outputs
All generated plots (`.png` files) and the `CNN_complete_summary.json` are saved to the `RESULTS_PATH` (`/content/drive/MyDrive/face_results`). Trained model weights (`.h5` files) are saved to `MODELS_PATH` (`/content/drive/MyDrive/face_models`).

## Next Steps
The notebook suggests that a BiLSTM (Bidirectional Long Short-Term Memory) model could be integrated to improve performance, especially under challenging lighting conditions, hinting at a potential next phase of research or development.
