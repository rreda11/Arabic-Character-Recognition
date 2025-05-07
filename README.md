# Arabic Character Recognition

This project aims to recognize handwritten Arabic characters using both traditional Machine Learning (SVM) and Deep Learning (CNN) techniques. The dataset is preprocessed, visualized, and used to train and evaluate multiple models.

---

## 📁 Dataset

- `csvTrainImages 13440x1024.csv`: Flattened grayscale images (32x32 resolution).
- `csvTrainLabel 13440x1.csv`: Labels corresponding to training images.
- `csvTestImages 3360x1024.csv`: Flattened grayscale test images.
- `csvTestLabel 3360x1.csv`: Labels corresponding to test images.

---

## 📊 Data Exploration & Preparation

- Loaded the training and test data.
- Normalized the image pixels by dividing by 255.
- Visualized sample images after rotating them 90° for correct orientation.
- Identified the number of unique Arabic character classes and their distribution.

---

## 🧪 Experiments & Results

### ✅ First Experiment: SVM

- Trained an SVM classifier using the training dataset.
- Evaluated using:
  - **Accuracy** on test set
  - **F1-score**
  - **Confusion Matrix** with Arabic labels

### 🧠 Second Experiment: Convolutional Neural Networks (CNN)

- Built **two distinct CNN architectures** with different layers and configurations.
- Applied **k-Fold Cross-Validation** to evaluate each architecture.
- Chose the best model based on **average validation accuracy**.
- Retrained the best model on the **entire training dataset**.
- Evaluated on the test dataset using:
  - Accuracy
  - F1-score
  - Confusion Matrix

---

## 🆚 Model Comparison

- Compared the final CNN model with the SVM model using:
  - Test accuracy
  - Weighted F1-score
  - Confusion matrix visualization

---

## 💾 Model Saving & Prediction

- Saved the best CNN model as `best_cnn_model.h5`.
- Reloaded the model in a separate script.
- Predicted Arabic letters from new (unlabeled) test images.
- Displayed the image along with its predicted Arabic character.

---

## 📌 Requirements

- Python
- TensorFlow / Keras
- scikit-learn
- matplotlib
- seaborn
- pandas
- numpy

---

## 📷 Sample Output

- Confusion matrices
- Predicted Arabic character for given images
- Image visualizations after rotation and preprocessing

---

## 🚀 Future Work

- Try data augmentation to boost CNN performance.
- Experiment with deeper CNN or attention-based models (like Transformers).
- Explore character-level language models for error correction or context understanding.



