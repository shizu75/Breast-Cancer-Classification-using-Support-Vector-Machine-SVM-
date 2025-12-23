# Breast Cancer Classification using Support Vector Machine (SVM)

## Project Overview
This project implements a **Support Vector Machine (SVM)** classifier to distinguish between **benign** and **malignant** breast cancer cell samples. Using numerical features extracted from cell images, the model learns a decision boundary that separates the two classes effectively.

The project includes data visualization, preprocessing, model training, and evaluation, making it a complete and educational machine learning workflow.

---

## Objectives
- Visualize the distribution of benign and malignant samples
- Clean and preprocess real-world medical data
- Train an SVM classifier for binary classification
- Evaluate model performance using standard metrics
- Understand how SVM uses support vectors for classification

---

## Technologies Used
- Python 3
- Pandas
- NumPy
- Matplotlib
- scikit-learn

---

## Dataset Description
The dataset (`cell_samples.csv`) contains measurements derived from breast cell images.

### Target Classes
- **Class = 2** → Benign
- **Class = 4** → Malignant

### Features Used
- Clump Thickness
- Uniformity of Cell Size
- Uniformity of Cell Shape
- Marginal Adhesion
- Single Epithelial Cell Size
- Bare Nuclei
- Bland Chromatin
- Normal Nucleoli
- Mitoses

These features are numerical and represent important medical indicators.

---

## Data Visualization
A scatter plot is used to visualize the relationship between:
- **Clump Thickness**
- **Uniformity of Cell Size**

Visualization highlights the separation between benign and malignant samples, providing intuition before model training.

---

## Data Preprocessing
- Removed invalid or non-numeric values from the `BareNuc` column
- Converted `BareNuc` to integer type
- Selected relevant features for model input
- Converted features into NumPy arrays for compatibility with scikit-learn

---

## Model Description
### Support Vector Machine (SVM)
- Uses a hyperplane to separate classes in feature space
- Relies on **support vectors**, the most critical data points
- Effective for high-dimensional and medical datasets

The SVM model is trained using scikit-learn’s `SVC` class with default parameters.

---

## Model Training and Testing
- Dataset split into training and testing sets (70% training, 30% testing)
- Model trained on training data
- Predictions generated on unseen test data

---

## Model Evaluation
The model outputs:
- **Support Vectors** used to define the decision boundary
- **Confusion Matrix** showing:
  - True Positives
  - True Negatives
  - False Positives
  - False Negatives
- **Accuracy Score** indicating overall classification performance

These metrics help assess the effectiveness of the classifier.

---

## Results
- High classification accuracy achieved on test data
- Clear separation between benign and malignant samples
- Support vectors provide insight into critical decision points

---

## How to Run the Project

### Prerequisites
Install the following Python libraries:
- pandas
- numpy
- matplotlib
- scikit-learn

### Steps
1. Place `cell_samples.csv` in the specified directory or update the file path
2. Run the Python script
3. Observe:
   - Scatter plot visualization
   - Printed support vectors
   - Confusion matrix
   - Accuracy score

---

## Learning Outcomes
- Practical understanding of Support Vector Machines
- Experience handling noisy and incomplete medical datasets
- Ability to visualize and interpret classification boundaries
- Familiarity with evaluation metrics for binary classification

---

## Future Improvements
- Hyperparameter tuning (kernel selection, C, gamma)
- Feature scaling for improved performance
- Cross-validation for robust evaluation
- ROC curve and AUC analysis
- Comparison with other classifiers (KNN, Logistic Regression)

---

## Use Case
This project is ideal for:
- Machine learning and data science portfolios
- Biomedical data analysis demonstrations
- Academic coursework and lab projects
- Understanding SVM applications in healthcare

---

## Author
Soban Saeed
Developed as an educational machine learning project for breast cancer classification using Support Vector Machines.
