🍽 Image Classification with GoogLeNet
This project focuses on image recognition using a deep learning approach with the GoogLeNet architecture. The task involves classifying food images from a reduced version of the popular Food-101 dataset.

📂 Dataset Overview
Source: A reduced version of the Food-101 dataset.
Classes: 12 distinct food categories, carefully selected to represent a variety of food types.
Objective: Accurately classify images into one of these 12 food categories using the GoogLeNet model.
🧠 Model Architecture
GoogLeNet (Inception v1):
GoogLeNet is a deep convolutional neural network (CNN) that introduces Inception modules to optimize performance and computational efficiency.
It enables multi-scale feature extraction through parallel convolution operations.
Its compact architecture allows effective image classification with reduced computational cost.
This makes GoogLeNet an excellent choice for the multi-class classification problem posed by the Food-101 dataset.

🛠 Tools and Libraries
Deep Learning Frameworks: TensorFlow / Pytorch
Data Handling: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Dataset Management: Scikit-learn for preprocessing and splitting
🔍 Methodology
Data Preparation:
Load the reduced Food-101 dataset with 12 classes.
Perform image preprocessing (e.g., resizing, normalization, augmentation).
Model Training:
Implement the GoogLeNet architecture for image classification.
Fine-tune hyperparameters to optimize model performance.
Evaluation:
Measure accuracy, precision, and recall on the validation and test sets.
Analyze model performance across the 12 food classes.
🚀 Results
The project demonstrates how GoogLeNet can effectively classify food images into distinct categories, leveraging its efficient multi-scale architecture to achieve high accuracy on the reduced Food-101 dataset.

📚 How to Run the Project
Clone the repository:
git clone <repository-url>
cd <project-directory>
Install the required libraries:
pip install -r requirements.txt
Run the training script:
python train.py
🧩 Dependencies
Python 3.x
TensorFlow / Keras
OpenCV
Pandas, NumPy
Matplotlib
🤝 Contributing
Contributions, feedback, and improvements are welcome! Open an issue or submit a pull request to help enhance this project. 🚀

🔗 References
Food-101 Dataset: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
GoogLeNet Paper: "Going Deeper with Convolutions" by Szegedy et al.
