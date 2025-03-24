# Predictive Maintenance System with Deep Learning

## Project Overview

This project implements a deep learning-based predictive maintenance system using the AI4I 2020 Predictive Maintenance Dataset. The system analyzes industrial sensor data to predict machine failures and classify them into specific types, enabling proactive maintenance before costly breakdowns occur.

### Failure Types Prediction

The model can predict five possible outcomes:
- No Failure (Functional machine)
- TWF: Tool Wear Failure
- HDF: Heat Dissipation Failure
- PWF: Power Failure
- OSF: Overstrain Failure

## Repository Structure
predictive_maintenance/
│
├── data/
│   └── ai4i2020.csv            # Main dataset file
│
├── neural_network/
│   ├── TP_IA_EMBARQUEE.ipynb   # Jupyter notebook with complete analysis and model development
│   ├── balanced_model.tflite   # TFLite model trained with balanced data
│   ├── imbalanced_model.tflite # TFLite model trained without balancing
│   ├── X_valid_mini.npy        # Validation input data for testing on device
│   └── Y_valid_mini.npy        # Validation labels for testing on device
│
├── cubeIDE/                    # STM32CubeIDE project files for embedded deployment
│   ├── Core/                   # Core application code
│   ├── Drivers/                # STM32 HAL drivers
│   ├── Middlewares/            # X-CUBE-AI middleware
│   ├── .project                # CubeIDE project file
│   └── ...                     # Other project files
│
└── README.md                   # This file
Copier
## Data Description

The AI4I 2020 Predictive Maintenance Dataset contains 10,000 instances of industrial machine operational data. Each instance represents the operating condition of a machine and includes:

- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

The dataset is highly imbalanced, with only 3.4% of instances representing machine failures.

## Methodology

Our approach involved:

1. **Exploratory Data Analysis**: Understanding the distributions and relationships within the data, particularly the severe class imbalance problem.

2. **Data Preprocessing**: 
   - Feature selection based on sensor measurements
   - One-hot encoding for target variables
   - Filtering for samples with exactly one failure type

3. **Model Development**: 
   - Comparison between balanced and imbalanced datasets
   - Implementation of SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance
   - Deep neural network architecture with multiple hidden layers

4. **Model Evaluation**: 
   - Confusion matrices
   - Precision, recall, and F1-scores for each failure type
   - Analysis of training and validation curves

5. **Deployment**: 
   - Conversion to TensorFlow Lite format
   - Integration with STM32 microcontroller using X-CUBE-AI

## Results

Our final model achieved:
- 98% overall accuracy
- Perfect recall (100%) for HDF, PWF, OSF, and Functional classes
- 90% recall for TWF
- High precision across all classes (94-100%)

The balanced model dramatically outperformed the imbalanced version, demonstrating the importance of addressing class imbalance in industrial machine failure prediction.

## Technical Implementation

The model was developed in Python using TensorFlow and scikit-learn on Google Colab. The architecture consists of:
- Input layer (5 features)
- Hidden layers (512 → 128 → 64 → 32 neurons)
- Output layer (5 classes)

Each hidden layer includes:
- BatchNormalization
- LeakyReLU activation
- Dropout regularization

The model was trained for up to 70 epochs with early stopping to prevent overfitting.

## Embedded Deployment

The trained model was converted to TensorFlow Lite format and deployed on an STM32 microcontroller using X-CUBE-AI, enabling real-time failure prediction directly on the hardware monitoring the machine.

## Usage

1. **Notebook Exploration**:
   - Open `neural_network/TP_IA_EMBARQUEE.ipynb` in Google Colab or Jupyter
   - Run all cells to reproduce the analysis and training process

2. **STM32 Deployment**:
   - Open the cubeIDE project in STM32CubeIDE
   - Build and flash to your STM32 device
   - Connect sensors according to the pin configuration in the code

## Requirements

- Python 3.7+
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- imbalanced-learn (for SMOTE)

For embedded deployment:
- STM32CubeIDE
- X-CUBE-AI extension
- Compatible STM32 microcontroller

## License

This project is available under the MIT License.

## Acknowledgments

- AI4I 2020 Predictive Maintenance Dataset creators
- STMicroelectronics for X-CUBE-AI tools
- The TensorFlow and scikit-learn communities
