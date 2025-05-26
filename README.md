# Oscillation Point Localization in Power Systems

## Project Overview
This project focuses on generating large-scale power system simulation data based on the WECC_179 bus model. Through PCA (Principal Component Analysis) dimensionality reduction, we obtain 2D visualization diagrams to identify oscillation point locations. The processed data is subsequently used to train a VGG-based deep learning model, ultimately establishing an effective oscillation point localization model for power systems.

## Key Features
- **Massive Simulation Data Generation**: Creates power system operation scenarios using WECC_179 bus prototype
- **PCA Visualization**: Implements dimension reduction to 2D space for oscillation pattern analysis
- **VGG Model Implementation**: Develops deep learning architecture for automatic oscillation localization

## Workflow
1. Power system simulation using WECC_179 bus model
2. Data preprocessing and feature extraction
3. PCA dimensionality reduction for 2D pattern visualization
4. Training VGG neural network classifier
5. Model validation and oscillation point localization

## Usage
```python
# Sample code library
from data_generator import WECC179Simulator
from processor import PCAVisualizer
from model import VGGClassifier
