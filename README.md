# Emotional State Analysis

This project analyzes EEG data to extract emotional states and stress levels, classify them using a machine learning model, and visualize the results. It also includes a REST API for serving processed data and alerts.

## Features
1. **Data Preprocessing**:
   - Applies a bandpass filter to EEG signals to remove noise.
   - Extracts frequency bands (delta, theta, alpha, beta, gamma).

2. **Feature Extraction**:
   - Computes power in each frequency band for sliding windows of EEG data.
   - Stores extracted features in a CSV file and an SQLite database.

3. **Emotion Classification**:
   - Implements a basic rule-based system for emotion classification.
   - Trains a Decision Tree Classifier on extracted features.
   - Alerts for high-stress levels.

4. **Visualization**:
   - Visualizes emotion distributions and stress levels using interactive Plotly graphs.

5. **REST API**:
   - Serves processed emotion and stress data via Flask.
   - Provides an alert history.

## Installation

### Prerequisites
- Python 3.11 or later
- `pip` (Python package manager)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/HafsaAyesha/EMOTIONAL_STATE_ANALYSIS.git
   cd EMOTIONAL_STATE_ANALYSIS
