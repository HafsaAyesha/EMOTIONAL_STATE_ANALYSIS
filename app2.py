import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, welch
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sqlite3
import warnings
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

#-------------PREPROCESSING DATA-----------------------
file_path = r"C:\Users\FA23-BSE-035.CUI\Pictures\EmotionalStateAnalysis\expanded_raw_eeg_signals.csv"

try:
    eeg_df = pd.read_csv(file_path)
    print("Successfully loaded the data.")

    def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=250, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    channels = eeg_df.columns[1:]  # Skip the timestamp column
    filtered_data = eeg_df.copy()

    for channel in channels:
        filtered_data[channel] = bandpass_filter(eeg_df[channel])

    print("Preprocessing complete. Signals filtered.")

except Exception as e:
    print("Failed to load the data:", e)
    filtered_data = None
    channels = []

#-------------FREQUENCY BAND EXTRACTION-----------------------
if filtered_data is not None:
    try:
        BANDS = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 50)
        }

        def calculate_band_power(signal, fs=250):
            freqs, psd = welch(signal, fs=fs, nperseg=250)  # Fixed nperseg to match signal length
            band_powers = {band: np.sum(psd[(freqs >= low) & (freqs <= high)]) 
                        for band, (low, high) in BANDS.items()}
            return band_powers

        fs = 250  # Sampling frequency (Hz)
        band_powers = {channel: calculate_band_power(filtered_data[channel].values, fs) 
                       for channel in channels}

        print("Frequency band extraction complete.")

    except Exception as e:
        print("Error in frequency band extraction:", e)

# -------------FEATURE EXTRACTION----------------------
if filtered_data is not None and len(channels) > 0:
    WINDOW_SIZE = 250  # 1 second at 250 Hz
    OVERLAP = 125      # 50% overlap

    try:
        all_band_powers = []

        for start_idx in range(0, len(filtered_data), WINDOW_SIZE - OVERLAP):
            end_idx = start_idx + WINDOW_SIZE
            if end_idx > len(filtered_data):
                break

            row_band_powers = {}
            for channel in channels:
                segment = filtered_data[channel].iloc[start_idx:end_idx].values
                band_power = calculate_band_power(segment, fs)
                for band, power in band_power.items():
                    row_band_powers[f"{channel}_{band}"] = power

            row_band_powers['Timestamp'] = eeg_df['Timestamp'].iloc[start_idx]
            all_band_powers.append(row_band_powers)

        features_df = pd.DataFrame(all_band_powers)
        features_df['Timestamp'] = pd.to_datetime(features_df['Timestamp'], errors='coerce')
        features_df.dropna(subset=['Timestamp'], inplace=True)

        # Save to CSV and SQLite database
        features_df.to_csv("emotion_features.csv", index=False)
        conn = sqlite3.connect("emotion_analysis.db")
        features_df.to_sql("emotions", conn, if_exists="replace", index=False)
        conn.close()

        print("Feature extraction complete. Data stored at emotion_features.csv.")

    except Exception as e:
        print("Error during feature extraction:", e)

# -------------EMOTION CLASSIFICATION & ALERT SYSTEM-------------
if 'features_df' in locals():
    try:
        # Define a basic alert system
        def classify_emotion(row):
            try:
                if row["Fp1_beta"] > row["Fp1_alpha"] and row["Fp1_beta"] > row["Fp1_theta"]:
                    if row["Fp1_beta"] > 2 * row["Fp1_delta"]:
                        return "Stress"
                elif row["Fp1_alpha"] > row["Fp1_beta"] and row["Fp1_alpha"] > row["Fp1_theta"]:
                    return "Calm"
                elif row["Fp1_theta"] > row["Fp1_beta"] and row["Fp1_theta"] > row["Fp1_alpha"]:
                    return "Drowsy"
                return "Neutral"
            except KeyError:
                return "Neutral"

        features_df['Emotion'] = features_df.apply(classify_emotion, axis=1)

        # Alert for high-stress levels
        alert_triggered = features_df[features_df['Emotion'] == "Stress"]
        if not alert_triggered.empty:
            print(f"ALERT: High stress detected at {alert_triggered['Timestamp'].iloc[0]}!")

        # Train a placeholder ML model
        X = features_df.drop(columns=['Timestamp', 'Emotion'])
        y = features_df['Emotion']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        # Save and evaluate the model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        print("Emotion classification and alert system complete.")

    except Exception as e:
        print("Error during emotion classification or alerting:", e)

# -------------VISUALIZATION--------------------
if 'features_df' in locals():  # Ensure that features_df is defined before plotting
    try:
        # Define the emotions to plot
        emotions = ['Joy', 'Anger', 'Surprise', 'Calm', 'Sad', 'Happy']

        # Simulate random values for these emotions (Replace with actual calculations)
        for emotion in emotions:
            features_df[emotion] = np.random.rand(len(features_df))  # Mock data for demonstration

        # Create a figure with two subplots
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
            subplot_titles=("Emotion Distribution Over Time", "Stress Level Over Time")
        )

        # Plot each emotion as a separate line in the first subplot
        for emotion in emotions:
            fig.add_trace(go.Scatter(
                x=features_df['Timestamp'],
                y=features_df[emotion],
                mode='lines',
                name=emotion
            ), row=1, col=1)

        # Calculate and plot Stress Level in the second subplot
        features_df['Stress_Level'] = features_df['Fp1_beta'] - features_df['Fp1_alpha']  # Replace with actual calculation
        fig.add_trace(go.Scatter(
            x=features_df['Timestamp'],
            y=features_df['Stress_Level'],
            mode='lines',
            name='Stress Level',
            line=dict(color='red')
        ), row=2, col=1)

        # Update the layout for clarity
        fig.update_layout(
            title="Emotion and Stress Level Distribution Over Time",
            xaxis_title="Time",
            yaxis_title="Emotion Intensity / Stress Level",
            legend_title="Legend",
            template="plotly_white",
            height=800  # Adjusted height for two subplots
        )

        # Display the plot
        fig.show()

    except Exception as e:
        print("Error while plotting emotion and stress level distribution:", e)



#--------------REST API---------------------


app = Flask(__name__)

# Simulate the data (replace this with your actual data)
def generate_mock_data():
    timestamps = pd.date_range(start="2024-11-29 00:00:00", periods=10, freq="T")
    emotions = ['Joy', 'Anger', 'Surprise', 'Calm', 'Sad', 'Happy']
    data = {emotion: np.random.rand(len(timestamps)) for emotion in emotions}
    data['Timestamp'] = timestamps
    data['Stress_Level'] = np.random.rand(len(timestamps))  # Simulate stress level
    return pd.DataFrame(data)

# Serve mock data as JSON
@app.route('/data', methods=['GET'])
def get_data():
    # Generate data
    features_df = generate_mock_data()
    
    # Convert emotion data to JSON
    emotion_data = [
        {
            "timestamp": row["Timestamp"].isoformat(),
            "emotion": emotion,
            "value": row[emotion]
        }
        for _, row in features_df.iterrows()
        for emotion in ['Joy', 'Anger', 'Surprise', 'Calm', 'Sad', 'Happy']
    ]

    # Convert stress level data to JSON
    stress_data = [
        {
            "timestamp": row["Timestamp"].isoformat(),
            "value": row["Stress_Level"]
        }
        for _, row in features_df.iterrows()
    ]

    # Example alert history (replace with your logic)
    alerts = [
        "High stress level detected at 00:02",
        "Calm state observed at 00:04"
    ]

    # Combine all data
    response = {
        "emotionData": emotion_data,
        "stressLevelData": stress_data,
        "alertHistory": alerts
    }
    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
