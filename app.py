import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import tempfile
import os

# Load your trained model (assuming you saved it)
@st.cache_resource
def load_model():
    # Replace with your actual model path
    model = joblib.load('best_xgb_model.pkl')
    scaler = joblib.load('vocaloid_scaler.pkl')
    return model, scaler

def extract_features(audio_path: str, sr=22050, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=sr)
    
    y_harmonic, _ = librosa.effects.hpss(y)
    duration = librosa.get_duration(y=y, sr=sr)

    centroid = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y_harmonic)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y_harmonic, sr=sr)[0]
    contrast = librosa.feature.spectral_contrast(y=y_harmonic, sr=sr)
    flux = librosa.onset.onset_strength(y=y_harmonic, sr=sr)  # not true spectral flux, but usable

    mfcc = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # Summary stats
    features = {
        "duration": duration,
        "zcr_mean": np.mean(zcr),
        "zcr_std": np.std(zcr),
        "centroid_mean": np.mean(centroid),
        "centroid_std": np.std(centroid),
        "flatness_mean": np.mean(flatness),
        "flatness_std": np.std(flatness),
        "rolloff_mean": np.mean(rolloff),
        "rolloff_std": np.std(rolloff),
        "onset_strength_mean": np.mean(flux),
        "onset_strength_std": np.std(flux),
    }

    for i in range(contrast.shape[0]):
        features[f"contrast_{i}_mean"] = np.mean(contrast[i])
        features[f"contrast_{i}_std"] = np.std(contrast[i])

    for i in range(n_mfcc):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])
        features[f"mfcc_delta_{i+1}_mean"] = np.mean(mfcc_delta[i])
        features[f"mfcc_delta2_{i+1}_mean"] = np.mean(mfcc_delta2[i])

    # print("Features dict:", features)
    # print("Number of features:", len(features))
    # print("Sample features:", list(features.keys())[:5])
    return pd.DataFrame([features])

def main():
    # Custom CSS for gradient background
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Make main content area transparent */
    .main .block-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Style the title */
    h1 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Style other text */
    .stMarkdown, .stText {
        color: white;
    }
    
    /* Style buttons */
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    /* Style file uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        border: 2px dashed rgba(255, 255, 255, 0.3);
    }
    
    /* Style success/error messages */
    .stSuccess {
        background: rgba(76, 175, 80, 0.2);
        border-left: 4px solid #4CAF50;
    }
    
    .stError {
        background: rgba(244, 67, 54, 0.2);
        border-left: 4px solid #f44336;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎵 Vocaloid vs Human Voice Classifier")
    st.write("Upload an MP3 file to classify whether it contains Vocaloid or human vocals!")
    
    # File upload
    uploaded_file = st.file_uploader("Choose an MP3 file", type=['mp3', 'wav'])
    
    if uploaded_file is not None:
        # Display file info
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**File size:** {uploaded_file.size / 1000000} MB")
        
        # Play the audio
        st.audio(uploaded_file, format='audio/mp3')
        
        if st.button("Classify Audio"):
            with st.spinner("Processing audio and extracting features..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Extract features
                    features_df = extract_features(tmp_file_path)
                    
                    # Load model and scaler
                    model, scaler = load_model()
                    
                    # Scale features (same as training)
                    features_scaled = scaler.transform(features_df)
                    
                    # Make prediction
                    prediction = model.predict(features_scaled)[0]
                    prediction_proba = model.predict_proba(features_scaled)[0]
                    
                    # Clean up temp file
                    os.unlink(tmp_file_path)
                    
                    # Display results
                    st.success("Classification complete!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 1:  # Assuming 1 = vocaloid
                            st.markdown("### 🤖 **Vocaloid**")
                        else:
                            st.markdown("### 👤 **Human**")
                    
                    with col2:
                        st.write("**Confidence:**")
                        st.write(f"Human: {prediction_proba[0]:.1%}")
                        st.write(f"Vocaloid: {prediction_proba[1]:.1%}")
                    
                    # # Progress bar for visual appeal
                    # st.write("**Prediction Confidence:**")
                    # if prediction == 1:
                    #     st.progress(prediction_proba[1])
                    # else:
                    #     st.progress(prediction_proba[0])
                    
                    # Show some feature info for debugging
                    with st.expander("🔍 Feature Details (for debugging)"):
                        st.write("**Extracted Features:**")
                        st.dataframe(features_df.head())
                        st.write(f"**Total features extracted:** {len(features_df.columns)}")
                        
                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")
                    st.write("Make sure the audio file is valid and try again.")

if __name__ == "__main__":
    main()