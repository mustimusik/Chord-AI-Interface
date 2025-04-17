import streamlit as st
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import librosa
import numpy as np
import pandas as pd
import tempfile
import os
import time
import requests
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Chord AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #12E678;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B9CD3;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .parameter-section {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .about-text {
        color: #FFFFFF;
    }
    .stButton button {
        background-color: #12E678;
        color: white;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #888888;
        font-size: 0.8rem;
    }
    /* Custom styling for the dataframe */
    .dataframe-container {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        overflow: hidden;
        max-height: 400px;
        overflow-y: auto;
        margin-bottom: 20px;
    }
    .stProgress > div > div > div > div {
        background-color: #12E678;
    }
    /* Sidebar text color */
    .sidebar .sidebar-content {
        color: white;
    }
    /* Instructions text color */
    .result-box {
        background-color: #212427;
        border: 1px solid #424549;
    }
    .instructions-text {
        color: #FFFFFF;
    }
    /* Limit file size text */
    .file-limit {
        color: rgba(250, 250, 250, 0.6);
        font-size: 0.8rem;
    }
    /* Sample links */
    .sample-link {
        color: #4B9CD3;
        text-decoration: none;
    }
    .sample-link:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# Header with music icon and application name
st.markdown('<div class="main-header">🎵 Chord AI</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #FFFFFF; margin-bottom: 20px;">Predict chords in your audio files! Transform your music experience with AI.</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the model and feature extractor (cached to avoid reloading)"""
    try:
        with st.spinner("Loading model..."):
            feature_extractor = AutoFeatureExtractor.from_pretrained("mustimusik/sec1-72label-v2_2")
            model = AutoModelForAudioClassification.from_pretrained("mustimusik/sec1-72label-v2_2")
        return feature_extractor, model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None

def detect_chords(audio_path, feature_extractor, model, onset_threshold=0.90, confidence_threshold=0.70, sr=16000, hop_length=512):
    """Detect chords from the uploaded audio file"""
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Onset detection function
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    # Find frames where onset strength exceeds threshold
    peak_idxs = librosa.util.peak_pick(onset_env, 
                                      pre_max=3, 
                                      post_max=3, 
                                      pre_avg=3, 
                                      post_avg=5, 
                                      delta=onset_threshold, 
                                      wait=10)
    
    # Convert to samples
    onset_samples = librosa.frames_to_samples(peak_idxs, hop_length=hop_length)
    
    # If no onsets detected, try alternative method
    if len(onset_samples) == 0:
        st.warning("No clear chord changes detected. Trying alternative detection method...")
        # Try alternative method
        onset_frames = np.where(onset_env > onset_threshold)[0]
        onset_samples = librosa.frames_to_samples(onset_frames, hop_length=hop_length)
        if len(onset_samples) == 0:
            st.error("Unable to detect chord changes. Try adjusting the sensitivity or using a clearer audio recording.")
            return None
    
    # Predict chord at each onset
    segment_length = 16000  # 1 second at sr=16000
    raw_predictions = []
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, onset in enumerate(onset_samples):
        # Update progress
        progress = (i + 1) / len(onset_samples)
        progress_bar.progress(progress)
        status_text.text(f"Processing chord {i+1}/{len(onset_samples)}")
        
        # Take 1 second of audio from onset
        start_sample = onset
        end_sample = min(start_sample + segment_length, len(y))
        
        # Skip if segment is too short
        if end_sample - start_sample < 8000:  # minimum half a second
            continue
            
        segment = y[start_sample:end_sample]
        
        # Pad if less than 1 second
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
        
        # Predict chord
        try:
            inputs = feature_extractor(segment, sampling_rate=sr, return_tensors="pt", max_length=segment_length, truncation=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                conf, classes = torch.max(probs, 1)
                predicted_class_id = classes[0].item()
                confidence = conf[0].item()
            
            # Calculate timestamp in seconds
            timestamp = librosa.samples_to_time(onset, sr=sr)
            
            # Save prediction if confidence is high enough
            if confidence >= confidence_threshold:
                chord = model.config.id2label[predicted_class_id]
            else:
                chord = "Unknown"
                
            raw_predictions.append((timestamp, chord, confidence))
            
        except Exception as e:
            st.error(f"Error processing chord {i}: {str(e)}")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Convert raw predictions to per-second format
    total_duration = len(y) / sr
    per_second_predictions = {}
    
    # Initialize with the first prediction
    if raw_predictions:
        current_chord, current_confidence = raw_predictions[0][1], raw_predictions[0][2]
    else:
        current_chord, current_confidence = "Unknown", 0.0
        
    # Fill in predictions for each second
    for second in range(int(total_duration) + 1):
        # Find the most recent prediction before this second
        relevant_preds = [p for p in raw_predictions if p[0] <= second]
        if relevant_preds:
            latest_pred = max(relevant_preds, key=lambda x: x[0])
            current_chord, current_confidence = latest_pred[1], latest_pred[2]
        
        # Format timestamp
        minutes = int(second // 60)
        seconds = int(second % 60)
        timestamp_str = f"{minutes:02d}:{seconds:02d}"
        
        per_second_predictions[second] = (timestamp_str, current_chord, current_confidence)
    
    # Create results dataframe
    results = []
    for second, (timestamp_str, chord, confidence) in per_second_predictions.items():
        results.append({
            "Time": timestamp_str,
            "Chord": chord,
            "Confidence": f"{confidence:.2f}"
        })
    
    return pd.DataFrame(results)

def download_file(url):
    """Download a file from Google Drive"""
    # Extract file ID from Google Drive URL
    file_id = url.split('/d/')[1].split('/')[0]
    
    # Create direct download link
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    response = requests.get(download_url)
    return BytesIO(response.content)

def main():
    # Load model
    feature_extractor, model = load_model()
    
    if not feature_extractor or not model:
        st.error("Failed to load the chord detection model. Please try again later.")
        return
    
    # Create sidebar for parameters
    st.sidebar.markdown('<div class="sub-header">Settings</div>', unsafe_allow_html=True)
    
    onset_threshold = st.sidebar.slider(
        "Chord Change Sensitivity",
        min_value=0.5,
        max_value=1.5,
        value=0.9,
        step=0.05,
        help="Lower values detect more subtle chord changes. Higher values only detect more prominent changes."
    )
    
    confidence_threshold = st.sidebar.slider(
        "Prediction Confidence Threshold",
        min_value=0.3,
        max_value=0.9,
        value=0.7,
        step=0.05,
        help="Minimum confidence required to accept a prediction. Lower values show more predictions but may be less accurate."
    )
    
    # About section in sidebar
    st.sidebar.markdown('<div class="sub-header">About</div>', unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="color: #FFFFFF;">
    Chord AI uses deep learning to detect and classify piano chords in audio recordings. 
    The system is trained on a dataset of piano chord recordings and can recognize major and minor triads across different octaves.
    
    Upload your audio file to get started!
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.sidebar.markdown('<div class="footer">© 2025 Chord AI</div>', unsafe_allow_html=True)
    
    # Main content - Upload audio file
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="sub-header">Upload Audio</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac", "m4a"])
        st.markdown('<p class="file-limit">Limit 200MB per file • WAV, MP3, OGG, FLAC, M4A</p>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="sub-header">Sample Audio</div>', unsafe_allow_html=True)
        st.markdown("Don't have an audio file? Use one of these samples:", unsafe_allow_html=False)
        
        # Sample audio links
        sample_urls = [
            "https://drive.google.com/file/d/1o3ouAGK_2aAn3bYV4RjBfEWR0WFRpumj/view?usp=sharing",
            "https://drive.google.com/file/d/1yGuu-gjSGHHeYnbCa4jdqJHyPGdvMLXA/view?usp=sharing",
            "https://drive.google.com/file/d/1TZwGZq16RdBpICRDjaVA3JaFcSkPiRWA/view?usp=sharing"
        ]
        
        # Create download buttons for each sample
        for i, url in enumerate(sample_urls, 1):
            file_id = url.split('/d/')[1].split('/')[0]
            direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            st.markdown(f'• <a href="{direct_url}" class="sample-link" download>Audio Sample {i}</a>', unsafe_allow_html=True)
    
    # Process audio if uploaded
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            audio_path = tmp_file.name
        
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Process button
        if st.button("Predict Chords"):
            with st.spinner("Analyzing audio..."):
                results_df = detect_chords(
                    audio_path, 
                    feature_extractor, 
                    model, 
                    onset_threshold=onset_threshold,
                    confidence_threshold=confidence_threshold
                )
            
            if results_df is not None:
                # Display results
                st.markdown('<div class="sub-header">Chord Predictions</div>', unsafe_allow_html=True)
                
                # Table view
                st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                st.dataframe(results_df, height=400)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Clean up the temporary file
            os.unlink(audio_path)
    else:
        # Display a welcome message or instructions when no file is uploaded
        st.markdown("""
        <div style="background-color: #262730; color: white; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
            <h3 style="color: #4B9CD3;">How to use Chord AI:</h3>
            <ol style="color: white;">
                <li>Upload an audio file (WAV, MP3, OGG, FLAC formats supported)</li>
                <li>Adjust sensitivity settings in the sidebar if needed</li>
                <li>Click "Predict Chords" to process your audio</li>
                <li>View results in table format</li>
            </ol>
            <p style="color: white;">For best results, use recordings with clear chord changes and minimal background noise.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()