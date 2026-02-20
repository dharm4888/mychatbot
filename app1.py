# voice_test_app.py — local mic capture (no WebRTC)
import sounddevice as sd
import numpy as np
import soundfile as sf
import tempfile
import streamlit as st
import os
from dotenv import load_dotenv
from backend import utils

load_dotenv()
st.set_page_config(page_title="Voice Test (Local)", layout="wide")

st.title("🎙️ Voice Test — Local Microphone Recorder")

st.markdown("""
1. Click **Start Recording** and speak clearly into your mic.  
2. Click **Stop Recording** to end recording.  
3. Click **Transcribe Audio** to convert speech → text using your model.
""")

# Global variables in Streamlit session
if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "fs" not in st.session_state:
    st.session_state.fs = 16000

duration_placeholder = st.empty()
record_btn = st.button("🎤 Start Recording" if not st.session_state.recording else "⏹ Stop Recording")

if record_btn:
    if not st.session_state.recording:
        st.session_state.recording = True
        st.info("Recording... speak now! Click Stop Recording to finish.")

        # Record audio until Stop is clicked
        duration = 10  # seconds (adjust if needed)
        st.session_state.audio_data = sd.rec(int(duration * st.session_state.fs),
                                             samplerate=st.session_state.fs,
                                             channels=1, dtype='float32')
        sd.wait()
        st.session_state.recording = False
        st.success(f"Recording complete ({duration} sec).")

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, st.session_state.audio_data, st.session_state.fs)
        st.session_state.audio_path = tmp.name
        st.audio(tmp.name)
        st.success(f"Saved audio to: {tmp.name}")
    else:
        st.session_state.recording = False
        sd.stop()
        st.success("Stopped recording.")

# --- Transcribe Button ---
if st.button("🧠 Transcribe Audio"):
    if "audio_path" not in st.session_state:
        st.warning("No audio recorded yet. Please record first.")
    else:
        try:
            transcription = utils.transcribe_audio(st.session_state.audio_path)
            st.subheader("🗣️ Transcription Result")
            st.write(transcription)
        except Exception as e:
            st.error(f"Transcription failed: {e}")
