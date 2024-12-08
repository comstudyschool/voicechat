import os
import subprocess
import streamlit as st

def check_ffmpeg():
    try:
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            st.success("FFmpeg is installed!")
        else:
            st.error("FFmpeg is not installed.")
    except FileNotFoundError:
        st.error("FFmpeg is not installed or not found in the system PATH.")

st.title("FFmpeg 설치 확인")
check_ffmpeg()
