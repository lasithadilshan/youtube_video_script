import pafy
import yt_dlp
import cv2
import os
import time
import imutils
import shutil
import img2pdf
import glob
import streamlit as st

# Set pafy to use yt-dlp as the backend
pafy.backend_shared.backend = "yt-dlp"

# Constants
FRAME_RATE = 3  # No. of frames per second to process
WARMUP = FRAME_RATE  # Initial frames to skip
FGBG_HISTORY = FRAME_RATE * 15  # Background frames
VAR_THRESHOLD = 16  # Threshold for background subtraction
DETECT_SHADOWS = False  # Detect shadows or not
MIN_PERCENT = 0.1  # Min percentage of frame difference for motion detection
MAX_PERCENT = 3  # Max percentage of frame difference for motion detection


def get_frames(url):
    """Returns frames from a video at the given URL."""
    video = pafy.new(url)
    video_stream = video.getbest(preftype="mp4")
    vs = cv2.VideoCapture(video_stream.url)
    if not vs.isOpened():
        raise Exception(f"Unable to open video stream: {video_stream.url}")
    
    frame_time = 0
    frame_count = 0
    while True:
        vs.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
        frame_time += 1 / FRAME_RATE
        ret, frame = vs.read()
        if not ret:
            break
        frame_count += 1
        yield frame_count, frame_time, frame
    vs.release()


def detect_unique_screenshots(video_url, output_folder):
    """Detect and save unique screenshots."""
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=FGBG_HISTORY,
        varThreshold=VAR_THRESHOLD,
        detectShadows=DETECT_SHADOWS
    )
    captured = False
    (W, H) = (None, None)
    screenshot_count = 0

    for frame_count, frame_time, frame in get_frames(video_url):
        orig = frame.copy()
        frame = imutils.resize(frame, width=600)
        mask = fgbg.apply(frame)

        if W is None or H is None:
            (H, W) = mask.shape[:2]

        p_diff = (cv2.countNonZero(mask) / float(W * H)) * 100

        if p_diff < MIN_PERCENT and not captured and frame_count > WARMUP:
            captured = True
            filename = f"{screenshot_count:03}_{round(frame_time/60, 2)}.png"
            path = os.path.join(output_folder, filename)
            cv2.imwrite(path, orig)
            screenshot_count += 1
        elif captured and p_diff >= MAX_PERCENT:
            captured = False


def initialize_output_folder():
    """Clean or create the output folder."""
    root_dir = os.getcwd()
    output_dir = os.path.join(root_dir, "output")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def convert_screenshots_to_pdf(output_folder):
    """Convert PNG screenshots to a single PDF."""
    output_pdf_path = os.path.join(output_folder, "output.pdf")
    with open(output_pdf_path, "wb") as f:
        f.write(img2pdf.convert(sorted(glob.glob(f"{output_folder}/*.png"))))
    return output_pdf_path


# Streamlit App
if __name__ == "__main__":
    st.title("YouTube Video to PDF Converter")
    video_url = st.text_input("Enter the URL of a YouTube video")

    if st.button("Convert"):
        try:
            output_folder = initialize_output_folder()
            detect_unique_screenshots(video_url, output_folder)
            pdf_path = convert_screenshots_to_pdf(output_folder)
            st.success("PDF created successfully!")

            with open(pdf_path, "rb") as pdf_file:
                PDFbyte = pdf_file.read()
            st.download_button(
                label="Download PDF",
                data=PDFbyte,
                file_name="converted_video.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
