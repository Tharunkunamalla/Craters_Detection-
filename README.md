# 🌙 Moon Crater Detection & Low Light Image Enhancement

This Project is a Problem Statement for SIH (Smart India Hackathon).

## Details

This Project is used to Detect Craters on Moon which are taken from Orbiter High Resolution Camera (OHRC). It aims to enhance (Low Light Image Enhancement) the feeble light reflected from PSR (Permanently Shadowed Regions) regions of Lunar craters into a better SNR (Signal to Noise Ratio) image for interpretations.

## Challenges

- Feeble signal to better signal image generation
- Low light image noise removal
- Usage: For generating first of its kind PSR image map of lunar poles captured by OHRC of Chandrayaan-2
- Users: Landing site selection users and geomorphological applications

## Outputs

- Software for generating low light image enhancement
- GrayScale Images of moon
- Gamma corrected (signal amplified) images

## 🚀 Streamlit Web Application

A web-based interface is available for easy image processing and crater detection.

### Installation

```bash
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Features

- **Image Preprocessing**: Apply CLAHE, noise reduction, and gamma correction
- **Edge Detection**: Detect crater edges using Sobel filter
- **Boulder Detection**: Identify boulders and craters using blob detection
- **Noise Removal**: Remove noise from low-light lunar images
- **SNR Analysis**: Calculate and improve Signal-to-Noise Ratio

### Screenshot

![Streamlit App](https://github.com/user-attachments/assets/7feafad0-8591-40d9-a608-6d13a75a01a5)

