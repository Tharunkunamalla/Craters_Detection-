import streamlit as st
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from skimage import restoration, exposure, filters
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.feature import blob_doh
import io

st.set_page_config(
    page_title="Moon Crater Detection",
    page_icon="🌙",
    layout="wide"
)

st.title("🌙 Moon Crater Detection & Low Light Image Enhancement")
st.markdown("""
This application enhances low light images from lunar surfaces and detects craters/boulders.
It processes images from Orbiter High Resolution Camera (OHRC) and enhances feeble light 
reflected from Permanently Shadowed Regions (PSR) into better SNR (Signal to Noise Ratio) images.
""")


def preprocess_image(image):
    """Apply CLAHE, noise reduction, and gamma correction."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    clahe = exposure.equalize_adapthist(image, clip_limit=0.03)

    sigma_est = np.mean(estimate_sigma(clahe, channel_axis=None))
    denoised = denoise_nl_means(
        clahe, h=1.15 * sigma_est, fast_mode=True,
        patch_size=5, patch_distance=3, channel_axis=None
    )

    tensor_img = T.ToTensor()(denoised.astype(np.float32))
    gamma = 1.2
    corrected_img = torch.pow(tensor_img, gamma)
    corrected_img = corrected_img.squeeze().numpy()

    return corrected_img


def detect_edges(preprocessed_image):
    """Edge detection using Sobel filter."""
    edges = filters.sobel(preprocessed_image)
    return edges


def detect_boulders(preprocessed_image, min_sigma=1, max_sigma=50, threshold=0.01):
    """Blob detection using Difference of Hessian."""
    blobs = blob_doh(
        preprocessed_image,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=10,
        threshold=threshold
    )
    return blobs


def low_light_noise_removal(image):
    """Apply contrast enhancement and noise removal for low-light images."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = exposure.equalize_adapthist(image, clip_limit=0.03)

    # Estimate noise and apply denoising directly to the enhanced image
    sigma_est = np.mean(restoration.estimate_sigma(clahe))
    denoised_image = restoration.denoise_nl_means(
        clahe, h=1.15 * sigma_est, fast_mode=True,
        patch_size=5, patch_distance=6
    )

    return clahe, denoised_image


def enhance_contrast_and_denoise(image):
    """Enhance contrast using CLAHE and denoise."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    clahe = exposure.equalize_adapthist(image, clip_limit=0.03)
    sigma_est = np.mean(restoration.estimate_sigma(clahe))
    denoised_image = restoration.denoise_nl_means(
        clahe, h=1.15 * sigma_est, fast_mode=True,
        patch_size=5, patch_distance=6
    )

    return denoised_image


def calculate_snr(denoised_image):
    """Calculate Signal-to-Noise Ratio."""
    signal_region = denoised_image[denoised_image > denoised_image.mean()]
    mu_signal = np.mean(signal_region)

    noise_region = denoised_image[denoised_image <= denoised_image.mean()]
    sigma_noise = np.std(noise_region)

    if sigma_noise == 0:
        return float('inf')

    snr = mu_signal / sigma_noise
    return snr


def amplify_signal(image, gamma=1.5):
    """Apply gamma correction to amplify signal."""
    gamma_corrected = exposure.adjust_gamma(image, gamma)
    return gamma_corrected


# Sidebar for file upload
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image file",
    type=['png', 'jpg', 'jpeg', 'tif', 'tiff']
)

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Display original image
    st.header("📷 Original Image")
    st.image(original_image_rgb, caption="Uploaded Image", use_container_width=True)

    # Create tabs for different processing options
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔧 Preprocessing",
        "📐 Edge Detection",
        "🔴 Boulder Detection",
        "🌑 Noise Removal",
        "📊 SNR Analysis"
    ])

    with tab1:
        st.subheader("Image Preprocessing")
        st.markdown("Apply CLAHE, noise reduction, and gamma correction.")

        if st.button("Apply Preprocessing", key="preprocess"):
            with st.spinner("Processing..."):
                preprocessed = preprocess_image(gray_image)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(gray_image, caption="Grayscale Image", use_container_width=True)
                with col2:
                    st.image(preprocessed, caption="Preprocessed Image", use_container_width=True, clamp=True)

    with tab2:
        st.subheader("Edge Detection")
        st.markdown("Detect edges using Sobel filter for crater identification.")

        if st.button("Detect Edges", key="edges"):
            with st.spinner("Detecting edges..."):
                preprocessed = preprocess_image(gray_image)
                edges = detect_edges(preprocessed)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(preprocessed, caption="Preprocessed Image", use_container_width=True, clamp=True)
                with col2:
                    st.image(edges, caption="Edge Detection", use_container_width=True, clamp=True)

    with tab3:
        st.subheader("Boulder/Crater Detection")
        st.markdown("Detect boulders and craters using Blob Detection (Difference of Hessian).")

        col1, col2, col3 = st.columns(3)
        with col1:
            min_sigma = st.slider("Min Sigma", 1, 20, 1)
        with col2:
            max_sigma = st.slider("Max Sigma", 20, 100, 50)
        with col3:
            threshold = st.slider("Threshold", 0.001, 0.1, 0.01, 0.001)

        if st.button("Detect Boulders", key="boulders"):
            with st.spinner("Detecting boulders..."):
                preprocessed = preprocess_image(gray_image)
                blobs = detect_boulders(preprocessed, min_sigma, max_sigma, threshold)

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(preprocessed, cmap='gray')

                for blob in blobs:
                    y, x, r = blob
                    circle = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
                    ax.add_patch(circle)

                ax.set_title(f"Detected {len(blobs)} Boulders/Craters")
                ax.axis('off')

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                buf.seek(0)
                st.image(buf, caption=f"Detected {len(blobs)} Boulders/Craters", use_container_width=True)
                buf.close()
                plt.close()

                st.info(f"Found {len(blobs)} potential boulders/craters in the image.")

    with tab4:
        st.subheader("Low Light Noise Removal")
        st.markdown("Remove noise from low-light lunar images.")

        if st.button("Apply Noise Removal", key="noise"):
            with st.spinner("Removing noise..."):
                enhanced, denoised = low_light_noise_removal(gray_image)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(enhanced, caption="Enhanced Image (CLAHE)", use_container_width=True, clamp=True)
                with col2:
                    st.image(denoised, caption="Denoised Image", use_container_width=True, clamp=True)

    with tab5:
        st.subheader("Signal-to-Noise Ratio (SNR) Analysis")
        st.markdown("Analyze and improve SNR through gamma correction.")

        gamma_value = st.slider("Gamma Value", 0.5, 3.0, 1.5, 0.1)

        if st.button("Analyze SNR", key="snr"):
            with st.spinner("Analyzing SNR..."):
                denoised = enhance_contrast_and_denoise(gray_image)
                snr_before = calculate_snr(denoised)

                gamma_corrected = amplify_signal(denoised, gamma=gamma_value)
                snr_after = calculate_snr(gamma_corrected)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(denoised, caption=f"Denoised (SNR: {snr_before:.2f})", use_container_width=True, clamp=True)
                    st.metric("SNR Before", f"{snr_before:.2f}")
                with col2:
                    st.image(gamma_corrected, caption=f"Signal Amplified (SNR: {snr_after:.2f})", use_container_width=True, clamp=True)
                    st.metric("SNR After", f"{snr_after:.2f}")

                improvement = ((snr_after - snr_before) / snr_before) * 100
                if improvement > 0:
                    st.success(f"SNR improved by {improvement:.1f}%")
                else:
                    st.warning(f"SNR changed by {improvement:.1f}%")

else:
    st.info("👈 Please upload an image using the sidebar to get started.")

    st.markdown("""
    ### Features:
    - **Preprocessing**: Apply CLAHE, noise reduction, and gamma correction
    - **Edge Detection**: Detect crater edges using Sobel filter
    - **Boulder Detection**: Identify boulders and craters using blob detection
    - **Noise Removal**: Remove noise from low-light lunar images
    - **SNR Analysis**: Calculate and improve Signal-to-Noise Ratio

    ### Use Cases:
    - Landing site selection for lunar missions
    - Geomorphological applications
    - PSR (Permanently Shadowed Regions) image enhancement
    - Chandrayaan-2 OHRC image processing
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This project enhances low light images from lunar surfaces, 
specifically from OHRC (Orbiter High Resolution Camera) of Chandrayaan-2.

Developed for Smart India Hackathon (SIH).
""")
