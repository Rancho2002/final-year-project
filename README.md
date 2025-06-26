# 🧠 Signal Denoising using Autoencoders

This project demonstrates signal denoising using multiple deep learning architectures like Conv1D Autoencoders, LSTM Autoencoders, and Enhanced Conv1D models. It aims to clean noisy 1D signals (such as audio, vibration, or synthetic time series) through effective neural filtering.

---

## 🚀 Features

- ✅ Autoencoder-based signal denoising
- ✅ Three model types:
  - Conv1D Autoencoder
  - LSTM Autoencoder
  - Enhanced Conv1D (deep stack with upsampling)
- ✅ ModelCheckpoint support to save best-performing models
- ✅ HTML-based output visualization (`denoise_signal.html`)
- ✅ Visual comparison of original vs noisy vs denoised signals
- ✅ Modular code, easily extendable for real-world signals

---

## 🗂️ Project Structure
```
├── checkpoints/ # Saved models (e.g., denoiser_model.h5)
├── denoise_signal.html # Final visualization of signal comparison
├── train_model.py # Training script for autoencoders
├── predict_denoised.py # Inference script to denoise signal
├── requirements.txt # All Python dependencies
└── README.md # This documentation file
```
---

## 📊 Model Architecture

This project explores and compares the following deep learning models for denoising:

### 1. Conv1D Autoencoder
- Encoder:
  - Conv1D → ReLU → Downsample
- Decoder:
  - Conv1D → UpSampling1D → ReLU

### 2. LSTM Autoencoder
- Encoder:
  - LSTM layers to capture temporal dependencies
- Decoder:
  - RepeatVector + LSTM + TimeDistributed Dense

### 3. Enhanced Conv1D Autoencoder
- Deeper architecture with multiple Conv1D and UpSampling1D layers
- Designed to reconstruct signals with finer detail

### 🔧 Loss & Optimizer
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
---

---

## 🛠 How to Run

Follow these steps to train the model and use it for signal denoising:

### 1. Install Dependencies

Ensure you have Python installed, then install the required libraries:

```bash
pip install -r requirements.txt
```

---

### 2. Train the Model

To train the autoencoder on noisy signal data:

```bash
python denoise_signal.ipynb
```

This will:
- Add synthetic noise to the clean signal
- Train the selected autoencoder model
- Save the best model using `ModelCheckpoint` to:

  ```
  checkpoints/denoiser_model.h5
  ```

---

### 3. Predict Denoised Signal

Once training is complete, you can use the saved model to denoise any input signal:

```bash
python denoise_signal.ipynb
```

This will:
- Load the trained model
- Run inference on noisy input data
- Output a cleaned (denoised) signal array
---

## 📈 Visualization

The project generates a signal comparison chart to visually evaluate denoising quality.

The output includes:
- Original (clean) signal
- Noised signal
- Denoised signals from:
  - Conv1D Autoencoder
  - LSTM Autoencoder
  - Enhanced Conv1D Autoencoder

To view the interactive HTML visualization:

```bash
Open denoise_signal.html in your browser
```

Example plot includes time-series overlay:
- X-axis: Time Steps
- Y-axis: Signal Amplitude

This makes it easy to visually compare how well each model removes noise.

---

## 📦 Dependencies

All required libraries are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

### Key Libraries Used:
- `tensorflow` – for building and training deep learning models
- `numpy` – for numerical operations
- `pandas` – for signal data handling
- `matplotlib` – for plotting signal comparisons
- `sklearn` – (if used for preprocessing or metrics)
-  Used platform: Google Collab

Make sure you're using Python 3.10 full compatibility.

---

## ✨ Future Improvements

This project can be enhanced further with the following additions:

- 🔊 **Real-world Datasets**  
  Apply models to ECG, urban sound, speech, or vibration sensor data.

- 🧪 **More Denoising Techniques**  
  Compare with traditional filters (low-pass, wavelet, Savitzky–Golay).

- 🧠 **Hybrid Architectures**  
  Combine LSTM with CNNs or use Transformers for improved temporal learning.

- 🌐 **Web App Interface**  
  Deploy a Streamlit or Flask app for real-time signal upload and denoising.

- ☁️ **Model Deployment**  
  Host the model on platforms like Render, Hugging Face, or AWS Lambda.

- 📊 **Evaluation Metrics**  
  Add SNR, PSNR, and MSE comparisons for more robust performance reporting.

These additions can help elevate the project into a publishable or production-ready tool.
