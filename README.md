# Real-Time Virtual Meeting Architecture for Deaf People

This project implements an inclusive virtual meeting platform designed to bridge the communication gap between hearing and deaf individuals. Our solution combines advanced audio processing with real-time sign language recognition using LSTM neural networks, enabling seamless, bidirectional communication in virtual meeting environments.

---

## Overview

The system consists of two main components:

1. **Speech-to-Text Transcription & Translation:**  
   - **Audio Processing:** Captures live audio, applies noise reduction and signal enhancement to improve clarity, and uses pre-trained models for real-time transcription.  
   - **Translation & Summarization:** Converts the transcribed text into the preferred language of the deaf participant and condenses lengthy conversations into concise summaries using NLP-based models.

2. **Sign Language Recognition & Conversion:**  
   - **Gesture Capture:** Utilizes a camera module to record sign language gestures.  
   - **LSTM-Based Recognition:** Processes captured gestures using a multi-layer LSTM neural network to accurately classify sign language into textual form, which is then converted to speech for hearing participants.

---

## Key Features

- **Real-Time Processing:**  
  Low-latency audio and video processing for immediate communication.

- **Robust Audio Enhancement:**  
  Integration of libraries such as Librosa and PyDub for effective noise reduction and signal enhancement.

- **Accurate Transcription:**  
  Utilizes Assembly AI's state-of-the-art ASR models trained on diverse accents and dialects.

- **Multilingual Support:**  
  Translation capabilities via Google Translate API (mBART and mT5) ensuring accessibility across language barriers.

- **Sign Language to Text Conversion:**  
  Leverages LSTM models for dynamic recognition of sign language gestures, translating them into text and subsequently into speech.

- **Scalability and Modularity:**  
  The architecture is designed as a set of microservices, making it adaptable for future upgrades, including VR/AR integration and emotion detection.

---

## Architecture

### 1. Speech-to-Text Pipeline

- **Audio Input Capture:**  
  Capturing clear audio through high-quality microphones.
  
- **Noise Reduction & Signal Enhancement:**  
  Preprocessing audio using techniques like the Wiener Filter and amplification for improved signal-to-noise ratio.

- **Transcription & Translation:**  
  Transcribing the cleaned audio using a pre-trained ASR model and converting it into text, then translating and summarizing the text for user comprehension.

### 2. Sign Language Recognition Pipeline

- **Gesture Input Capture:**  
  Recording sign language gestures with a high-definition camera.
  
- **Key Point Detection:**  
  Using tools like MediaPipe to extract hand, finger, and body landmarks.

- **LSTM-Based Gesture Classification:**  
  Processing gesture sequences through an LSTM model trained on extensive sign language datasets to generate accurate textual representations.

- **Text-to-Speech Conversion:**  
  Converting the recognized text into speech for hearing participants using TTS engines.

## Future Directions

The roadmap for this project includes:

- **Optimization:** Enhancing transcription accuracy and reducing latency.  
- **Extended Multilingual Support:** Training models to support a broader range of sign languages.  
- **Emotion Detection:** Integrating emotion analysis from vocal tone and facial expressions.  
- **AR/VR Integration:** Expanding the platform to immersive virtual reality environments.  
- **Model Upgrades:** Investigating the use of transformer-based architectures for both audio and gesture recognition to further improve accuracy.  

---

## Contributing

Contributions are welcome! If you’re interested in helping improve this project, please fork the repository and submit a pull request. For major changes, open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

*Based on the research paper "Real-Time Virtual Meeting Architecture for Deaf People: Audio Transcription, Noise Processing, and Sign Language Translation Using LSTM Neural Networks" by Aaryan Chaurasia and Sivakumar Rajagopal from Vellore Institute of Technology.*
