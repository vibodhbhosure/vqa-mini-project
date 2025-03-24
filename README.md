# 🧠 Visual Question Answering System (IPCV Project)

This project implements a **Visual Question Answering (VQA)** system that can analyze a video, extract key scene frames, and answer natural language questions about any selected frame using a **pretrained BLIP model** from Hugging Face.

---

## 🎯 Project Goal

To build a lightweight and interactive VQA application that:

- Accepts a video as input and automatically detects scene changes.
- Allows the user to select any scene frame from the video.
- Lets the user ask a natural language question about the selected frame.
- Uses a pre-trained transformer-based model (BLIP) to generate relevant answers.
- Runs efficiently on a MacBook Air (Apple Silicon, 8GB RAM).

---

## ⚙️ Technologies Used

- Python
- PyTorch (with MPS backend for Apple Silicon)
- Hugging Face `transformers`
- `Salesforce/blip-vqa-capfilt-large` pretrained model
- Gradio (for user interface)
- PIL (for image handling)

---

## 🚀 How to Run

### 1. Clone the Repository (or Download Files)
```bash
git clone https://github.com/vibodhbhosure/vqa-mini-project.git
cd vqa-mini-project
```
### 2. Install Dependencies
```bash
python -m venv vqa_env
source vqa_env/bin/activate
```
### 3. Install required packages
```bash
pip install -r requirements.txt
```
### 4. Run VQA App
```bash
python vid_app.py
```
---

## 🖥️ User Interface (Gradio)

- **Upload Video**: MP4 or any supported video format.
- **Extract Scene Frames**: Automatically detects scene changes and extracts representative frames.
- **Select Frame**: Click on any frame from the scene gallery to ask a question.
- **Enter Question**: e.g., “What is the person doing?” or “How many people are in the scene?”
- **Output**: The system generates a relevant answer using a pretrained BLIP model.

---

## 📚 Model Details

- **Model**: `Salesforce/blip-vqa-capfilt-large`
- **Architecture**: Vision-Language Transformer
- **Trained on**: VQA datasets using image captioning + QA tasks
- **Hosted on**: [Hugging Face 🤗](https://huggingface.co/Salesforce/blip-vqa-capfilt-large)

---

