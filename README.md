# 🧠 Visual Question Answering System (Mini Project)

This project implements a **Visual Question Answering (VQA)** system that can analyze an image and answer natural language questions about it using a **pretrained BLIP model** from Hugging Face.

---

## 🎯 Project Goal

To build a lightweight and interactive VQA application that:

- Accepts an image and a question as input.
- Uses a pre-trained transformer-based model to process the inputs.
- Returns a relevant and human-like answer.
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
git clone https://github.com/yourusername/vqa-mini-project.git
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
python app.py
```
---

## 🖥️ User Interface (Gradio)

- **Upload Image**: JPG or PNG format.
- **Enter Question**: e.g., “What is the person doing?” or “How many people are there?”
- **Output**: Answer appears instantly below.

---

## 📚 Model Details

- **Model**: `Salesforce/blip-vqa-capfilt-large`
- **Architecture**: Vision-Language Transformer
- **Trained on**: VQA datasets using image captioning + QA tasks
- **Hosted on**: [Hugging Face 🤗](https://huggingface.co/Salesforce/blip-vqa-capfilt-large)

---

