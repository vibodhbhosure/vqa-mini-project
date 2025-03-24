import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import gradio as gr

# Load model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Define the function for answering questions
def answer_question(image, question):
    if not image or not question:
        return "Please provide both an image and a question."
    
    image = image.convert("RGB")
    inputs = processor(image, question, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs)
    answer = processor.decode(output[0], skip_special_tokens=True)
    with open("qa_log.txt", "a") as f:
        f.write(f"Q: {question}\nA: {answer}\n\n")
    return answer

# Launch Gradio app
demo = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Image(type="pil", label="Upload an Image"),
        gr.Textbox(lines=1, placeholder="Ask a question...", label="Question")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Visual Question Answering (Mini Project)",
    description="Upload an image and ask a question about it. The system will try to answer using a pretrained BLIP model."
)

demo.launch()