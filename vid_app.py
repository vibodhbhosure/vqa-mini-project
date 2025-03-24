import torch
import cv2
from PIL import Image
import gradio as gr
from transformers import BlipProcessor, BlipForQuestionAnswering
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Scene detection and frame extraction
def extract_scenes(video_path):
    print(f"[INFO] Processing video: {video_path}")
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=15.0))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list()
    print(f"[INFO] Detected {len(scene_list)} scenes.")

    if not scene_list:
        return []

    cap = cv2.VideoCapture(video_path)
    frames = []
    for start_time, _ in scene_list:
        frame_num = start_time.get_frames()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)
    cap.release()
    return frames

# VQA function
def answer_question(image, question):
    if image is None or question.strip() == "":
        return "Please upload an image and enter a question."
    inputs = processor(image, question, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

# Auto-update image when selected from gallery
def update_selected(evt: gr.SelectData):
    image_path = evt.value["image"]["path"]
    img = Image.open(image_path).convert("RGB")
    return img

# Updated UI to match the image-based app layout
with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("# Visual Question Answering (Mini Project)")
    gr.Markdown("Upload a video, extract scene frames, and ask a question about any selected frame.")

    with gr.Row():
        with gr.Column(scale=2):
            video_input = gr.Video(label="🎬 Upload a Video")
            extract_button = gr.Button("🎞️ Extract Scene Frames")
            scene_gallery = gr.Gallery(label="📸 Scene Frames", columns=4, height="auto", allow_preview=True)

        with gr.Column(scale=3):
            selected_frame = gr.Image(label="🖼️ Selected Frame for VQA")
            question = gr.Textbox(label="Question", placeholder="Ask a question...", lines=1)
            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Submit")
            answer = gr.Textbox(label="Answer", interactive=False)

    extract_button.click(fn=extract_scenes, inputs=video_input, outputs=scene_gallery)
    scene_gallery.select(fn=update_selected, outputs=selected_frame)
    submit_btn.click(fn=answer_question, inputs=[selected_frame, question], outputs=answer)
    clear_btn.click(fn=lambda: [None, ""], outputs=[selected_frame, question])

if __name__ == "__main__":
    demo.launch(share=False)