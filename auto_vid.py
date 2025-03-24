import torch
import cv2
from PIL import Image
import gradio as gr
from transformers import BlipProcessor, BlipForQuestionAnswering
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from collections import Counter

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

# VQA function for a single frame
def get_answer_for_frame(image, question):
    inputs = processor(image, question, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

# VQA over all frames
def answer_question_over_video(video, question):
    if video is None or question.strip() == "":
        return "Please upload a video and enter a question."

    frames = extract_scenes(video)
    if not frames:
        return "No scene frames detected in video."

    answers = []
    for frame in frames:
        answer = get_answer_for_frame(frame, question)
        if answer:
            answers.append(answer.strip().lower())

    if not answers:
        return "No valid answers found from scene frames."

    # Use most common answer
    most_common_answer = Counter(answers).most_common(1)[0][0]
    return most_common_answer.capitalize()

# Gradio UI
with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("# Visual Question Answering (Mini Project)")
    gr.Markdown("Upload a video and ask a question about it. The system will analyze key scene frames and give a single consolidated answer.")

    video_input = gr.Video(label="🎬 Upload a Video")
    question = gr.Textbox(label="Question", placeholder="Ask a question...", lines=1)
    with gr.Row():
        clear_btn = gr.Button("Clear")
        submit_btn = gr.Button("Submit")
    answer = gr.Textbox(label="Answer", interactive=False)

    submit_btn.click(fn=answer_question_over_video, inputs=[video_input, question], outputs=answer)
    clear_btn.click(fn=lambda: [None, ""], outputs=[video_input, question])

if __name__ == "__main__":
    demo.launch(share=False)