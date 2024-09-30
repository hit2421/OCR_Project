import streamlit as st
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import easyocr
import numpy as np
import cv2

# Load the Qwen2-VL model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Initialize EasyOCR reader for Hindi
easyocr_reader = easyocr.Reader(['hi'])  # Only for Hindi

# Streamlit app layout
st.title("OCR for Hindi and English Text")
st.write("Upload an image containing text in both Hindi and English:")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to a PIL image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Prepare the message for the model
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": "Extract text from this image."},
            ],
        }
    ]
    
    # Prepare for inference with Qwen2-VL
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

    # Inference: Generate the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Display the extracted English text from Qwen2-VL
    st.subheader("Extracted English Text :")
    st.write(output_text[0])  # Display the extracted text from the image
    
    # Extract Hindi text using EasyOCR
    st.subheader("Extracted Hindi Text:")
    
    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Perform OCR using EasyOCR
    results = easyocr_reader.readtext(image_cv)

    # Extract Hindi text
    hindi_text = ""
    for (bbox, text, prob) in results:
        # Check if the detected text is in Hindi
        if any('\u0900' <= char <= '\u097F' for char in text):  # Unicode range for Devanagari script
            hindi_text += f"{text})\n"

    # Display extracted Hindi text
    if hindi_text.strip():
        st.write(hindi_text)
    else:
        st.write("No Hindi text detected.")

    # Store extracted text for search
    extracted_text = output_text[0] + "\n" + hindi_text

    # Keyword search
    st.subheader("Keyword Search:")
    keyword = st.text_input("Enter a keyword to search in the extracted text (Hindi/English):")

    if keyword:
        # Search for the keyword in the extracted text (case-insensitive)
        if keyword.lower() in extracted_text.lower():
            st.write(f"Keyword '{keyword}' found in the extracted text.")
        else:
            st.write(f"Keyword '{keyword}' not found in the extracted text.")


