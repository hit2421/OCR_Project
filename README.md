# OCR for Hindi and English Text Web Application

## Objective
The goal of this project is to develop and deploy a web-based prototype that demonstrates the ability to perform Optical Character Recognition (OCR) on an uploaded image containing text in both Hindi and English. The application also includes a basic keyword search functionality for the extracted text. The final web application is accessible via a live URL.

## Scope of the Assignment
- Create a web application that allows users to upload an image.
- The application processes the image and extracts text using OCR.
- The extracted text can be searched using a basic keyword search functionality.
- The application is deployed online and accessible via a public URL.

## Steps Taken During Development

### 1. Environment Setup:
- **GPU Setup:** 
  To optimize the processing speed, the system's GPU was set up with the following installations:
  - **NVIDIA Video Driver**
  - **CUDA Toolkit**
  - **cuDNN**
  - **PyTorch** (with GPU support enabled)
- A `test.py` file was created to verify that the GPU is functioning correctly. Run this file to ensure the environment is properly configured.

### 2. OCR Model Integration:
- Installed necessary Python libraries:
  - Huggingface Transformers
  - PyTorch
  - EasyOCR
  - Streamlit
- Integrated two OCR models:
  - **Qwen2-VL**: Used for extracting English text from images.
  - **EasyOCR**: Used for extracting Hindi text from images.

### 3. Web Application Development:
- Developed a simple web application using **Streamlit** with the following features:
  - **Image Upload**: Users can upload an image file (JPEG, PNG, etc.).
  - **Text Extraction**: The OCR models extract both Hindi and English text from the uploaded image.
  - **Keyword Search**: Users can search for specific keywords within the extracted text, and matching results are highlighted.

### 4. Deployment:
- The web application was deployed on **Hugging Face Space** under the name **qweve**. This platform provides an easy way to host machine learning models and web applications with public access.

#### Deployment Process:
1. **Environment Setup:**
   - Created a Hugging Face Space and selected the appropriate environment for running Python applications.
   - Uploaded all necessary files including `app.py`, `test.py`, and `requirements.txt`.

2. **Model Selection:**
   - The OCR model chosen for this project is **Qwen/Qwen2-VL-7B-Instruct**. This model is integrated into the application to extract text from images in both Hindi and English efficiently.

3. **Dependencies:**
   - Configured the deployment environment to install the required libraries as specified in the `requirements.txt` file. This ensures that all necessary dependencies are available for the application to run smoothly.

4. **Starting the Application:**
   - After the environment setup and library installation, the application was started, and it became accessible via a public URL.

5. **Accessing the Application:**
   - The live application can now be accessed at the URL: [https://hitesh2124-qweve.hf.space/]

This deployment allows users to interact with the application seamlessly, utilizing the chosen OCR model for effective text extraction and search functionality.

## How to Set Up the Environment Locally

### Pre-requisites:
- Python 3.x
- NVIDIA GPU setup with drivers, CUDA Toolkit, and cuDNN (if you want GPU acceleration)
- Huggingface Transformers
- PyTorch
- Streamlit
- EasyOCR

### Steps to Install:
1. Clone the repository:
    ```bash
    git clone <https://github.com/hit2421/OCR_Project.git>
    cd <OCR>
    ```
2. Install the required dependencies from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
    Your `requirements.txt` should include:
    - `torchvision`
    - `transformers`
    - `pillow`
    - `streamlit`
    - `pytesseract`
    - `git+https://github.com/huggingface/transformers.git`
    - `qwen-vl-utils`
    - `accelerate>=0.26.0`
    - `sentencepiece`
    - `easyocr`
    - `opencv-python`

4. Run the GPU test script to check if your setup is correct:
    ```bash
    python test.py
    ```

5. Run the web application locally:
    ```bash
    streamlit run app.py
    ```

## Usage

1. Open the web application.
2. Upload an image file containing text in both Hindi and English.
3. The extracted text from both languages will be displayed.
4. You can search for specific keywords within the extracted text.

## Files Included

- `app.py`: The main Python script for running the Streamlit web application.
- `test.py`: A script to verify if the GPU is functioning properly.
- `requirements.txt`: Lists the necessary dependencies for running the project.
- `README.md`: Documentation explaining the project and how to set it up.

## Deliverables

1. Python scripts for the OCR process and search functionality.
2. Deployed web application (accessible via live URL).
3. Example outputs for both the extracted text and search functionality.

### Code Explanation

The provided code implements a web-based Optical Character Recognition (OCR) application using Streamlit, leveraging advanced OCR models to extract text from images containing both Hindi and English text. Below is a detailed breakdown of the code's functionality:

1. **Imports**:  
   The necessary libraries are imported, including:
   - **Streamlit**: Used to create the web interface.
   - **Torch**: Required for PyTorch functionalities.
   - **Transformers**: From Hugging Face, used to load the Qwen model.
   - **PIL**: For image processing tasks.
   - **EasyOCR**: For performing OCR on images.
   - **NumPy** and **OpenCV**: For image manipulation and processing.

2. **Model Loading**:  
   The Qwen2-VL model is loaded using the `from_pretrained()` method, which fetches the model weights and configurations needed for text generation from images. An `AutoProcessor` is instantiated to manage the pre-processing of input data.

3. **OCR Initialization**:  
   An EasyOCR reader is initialized specifically for recognizing Hindi text. This setup allows the application to extract Hindi characters from images effectively.

4. **Streamlit Layout**:  
   The web application layout is defined, with a title and description prompting users to upload images that contain text in both Hindi and English.

5. **Image Upload**:  
   A file uploader widget is created to enable users to upload image files in supported formats (JPEG and PNG).

6. **Image Processing**:  
   Once an image is uploaded, it is opened as a PIL image and displayed on the web interface for user confirmation.

7. **Input Preparation**:  
   A structured message is constructed for the model, indicating that the input comprises the uploaded image and a command to extract text from it.

8. **Inference Preparation**:  
   The input messages are processed to ensure they are formatted correctly for the model. The function `process_vision_info()` handles necessary transformations for the images. The inputs are moved to the appropriate device (GPU or CPU) for efficient processing.

9. **Text Generation**:  
   The model generates text based on the provided inputs. The output is processed to remove any special tokens and decode the generated IDs back into human-readable text.

10. **Displaying Output**:  
    The extracted English text is displayed in the web application, providing users with immediate feedback on the results of the OCR process.

11. **Hindi Text Extraction**:  
    The uploaded image is converted to OpenCV format for processing with EasyOCR, which extracts Hindi text from the image.

12. **Text Filtering**:  
    The extracted text is filtered to include only Hindi characters, ensuring that the displayed text contains relevant content. If no Hindi text is detected, an appropriate message is shown.

13. **Combined Text Storage**:  
    The extracted English and Hindi texts are combined to facilitate later keyword searches within the application.

14. **Keyword Search Input**:  
    A text input widget is provided for users to enter keywords they wish to search for within the extracted text.

15. **Keyword Search Functionality**:  
    When a keyword is entered, the application checks for its presence in the combined extracted text (case-insensitive). The results are displayed, indicating whether the keyword was found or not.

### Summary
This code successfully creates an interactive web application that allows users to upload images and extract text in both Hindi and English using state-of-the-art OCR models. The application not only extracts text but also offers functionality for users to search keywords within the extracted content, enhancing user engagement with the text extraction process.
