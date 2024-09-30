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
   - The live application can now be accessed at the URL: [Live application URL goes here]

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
    git clone <repository-url>
    cd <repository-directory>
    ```
2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. Install the required dependencies from `requirements.txt`:
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

## Evaluation Criteria
- **Accuracy:** How accurately the application extracts Hindi and English text from images.
- **Functionality:** Ability to upload images, extract text, and search the text for keywords.
- **User Interface:** A simple and functional UI.
- **Deployment:** The application must be accessible via a live URL.
- **Clarity:** Clear and concise documentation and code.
- **Completeness:** All tasks are successfully demonstrated.

## Live URL:
[https://hitesh2124-qweve.hf.space/]
