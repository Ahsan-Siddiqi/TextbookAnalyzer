# BookAnalyzer

BookAnalyzer is a Python program that captures images from a camera, compares them using Structural Similarity Index (SSIM), and extracts text from the images using EasyOCR.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- scikit-image
- EasyOCR

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Ahsan-Siddiqi/BookAnalyzer.git
    cd BookAnalyzer
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```


## Usage

1. Run the main script:
    ```bash
    python main.py
    ```

2. Follow the prompts to capture two images using your camera.

3. The program will compare the two images using SSIM and print the detected text from the first image if the similarity score is above the threshold.

## Configuration

- You can adjust the minimum similarity score threshold by modifying the `min_score` variable in `main.py`.

- To add more languages for text recognition, modify the `easyocr.Reader` initialization in the `get_text` function in `main.py`.

## Next Steps

- Make sure we can see live camera feed.
    - At time of taking picture
    - Pictures taken

- We store that data in a variable and classify it as a title, author, etc....
    - May require training a computer vision model

- Testing framework
    - Make sure min_score is adjusted properly
    - Make sure OCR is accurate
    - Complete testing framework including unit testing (ex. pytest)

## Sources

Explaination on how to compare 2 images accurately using scikit
- https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python

scikit-image (the library and specific method from the stack overflow post)
- https://scikit-image.org/docs/stable/api/skimage.metrics#skimage.metrics.structural_similarity

Another explanation on how to compare 2 images using scikit but also another method using vector comparison
- https://stackoverflow.com/questions/11541154/checking-images-for-similarity-with-opencv/71634759#71634759

EasyOCR (to extract text from an image)
- https://pypi.org/project/easyocr/

OpenCV
- https://www.youtube.com/watch?v=oXlwWbU8l2o
