# OpenCV Experimenting

This project is an exploration of computer vision techniques using OpenCV and TensorFlow's MobileNetV2 model. The goal is to capture video from a webcam, preprocess the frames, and perform object detection using a pre-trained MobileNetV2 model.

## Requirements

- Python 3.x
- OpenCV
- TensorFlow
- NumPy

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/OpenCVExperimenting.git
    cd OpenCVExperimenting
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Ensure your webcam is connected.

2. Run the notebook:
    ```sh
    jupyter notebook notebook.ipynb
    ```

3. Follow the instructions in the notebook to start the video capture and object detection.

## Project Structure

- `notebook.ipynb`: Jupyter notebook containing the main code for video capture and object detection.
- `requirements.txt`: List of required Python packages.
- `venv/`: Virtual environment directory. (You need to make this with the requirements.txt)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
