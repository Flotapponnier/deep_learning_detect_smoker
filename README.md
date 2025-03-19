# Deep Learning Smoker Detection

This project uses PyTorch to create a deep learning model capable of classifying images into two categories: smokers and non-smokers.

## Installation

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `data/` : Data storage directory
  - `Training/` : Training images
    - `smoker/` : Smoker images
    - `non_smoker/` : Non-smoker images
  - `Validation/` : Validation images
    - `smoker/` : Smoker images
    - `non_smoker/` : Non-smoker images
  - `Testing/` : Test images
    - `smoker/` : Smoker images
    - `non_smoker/` : Non-smoker images
- `models/` : Directory for saving trained models
- `train.py` : Model training script
- `predict.py` : Prediction script
- `utils.py` : Utility functions
- `organize_data.py` : Script to organize data into appropriate directories

## Usage

1. Place your data in the `data/` directory following the structure above
2. Organize the data:
```bash
python organize_data.py
```
3. Train the model:
```bash
python train.py
```
4. Make predictions:
```bash
python predict.py --image path/to/image.jpg
```

## Notes

- Images should be of good quality and clearly show whether the person is smoking or not
- For best results, use images with clearly visible faces
- The model uses ResNet50 pre-trained on ImageNet as its base
