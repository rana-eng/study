# Hand writing Digit Recognizer

A handwritten digit recognition project built with Python, PyTorch, and Tkinter. The app lets you draw a digit on a canvas and predicts the number using a trained neural network.

## Features

- Tkinter-based drawing canvas for handwritten digits
- 28x28 normalized input preprocessing before prediction
- PyTorch neural network trained on the digit dataset
- Real-time preview of the processed 28x28 image
- Prediction confidence output

## Project Structure

- `canvas_app.py` – GUI app for drawing and predicting digits
- `train_model.py` – model training script
- `test.py` – dataset inspection and quick visualization script
- `digit_model.pth` – saved trained model weights
- `data/train.csv` – training dataset
- `data/test.csv` – test dataset

## Model Architecture

The model is a simple feedforward neural network:

- Input layer: 784 features (28 x 28 pixels)
- Hidden layer 1: 128 units with ReLU
- Hidden layer 2: 64 units with ReLU
- Output layer: 10 classes (digits 0–9)

## Requirements

Install the dependencies with:

```bash
pip install torch pandas matplotlib pillow
```

## Training the Model

To train the model from scratch, run:

```bash
python train_model.py
```

This script loads the dataset from `data/train.csv`, normalizes the pixel values, trains the network, and saves the trained weights to `digit_model.pth`.

> Note: the training script currently uses an absolute Windows path to the dataset. If you run it on another machine or in a different folder layout, update the CSV path accordingly.

## Running the App

Start the drawing application with:

```bash
python canvas_app.py
```

Then:

1. Draw a digit in the canvas
2. Click Predict Digit
3. View the recognized number and confidence score

## Example Workflow

1. Train the model using `train_model.py`
2. Run `canvas_app.py`
3. Draw a digit on the canvas
4. Let the network predict the digit

## Notes

- Pixel values are normalized to the range 0 to 1 before model input
- The drawing app resizes the canvas drawing to 28x28 before inference
- The model predicts digits from 0 to 9 using a softmax-based classification output

## License

This project is intended for educational and personal use.
