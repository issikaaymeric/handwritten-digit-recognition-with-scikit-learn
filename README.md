 🧠 Handwritten Digit Recognition with Random Forest & Gradio

This project is a simple web-based application that recognizes handwritten digits (0–9) using the MNIST dataset and a Random Forest classifier. The interface is built with [Gradio](https://www.gradio.app/), making it easy to draw and test the model in real time.

**Features**

* Trained on the **MNIST** dataset (70,000 images)
* Uses **Random Forest** classifier (faster than SVM for deployment)
* Supports real-time digit prediction through a **Gradio GUI**
* Preprocesses input using **PIL** and **scikit-learn’s StandardScaler**

**Dependencies**

Make sure you have the following Python packages installed:
pip install numpy gradio scikit-learn pillow

## How It Works

1. **Load Dataset**: The MNIST dataset is fetched via `fetch_openml`.
2. **Preprocess Data**:

   * Convert features to `float32` and labels to `int`
   * Normalize with `StandardScaler`
   * Split into training and test sets (80/20)
3. **Train Model**: A `RandomForestClassifier` with 100 trees is trained on the scaled data.
4. **Make Predictions**:

   * User draws a digit using Gradio's canvas
   * Image is resized to 28x28 pixels, converted to grayscale, and colors are inverted (white background to black)
   * Image is flattened and normalized
   * Model returns the predicted digit
5. **Launch UI**: A simple Gradio app allows users to draw and test digits.

---

## Interface

Once launched, you'll see a drawing canvas where you can draw digits. The app will output the predicted digit below the canvas.

---

## 📂 Code Structure

```python
import numpy as np
import gradio as gr
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image
```

* **Data Loading and Preprocessing**
* **Model Training**
* **Prediction Function (`predict_digit`)**
* **Gradio UI Setup (`gr.Interface`)**

---

## 📊 Example

Draw a digit like this:

```
[ 5 ]
```

Get an output like:

```
Prediction: 5
```


## 💡 Why Random Forest?

While many MNIST projects use Convolutional Neural Networks (CNNs), this project uses a **Random Forest** to demonstrate 
that even traditional machine learning methods can achieve high accuracy with proper preprocessing.

---

## 📈 Accuracy

Random Forests can achieve \~96% accuracy on MNIST, making it a good trade-off between speed
and performance for small projects and demos.

---

## 🧪 Future Improvements

* Replace with CNN for higher accuracy
* Add confidence scores
* Allow uploading images from disk
* Extend to recognize letters (EMNIST)

---


---

## 🔗 License

This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).
