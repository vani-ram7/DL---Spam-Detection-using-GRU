# 📩 SMS Spam Detection using GRU

A Deep Learning-based **SMS Spam Detection** project using **Natural Language Processing (NLP)** and a **Gated Recurrent Unit (GRU)** neural network.

The model analyzes SMS text messages and classifies them into two categories:

* **Ham** – Normal / legitimate message
* **Spam** – Unwanted or potentially fraudulent message

## 📌 Project Overview

Spam messages are a common problem in digital communication. This project demonstrates how deep learning can be used to automatically identify spam messages based on their textual content.

The project uses the **SMS Spam Collection dataset**, containing **5,572 SMS messages** with two classes:

| Class     | Number of Messages |
| --------- | -----------------: |
| Ham       |              4,825 |
| Spam      |                747 |
| **Total** |          **5,572** |

The text is converted into numerical sequences using a Keras tokenizer and then processed by a GRU-based neural network.

## 🧠 Model Architecture

The implemented model follows this architecture:

```text
SMS Text
   ↓
Tokenization
   ↓
Sequence Padding
   ↓
Embedding Layer
   ↓
GRU Layer (64 units)
   ↓
Dense Layer (32 units, ReLU)
   ↓
Dropout (0.3)
   ↓
Dense Layer (1 unit, Sigmoid)
   ↓
Spam / Ham Prediction
```

### Model Configuration

* **Vocabulary Size:** 5,000 words
* **Maximum Sequence Length:** 50
* **Embedding Dimension:** 64
* **GRU Units:** 64
* **Dense Units:** 32
* **Output Activation:** Sigmoid
* **Optimizer:** Adam
* **Loss Function:** Binary Crossentropy
* **Batch Size:** 64
* **Maximum Epochs:** 10
* **Early Stopping:** Enabled

The resulting model contains **347,073 trainable parameters**.

## 🔄 Data Preprocessing

The dataset is loaded from `spam.csv` and the relevant columns are renamed to:

* `label`
* `text`

The labels are converted into numerical values using `LabelEncoder`.

The dataset is then divided into training and testing sets using an **80/20 split**, with stratification to preserve the class distribution.

Text preprocessing includes:

1. Creating a Keras `Tokenizer`
2. Limiting the vocabulary to 5,000 words
3. Converting SMS messages into numerical sequences
4. Padding sequences to a maximum length of 50
5. Using an `<OOV>` token for unknown words

## 📊 Training Results

During training, the model achieved strong validation performance.

The best recorded validation accuracy in the notebook was approximately:

**98.65%**

The corresponding validation loss was approximately **0.0527**.

The training process uses **EarlyStopping** with:

```python
EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```

This helps prevent unnecessary training once validation performance stops improving.

## 🛠️ Technologies Used

* 🐍 Python
* 🧠 TensorFlow
* 🧩 Keras
* 📊 Pandas
* 🔢 NumPy
* 🤖 Scikit-learn
* 📈 Matplotlib
* 📝 Natural Language Processing
* 🔄 GRU / Recurrent Neural Networks

## 📂 Project Structure

```text
SMS-Spam-Detection/
│
├── DLMA - GRU.ipynb
├── spam.csv
└── README.md
```

> Make sure `spam.csv` is available in the project directory before running the notebook.

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the project

```bash
cd SMS-Spam-Detection
```

### 3. Install dependencies

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open

```text
DLMA - GRU.ipynb
```

Run the notebook cells sequentially.

## 🎯 Project Objective

The primary objective of this project is to demonstrate the application of **deep learning and NLP techniques for binary text classification**.

The project provides practical experience with:

* Text preprocessing
* Tokenization
* Sequence padding
* Word embeddings
* GRU networks
* Binary classification
* Model training and validation
* Early stopping

## 🚀 Future Improvements

Possible improvements for future versions include:

* Building a user-friendly web interface
* Adding real-time SMS prediction
* Saving and loading the trained model
* Adding confusion matrix and classification metrics
* Comparing GRU with LSTM and Transformer models
* Improving handling of class imbalance
* Deploying the model as a web application or API

## 📈 Key Learning

This project demonstrates how **sequence-based deep learning models can understand patterns in textual data and perform automated classification**.

The use of GRU provides an effective approach for learning sequential relationships within SMS messages while maintaining a relatively compact architecture.

## 👩‍💻 Author

**Vani**

This project is part of my Data Science and Deep Learning portfolio.
