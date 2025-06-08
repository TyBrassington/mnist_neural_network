# MNIST Handwritten Digit Classifier

This project is a fully custom neural network built from scratch using NumPy to classify handwritten digits from the MNIST dataset. 
It has been **modularized and generalized** to also work with the **Fashion MNIST** dataset with minimal modification — simply change the dataset input and class labels.

> Note: While the neural network itself is implemented entirely with NumPy, additional libraries such as **Pandas** (for CSV parsing) and **Matplotlib/Seaborn** (for plotting and evaluation) are used for other supporting functionality like data visualization and generalized accuracy evaluation.

- 3-layer neural network
- Dropout regularization
- Learning rate decay
- Early stopping with autosave/resume
- Accuracy convergence plots
- Confusion matrix visualization
- Easily switchable dataset backend

- Automatically saves model state to autosave_state/ every 200 iterations.
- Resumes from that state if resume_autosave_state = True.
- Only saves final model if validation accuracy improves by >0.3%.

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/mnist-numpy-classifier.git
   cd mnist-numpy-classifier
   ```
2. **Install required libraries**
    ```bash
    pip install numpy pandas matplotlib seaborn
    ```
3. **Run the program**
```bash
  python main.py
```

### `config.py` Overview
Modify global settings from config.py to control training behavior, architecture, and dataset handling.

```python
# Do not modify these unless you are training your own model
INPUT_SIZE = 784
HIDDEN_LAYER_1_SIZE = 312
HIDDEN_LAYER_2_SIZE = 55
OUTPUT_CLASSES = 10

# Learning Parameters
NUM_ITERATIONS = 50000
INITIAL_LEARNING_RATE = 0.1
DROPOUT_RATE = 0.25
DECAY_RATE = 0.0001
ENABLE_DROPOUT = False

# Dataset and Save Paths
# Modify these directories if you wish to switch datasets to MNIST fashion training datasets
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
dataset_path = "data/train.csv"
param_dir = "model_params"
generalized_accuracy_filepath = f"{param_dir}/generalized_accuracy.csv"

# Execution Options
train_model = True                 # True to train, False to only evaluate
resume_autosave_state = False     # Resume from last autosaved checkpoint
```

To use Fashion-MNIST, simply:
Replace `train.csv` with the Fashion-MNIST version.

Replace the `labels` list:
```python
labels = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
          "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
```

### Evaluation Mode
To run the program in evaluation mode, simply set the following flag in `config.py`:

```python
train_model = False
```

In this mode, the model will:

- Load previously saved weights from model_params/
- Generate 5 random predictions on the test set
- Display the predicted label alongside the true label for each image

**Examples:
**
![image](https://github.com/user-attachments/assets/d20eaa97-b421-4c86-8676-07445d0d4d45)

![image](https://github.com/user-attachments/assets/fbe02ab8-52c9-47f0-8cec-fbec7d67409f)

![image](https://github.com/user-attachments/assets/d16bee05-0abc-488e-ba1c-58fd485ff121)


A **confusion matrix** will also be plotted, where:
- The x-axis represents predicted labels
- The y-axis represents true (actual) labels
- Correct predictions appear along the diagonal from top-left to bottom-right
- Off-diagonal entries indicate misclassifications

**Example:**

![image](https://github.com/user-attachments/assets/a4dc9840-9820-49ad-852c-7c81224292f8)






---

### MNIST Dataset Links

**Handwritten Digits:** https://www.kaggle.com/datasets/hojjatk/mnist-dataset
**Fashion:** https://www.kaggle.com/datasets/zalando-research/fashionmnist

