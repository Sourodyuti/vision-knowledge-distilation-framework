# DistillKit: A PyTorch Knowledge Distillation Research Harness

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?style=for-the-badge&logo=pytorch)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

A flexible and reusable PyTorch research harness for exploring knowledge distillation with a pre-trained teacher (ResNet-18) and a lightweight student (MobileNetV2) on CIFAR-10.

This framework is designed for researchers and practitioners to easily experiment with and understand the dynamics of knowledge distillation, a powerful model compression technique. The entire experimental setup is controlled through a single, easy-to-use configuration dictionary.

---

## 🔬 Key Features

* **Teacher-Student Framework**: Implements a classic knowledge distillation pipeline with a powerful, pre-trained teacher model (`ResNet-18`) and a lightweight student model (`MobileNetV2`).
* **Hybrid Distillation Loss**: Utilizes a sophisticated loss function that combines standard cross-entropy with "soft targets" from the teacher and a "hint-based" loss to guide the student's intermediate feature representations.
* **Centralized Configuration**: All hyperparameters, model names, and paths are managed in a single `CONFIG` dictionary in `research_harness.py` for easy experimentation.
* **Automated Results Tracking**: Automatically saves all experimental artifacts, including:
    * Training & validation plots (Loss & Accuracy).
    * Final confusion matrices.
    * The best-performing student model weights (`.pth`).
    * A comprehensive `experiment_log.xlsx` file to track all runs and their configurations.
* **Reproducibility**: Designed to make experiments easy to run, track, and reproduce.

---

## 🚀 Getting Started

Follow these instructions to get the project set up and running on your local machine.

### Prerequisites

* Python 3.8 or higher
* NVIDIA GPU with CUDA support (recommended)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/DistillKit.git](https://github.com/your-username/DistillKit.git)
    cd DistillKit
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file. See the section below.)*

---

## ⚙️ How to Run

Running an experiment is as simple as executing the main script.

1.  **Configure your experiment:**
    Open `research_harness.py` and modify the `CONFIG` dictionary to set your desired hyperparameters (e.g., learning rate, batch size, epochs, distillation alpha/temperature).

2.  **Run the script:**
    ```bash
    python research_harness.py
    ```

The script will automatically set up the required directories, download the CIFAR-10 dataset (if not present), fine-tune the teacher model, and then train the student model using distillation. All results will be saved in the `results/` directory, timestamped with a unique `run_id`.

---

## 🔧 Configuration

The `CONFIG` dictionary is the control center for all experiments.

```python
CONFIG = {
    # --- Project Paths ---
    "base_dir": os.path.dirname(os.path.abspath(__file__)),
    "data_dir": "data",
    "results_dir": "results",

    # --- Model & Architecture ---
    "teacher_model_name": "ResNet18_Teacher_Pretrained",
    "student_model_name": "MobileNetV2_Student",
    "num_classes": 10,

    # --- Training Parameters ---
    "epochs": 100,
    "batch_size": 64,
    "optimizer": "SGD",
    "learning_rate": 0.01,

    # --- Knowledge Distillation Parameters ---
    "distillation_alpha": 0.7,
    "distillation_temperature": 5.0,
    
    # --- Execution Control ---
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
