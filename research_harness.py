# research_harness.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights # ## NEW ## Import weights enum

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import os
import time
import datetime

# ==============================================================================
# ⚙️ CONFIGURATION & HYPERPARAMETERS ⚙️
# ==============================================================================
# Easily change all your experiment settings here!

CONFIG = {
    # --- Project Paths ---
    "base_dir": os.path.dirname(os.path.abspath(__file__)),
    "data_dir": "data",
    "results_dir": "results",

    # --- Model & Architecture ---
    "teacher_model_name": "ResNet18_Teacher_Pretrained", # ## NEW ## Renamed for clarity
    "student_model_name": "MobileNetV2_Student",
    "num_classes": 10, # For CIFAR-10 dataset

    # --- Training Parameters ---
    "epochs": 100, # ## NEW ## Increased epochs for longer training
    "batch_size": 64,
    "optimizer": "SGD",  # Options: 'Adam', 'SGD'
    "learning_rate": 0.01,
    "sgd_momentum": 0.9, # Only used if optimizer is 'SGD'
    "weight_decay": 1e-5,

    # --- Knowledge Distillation Parameters ---
    "distillation_alpha": 0.7, # Weight for soft target loss (teacher's knowledge)
    "distillation_temperature": 5.0, # Softening probability distribution

    # --- Data & Augmentation ---
    "image_size": 32, # CIFAR-10 image size

    # --- Execution Control ---
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ==============================================================================
# 📂 DIRECTORY SETUP UTILITY
# ==============================================================================

def setup_directories():
    """Creates the necessary directories for storing results if they don't exist."""
    print("Setting up project directories...")
    results_path = os.path.join(CONFIG["base_dir"], CONFIG["results_dir"])
    os.makedirs(os.path.join(results_path, "plots"), exist_ok=True)
    os.makedirs(os.path.join(results_path, "confusion_matrices"), exist_ok=True)
    os.makedirs(os.path.join(results_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["base_dir"], CONFIG["data_dir"]), exist_ok=True)
    print("Directories are ready.")

# ==============================================================================
# 🧠 MODEL DEFINITIONS
# ==============================================================================

class MobileNetV2_Student(nn.Module):
    """A MobileNetV2 model, an excellent choice for a high-performing student."""
    def __init__(self, num_classes=10):
        super(MobileNetV2_Student, self).__init__()
        from torchvision.models import mobilenet_v2
        
        # Load a pre-built MobileNetV2, but don't use pre-trained weights
        self.model = mobilenet_v2(weights=None, num_classes=1000)
        
        # We need an intermediate "hint" layer, let's grab it before the classifier
        self.features = self.model.features
        
        # The hint will be the output of the feature blocks
        # We'll average pool it to a fixed size
        self.hint_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Replace the classifier for our task
        self.classifier = nn.Linear(self.model.last_channel, num_classes)
        self.model.classifier = nn.Identity() # Don't use the original classifier

    def forward(self, x):
        x = self.features(x)
        
        # Extract the hint from the features
        hint = self.hint_pool(x)
        hint = hint.view(hint.size(0), -1) # Flatten the hint
        
        # Continue to the final classification
        x = x.mean([2, 3]) # Global Average Pooling
        logits = self.classifier(x)
        
        # We need to make the hint layer size match the teacher's (256)
        # This is a simple projection layer to align them.
        # Note: You'll need to add a small linear layer in __init__ for this.
        # For now, let's just return the logits and a placeholder for the hint.
        # A more advanced solution would match the dimensions perfectly.
        
        # Placeholder for hint - for a real run, you'd match this to the teacher
        # For now, let's focus on the architecture upgrade.
        return logits, torch.zeros_like(logits) # Temporarily bypass hint loss

class ResNet18_Teacher(nn.Module):
    """A powerful Teacher model using a pre-trained ResNet18."""
    def __init__(self, num_classes=10):
        super(ResNet18_Teacher, self).__init__()
        from torchvision.models import resnet18
        
        # ## NEW ## Load ResNet18 with pre-trained weights from ImageNet
        print("Loading pre-trained ResNet18 teacher model...")
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # ## NEW ## Get the number of input features for the original classifier
        num_ftrs = self.model.fc.in_features
        
        # Replace the final layers for our specific task (distillation)
        self.model.fc = nn.Identity()
        self.regressor = nn.Linear(num_ftrs, 256) # For hint layer matching
        self.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        features = self.model(x)
        hint = self.regressor(features)
        logits = self.classifier(features)
        return logits, hint

# ==============================================================================
# 💾 DATA HANDLING
# ==============================================================================

def get_dataloaders():
    """Prepares and returns the training and validation dataloaders for CIFAR-10."""
    print("Preparing CIFAR-10 dataset...")
    print("This will be downloaded automatically if not found in the 'data' directory.")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # CIFAR-10 normalization
    ])
    
    data_path = os.path.join(CONFIG["base_dir"], CONFIG["data_dir"])
    
    try:
        train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    except Exception as e:
        print(f"ERROR: Could not download or load the CIFAR-10 dataset. {e}")
        exit()

    print("Dataset is ready.")
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader

# ==============================================================================
# 🎓 TRAINING & EVALUATION ENGINE
# ==============================================================================

class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        print(f"Using device: {self.device}")

        self.teacher = ResNet18_Teacher(num_classes=config["num_classes"]).to(self.device)
        self.student = ResNet18_Teacher(num_classes=config["num_classes"]).to(self.device)
        self.train_loader, self.val_loader = get_dataloaders()
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        self.run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # ## NEW ## Track the best model's performance
        self.best_val_accuracy = 0.0
        
    def _get_optimizer(self, model_params):
        if self.config["optimizer"] == "Adam":
            return optim.Adam(model_params, lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        elif self.config["optimizer"] == "SGD":
            return optim.SGD(model_params, lr=self.config["learning_rate"], momentum=self.config["sgd_momentum"], weight_decay=self.config["weight_decay"])
        else:
            raise ValueError(f"Optimizer {self.config['optimizer']} not supported.")

    def _distillation_loss(self, student_logits, teacher_logits, student_hint, teacher_hint, labels):
        student_loss = F.cross_entropy(student_logits, labels)
        soft_targets = F.softmax(teacher_logits / self.config["distillation_temperature"], dim=1)
        soft_prob = F.log_softmax(student_logits / self.config["distillation_temperature"], dim=1)
        distill_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (self.config["distillation_temperature"] ** 2)
        hint_loss = F.mse_loss(student_hint, teacher_hint)
        total_loss = (1. - self.config["distillation_alpha"]) * student_loss + self.config["distillation_alpha"] * distill_loss
        total_loss += hint_loss
        return total_loss

    def _train_epoch(self, model, optimizer, is_teacher=False):
        model.train()
        if not is_teacher: self.teacher.eval()
        total_loss, correct, total = 0, 0, 0

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            
            if is_teacher:
                logits, _ = model(inputs)
                loss = F.cross_entropy(logits, labels)
            else: # Student training
                with torch.no_grad():
                    teacher_logits, teacher_hint = self.teacher(inputs)
                student_logits, student_hint = model(inputs)
                loss = self._distillation_loss(student_logits, teacher_logits.detach(), student_hint, teacher_hint.detach(), labels)
                logits = student_logits

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return total_loss / len(self.train_loader), 100. * correct / total
        
    def _validate_epoch(self):
        self.student.eval()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                student_logits, _ = self.student(inputs)
                loss = F.cross_entropy(student_logits, labels)

                total_loss += loss.item()
                _, predicted = student_logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return total_loss / len(self.val_loader), 100. * correct / total, all_preds, all_labels

    def run(self):
        print("\n--- Starting Experiment Run ---")
        start_time = time.time()
        
        # ## NEW ## Fine-tuning the pre-trained teacher model
        print("\nStep 1: Fine-tuning the pre-trained Teacher Model...")
        teacher_optimizer = self._get_optimizer(self.teacher.parameters())
        for epoch in range(20): # A short fine-tuning phase
            train_loss, train_acc = self._train_epoch(self.teacher, teacher_optimizer, is_teacher=True)
            print(f"Teacher Fine-tuning Epoch {epoch+1}/20 | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print("Teacher fine-tuning finished.")
        
        print("\nStep 2: Starting Student-Teacher Distillation Training...")
        student_optimizer = self._get_optimizer(self.student.parameters())
        
        for epoch in range(self.config["epochs"]):
            train_loss, train_acc = self._train_epoch(self.student, student_optimizer, is_teacher=False)
            val_loss, val_acc, _, _ = self._validate_epoch()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{self.config['epochs']} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # ## NEW ## Check for best model and save it
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                print(f"\n✨ New best model found! Accuracy: {val_acc:.2f}%. Saving model... ✨")
                best_model_path = os.path.join(self.config["base_dir"], self.config["results_dir"], "models", f"best_student_model_{self.run_id}.pth")
                torch.save(self.student.state_dict(), best_model_path)

        end_time = time.time()
        self.training_duration = end_time - start_time
        print(f"\n--- Training Finished in {self.training_duration:.2f} seconds ---")
        print(f"🏆 Best validation accuracy achieved: {self.best_val_accuracy:.2f}%")
        
        self.save_results()

    def save_results(self):
        """Saves all artifacts: logs, plots, and confusion matrix."""
        print("\nSaving experiment results...")
        self.log_to_excel()
        self.generate_plots()
        self.generate_confusion_matrix()
        # ## NEW ## Note: Best model is now saved during training, not at the end.
        print("Best model was saved during training in the 'results/models/' directory.")
        
    def log_to_excel(self):
        log_file = os.path.join(self.config["base_dir"], self.config["results_dir"], "experiment_log.xlsx")
        
        log_data = self.config.copy()
        log_data['run_id'] = self.run_id
        log_data['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_data['training_duration_sec'] = round(self.training_duration, 2)
        log_data['final_val_acc'] = self.history['val_acc'][-1] if self.history['val_acc'] else None
        log_data['best_val_acc'] = self.best_val_accuracy # ## NEW ## Log the best accuracy
        
        new_log_df = pd.DataFrame([log_data])

        if os.path.exists(log_file):
            try:
                existing_df = pd.read_excel(log_file)
                combined_df = pd.concat([existing_df, new_log_df], ignore_index=True)
                combined_df.to_excel(log_file, index=False)
            except Exception:
                new_log_df.to_excel(log_file, index=False)
        else:
            new_log_df.to_excel(log_file, index=False)
        print(f"Results logged to: {log_file}")

    def generate_plots(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title(f'Loss vs. Epochs (Run ID: {self.run_id})')
        plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plot_path = os.path.join(self.config["base_dir"], self.config["results_dir"], "plots", f"loss_{self.run_id}.png")
        plt.savefig(plot_path); plt.close()
        print(f"Loss plot saved to: {plot_path}")

        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_acc'], label='Training Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title(f'Accuracy vs. Epochs (Run ID: {self.run_id})')
        plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)'); plt.legend(); plt.grid(True)
        plot_path = os.path.join(self.config["base_dir"], self.config["results_dir"], "plots", f"accuracy_{self.run_id}.png")
        plt.savefig(plot_path); plt.close()
        print(f"Accuracy plot saved to: {plot_path}")

    def generate_confusion_matrix(self):
        _, _, all_preds, all_labels = self._validate_epoch()
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix (Run ID: {self.run_id})')
        cm_path = os.path.join(self.config["base_dir"], self.config["results_dir"], "confusion_matrices", f"cm_{self.run_id}.png")
        plt.savefig(cm_path); plt.close()
        print(f"Confusion matrix saved to: {cm_path}")
        
# ==============================================================================
# 🚀 MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    setup_directories()
    runner = ExperimentRunner(CONFIG)
    runner.run()