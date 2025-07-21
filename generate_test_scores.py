from enum import Enum
import os
import time
import gc
import random
import warnings
import hashlib
from copy import deepcopy
import pandas as pd
import numpy as np
import psutil
import matplotlib
matplotlib.use('Agg')  # for headless environments (e.g., saving figures)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import defaultdict

# TQDM
from tqdm.notebook import tqdm  # Or replace with `from tqdm import tqdm` if not using notebook
from tqdm import trange

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Transformers / Hugging Face
from transformers import (
    RobertaTokenizer,
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    BertForPreTraining,
    BertModel,
    get_linear_schedule_with_warmup
)

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import AdamW

# Filter warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class DatasetType(Enum):
    EDGEIIOT = "edgeiiot"
    CICIDS2017 = "cicids2017"

class ValueType(Enum):
    PPFLE = "ppfle"
    FLAT = "flat"
    P_INT = "p_int"

class CustomDataset(Dataset):
  def __init__(self, df, tokenizer, max_len, value_type=ValueType):
    self.df = df
    self.tokenizer = tokenizer
    self.max_len = max_len

    if value_type == ValueType.PPFLE:
        self.sequence = self.df['encoded_PPFLE'].tolist()
    else:
        self.sequence = self.df['raw_input'].tolist()

    self.targets = self.df['target'].tolist()

  def __len__(self):
    return len(self.df)

  def __getitem__(self,idx):
    sequence = str(self.sequence[idx])
    target = self.targets[idx]
    encoding = self.tokenizer.encode_plus(
        sequence,
        add_special_tokens=True,
        max_length=self.max_len,
        padding='max_length',
        return_token_type_ids=False,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    return {
        'input_ids':encoding['input_ids'].flatten(),
        'attention_mask':encoding['attention_mask'].flatten(),
        'targets':torch.tensor(target,dtype=torch.long)
    }
  
class SecurityBERT(nn.Module):
  def __init__(self,finetunedBERT,n_classes):
    super(SecurityBERT,self).__init__()
    self.bert = finetunedBERT
    self.dropout = nn.Dropout(p=0.1)
    self.out = nn.Linear(self.bert.config.hidden_size,n_classes)

  def forward(self,input_ids,attention_mask):
    pooled_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
    ).pooler_output

    output = self.dropout(pooled_output)

    return self.out(output)
  
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use the last hidden state
        out = self.fc(lstm_out[:, -1, :])
        return out

class GenerateTestScores():

    def save_scaler(model_directory, X_train, y_train, X_val, y_val, dataset_type = DatasetType):
        file_name = "_EDGEIIOT"
        if (dataset_type == DatasetType.CICIDS2017):
            file_name = "_CICIDS2017"

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Save the scaler to a file
        joblib.dump(scaler, f'./{model_directory}scaler{file_name}.joblib')

    def get_predictions_ml(model, X_test, y_test):
        predictions = []
        predictions_probs = []
        real_values = []
        total_time = 0.0
        total_samples = len(X_test)

        for i in tqdm(range(total_samples), desc="Predictions"):
            x = X_test[i].reshape(1, -1)  # Reshape single sample
            y = y_test[i]

            start_time = time.perf_counter()

            # Predict class and probability
            pred = model.predict(x)[0]
            prob = model.predict_proba(x)[0]

            end_time = time.perf_counter()
            total_time += (end_time - start_time)

            predictions.append(pred)
            predictions_probs.append(prob)
            real_values.append(y)

        predictions = np.array(predictions)
        predictions_probs = np.array(predictions_probs)
        real_values = np.array(real_values)

        # Calculate average inference time per sample in milliseconds
        avg_inference_time_ms = (total_time / total_samples) * 1000

        print(f"Average per-sample inference time: {avg_inference_time_ms:.4f} ms")

        return predictions, predictions_probs, real_values, avg_inference_time_ms
    
    def get_predictions_dl(model, test_loader):
        model.eval()

        predictions, predictions_probs, real_values = [], [], []
        total_time = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predictions"):
                inputs = batch[0].to(device)
                targets = batch[1].to(device)

                start_time = time.perf_counter()

                outputs = model(inputs)                     # Forward pass
                                                            # Should output logits: (batch_size, num_classes)

                end_time = time.perf_counter()

                # Tracking time and sample count
                batch_size = inputs.size(0)
                total_samples += batch_size
                total_time += (end_time - start_time)

                probs = F.softmax(outputs, dim=1)           # Get class probabilities
                _, preds = torch.max(probs, dim=1)          # Predicted class labels

                predictions.extend(preds.cpu())
                predictions_probs.extend(probs.cpu())
                real_values.extend(targets.cpu())

        predictions = torch.stack(predictions)
        predictions_probs = torch.stack(predictions_probs)
        real_values = torch.stack(real_values)

        avg_inference_time_ms = (total_time / total_samples) * 1000
        print(f"Average per-sample inference time: {avg_inference_time_ms:.4f} ms")

        return predictions, predictions_probs, real_values, avg_inference_time_ms

    def get_predictions_tt(model, X_test, y_test, device):
        """
        Vectorized prediction for TabTransformer.
        
        Args:
            model: Trained TabularModel.
            X_test: Test features (DataFrame or NumPy array).
            y_test: True labels (array-like).
            device: "cpu" or "cuda".
            
        Returns:
            predictions: Tensor of predicted class labels.
            predictions_probs: Tensor of predicted probabilities.
            real_values: Tensor of true labels.
            avg_inference_time_ms: Average inference time per sample in ms.
        """

        model.model.eval()

        # Ensure DataFrame input
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test)
        if isinstance(y_test, np.ndarray):
            y_test = pd.Series(y_test)

        test_df = X_test.copy()
        test_df["target"] = y_test.reset_index(drop=True)

        # Inference timing
        start_time = time.perf_counter()
        pred_df = model.predict(test_df, ret_logits=False)  # batch prediction
        end_time = time.perf_counter()

        # Extract probability columns
        prob_cols = [col for col in pred_df.columns if col.endswith("_probability")]
        probs_array = pred_df[prob_cols].values
        preds_array = np.argmax(probs_array, axis=1)

        # Convert to tensors
        predictions = torch.tensor(preds_array, dtype=torch.long)
        predictions_probs = torch.tensor(probs_array, dtype=torch.float32)
        real_values = torch.tensor(y_test.values, dtype=torch.long)

        avg_inference_time_ms = ((end_time - start_time) / len(test_df)) * 1000
        print(f"Batch inference complete. Avg inference time per sample: {avg_inference_time_ms:.4f} ms")
        return predictions, predictions_probs, real_values, avg_inference_time_ms

    def plot_test_metrics_table(test_metrics, figure_version, num_epochs, figure_file_name):
        """
        Plots and saves a table of test metrics as a figure.

        Parameters:
        - test_metrics: defaultdict or dict with scalar values
        - figure_version: str (e.g., 'v1.0')
        - num_epochs: int
        - figure_file_name: str (e.g., 'tab-transformer')
        """

        # Define a mapping from metric keys to display names
        metric_name_map = {
            "micro_acc": "Overall Accuracy",
            "micro_rec": "Overall Recall",
            "micro_prec": "Overall Precision",
            "micro_f1": "Overall F1 Score",
            "micro_roc_auc": "Overall ROC AUC",
            "macro_acc": "Class-Averaged Accuracy",
            "macro_rec": "Class-Averaged Recall",
            "macro_prec": "Class-Averaged Precision",
            "macro_f1": "Class-Averaged F1 Score",
            "macro_roc_auc": "Class-Averaged ROC AUC",
            "pretrain_time": "Pre-training Time (h)",
            "finetune_time": "Classification Training Time (h)",
            "inference_time": "Inference Latency (ms)"
            # Add more mappings as needed
        }

        # Convert metrics to a list of [metric_name, value] rows
        data = []
        for key, value in test_metrics.items():
            label = metric_name_map.get(key, key)  # Use key as fallback if not in map
            if (value > -100.0):
                data.append([label, f"{value:.4f}"])
            else:
                data.append([label, "-"])

        fig, ax = plt.subplots(figsize=(8, len(data) * 0.4 + 1))
        ax.axis('off')

        table = ax.table(
            cellText=data,
            colLabels=["Metric", "Score"],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)

        plt.title(f"{figure_version} {num_epochs} Epochs Test Scores", fontsize=14, pad=20)

        output_path = f"./figures/{figure_file_name}-{num_epochs}ep-test-scores.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.clf()
    
    def latex_escape(text: str) -> str:
        """Escapes LaTeX special characters for safe rendering inside $\\textit{}$."""
        replacements = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\^{}',
        }
        for char, escaped in replacements.items():
            text = text.replace(char, escaped)
        return text

    def plot_all_test_metrics_tables(
        test_metrics_list: list[dict],
        figure_version: str,
        figure_file_name: str,
        row_labels: list[str]
    ):
        """
        Plots and saves a transposed table of test metrics from multiple runs.

        Parameters:
        - test_metrics_list: List of dicts (or defaultdicts) with scalar values.
        - figure_version: str (e.g., 'v1.0')
        - num_epochs: int
        - figure_file_name: str (e.g., 'tab-transformer')
        - row_labels: list of str identifying each row (e.g., model names)
        """

        metric_name_map = {
            "micro_acc": "Overall\nAccuracy",
            "micro_rec": "Overall\nRecall",
            "micro_prec": "Overall\nPrecision",
            "micro_f1": "Overall\nF1 Score",
            "micro_roc_auc": "Overall\nROC AUC",
            "macro_acc": "Class-Averaged\nAccuracy",
            "macro_rec": "Class-Averaged\nRecall",
            "macro_prec": "Class-Averaged\nPrecision",
            "macro_f1": "Class-Averaged\nF1 Score",
            "macro_roc_auc": "Class-Averaged\nROC AUC",
            "pretrain_time": "Pre-training\nTime (h)",
            "finetune_time": "Classification\nTraining Time (h)",
            "inference_time": "Inference\nLatency (ms)"
        }

        # All unique metric keys from all test_metrics dictionaries
        if test_metrics_list:
            all_keys = list(test_metrics_list[0].keys())
            # Optionally add keys from other dicts not in the first one, appended at the end:
            other_keys = {k for metrics in test_metrics_list for k in metrics.keys()} - set(all_keys)
            all_keys.extend(sorted(other_keys))
        else:
            all_keys = []
        headers = [""] + [metric_name_map.get(k, k.replace("_", "\n")) for k in all_keys]
        print(len(headers))
        # Prepare headers safely:
        clean_headers = []
        for h in headers:
            lines = h.split('\n')  # Split multi-line headers
            lines_escaped = [f"\\textit{{{GenerateTestScores.latex_escape(line.strip())}}}" for line in lines]
            stacked = " \\\\ ".join(lines_escaped)
            clean_headers.append(f"$\\shortstack{{{stacked}}}$")

        headers = clean_headers

        # Data rows: one row per test_metrics instance, including row label
        data = []
        for label, metrics in zip(row_labels, test_metrics_list):
            row = [f"\\textit{{{GenerateTestScores.latex_escape(label)}}}"]  # Italicized row label
            for k in all_keys:
                val = metrics.get(k, float("nan"))
                if isinstance(val, float) and val > -100.0:
                    row.append(f"{val:.4f}")
                else:
                    row.append("-")
            data.append(row)

        # Enable LaTeX rendering
        plt.rcParams['text.usetex'] = True

        base_col_width = 1.2
        num_cols = len(headers)  # includes the empty first header
        first_col_width = 2.5 * base_col_width
        other_cols_width = base_col_width * (num_cols - 1)
        total_width = first_col_width + other_cols_width

        # Plot setup
        fig, ax = plt.subplots(figsize=(total_width, len(data) * 0.6 + 1))
        
        ax.axis("off")

        table = ax.table(
            cellText=data,
            colLabels=headers,
            loc="center",
            cellLoc="center",
            colColours=["#f2f2f2"] * len(headers),
        )

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)  # general scaling for readability

        # ⬇ Double the height of just the header row (row=0)
        for (row, col), cell in table.get_celld().items():
            if row == 0:  # header row
                cell.set_height(cell.get_height() * 2.0)
            if col == 0:
                cell.set_width(cell.get_width() * 2.0)

        plt.title(f"{figure_version} Test Metrics", fontsize=14, pad=20)
        output_path = f"./figures/{figure_file_name}-multi-test-scores.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.clf()

    def plot_model_histories_extended(histories, model_names):
        """
        Plots training/validation loss and accuracy for 8 histories on 3x2 subplots.

        Parameters:
        - histories: List of 8 defaultdict(list) objects.
        - model_names: List of 6 strings corresponding to the first 6 histories.

        Returns:
        - None (displays the plot)
        """
        assert len(histories) == 23, "Expected 23 history objects"
        assert len(model_names) == 19, "Expected 19 model names"

        # Color palette: c1-c6
        c1, c2, c3, c4, c5, c6, c7 = "crimson","darkorange","gold","limegreen","turquoise","dodgerblue","slateblue"
        c8, c9, c10, c11, c12, c13, c14 = "indigo","orchid","deeppink","saddlebrown","dimgray","black","mediumseagreen"
        colors = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c1, c2, c3, c4, c5, c6, c7, c9, c10, c11, c12, c13, c14]  # hist1–hist8

        fig, axs = plt.subplots(3, 2, figsize=(14, 15))
        axs = axs.flatten()

        # Titles for subplots
        pretrain_title = "Pre-training"

        titles = [
            "EdgeIIoT - Loss",                 # subplot 0
            "EdgeIIoT - Accuracy",             # subplot 1
            pretrain_title + " - Loss",        # subplot 2
            pretrain_title + " - Accuracy",    # subplot 3
            "CIC-IDS2017 - Loss",              # subplot 4
            "CIC-IDS2017 - Accuracy"           # subplot 5
        ]

        for i, ax in enumerate(axs):
            ax.set_title(GenerateTestScores.latex_escape(titles[i]))
            ax.set_xlabel("Epoch")
            ax.grid(True)

            # Top row: EdgeIIoT (histories 0–2)
            if i in [0, 1]:
                idxs = range(10)

            # Bottom row: CIC-IDS2017 (histories 3–5)
            elif i in [4, 5]:
                idxs = range(10, 19)

            # Middle row (new): hist7 + hist8
            elif i in [2, 3]:
                idxs = range(19, 23)  # both histories shown in same subplot

            handles, labels = [], []

            for j in idxs:
                hist = histories[j]
                color = colors[j]

                # Use dataset name for hist7/hist8
                if j == 19 or j == 21:
                    if j == 19:
                        base_label = GenerateTestScores.latex_escape(model_names[5]) + " EdgeIIoT"
                    else:
                        base_label = GenerateTestScores.latex_escape(model_names[8]) + " EdgeIIoT"
                elif j == 20 or j == 22:
                    if j == 20:
                        base_label = GenerateTestScores.latex_escape(model_names[5]) + " CIC-IDS2017"
                    else:
                        base_label = GenerateTestScores.latex_escape(model_names[8]) + " CIC-IDS2017"
                else:
                    base_label = GenerateTestScores.latex_escape(model_names[j])

                # Choose metric based on column (even index = loss, odd = acc)
                if i % 2 == 0:
                    tr_vals = hist['train_loss']
                    val_vals = hist['val_loss']
                else:
                    tr_vals = hist['train_acc']
                    val_vals = hist['val_acc']

                tr_line, = ax.plot(range(1, 4), tr_vals, '-', color=color, label=f"{base_label} (train)")
                val_line, = ax.plot(range(1, 4), val_vals, '--', color=color, label=f"{base_label} (val)")

                handles.extend([tr_line, val_line])
                labels.extend([f"{base_label} (train)", f"{base_label} (val)"])

            ax.legend(handles, labels, fontsize='small', ncol=1)

        plt.tight_layout()
        plt.savefig(f'./figures/training-and-validation-loss-and-accuracy.png',dpi=300)
        plt.clf()


    def show_confusion_matrix(confusion_matrix, annot, confusion_matrix_fig_size, figure_version, figure_file_name, num_epochs):

        plt.figure(figsize=confusion_matrix_fig_size)
        sns.heatmap(confusion_matrix, annot=annot, cmap='Blues', fmt='')
        plt.xticks(rotation=90)
        plt.title(f"{figure_version} {num_epochs} Epochs Confusion Matrix")
        plt.ylabel('Real threats')
        plt.xlabel('Predicted threats')
        plt.savefig(f'./figures/{figure_file_name}-{num_epochs}ep-confusion-matrix.png',bbox_inches="tight",dpi=300)
        plt.clf()

    def get_predictions(model, test_loader):
        model = model.eval()

        predictions, predictions_probs, real_values = [], [], []
        total_time = 0.0
        total_samples = 0

        with torch.no_grad():
            for data in tqdm(test_loader, desc="Predictions"):
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                targets = data['targets'].to(device)

                start_time = time.perf_counter()

                outputs = model(input_ids, attention_mask)

                end_time = time.perf_counter()

                # Track time and sample count
                batch_size = input_ids.size(0)
                total_samples += batch_size
                total_time += (end_time - start_time)

                _, preds = torch.max(outputs, dim=1)
                probs = F.softmax(outputs, dim=1)

                predictions.extend(preds)
                predictions_probs.extend(probs)
                real_values.extend(targets)

        # Stack outputs
        predictions = torch.stack(predictions).cpu()
        predictions_probs = torch.stack(predictions_probs).cpu()
        real_values = torch.stack(real_values).cpu()

        # Calculate average inference time per sample (in milliseconds)
        avg_inference_time_ms = (total_time / total_samples) * 1000

        print(f"Average per-sample inference time: {avg_inference_time_ms:.4f} ms")

        return predictions, predictions_probs, real_values, avg_inference_time_ms


    def safe_load_tensor(path):
        try:
            data = torch.load(path, weights_only=False)
            return data
        except FileNotFoundError:
            print(f"File not found: {path}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
        return None  # fallback value
    
    def get_history_value(history: dict, key: str, default=None):
        try:
            return history[key]
        except (KeyError, IndexError):
            print(f"Missing entry: key='{key}'. Returning default: {default}")
            return default
        
    def replace_finetuned_with_pretrained(s: str) -> str:
        if "bertFinetuned_" in s:
            return s.replace("bertFinetuned_", "bertPretrained_")
        return s

    def collect_test_results(
        model_directory: str,
        model_version: str,
        num_epochs: str,
        keep_frac: float,
        figure_version: str,
        figure_file_name: str,
        test_set,
        test_loader,
        TARGET_LIST: list,
        dataset_type: DatasetType,
        le: LabelEncoder,
    ):
        if (dataset_type == DatasetType.EDGEIIOT):
            confusion_matrix_fig_size = (10, 8)
            VOCAB_SIZE = 5000
        else:
            confusion_matrix_fig_size = (20,12)
            VOCAB_SIZE = 10000
        
        history = GenerateTestScores.safe_load_tensor(f"{model_directory}history_{model_version}.pt")

        # Inspect keys and sample values
        history_keys = list(history.keys())
        sample_values = {
            k: history[k][:3] if isinstance(history[k], list) else history[k]
            for k in history_keys
        }

        print(f"Available keys for history of {model_version}:", history_keys)
        for key, value in sample_values.items():
            print(f"{key}: {value}")

        epochs = range(1, 3+1)
        train_losses = history['train_loss']
        val_losses = history['val_loss']

        # Création du graphique
        plt.figure(figsize=(10, 6))

        # Tracé des courbes de training et de validation losses
        sns.lineplot(x=epochs, y=train_losses, label='Training Loss',marker="o")
        sns.lineplot(x=epochs, y=val_losses, label='Validation Loss',marker="o")

        # Ajout de titres et de légendes
        plt.title(f'{figure_version} Training and Validation Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'./figures/{figure_file_name}-{num_epochs}ep-training-validation-losses.png',dpi=300)
        plt.clf()
        # Affichage du graphique
        
        if isinstance(history["train_acc"][0], float):
            train_accuracies = [float_ for float_ in history['train_acc']]
            val_accuracies = [float_ for float_ in history['val_acc']]
        else:
            train_accuracies = [tens.item() for tens in history['train_acc']]
            val_accuracies = [tens.item() for tens in history['val_acc']]

        # Création du graphique
        plt.figure(figsize=(10, 6))

        # Tracé des courbes de training et de validation losses
        sns.lineplot(x=epochs, y=train_accuracies, label='Training accuracy',marker='o')
        sns.lineplot(x=epochs, y=val_accuracies, label='Validation accuracy',marker='o')

        # Ajout de titres et de légendes
        plt.title(f'{figure_version} Training and Validation Accuracies')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'./figures/{figure_file_name}-{num_epochs}ep-training-validation-accuracies.png',dpi=300)
        plt.clf()
        # Affichage du graphique
        
        y_pred_path = os.path.join(model_directory, f"y_pred_{model_version}.pt")
        y_proba_path = os.path.join(model_directory, f"y_proba_{model_version}.pt")
        real_values_path = os.path.join(model_directory, f"real_values_{model_version}.pt")
        inference_time_path = os.path.join(model_directory, f"inference_time_{model_version}.pt")

        y_pred = GenerateTestScores.safe_load_tensor(y_pred_path)
        y_proba = GenerateTestScores.safe_load_tensor(y_proba_path)
        real_values = GenerateTestScores.safe_load_tensor(real_values_path)
        inference_time = GenerateTestScores.safe_load_tensor(inference_time_path)

        if ((y_pred is None) or (y_proba is None) or (real_values is None) or (inference_time is None)):
            checkpoint_path = f"./{model_directory}{model_version}_{num_epochs}.0.pt"
            model_path = f"./{model_directory}{model_version}_{num_epochs}.0.pkl"

            if model_version.startswith("tabTransformer"):
                # Do something
                print("This is a tabTransformer model.")
            elif model_version.startswith("baseline_LSTM"):
                print("This is an LSTM model.")

                input_dim = test_set.shape[1]
                test_set = test_set.reshape((test_set.shape[0], 1, test_set.shape[1]))

                test_dataset = TensorDataset(
                    torch.tensor(test_set, dtype=torch.float32),
                    torch.tensor(test_loader, dtype=torch.long)
                )
                batch_size=64
                test_dataset_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                hidden_dim=64
                lr=0.001
                num_classes = len(le.classes_)
                lstm_cls = LSTMClassifier(input_dim, hidden_dim, num_classes).to(device)
                lstm_cls.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
                try:
                    lstm_cls.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
                    print(f"Model state_dict loaded successfully at \"{checkpoint_path}\"")
                except FileNotFoundError:
                    print(f"Checkpoint not found: \"{checkpoint_path}\"")
                    return None
                except Exception as e:
                    print(f"Error loading model checkpoint: {e}")
                    return None
                y_pred,y_proba,real_values,inference_time = GenerateTestScores.get_predictions_dl(lstm_cls,test_dataset_loader)

                torch.save(y_pred,f"./{model_directory}y_pred_{model_version}.pt")
                torch.save(y_proba,f"./{model_directory}y_proba_{model_version}.pt")
                torch.save(real_values,f"./{model_directory}real_values_{model_version}.pt")
                torch.save(inference_time,f"./{model_directory}inference_time_{model_version}.pt")
            elif model_version.startswith("baseline"):
                print("This is a classical ML baseline model.")
                model = None

                try:
                    model = joblib.load(model_path)
                    print(f"Model state_dict loaded successfully at \"{model_path}\"")
                except FileNotFoundError:
                    print(f"Checkpoint not found: \"{checkpoint_path}\"")
                    return None
                except Exception as e:
                    print(f"Error loading model checkpoint: {e}")
                    return None
                y_pred,y_proba,real_values,inference_time = GenerateTestScores.get_predictions_ml(model,test_set,test_loader)

                torch.save(y_pred,f"./{model_directory}y_pred_{model_version}.pt")
                torch.save(y_proba,f"./{model_directory}y_proba_{model_version}.pt")
                torch.save(real_values,f"./{model_directory}real_values_{model_version}.pt")
                torch.save(inference_time,f"./{model_directory}inference_time_{model_version}.pt")

            else:
                config = AutoConfig.from_pretrained("bert-base-uncased")
                config.hidden_size = 256
                config.num_hidden_layers = 4
                config.num_attention_heads = 16
                config.intermediate_size = 512
                config.vocab_size=VOCAB_SIZE

                finetunedBERT = AutoModel.from_config(config)
                securityBert = SecurityBERT(finetunedBERT=finetunedBERT,n_classes=len(TARGET_LIST)).to(device)

                try:
                    securityBert.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
                    print(f"Model state_dict loaded successfully at \"{checkpoint_path}\"")
                except FileNotFoundError:
                    print(f"Checkpoint not found: \"{checkpoint_path}\"")
                    return None
                except Exception as e:
                    print(f"Error loading model checkpoint: {e}")
                    return None
                y_pred,y_proba,real_values,inference_time = GenerateTestScores.get_predictions(securityBert,test_loader)

                torch.save(y_pred,f"./{model_directory}y_pred_{model_version}.pt")
                torch.save(y_proba,f"./{model_directory}y_proba_{model_version}.pt")
                torch.save(real_values,f"./{model_directory}real_values_{model_version}.pt")
                torch.save(inference_time,f"./{model_directory}inference_time_{model_version}.pt")

        s = set()

        for elt in y_pred:
            s.add(elt.item())

        actual_considered_classes = [TARGET_LIST[i] for i in s]
        
        cm = confusion_matrix(real_values,y_pred)

        def format_k(x):
            return f"{x/1000:.1f}k" if x >= 1000 else str(x)

        # Create formatted annotations
        annot = np.array([[format_k(val) for val in row] for row in cm])
        df_cm = pd.DataFrame(cm,index=TARGET_LIST,columns=TARGET_LIST)
        GenerateTestScores.show_confusion_matrix(df_cm, annot, confusion_matrix_fig_size, figure_version, figure_file_name, num_epochs)
        
        print(classification_report(real_values,y_pred,target_names=TARGET_LIST))

        print(actual_considered_classes)

        counts = None
        if model_version.startswith("baseline"):
            counts = pd.Series(test_loader).value_counts().sort_index()
        else:
            counts = test_set['target'].value_counts().sort_index()
        class_names = le.inverse_transform(counts.index)
        class_counts = pd.Series(counts.values, index=class_names)
        class_counts = class_counts.sort_values(ascending=False)  # or .sort_index()
        print(class_counts)

        y_true = real_values.cpu().numpy() if hasattr(real_values, 'cpu') else real_values
        y_score = y_pred.cpu().numpy() if hasattr(y_pred, 'cpu') else y_pred

        n_classes = len(np.unique(y_true))
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

        test_metrics = defaultdict(float)

        # Micro metrics
        test_metrics["micro_acc"] = accuracy_score(y_true, y_pred)
        try:
            test_metrics["micro_roc_auc"] = roc_auc_score(
                y_true_bin, y_proba, average="micro", multi_class="ovr"
            )
        except ValueError:
            test_metrics["micro_roc_auc"] = float("nan")

        test_metrics["macro_prec"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        test_metrics["macro_rec"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        test_metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro")

        # ROC AUC Scores
        try:
            test_metrics["macro_roc_auc"] = roc_auc_score(
                y_true_bin, y_proba, average="macro", multi_class="ovr"
            )
        except ValueError:
            test_metrics["macro_roc_auc"] = float("nan")

        pretrain_time = -200.0
        finetune_time = -200.0
        pretrain_model_version = GenerateTestScores.replace_finetuned_with_pretrained(model_version)
        pretrain_history = GenerateTestScores.safe_load_tensor(f"./securityBERT/pretrained_model/history_{pretrain_model_version}.pt")

        if (pretrain_history != None):
            if (GenerateTestScores.get_history_value(pretrain_history, key = "training_time") != None and isinstance(pretrain_history["training_time"], list) and len(pretrain_history["training_time"]) > 0):
                pretrain_time = pretrain_history["training_time"][0] / 3600.0
            elif (GenerateTestScores.get_history_value(pretrain_history, key = "training_time") != None and isinstance(pretrain_history["training_time"], float)):
                pretrain_time = pretrain_history["training_time"] / 3600.0
            elif (GenerateTestScores.get_history_value(pretrain_history, key = "total_time") != None and isinstance(pretrain_history["total_time"], list) and len(pretrain_history["total_time"]) > 0):
                pretrain_time = pretrain_history["total_time"][0] / 3600.0
            elif (GenerateTestScores.get_history_value(pretrain_history, key = "total_time") != None and isinstance(pretrain_history["total_time"], float)):
                pretrain_time = pretrain_history["total_time"] / 3600.0

        if (GenerateTestScores.get_history_value(history, key = "training_time") != None and isinstance(history["training_time"], list) and len(history["training_time"]) > 0):
            finetune_time = history["training_time"][0] / 3600.0
        elif (GenerateTestScores.get_history_value(history, key = "training_time") != None and isinstance(history["training_time"], float)):
            finetune_time = history["training_time"] / 3600.0
        elif (GenerateTestScores.get_history_value(history, key = "total_time") != None and isinstance(history["total_time"], list) and len(history["total_time"]) > 0):
            finetune_time = history["total_time"][0] / 3600.0
        elif (GenerateTestScores.get_history_value(history, key = "total_time") != None and isinstance(history["total_time"], float)):
            finetune_time = history["total_time"] / 3600.0

        test_metrics["pretrain_time"] = pretrain_time
        test_metrics["finetune_time"] = finetune_time
        test_metrics["inference_time"] = inference_time

        # Save metrics
        torch.save(dict(test_metrics), f"./{model_directory}test_metrics_{model_version}.pt")

        # Plot
        GenerateTestScores.plot_test_metrics_table(
            test_metrics,
            figure_version=figure_version,
            num_epochs=num_epochs,
            figure_file_name=figure_file_name
        )

        return test_metrics, figure_version, history
    
    def hex_to_byte_array(hex_str):
        if not isinstance(hex_str, str):
            # Handle NaNs or non-strings gracefully
            print(f"{hex_str} not instance of str")
            return []
        # Remove 0x prefix if present
        hex_str = hex_str.strip().lower()
        if hex_str.startswith('0x'):
            hex_str = hex_str[2:]
        # Remove any whitespace
        hex_str = hex_str.replace(' ', '')
        # If empty after cleaning, return empty list
        if len(hex_str) == 0:
            print("length is 0")
            return []
        # If odd length, pad with leading zero
        if len(hex_str) % 2 != 0:
            hex_str = '0' + hex_str
        # Validate characters are hex digits
        if any(c not in '0123456789abcdef' for c in hex_str):
            # Handle invalid characters by returning empty or raise custom error
            print(f"{hex_str} not all in 0123456789abcdef")
            return []
        # Convert hex string to byte list
        byte_array = bytes.fromhex(hex_str)
        return list(byte_array)

    def generate_test_split(data: pd.DataFrame, dataset_type: DatasetType, value_type: ValueType):
        if (dataset_type == DatasetType.EDGEIIOT):
            label_col = "Attack_type"
            if value_type == ValueType.PPFLE:
                tokenizer_file_name = f"./securityBERT/tokenizer"
            elif value_type == ValueType.FLAT:
                tokenizer_file_name = f"./securityBERT/tokenizer_raw"
            data_figure_title = "Edge-IIoT"
            data_figure_file_name = "./figures/edgeiiot"
            scaler_name = "_EDGEIIOT"
        else:
            label_col = "Label"
            if value_type == ValueType.PPFLE:
                tokenizer_file_name = f'./securityBERT/tokenizer_CICIDS2017_0.02samples'
            elif value_type == ValueType.FLAT:
                tokenizer_file_name = f"./securityBERT/tokenizer_CICIDS2017_raw_0.02samples"
            data_figure_title = "Downsampled CIC-IDS2017 2% Samples"
            data_figure_file_name = "./figures/cicids2017-0.02samples"
            scaler_name = "_CICIDS2017"
        
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15

        if (value_type != ValueType.P_INT):
            data_order = data[label_col].value_counts().index
            sns.countplot(data,x=label_col, order=data_order)
            plt.xticks(rotation=90)
            plt.title(f"{data_figure_title} Population by Class")
            plt.savefig(f'{data_figure_file_name}-full-data-distribution.png',bbox_inches="tight",dpi=300)
            plt.clf()

            le = LabelEncoder()
            data['target'] = le.fit_transform(data[label_col])

            train_set, test_set = train_test_split(data, test_size=test_ratio,stratify=data.iloc[:,-1], random_state=42)
            train_set, val_set = train_test_split(train_set, test_size=val_ratio/(val_ratio+train_ratio),stratify=train_set.iloc[:,-1], random_state=42)

            train_order = train_set[label_col].value_counts().index
            sns.countplot(train_set,x=label_col, order=train_order)
            plt.xticks(rotation=90)
            plt.title(f"{data_figure_title} Training Data Distribution")
            plt.savefig(f'{data_figure_file_name}-train-data-distribution.png',bbox_inches="tight",dpi=300)
            plt.clf()
        
            val_order = val_set[label_col].value_counts().index
            sns.countplot(val_set,x=label_col, order=val_order)
            plt.xticks(rotation=90)
            plt.title(f"{data_figure_title} Validation Data Distribution")
            plt.savefig(f'{data_figure_file_name}-val-data-distribution.png',bbox_inches="tight",dpi=300)
            plt.clf()
        
            test_order = test_set[label_col].value_counts().index
            sns.countplot(test_set,x=label_col, order=test_order)
            plt.xticks(rotation=90)
            plt.title(f"{data_figure_title} Testing Data Distribution")
            plt.savefig(f'{data_figure_file_name}-test-data-distribution.png',bbox_inches="tight",dpi=300)
            plt.clf()
        
            TARGET_LIST = le.classes_

            tokenizer = RobertaTokenizer.from_pretrained(tokenizer_file_name)
            MAX_LEN=512
            BATCH_SIZE=32

            test_dataset = CustomDataset(test_set,tokenizer=tokenizer,max_len=MAX_LEN,value_type=value_type)
            test_loader = DataLoader(
                test_dataset,
                shuffle=False,
                batch_size=BATCH_SIZE,
                num_workers=0
            )
        
        else:
            le = LabelEncoder()
            data['target'] = le.fit_transform(data[label_col])
            data = data.drop(columns=[label_col])
            
            feature_cols = data.drop(columns='target').columns
            byte_matrix_df = data[feature_cols].applymap(GenerateTestScores.hex_to_byte_array)
            X = byte_matrix_df.apply(lambda row: sum(row, []), axis=1).to_list()
            X = np.array(X)
            y = data['target'].values

            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=test_ratio,
                stratify=y,
                random_state=42
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_ratio / (train_ratio + val_ratio),
                stratify=y_temp,
                random_state=42
            )

            GenerateTestScores.save_scaler('securityBERT/baseline_model/', X_train, y_train, X_val, y_val, dataset_type)
            scaler = joblib.load(f'./securityBERT/baseline_model/scaler{scaler_name}.joblib')
            X_test_scaled = scaler.transform(X_test)

            TARGET_LIST = le.classes_

            return X_test_scaled, y_test, TARGET_LIST, le

        return test_set, test_loader, TARGET_LIST, le
    
    def collect_test_results_helper(models_to_score):
        encoded_data_file_edgeiiot = "./securityBERT/saved_data/encoded_data"
        encoded_data_file_cicids2017 = "./securityBERT/saved_data/encoded_data_CICIDS2017_0.02samples"
        raw_data_file_edgeiiot = "./securityBERT/saved_data/raw_data"
        raw_data_file_cicids2017 = "./securityBERT/saved_data/raw_data_CICIDS2017_0.02samples"
        encoded_baseline_data_file_edgeiiot = "./securityBERT/saved_data/encoded_baseline_data"
        encoded_baseline_data_file_cicids2017 = "./securityBERT/saved_data/encoded_baseline_data_CICIDS2017_0.02samples"

        test_metrics_list = []
        test_metrics_row_labels = []
        history_list = []
        previous_dataset_str = ''
        for model_to_score in models_to_score:
            # Load datasets as-needed to reduce memory requirements
            if model_to_score[6] == 'edgeiiot_ppfle' and previous_dataset_str != 'edgeiiot_ppfle':
                previous_dataset_str = 'edgeiiot_ppfle'
                test_set, test_loader, TARGET_LIST, le = GenerateTestScores.generate_test_split(pd.read_pickle(f'{encoded_data_file_edgeiiot}.pck'), DatasetType.EDGEIIOT, ValueType.PPFLE)
            elif model_to_score[6] == 'edgeiiot_flat' and previous_dataset_str != 'edgeiiot_flat':
                previous_dataset_str = 'edgeiiot_flat'
                test_set, test_loader, TARGET_LIST, le = GenerateTestScores.generate_test_split(pd.read_pickle(f'{raw_data_file_edgeiiot}.pck'), DatasetType.EDGEIIOT, ValueType.FLAT)
            elif model_to_score[6] == 'edgeiiot_int' and previous_dataset_str != 'edgeiiot_int':
                previous_dataset_str = 'edgeiiot_int'
                test_set, test_loader, TARGET_LIST, le = GenerateTestScores.generate_test_split(pd.read_pickle(f'{encoded_baseline_data_file_edgeiiot}.pck'), DatasetType.EDGEIIOT, ValueType.P_INT)
            elif model_to_score[6] == 'cicids2017_ppfle' and previous_dataset_str != 'cicids2017_ppfle':
                previous_dataset_str = 'cicids2017_ppfle'
                test_set, test_loader, TARGET_LIST, le = GenerateTestScores.generate_test_split(pd.read_pickle(f'{encoded_data_file_cicids2017}.pck'), DatasetType.CICIDS2017, ValueType.PPFLE)
            elif model_to_score[6] == 'cicids2017_flat' and previous_dataset_str != 'cicids2017_flat':
                previous_dataset_str = 'cicids2017_flat'
                test_set, test_loader, TARGET_LIST, le = GenerateTestScores.generate_test_split(pd.read_pickle(f'{raw_data_file_cicids2017}.pck'), DatasetType.CICIDS2017, ValueType.FLAT)
            elif model_to_score[6] == 'cicids2017_int' and previous_dataset_str != 'cicids2017_int':
                previous_dataset_str = 'cicids2017_int'
                test_set, test_loader, TARGET_LIST, le = GenerateTestScores.generate_test_split(pd.read_pickle(f'{encoded_baseline_data_file_cicids2017}.pck'), DatasetType.CICIDS2017, ValueType.P_INT)
            
            test_metrics, row_label, history = GenerateTestScores.collect_test_results(
                model_directory = model_to_score[0],
                model_version = model_to_score[1],
                num_epochs = model_to_score[2],
                keep_frac = model_to_score[3],
                figure_version = model_to_score[4],
                figure_file_name = model_to_score[5],
                test_set = test_set,
                test_loader = test_loader,
                TARGET_LIST = TARGET_LIST,
                dataset_type = model_to_score[7],
                le = le,
            )
            test_metrics_list.append(test_metrics)
            test_metrics_row_labels.append(row_label)
            history_list.append(history)
        return test_metrics_list, test_metrics_row_labels, history_list

def main():
    os.makedirs(os.path.dirname("./figures/"), exist_ok=True)

    models_to_score_edgeiiot = [
        ("securityBERT/baseline_model/", "baseline_DT_PPFLE", 3, 1.0, "DT", 'edgeiiot-ppfle-dt', 'edgeiiot_int', DatasetType.EDGEIIOT),
        ("securityBERT/baseline_model/", "baseline_RF_PPFLE", 3, 1.0, "RF", 'edgeiiot-ppfle-rf', 'edgeiiot_int', DatasetType.EDGEIIOT),
        ("securityBERT/baseline_model/", "baseline_KNN_PPFLE", 1, 1.0, "KNN", 'edgeiiot-ppfle-knn', 'edgeiiot_int', DatasetType.EDGEIIOT),
        ("securityBERT/baseline_model/", "baseline_LSTM_PPFLE", 1, 1.0, "LSTM", 'edgeiiot-ppfle-lstm', 'edgeiiot_int', DatasetType.EDGEIIOT),
        ("securityBERT/baseline_model/", "baseline_tabTransformer", 3, 1.0, "TabTransformer", 'edgeiiot-ppfle-tabtransformer', 'edgeiiot_int', DatasetType.EDGEIIOT),
        ("securityBERT/saved_model/", "securityBert3_mod_raw", 3, 1.0, "FLAT-BERT", 'edgeiiot-flat-mod', 'edgeiiot_flat', DatasetType.EDGEIIOT),
        ("securityBERT/finetuned_model/", "bertFinetuned_securityBert4_mod_raw_1.0samples", 3, 1.0, "FLAT-BERT-SEM", 'edgeiiot-flat-sem', 'edgeiiot_flat', DatasetType.EDGEIIOT),
        ("securityBERT/saved_model/", "securityBert3", 3, 1.0, "PPFLE-BERT (Adjh)", 'edgeiiot-ppfle-orig', 'edgeiiot_ppfle', DatasetType.EDGEIIOT),
        ("securityBERT/saved_model/", "securityBert3_mod", 3, 1.0, "PPFLE-BERT (Mine)", 'edgeiiot-ppfle-mod', 'edgeiiot_ppfle', DatasetType.EDGEIIOT),
        ("securityBERT/finetuned_model/", "bertFinetuned_securityBert4_mod_1.0samples", 3, 1.0, "PPFLE-BERT-SEM", 'edgeiiot-ppfle-sem', 'edgeiiot_ppfle', DatasetType.EDGEIIOT),
    ]
    
    test_metrics_list_edgeiiot, test_metrics_row_labels_edgeiiot, histories = GenerateTestScores.collect_test_results_helper(models_to_score_edgeiiot)

    models_to_score_cicids2017 = [
        ("securityBERT/baseline_model/", "baseline_DT_CICIDS2017_PPFLE", 2, 0.02, "DT 2% Samples", 'cicids2017-ppfle-dt-0.02samples', 'cicids2017_int', DatasetType.CICIDS2017),
        ("securityBERT/baseline_model/", "baseline_RF_CICIDS2017_PPFLE", 3, 0.02, "RF 2% Samples", 'cicids2017-ppfle-rf-0.02samples', 'cicids2017_int', DatasetType.CICIDS2017),
        ("securityBERT/baseline_model/", "baseline_KNN_CICIDS2017_PPFLE", 1, 0.02, "KNN 2% Samples", 'cicids2017-ppfle-knn-0.02samples', 'cicids2017_int', DatasetType.CICIDS2017),
        ("securityBERT/baseline_model/", "baseline_LSTM_CICIDS2017_PPFLE", 3, 0.02, "LSTM 2% Samples", 'cicids2017-ppfle-lstm-0.02samples', 'cicids2017_int', DatasetType.CICIDS2017),
        ("securityBERT/baseline_model/", "baseline_tabTransformer_cicids2017_0.02samples", 3, 0.02, "TabTransformer 2% Samples", 'cicids2017-ppfle-tabtransformer-0.02samples', 'cicids2017_int', DatasetType.CICIDS2017),
        ("securityBERT/saved_model/", "securityBERT3_mod_raw_CICIDS2017_0.02samples", 3, 0.02, "FLAT-BERT 2% Samples", 'cicids2017-flat-0.02samples', 'cicids2017_flat', DatasetType.CICIDS2017),
        ("securityBERT/finetuned_model/", "bertFinetuned_securityBERT4_mod_raw_CICIDS2017_0.02samples", 3, 0.02, "FLAT-BERT-SEM 2% Samples", 'cicids2017-flat-sem-0.02samples', 'cicids2017_flat', DatasetType.CICIDS2017),
        ("securityBERT/saved_model/", "securityBERT3_mod_CICIDS2017_0.02samples", 3, 0.02, "PPFLE-BERT 2% Samples", 'cicids2017-ppfle-0.02samples', 'cicids2017_ppfle', DatasetType.CICIDS2017),
        ("securityBERT/finetuned_model/", "bertFinetuned_securityBERT4_mod_CICIDS2017_0.02samples", 3, 0.02, "PPFLE-BERT-SEM 2% Samples", 'cicids2017-ppfle-sem-0.02samples', 'cicids2017_ppfle', DatasetType.CICIDS2017),
    ]
    test_metrics_list_cicids2017, test_metrics_row_labels_cicids2017, cicids2017_histories = GenerateTestScores.collect_test_results_helper(models_to_score_cicids2017)
    
    GenerateTestScores.plot_all_test_metrics_tables(test_metrics_list_edgeiiot, "EdgeIIoT", "edgeiiot", test_metrics_row_labels_edgeiiot)
    GenerateTestScores.plot_all_test_metrics_tables(test_metrics_list_cicids2017, "CIC-IDS2017", "cicids2017", test_metrics_row_labels_cicids2017)

    for tmp_history in cicids2017_histories:
        histories.append(tmp_history)
    
    histories.append(GenerateTestScores.safe_load_tensor("./securityBERT/pretrained_model/history_bertPretrained_securityBERT4_mod_raw_1.0samples.pt"))
    histories.append(GenerateTestScores.safe_load_tensor("./securityBERT/pretrained_model/history_bertPretrained_securityBERT4_mod_raw_CICIDS2017_0.02samples.pt"))
    histories.append(GenerateTestScores.safe_load_tensor("./securityBERT/pretrained_model/history_bertPretrained_securityBERT4_mod_1.0samples.pt"))
    histories.append(GenerateTestScores.safe_load_tensor("./securityBERT/pretrained_model/history_bertPretrained_securityBERT4_mod_CICIDS2017_0.02samples.pt"))

    row_labels = deepcopy(test_metrics_row_labels_edgeiiot)

    for tmp_row_label in test_metrics_row_labels_cicids2017:
        row_labels.append(tmp_row_label)

    GenerateTestScores.plot_model_histories_extended(histories=histories, model_names=row_labels)

# Optional: Example stub if you run this as a script directly
if __name__ == "__main__":
    main()
        