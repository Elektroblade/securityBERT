from sklearn.metrics import confusion_matrix,classification_report
from enum import Enum
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # for headless environments (e.g., saving figures)
import warnings
import hashlib
from tqdm.notebook import tqdm
import os
warnings.filterwarnings('ignore')
import psutil
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer,AutoTokenizer,AutoConfig,BertForPreTraining,AutoModel,BertModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,TensorDataset
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from collections import defaultdict
from sklearn.model_selection import train_test_split
import random
from torch.optim import AdamW
import time
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class DatasetType(Enum):
    EDGEIIOT = "edgeiiot"
    CICIDS2017 = "cicids2017"

class CustomDataset(Dataset):
  def __init__(self,df,tokenizer,max_len):
    self.df = df
    self.tokenizer = tokenizer
    self.max_len=max_len
    self.sequence = self.df['encoded_PPFLE'].tolist()
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

class GenerateTestScores():

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
        plt.savefig(output_path, dpi=780, bbox_inches="tight")
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
        assert len(histories) == 8, "Expected 8 history objects"
        assert len(model_names) == 6, "Expected 6 model names"

        # Color palette: c1-c6
        c1, c2, c3, c4, c5, c6 = 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown'
        colors = [c1, c2, c3, c2, c3, c4, c5, c6]  # hist1–hist8

        fig, axs = plt.subplots(3, 2, figsize=(14, 15))
        axs = axs.flatten()

        # Titles for subplots
        pretrain_title = model_names[4] + " Pre-training"

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
                idxs = range(3)

            # Bottom row: CIC-IDS2017 (histories 3–5)
            elif i in [4, 5]:
                idxs = range(3, 6)

            # Middle row (new): hist7 + hist8
            elif i in [2, 3]:
                idxs = [6, 7]  # both histories shown in same subplot

            handles, labels = [], []

            for j in idxs:
                hist = histories[j]
                color = colors[j]

                # Use dataset name for hist7/hist8
                if j == 6:
                    base_label = "EdgeIIoT"
                elif j == 7:
                    base_label = "CIC-IDS2017"
                else:
                    base_label = GenerateTestScores.latex_escape(model_names[j])

                # Choose metric based on column (even index = loss, odd = acc)
                if i % 2 == 0:
                    tr_vals = hist['train_loss']
                    val_vals = hist['val_loss']
                else:
                    tr_vals = hist['train_acc']
                    val_vals = hist['val_acc']

                tr_line, = ax.plot(tr_vals, '-', color=color, label=f"{base_label} (train)")
                val_line, = ax.plot(val_vals, '--', color=color, label=f"{base_label} (val)")

                handles.extend([tr_line, val_line])
                labels.extend([f"{base_label} (train)", f"{base_label} (val)"])

            ax.legend(handles, labels, fontsize='small', ncol=1)

        plt.tight_layout()
        plt.savefig(f'./figures/training-and-validation-loss-and-accuracy.png',dpi=780)
        plt.clf()


    def show_confusion_matrix(confusion_matrix, annot, confusion_matrix_fig_size, figure_version, figure_file_name, num_epochs):

        plt.figure(figsize=confusion_matrix_fig_size)
        sns.heatmap(confusion_matrix, annot=annot, cmap='Blues', fmt='')
        plt.xticks(rotation=90)
        plt.title(f"{figure_version} {num_epochs} Epochs Confusion Matrix")
        plt.ylabel('Real threats')
        plt.xlabel('Predicted threats')
        plt.savefig(f'./figures/{figure_file_name}-{num_epochs}ep-confusion-matrix.png',bbox_inches="tight",dpi=780)
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
        test_set: pd.DataFrame,
        test_loader: DataLoader,
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

        epochs = range(1, num_epochs+1)
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
        plt.savefig(f'./figures/{figure_file_name}-{num_epochs}ep-training-validation-losses.png',dpi=780)
        plt.clf()
        # Affichage du graphique
        


        epochs = range(1, num_epochs+1)
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
        plt.savefig(f'./figures/{figure_file_name}-{num_epochs}ep-training-validation-accuracies.png',dpi=780)
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

        if (y_pred == None or y_proba == None or real_values == None or inference_time == None):
            config = AutoConfig.from_pretrained("bert-base-uncased")
            config.hidden_size = 256
            config.num_hidden_layers = 4
            config.num_attention_heads = 16
            config.intermediate_size = 512
            config.vocab_size=VOCAB_SIZE

            finetunedBERT = AutoModel.from_config(config)

            checkpoint_path = f"./{model_directory}{model_version}_{num_epochs}.0.pt"
            securityBert = SecurityBERT(finetunedBERT=finetunedBERT,n_classes=len(TARGET_LIST)).to(device)

            if model_version.startswith("tabTransformer"):
                # Do something
                print("This is a tabTransformer model.")
            else:
                try:
                    securityBert.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
                    print("Model state_dict loaded successfully at \"checkpoint_path\"")
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

        counts = test_set['target'].value_counts().sort_index()
        class_names = le.inverse_transform(counts.index)
        class_counts = pd.Series(counts.values, index=class_names)
        class_counts = class_counts.sort_values(ascending=False)  # or .sort_index()
        print(class_counts)

        y_true = real_values.cpu().numpy()
        y_score = y_pred.cpu().numpy()

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
            finetune_time = history["training_time"][0] / 3600.0
        elif (GenerateTestScores.get_history_value(history, key = "total_time") != None and isinstance(history["total_time"], float)):
            finetune_time = history["training_time"] / 3600.0

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

    def generate_test_split(data: pd.DataFrame, dataset_type: DatasetType):
        if (dataset_type == DatasetType.EDGEIIOT):
            label_col = "Attack_type"
            tokenizer_file_name = f"./securityBERT/tokenizer"
            data_figure_title = "Edge-IIoT"
            data_figure_file_name = "./figures/edgeiiot"
        else:
            label_col = "Label"
            tokenizer_file_name = f'./securityBERT/tokenizer_CICIDS2017_0.02samples'
            data_figure_title = "Downsampled CIC-IDS2017 0.02samples"
            data_figure_file_name = "./figures/cicids2017-0.02samples"
            
        data_order = data[label_col].value_counts().index
        sns.countplot(data,x=label_col, order=data_order)
        plt.xticks(rotation=90)
        plt.title(f"{data_figure_title} Population by Class")
        plt.savefig(f'{data_figure_file_name}-full-data-distribution.png',bbox_inches="tight",dpi=780)
        plt.clf()

        le = LabelEncoder()
        data['target'] = le.fit_transform(data[label_col])

        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15

        train_set, test_set = train_test_split(data, test_size=test_ratio,stratify=data.iloc[:,-1], random_state=42)
        train_set, val_set = train_test_split(train_set, test_size=val_ratio/(val_ratio+train_ratio),stratify=train_set.iloc[:,-1], random_state=42)

        train_order = train_set[label_col].value_counts().index
        sns.countplot(train_set,x=label_col, order=train_order)
        plt.xticks(rotation=90)
        plt.title(f"{data_figure_title} Training Data Distribution")
        plt.savefig(f'{data_figure_file_name}-train-data-distribution.png',bbox_inches="tight",dpi=780)
        plt.clf()
    
        val_order = val_set[label_col].value_counts().index
        sns.countplot(val_set,x=label_col, order=val_order)
        plt.xticks(rotation=90)
        plt.title(f"{data_figure_title} Validation Data Distribution")
        plt.savefig(f'{data_figure_file_name}-val-data-distribution.png',bbox_inches="tight",dpi=780)
        plt.clf()
    
        test_order = test_set[label_col].value_counts().index
        sns.countplot(test_set,x=label_col, order=test_order)
        plt.xticks(rotation=90)
        plt.title(f"{data_figure_title} Testing Data Distribution")
        plt.savefig(f'{data_figure_file_name}-test-data-distribution.png',bbox_inches="tight",dpi=780)
        plt.clf()
    
        TARGET_LIST = le.classes_

        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_file_name)
        MAX_LEN=512
        BATCH_SIZE=32

        test_dataset = CustomDataset(test_set,tokenizer=tokenizer,max_len=MAX_LEN)
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=BATCH_SIZE,
            num_workers=0
        )

        return test_set, test_loader, TARGET_LIST, le
    
    def collect_test_results_helper(models_to_score):
        test_metrics_list = []
        test_metrics_row_labels = []
        history_list = []
        for model_to_score in models_to_score:
            test_metrics, row_label, history = GenerateTestScores.collect_test_results(
                model_directory = model_to_score[0],
                model_version = model_to_score[1],
                num_epochs = model_to_score[2],
                keep_frac = model_to_score[3],
                figure_version = model_to_score[4],
                figure_file_name = model_to_score[5],
                test_set = model_to_score[6],
                test_loader = model_to_score[7],
                TARGET_LIST = model_to_score[8],
                dataset_type = model_to_score[9],
                le = model_to_score[10],
            )
            test_metrics_list.append(test_metrics)
            test_metrics_row_labels.append(row_label)
            history_list.append(history)
        return test_metrics_list, test_metrics_row_labels, history_list

def main():
    os.makedirs(os.path.dirname("./figures/"), exist_ok=True)

    encoded_data_file_edgeiiot = "./securityBERT/saved_data/encoded_data"
    encoded_data_file_cicids2017 = "./securityBERT/saved_data/encoded_data_CICIDS2017_0.02samples"
    test_set_edgeiiot, test_loader_edgeiiot, target_list_edgeiiot, le_edgeiiot = GenerateTestScores.generate_test_split(pd.read_pickle(f'{encoded_data_file_edgeiiot}.pck'), DatasetType.EDGEIIOT)
    test_set_cicids2017, test_loader_cicids2017, target_list_cicids2017, le_cicids2017 = GenerateTestScores.generate_test_split(pd.read_pickle(f'{encoded_data_file_cicids2017}.pck'), DatasetType.CICIDS2017)

    models_to_score_edgeiiot = [
        ("securityBERT/saved_model/", "securityBert3", 3, 1.0, "PPFLE-BERT (Adjh)", 'edgeiiot-ppfle-orig', test_set_edgeiiot, test_loader_edgeiiot, target_list_edgeiiot, DatasetType.EDGEIIOT, le_edgeiiot),
        ("securityBERT/saved_model/", "securityBert3_mod", 3, 1.0, "PPFLE-BERT (Mine)", 'edgeiiot-ppfle-mod', test_set_edgeiiot, test_loader_edgeiiot, target_list_edgeiiot, DatasetType.EDGEIIOT, le_edgeiiot),
        ("securityBERT/finetuned_model/", "bertFinetuned_securityBert4_mod_1.0samples", 3, 1.0, "PPFLE-BERT-SEM", 'edgeiiot-ppfle-sem', test_set_edgeiiot, test_loader_edgeiiot, target_list_edgeiiot, DatasetType.EDGEIIOT, le_edgeiiot),
    ]
    
    test_metrics_list_edgeiiot, test_metrics_row_labels_edgeiiot, histories = GenerateTestScores.collect_test_results_helper(models_to_score_edgeiiot)
    models_to_score_cicids2017 = [
        ("securityBERT/saved_model/", "securityBERT3_mod_CICIDS2017_0.02samples", 3, 0.02, "PPFLE-BERT 2% Samples", 'cicids2017-ppfle-0.02samples', test_set_cicids2017, test_loader_cicids2017, target_list_cicids2017, DatasetType.CICIDS2017, le_cicids2017),
        ("securityBERT/finetuned_model/", "bertFinetuned_securityBERT4_mod_CICIDS2017_0.02samples", 3, 0.02, "PPFLE-BERT-SEM 2% Samples", 'cicids2017-ppfle-sem-0.02samples', test_set_cicids2017, test_loader_cicids2017, target_list_cicids2017, DatasetType.CICIDS2017, le_cicids2017),
        ("languageClass/languageClass/pretrained_model/", "tabTransformer_cicids2017_0.02samples", 3, 0.02, "TabTransformer 2% Samples", 'cicids2017-tabtransformer-0.02samples', test_set_cicids2017, test_loader_cicids2017, target_list_cicids2017, DatasetType.CICIDS2017, le_cicids2017),
    ]
    test_metrics_list_cicids2017, test_metrics_row_labels_cicids2017, cicids2017_histories = GenerateTestScores.collect_test_results_helper(models_to_score_cicids2017)
    
    GenerateTestScores.plot_all_test_metrics_tables(test_metrics_list_edgeiiot, "EdgeIIoT", "edgeiiot", test_metrics_row_labels_edgeiiot)
    GenerateTestScores.plot_all_test_metrics_tables(test_metrics_list_cicids2017, "CIC-IDS2017", "cicids2017", test_metrics_row_labels_cicids2017)

    for tmp_history in cicids2017_histories:
        histories.append(tmp_history)
    
    histories.append(GenerateTestScores.safe_load_tensor("./securityBERT/pretrained_model/history_bertPretrained_securityBERT4_mod_1.0samples.pt"))
    histories.append(GenerateTestScores.safe_load_tensor("./securityBERT/pretrained_model/history_bertPretrained_securityBERT4_CICIDS2017_0.02samples.pt"))

    row_labels = deepcopy(test_metrics_row_labels_edgeiiot)

    for tmp_row_label in test_metrics_row_labels_cicids2017:
        row_labels.append(tmp_row_label)

    GenerateTestScores.plot_model_histories_extended(histories=histories, model_names=row_labels)

# Optional: Example stub if you run this as a script directly
if __name__ == "__main__":
    main()
        