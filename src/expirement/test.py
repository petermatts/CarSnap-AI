import os
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Tester:
    # todo trim out unecessary pieces (this is effectively a copy of the trainer class)
    # ? this may not even be necessary if I slightly alter the trainer class
    # maybe change the _validate() to a public test() function

    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader = None,
                 optimizer: torch.optim.Optimizer = None,
                 scheduler=None,
                 loss_fn: callable = None,
                 device: torch.device | str = None,
                 num_epochs: int = 10,
                 checkpoint_path: os.PathLike = None,
                 early_stopping_patience: int = None,
                 use_amp: bool = False,
                 plot_loss: bool = True,
                 show_metrics: bool = True,
                 show_gradients: bool = False):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.checkpoint_path = checkpoint_path
        self.early_stopping_patience = early_stopping_patience
        self.use_amp = use_amp
        self.plot_loss = plot_loss
        self.show_metrics = show_metrics
        self.show_gradients = show_gradients

        self.model.to(self.device)

        # AMP components
        self.scaler = torch.amp.GradScaler(
            device=self.device, enabled=self.use_amp)

        # Live plot data
        self.train_losses = []
        self.val_losses = []

        if self.plot_loss:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.train_line, = self.ax.plot([], [], label="Train Loss")
            self.val_line, = self.ax.plot([], [], label="Val Loss")
            self.ax.set_xlabel("Epoch")
            self.ax.set_ylabel("Loss")
            self.ax.set_title("Training/Validation Loss")
            self.ax.legend()

    def test(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                with torch.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                total_correct += (preds == targets).sum().item()
                total_samples += targets.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        if self.show_metrics:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_preds, average='macro', zero_division=0
            )
            print(
                f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
        else:
            precision = recall = f1 = None

        return avg_loss, accuracy
