from sklearn.metrics import precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import Iterable


class Trainer:
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

    def _save_checkpoint(self, epoch: int, best_val_loss: float):
        if self.checkpoint_path:
            torch.save({
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, self.checkpoint_path)  # ! seperate files per checkpoint?

    def _validate(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                if not isinstance(targets, (list, tuple)):
                    targets = [targets.to(self.device)]
                else:
                    for i, t in enumerate(targets):
                        targets[i] = t.to(self.device)

                with torch.amp.autocast(str(self.device), enabled=self.use_amp):
                    outputs = self.model(inputs)
                    if not isinstance(outputs, (list, tuple)):
                        outputs = [outputs]
                    losses = [self.loss_fn(o, t)
                              for o, t in zip(outputs, targets)]
                loss = sum(losses)
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

    def _update_plot(self):
        self.train_line.set_data(
            range(1, len(self.train_losses)+1), self.train_losses)
        if self.val_losses:
            self.val_line.set_data(
                range(1, len(self.val_losses)+1), self.val_losses)

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _plot_gradients_and_weights(self):
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        axes[0].set_title("Gradient Magnitudes")
        axes[1].set_title("Weight Magnitudes")

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                axes[0].hist(param.grad.detach().cpu().numpy(
                ).flatten(), bins=100, alpha=0.5, label=name)
            axes[1].hist(param.detach().cpu().numpy().flatten(),
                         bins=100, alpha=0.5, label=name)

        for ax in axes:
            ax.legend(fontsize='small', loc='upper right')
            ax.set_yscale('log')
        plt.tight_layout()
        plt.show()

    def learn(self):  # just for funzies
        self.train()

    def train(self):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            running_loss = 0.0

            for inputs, targets in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}"):
                inputs = inputs.to(self.device)
                if not isinstance(targets, (list, tuple)):
                    targets = [targets.to(self.device)]
                else:
                    for i, t in enumerate(targets):
                        targets[i] = t.to(self.device)

                self.optimizer.zero_grad()
                with torch.amp.autocast(str(self.device), enabled=self.use_amp):
                    outputs = self.model(inputs)
                    if not isinstance(outputs, (list, tuple)):
                        outputs = [outputs]
                    losses = [self.loss_fn(o, t)
                              for o, t in zip(outputs, targets)]

                loss = sum(losses)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss += loss.item() * inputs.size(0)

            train_loss = running_loss / len(self.train_loader.dataset)
            self.train_losses.append(train_loss)
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")

            if self.val_loader:
                val_loss, val_acc = self._validate()
                self.val_losses.append(val_loss)
                print(
                    f"[Epoch {epoch}] Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

                if self.scheduler:
                    self.scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_checkpoint(epoch, best_val_loss)
                else:
                    patience_counter += 1

                if self.early_stopping_patience and patience_counter >= self.early_stopping_patience:
                    print("Early stopping triggered.")
                    break

            else:
                self.val_losses.append(None)

            if self.plot_loss:
                self._update_plot(epoch)

        if self.plot_loss:
            plt.ioff()
            plt.show()


if __name__ == "main":
    # Usage:
    # trainer = Trainer(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     loss_fn=torch.nn.CrossEntropyLoss(),
    #     num_epochs=10,
    #     checkpoint_path='model.pt',
    #     early_stopping_patience=5,
    #     use_amp=True,
    #     plot_loss=True
    # )

    # trainer.train()
    pass
