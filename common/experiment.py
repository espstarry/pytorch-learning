import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("runs/.matplotlib").resolve()))

import matplotlib.pyplot as plt
import torch


def seed_everything(seed):
    """Seed Python and PyTorch, and request deterministic PyTorch operations."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class ClassificationExperiment:
    """Run and record one reproducible classification experiment."""

    def __init__(self, model, criterion, optimizer, config, output_root="runs"):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = dict(config)
        self.history = []
        self.best_epoch = -1
        self.best_valid_loss = float("inf")
        self.epochs_without_improvement = 0

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(output_root) / self.config["experiment_name"] / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.best_model_path = self.run_dir / "best_model.pt"
        self.last_checkpoint_path = self.run_dir / "last_checkpoint.pt"
        self.config_path = self.run_dir / "config.json"
        self.environment_path = self.run_dir / "environment.json"
        self.history_path = self.run_dir / "history.json"
        self.summary_path = self.run_dir / "summary.json"
        self.loss_curve_path = self.run_dir / "loss_curve.png"
        self.accuracy_curve_path = self.run_dir / "accuracy_curve.png"

    def run(self, train_loader, valid_loader):
        """Train, validate, checkpoint, and write all experiment artifacts."""
        self._save_json(self.config_path, self.config)
        self._save_json(self.environment_path, self._environment_info())

        for epoch in range(self.config["epochs"]):
            train_loss = self._train_one_epoch(train_loader)
            valid_loss, valid_accuracy = self._evaluate(valid_loader)
            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "valid_accuracy": valid_accuracy,
                }
            )

            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.best_model_path)
            else:
                self.epochs_without_improvement += 1

            self._save_last_checkpoint(epoch, train_loader)
            self._save_json(self.history_path, self.history)
            self._print_progress(epoch, train_loss, valid_loss, valid_accuracy)

            if self.epochs_without_improvement >= self.config["patience"]:
                print(f"early stopping at epoch={epoch:03d}")
                break

        self.model.load_state_dict(
            torch.load(self.best_model_path, weights_only=True)
        )
        best_valid_loss, best_valid_accuracy = self._evaluate(valid_loader)
        summary = {
            "best_epoch": self.best_epoch,
            "best_valid_loss": best_valid_loss,
            "best_valid_accuracy": best_valid_accuracy,
            "epochs_completed": len(self.history),
        }
        self._save_json(self.summary_path, summary)
        self._plot_history()
        return summary

    def _train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_x, batch_labels in loader:
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = batch_labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / total_samples

    def _evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        correct = 0

        with torch.no_grad():
            for batch_x, batch_labels in loader:
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_labels)
                predictions = logits.argmax(dim=1)

                batch_size = batch_labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                correct += (predictions == batch_labels).sum().item()

        return total_loss / total_samples, correct / total_samples

    def _save_last_checkpoint(self, epoch, train_loader):
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_valid_loss": self.best_valid_loss,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
            "config": self.config,
            "history": self.history,
            "torch_rng_state": torch.get_rng_state(),
        }
        if train_loader.generator is not None:
            checkpoint["loader_rng_state"] = train_loader.generator.get_state()
        torch.save(checkpoint, self.last_checkpoint_path)

    def _plot_history(self):
        epochs = [record["epoch"] for record in self.history]
        train_losses = [record["train_loss"] for record in self.history]
        valid_losses = [record["valid_loss"] for record in self.history]
        valid_accuracies = [record["valid_accuracy"] for record in self.history]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="train loss")
        plt.plot(epochs, valid_losses, label="valid loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Training and validation loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.loss_curve_path)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, valid_accuracies, label="valid accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title("Validation accuracy")
        plt.ylim(0.0, 1.05)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.accuracy_curve_path)
        plt.close()

    def _environment_info(self):
        git_info = self._git_info()
        return {
            "python": sys.version,
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
            **git_info,
        }

    @staticmethod
    def _git_info():
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        status_result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True,
            check=False,
        )
        status_lines = status_result.stdout.splitlines()
        return {
            "git_commit": (
                commit_result.stdout.strip()
                if commit_result.returncode == 0
                else None
            ),
            "git_dirty": bool(status_lines),
            "git_status": status_lines,
        }

    @staticmethod
    def _save_json(path, data):
        with Path(path).open("w", encoding="utf-8") as output_file:
            json.dump(data, output_file, indent=2)

    @staticmethod
    def _print_progress(epoch, train_loss, valid_loss, valid_accuracy):
        if epoch % 20 == 0:
            print(
                f"epoch={epoch:03d}",
                f"train_loss={train_loss:.6f}",
                f"valid_loss={valid_loss:.6f}",
                f"valid_accuracy={valid_accuracy:.2f}",
            )
