import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from haptic_exploration.visualization import plot_training_performance
from haptic_exploration.ml_util import get_device, ModelTrainingMonitor


def train_cls_random(train_datasets, validation_datasets, dataset_names, model, num_epochs, batch_size=32, lr=0.001) -> ModelTrainingMonitor:
    """ Train a model with the given training and test data """

    assert len(set(len(td) for td in train_datasets)) == 1
    assert len(set(len(vd) for vd in validation_datasets)) == 1

    train_loaders = [DataLoader(train_dataset, batch_size=batch_size, shuffle=True) for train_dataset in train_datasets]
    validation_loaders = [DataLoader(validation_dataset, batch_size=batch_size, shuffle=True) for validation_dataset in validation_datasets]

    device = get_device()
    model = model.to(device)

    training_monitor = ModelTrainingMonitor()

    # Training loop
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        train_stats = [[0.0, 0] for _ in train_loaders]  # train_loss, train_correct
        for batches in tqdm(zip(*train_loaders), desc=f"Train epoch {epoch + 1}/{num_epochs}", total=len(train_loaders[0])):
            for batch_idx, batch in enumerate(batches):
                x1, x2, y = batch
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                y_pred = model((x1, x2))
                loss = criterion(y_pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_stats[batch_idx][0] += loss.detach().cpu().item()
                train_stats[batch_idx][1] += torch.sum(torch.argmax(y_pred, dim=1) == y).detach().cpu().item()

        train_total_batches = len(train_loaders[0])
        train_accuracies = [correct/(train_total_batches*batch_size) for _, correct in train_stats]
        train_losses = [loss/train_total_batches for loss, _ in train_stats]
        train_loss = sum(train_losses)/len(train_loaders)
        train_accuracy = sum(train_accuracies)/len(train_loaders)
        print(f"Train Accuracy: {(100*train_accuracy):.2f}%", "(" + ", ".join(f"{dataset_names[i]}: {(100*acc):.2f}%" for i, acc in enumerate(train_accuracies)) + ")")
        print(f"Train Loss: {train_loss:.2f}", "(" + ", ".join(f"{dataset_names[i]}: {loss:.2f}" for i, loss in enumerate(train_losses)) + ")")


        # Validation loop
        model.eval()
        with torch.no_grad():
            val_stats = [[0.0, 0] for _ in validation_loaders]  # val_loss, val_correct
            for batches in tqdm(zip(*validation_loaders), desc="Validation", total=len(validation_loaders[0])):
                for batch_idx, batch in enumerate(batches):
                    x1, x2, y = batch
                    x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                    y_pred = model((x1, x2))
                    loss = criterion(y_pred, y)

                    val_stats[batch_idx][0] += loss.detach().cpu().item()
                    val_stats[batch_idx][1] += torch.sum(torch.argmax(y_pred, dim=1) == y).detach().cpu().item()

        val_total_batches = len(validation_loaders[0])
        val_accuracies = [correct/(val_total_batches*batch_size) for _, correct in val_stats]
        val_losses = [loss/val_total_batches for loss, _ in val_stats]
        val_loss = sum(val_losses)/len(validation_loaders)
        val_accuracy = sum(val_accuracies)/len(validation_loaders)
        print(f"Validation Accuracy: {(100*val_accuracy):.2f}%", "(" + ", ".join(f"{dataset_names[i]}: {(100*acc):.2f}%" for i, acc in enumerate(val_accuracies)) + ")")
        print(f"Validation Loss: {val_loss:.2f}", "(" + ", ".join(f"{dataset_names[i]}: {loss:.2f}" for i, loss in enumerate(val_losses)) + ")")

        training_monitor.process_episode(model.state_dict(), val_loss, val_accuracy)

    training_monitor.print_results()
    plot_training_performance(training_monitor)

    return training_monitor
