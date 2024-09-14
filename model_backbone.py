import torch
from torch import nn

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

#Now we will build training and testing functions to use with models
def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer, device: torch.device):
    """
       Performs one training step for the given model on the provided dataloader.

       Args:
           model (torch.nn.Module): The neural network model to train.
           dataloader (torch.utils.data.DataLoader): DataLoader providing the training data.
           loss_fn (torch.nn.Module): Loss function to compute the loss between predictions and targets.
           optimizer (torch.optim.Optimizer): Optimizer to adjust model parameters based on gradients.
           device (torch.device): Device (CPU/GPU) to which the data and model should be moved.

       Returns:
           tuple: A tuple containing:
               - avg_loss (float): The average loss over the dataloader.
               - avg_acc (float): The average accuracy over the dataloader.
    """


    # Set the model to training mode
    model.train()

    # Initialize accumulators for loss and accuracy
    total_loss, total_acc = 0.0, 0.0

    # Iterate over batches in the dataloader
    for X_batch, y_batch in dataloader:
        # Move data to the specified device
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass: compute predictions
        y_pred = model(X_batch)

        # Compute loss
        loss = loss_fn(y_pred, y_batch)
        total_loss += loss.item()

        # Zero out gradients from previous steps
        optimizer.zero_grad()

        # Backward pass: compute gradients
        loss.backward()

        # Perform a single optimization step
        optimizer.step()

        # Compute accuracy for the batch

        # _, ignores the first return value we dont need, which in this case is the actual max value itself (we only
        #want the index since it tells us what class the model most thought was correct)
        #Technically, you are suppose to call argmax on softmax to turn logits into probs and then look at what value gave
        #us the greatest value when INPUTTED INTO softmax, but this comment here explains that unneccesary complexity anyways
        _, y_pred_labels = torch.max(y_pred, dim=1)


        total_acc += (y_pred_labels == y_batch).sum().item() / len(y_batch)

    # Compute average loss and accuracy across all batches
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)

    return avg_loss, avg_acc

#our testing method will be a little bit shorter since we dont need to gradient descent (obviously wont change parameters)
#in this step, and we will run different modes

#Because pycharm is annoying af, calling this test_step will cause the interpreter to immediately run this as if it was a test
def run_test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
              device: torch.device):
    """
    Perform a testing step with the given model, dataloader, and loss function.

    Args:
    - model (torch.nn.Module): The model to evaluate.
    - dataloader (torch.utils.data.DataLoader): The data loader with test data.
    - loss_fn (torch.nn.Module): The loss function to calculate test loss.
    - device (torch.device): The device (CPU/GPU) to run the computation on.

    Returns:
    - (float, float): Tuple of average test loss and accuracy.
    """
    # Set the model to evaluation mode
    model.eval()

    # Initialize metrics to track test loss and accuracy
    total_loss = 0.0
    total_acc = 0.0

    # Disable gradient calculation for inference
    with torch.inference_mode():
        # Iterate over batches in the dataloader
        for X, y in dataloader:
            # Move inputs and labels to the target device
            X, y = X.to(device), y.to(device)

            # Forward pass: Compute predictions
            logits = model(X)

            # Compute loss for the batch
            loss = loss_fn(logits, y)
            total_loss += loss.item()

            # Compute accuracy for the batch
            # We do not need to apply softmax here because using probabilites to look at how much a model thinks each class
            # matches matter far more in training since multiplte other classes could be somewhat considered by the training model,
            # but we only care about the end result in testing
            preds = logits.argmax(dim=1)
            total_acc += (preds == y).sum().item() / len(preds)

    # Compute average loss and accuracy over all batches
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)

    return avg_loss, avg_acc

#Now lets make a method to combine these too:
def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),epochs: int = 5,
          device: torch.device = torch.device("cpu")):
    """
    Train and evaluate a model for a specified number of epochs, logging results.

    Args:
    - model (torch.nn.Module): The model to be trained and evaluated.
    - train_dataloader (torch.utils.data.DataLoader): Dataloader for training data.
    - test_dataloader (torch.utils.data.DataLoader): Dataloader for testing data.
    - optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    - loss_fn (torch.nn.Module, optional): Loss function (default: CrossEntropyLoss).
    - epochs (int, optional): Number of training epochs (default: 5).
    - device (torch.device, optional): Device to run training and testing on (default: CPU).

    Returns:
    - dict: A dictionary containing the training and testing loss and accuracy for each epoch.
    """

    # Move the model to the target device
    model.to(device)

    # Initialize a results dictionary to store training and testing metrics
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # Iterate over the number of epochs
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        # Perform a single training step
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        # Perform a single test step
        test_loss, test_acc = run_test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        # Log the current epoch's results
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

        # Append results to the dictionary
        results["train_loss"].append(float(train_loss))
        results["train_acc"].append(float(train_acc))
        results["test_loss"].append(float(test_loss))
        results["test_acc"].append(float(test_acc))

    return results
