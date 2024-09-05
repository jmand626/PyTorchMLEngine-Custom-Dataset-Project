import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn


from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

#torchvision since this is a computer vision problem


from pathlib import Path
import os
import random
from tqdm.auto import tqdm
from timeit import default_timer as timer
from typing import Dict, List

##Inspired by and working with mrdbourke's wonderful pytorch tutorials
##


#from create_custom_dataset import inspect_dir

#checking version
print(torch.__version__)

# Setting up device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


#We will use a subset of the FGVC_Aircraft dataset from PyTorch

#we need to do this because my code put the data folder next to the FGVC_Aircraft folder, but since this file is the latter,
#we need to change the directory to go to its parent one step in order to also see the data folder
os.chdir("C:\\Users\\joban\\PycharmProjects\\PyCustomDatasetPractice")

#as mentioned before, this is from the perspective of the PyCustomDatasetPractice directory
data_path = Path("data/")
image_file_path = data_path / "eleven_group_subset_90_percent"
if image_file_path.is_dir():
    print(f"{image_file_path} directory exists.")

#Struggling a bit to get this working where it will only execute once when importing, but its not a big deal since
#we can just run the other file to see the output of the call here
#inspect_dir(image_files_path)

train_dir = image_file_path / "train"
test_dir = image_file_path / "test"


#Visualization
random.seed(62)
image_path_list = list(image_file_path.glob("*/*/*.jpg"))
random_image_path = random.choice(image_path_list)
image_class = random_image_path.parent.stem
img = Image.open(random_image_path)
#print(f"Random image path: {random_image_path}")
#print(f"Image class: {image_class}")
#print(f"Image height: {img.height}")
#print(f"Image width: {img.width}")
#img.show()



#Using torchvision.transforms to modify our data (and finally turn them into PyTorch-compatible tensors)
data_transform = transforms.Compose([
    transforms.Resize(size=(72, 72)),
    transforms.RandomVerticalFlip(p=0.4), #p -> probability of the image flipping
    # Turn the image into a torch.Tensor
    transforms.ToTensor()
])#btw, this is variable, but its kinda defined how you define a struct in C

#We use that var/struct to transform multiple images at once
def plot_transformed_images(image_paths, transform, n=4, seed=61):
    """
    Plots a comparison of original and transformed images from a list of image file paths.

    Args:
        image_paths (list of str or Path): List of paths to the image files to be displayed.
        transform (callable): A transformation function (e.g., from torchvision.transforms)
                              that will be applied to the original image.
        n (int, optional): Number of images to randomly select and display. Default is 4.
        seed (int, optional): Random seed for selecting the images. Default is 61.

    Returns:
        None: Displays the original and transformed images in matplotlib figures.

    Example:
        plot_transformed_images(image_paths, transform, n=4)
    """
    random.seed(seed)
    selected_images = random.sample(image_paths, k=n)

    for image_path in selected_images:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        with Image.open(image_path) as img:
            original_image = img.copy()
            transformed_image = transform(original_image).permute(1, 2, 0)

            # Plot original image
            axes[0].imshow(original_image)
            axes[0].set_title(f"Original\nSize: {original_image.size}")
            axes[0].axis("off")

            # Plot transformed image
            axes[1].imshow(transformed_image)
            axes[1].set_title(f"Transformed\nSize: {tuple(transformed_image.shape[:2])}")
            axes[1].axis("off")

        fig.suptitle(f"Class: {Path(image_path).parent.stem}", fontsize=16)
        plt.tight_layout()
        plt.show()

# Example usage
#plot_transformed_images(image_path_list, transform=data_transform, n=3)


# Use ImageFolder to create dataset(s). This is where actually put our data into a form that PyTorch can understand
train_data = datasets.ImageFolder(root=train_dir, transform=data_transform, target_transform=None)
#root = where we the images to go (a folder)
#transform = what kinds of modifications we wish to do on the data
#target_transform = targets specifcally refer to labels in this situation
test_data = datasets.ImageFolder(root=test_dir, transform=data_transform, target_transform=None)

#print(f"Train data:\n{train_data}\nTest data:\n{test_data}")

#can be useful later, obviously these are same for test_data
c = train_data.classes
c_dict = train_data.class_to_idx
#print(c)
#print(c_dict)





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

#Next, lets write a function to display loss plots so we can see when our model sucks and when it doesnt
#With these visualizations, it makes it much easier to identify overfitting and underfitting
def visualize_loss_curves(results: Dict[str, List[float]]):
    """
    Plot training and testing loss and accuracy curves from the results dictionary.

    Args:
    - results (Dict[str, List[float]]): A dictionary containing loss and accuracy values
      for training and testing across epochs. The dictionary should have the following keys:
        - 'train_loss': List of training loss values per epoch.
        - 'test_loss': List of testing loss values per epoch.
        - 'train_acc': List of training accuracy values per epoch.
        - 'test_acc': List of testing accuracy values per epoch.

    Displays:
    - Two subplots:
      1. Training and testing loss vs. epochs.
      2. Training and testing accuracy vs. epochs.
    """

    # Extract loss and accuracy values
    train_loss = results['train_loss']
    test_loss = results['test_loss']
    train_acc = results['train_acc']
    test_acc = results['test_acc']

    # Determine the number of epochs based on the length of loss data
    epochs = range(len(train_loss))

    # Set up a figure for loss and accuracy plots
    plt.figure(figsize=(15, 7))

    # Plot training and testing loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Training Loss', color='blue', marker='o')
    plt.plot(epochs, test_loss, label='Testing Loss', color='red', marker='x')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot training and testing accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Training Accuracy', color='blue', marker='o')
    plt.plot(epochs, test_acc, label='Testing Accuracy', color='red', marker='x')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.tight_layout()
    plt.show()


#Now we can finally start with our model, oops we have to transform our data first!
train_transform = transforms.Compose([transforms.Resize((72, 72)), transforms.TrivialAugmentWide(num_magnitude_bins=27),
                                      transforms.ToTensor()])

#Obviously we do not want to augment our test data since it ours training data that needs to be harder for our model to get better
test_transform = transforms.Compose([transforms.Resize((72, 72)),
                                      transforms.ToTensor()])

#Now create the datasets with these new transformations
train_augmented = datasets.ImageFolder(train_dir, transform=train_transform)
test_transformed = datasets.ImageFolder(test_dir, transform=test_transform)


#Turn datasets into dataloaders
# Constants
BATCH_SIZE = 48

NUM_WORKERS = 0
#NUM_WORKERS = os.cpu_count()
#doesnt work for some reason. This will certainly make it slower but I'm not sure what else I can do

# Set random seed for reproducibility
torch.manual_seed(61)

# DataLoader for training data with augmentation
train_dataloader_augmented = DataLoader(
    train_augmented,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

# DataLoader for test data without augmentation
test_dataloader_simple = DataLoader(
    test_transformed,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

#Now, we finally work on the model. We will focus on recreating the model from TinyVGG
class VGGImposter(nn.Module):
    """
    A simplified VGG-like architecture with two convolutional blocks followed by
    a fully connected layer for classification.

    The architecture includes:
    - Two convolutional blocks, each consisting of two convolutional layers
      followed by ReLU activations and a max pooling layer.
    - A final classifier consisting of a flattening layer and a linear layer
      to produce class scores.

    Args:
        input_channels (int): Number of input channels (e.g., 3 for RGB images).
        hidden_units (int): Number of filters in the convolutional layers.
        output_classes (int): Number of output classes for classification.

    Example:
        model = TinyVGG(input_channels=3, hidden_units=32, output_classes=10)
    """

    def __init__(self, input_channels: int, hidden_units: int, output_classes: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 18 * 18,  # Assuming input images are 64x64
                      out_features=output_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

torch.manual_seed(61)
model = VGGImposter(input_channels=3, hidden_units=12, output_classes=len(train_augmented.classes)).to(device)


#Run our training and testing loop!
# Set number of epochs
EPOCHS = 5

# Setup loss function and optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
loss_func = nn.CrossEntropyLoss()

# Start the timer
start_time = timer()

# Train modelf
model_results = train(model=model, train_dataloader=train_dataloader_augmented,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer, loss_fn=loss_func, epochs=EPOCHS)

# End the timer and print total training time
end_time = timer()
print(f"Total time: {end_time - start_time:.3f} seconds")

# Finally, show the loss curve plots
visualize_loss_curves(model_results)







