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

from torchvision.datasets import FGVCAircraft
from tqdm.auto import tqdm
from timeit import default_timer as timer
from typing import Dict, List

from FGVC_Aircraft import setup_dataholders
from FGVC_Aircraft import firsttry_model
from FGVC_Aircraft import model_backbone

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



#Start the training with your own parameters here
model_backbone.train();



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


#Can make dataloaders here
train_dataloader_augmented, test_dataloader_simple, class_names = FGVCAircraft.create_dataloaders(...)



#Make model here, code in another file

torch.manual_seed(61)
model = firsttry_model.VGGImposter(input_channels=3, hidden_units=12, output_classes=len(train_augmented.classes)).to(device)


#Run our training and testing loop!
# Set number of epochs
EPOCHS = 5

# Setup loss function and optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
loss_func = nn.CrossEntropyLoss()

# Start the timer
start_time = timer()

# Train modelf
model_results = model_backbone.train(model=model, train_dataloader=train_dataloader_augmented,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer, loss_fn=loss_func, epochs=EPOCHS)

# End the timer and print total training time
end_time = timer()
print(f"Total time: {end_time - start_time:.3f} seconds")

# Finally, show the loss curve plots
visualize_loss_curves(model_results)







