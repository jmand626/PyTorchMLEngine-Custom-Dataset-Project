import torch
import torchvision
import random
import os
import shutil
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from PIL import Image
import pathlib

##Inspired by and working with mrdbourke's wonderful pytorch tutorials
##

data_dir = pathlib.Path("../data")


train_data = datasets.FGVCAircraft(root=data_dir, split="train", download=True)
test_data = datasets.FGVCAircraft(root=data_dir, split="test", download=True)


class_names = train_data.classes
#print(class_names[:12])
#we print these out to make sure we know what the classes for this dataset look like. People have downloaded
#the dataset on public repos, which means we can look there for an actual class list. This might be a little bit more
#complicated for this dataset since you can also split by family or manufacturer, but we want to split by variant

#print(class_names[train_data[2][1]])
#image = train_data[0][0]
#image.show()


#getting a subset of all of this data (path can be unique for each dataset)
data_path = data_dir / "fgvc-aircraft-2013b" / "data" / "images"
wanted_classes = ["707-320", "777-300", "A321", "A380", "C-47", "DC-3", "Yak-42" "DR-400", "F-16A_B", "MD-90" "Metroliner"]

percentage = 0.9


# Create function to separate a random amount of data
def get_subset(image_file_path=data_path, data_splits=["train", "test"],
               wanted_classes=["707-320", "777-300", "A321", "A380", "C-47", "DC-3", "Yak-42", "DR-400", "F-16A_B", "MD-90", "Metroliner"],
               seed=61, percent=0.9):
    """
    Creates a subset of the dataset by selecting a random group of images from specified classes
    and data splits (e.g., train and test).

    Args:
        image_file_path (Path or str, optional): The root directory containing the image files.
        data_splits (list of str, optional): List of dataset splits (e.g., "train", "test") to process.
                                             Default is ["train", "test"].
        wanted_classes (list of str, optional): List of class names to filter the dataset by.
                                                Only images from these classes will be selected.
        seed (int, optional): Random seed for reproducibility. Default is 61.
        percent (float, optional): Percentage of the filtered dataset to randomly sample.
                                   Must be between 0 and 1. Default is 0.9.

    Returns:
        dict: A dictionary where keys are the data splits (e.g., "train", "test") and values are lists
              of tuples containing the image paths and their corresponding class names.

    Example:
        subset = get_subset(data_splits=["train"], wanted_classes=["A321", "F-16A_B"], percent=0.8)
    """
    random.seed(seed)  # Ensure reproducibility

    label_splits = {}

    # Process each data split (e.g., "train", "test")
    for data_split in data_splits:
        print(f"Creating image split for: {data_split}...")

        label_path = image_file_path / "fgvc-aircraft-2013b" / "data" / f"images_variant_{data_split}.txt"

        # Parse the label file to extract image IDs and their corresponding class names
        with open(label_path, "r") as f:
            labels = [line.strip().split(maxsplit=1) for line in f.readlines()]

        # Filter labels based on the specified wanted classes
        filtered_labels = [(image_id, class_name) for image_id, class_name in labels if class_name in wanted_classes]

        # Calculate the number of samples to extract based on the percentage
        number_from_sample = round(percent * len(filtered_labels))
        print(f"Getting a random group of {number_from_sample} images for our {data_split} split...")

        # Randomly select a subset of filtered labels
        samples = random.sample(filtered_labels, k=number_from_sample)

        # Generate paths to the selected image files
        image_paths = [(image_file_path / f"{sample_image}.jpg", class_name) for sample_image, class_name in samples]
        label_splits[data_split] = image_paths

    return label_splits


label_splits = get_subset(percent=percentage)


#print(label_splits.keys())
#debugger


#Moving images to folders to separate subset from the full set

target_dir_name = f"../data/eleven_group_subset_{str(int(percentage * 100))}_percent"

#print(f"Creating new directory for {target_dir_name}...")

target_dir = pathlib.Path(target_dir_name)

target_dir.mkdir(parents=True, exist_ok=True)

#actually copy the data over

#we know from earlier and by looking at the code that the two keys are just train and test
for image_split in label_splits.keys():

    #since it goes through every image path, it also just goes through every image
    for image_path, class_name in label_splits[str(image_split)]:
        #We are creating a path, not dividing -> (eleven... / train or test / class(variant) / image jpg id)
        destination_dir = target_dir / image_split / class_name / image_path.name

        #in case the destination.dir does not already exist, defensive programming
        if not destination_dir.parent.is_dir():
            destination_dir.mkdir(parents=True, exist_ok=True)

        #I honestly use shutil here because I know its popular, I'm not experienced enough to know much on why

        #print(f"Copying {image_path} to {destination_dir}...")
        shutil.copy2(image_path, destination_dir)
        #copy2 to preserve metadata



#Now we just count everything in our new dir to make sir we have everything
def inspect_dir(dir_path):
    """
    Inspects the specified directory and prints the number of subdirectories and images (files) in each folder.

    Args:
        dir_path (Path or str): The path to the directory to inspect.

    Returns:
        None: The function prints the inspection results directly.

    Example:
        inspect_dir("/path/to/data")
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


#by setting this method to run on our parent directory (the one that includes all subsections of test and train),
#we get a fuller expose of the entire directory we created

inspect_dir(target_dir)



