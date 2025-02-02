# Load MNIST using sklearn.datasets.fetch_openml
from sklearn.datasets import fetch_openml

import matplotlib.pyplot as plt
import numpy as np

# Load data from https://www.openml.org/d/554
# Returns a tuple
Data, Labels = fetch_openml("mnist_784", version=1, return_X_y=True, parser="auto")

#Split: 6,000 images for training, 1,000 for testing.
# Data: contains the pixel values of the images.
Data_train = Data[:60000]
# Labels: contains the corresponding digit labels (0-9).
Data_test = Data[60000:]

#Split: 6,000 lables for each image for training, 1,000 for testing.
Label_train = Labels[:60000]
Label_test = Labels[60000:]

print(f"Training data size = {len(Data_train)}")
print(f"Testing data size = {len(Data_test)}")
print(f"first image data: {Data_train[:1]}")
print(f"first image label: {Label_train[0]}")