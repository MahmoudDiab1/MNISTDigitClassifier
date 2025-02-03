# Import necessary libraries
import numpy
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np

# Fetch MNIST dataset from OpenML
# Returns two separate arrays: one for image data (Data) and one for the corresponding labels (Labels)
Data, Labels = fetch_openml("mnist_784", version=1, return_X_y=True, parser="auto")

# Split the dataset: 60,000 images for training, and the remaining 10,000 for testing.
# Data array holds pixel values of the images.
Data_train = Data[:60000]
Data_test = Data[60000:]

# Split the labels: the first 60,000 labels for training, and the remaining for testing.
Label_train = Labels[:60000]
Label_test = Labels[60000:]

# Display the size of the training and test data
print(f"Training data size = {len(Data_train)}")
print(f"Testing data size = {len(Data_test)}")


# Normalize the pixel values to a range from 0 to 1 (instead of the original 0-255 range)
# This is done by dividing the pixel values by 255
# Convert the data from a pandas DataFrame to NumPy arrays, which are more efficient for numerical computations
Data_train = numpy.array(Data_train) / 255
Data_test = numpy.array(Data_test) / 255

# Convert labels to a smaller integer type (int8) since the label values range from 0 to 9
Label_train = numpy.array(Label_train, dtype=np.int8)
Label_test = numpy.array(Label_test, dtype=np.int8)

# Show the pixel data of the first image in the training set
print(f"First image data: {Data_train[0]}")

# Show the corresponding label of the first image
print(f"First image label: {Label_train[0]}")