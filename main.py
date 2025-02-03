from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import  MLPClassifier
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
Data_train = np.array(Data_train) / 255
Data_test = np.array(Data_test) / 255

# Convert labels to a smaller integer type (int8) since the label values range from 0 to 9
Label_train = np.array(Label_train, dtype=np.int8)
Label_test = np.array(Label_test, dtype=np.int8)

# Show the pixel data of the first image in the training set
print(f"First image data: {Data_train[0]}")

# Show the corresponding label of the first image
print(f"First image label: {Label_train[0]}")

# Show the first 3 images
plt.figure(figsize=(4, 4))
for index, (image, label) in enumerate(zip(Data_train[0:3], Label_train[0:3])):
    plt.subplot(1, 3, index + 1)
    plt.imshow(np.reshape(image, (28, 28)), cmap=plt.cm.gray)
    plt.title("Label: %s\n" % label, fontsize=20)

# plt.show()


# Create and train a Multi-Layer Perceptron

mlp = MLPClassifier(
    hidden_layer_sizes=(50), # 50 neurons in the hidden layer
    max_iter=10,             # Maximum number of iterations
    alpha=1e-4,              # Regularization strength (L2 penalty)
    solver="sgd",            # Use Stochastic Gradient Descent
    verbose=1,              # after every 10 iterations, you will see detailed output
    random_state=1,          # Random seed for reproducibility
    learning_rate_init=0.1   # Initial learning rate
)

mlp.fit(Data_train, Label_train)



print(f"Training set score: {mlp.score(Data_train, Label_train)}")
print(f"Test set score: {mlp.score(Data_test, Label_test)}")

predictions = mlp.predict(Data_test)

# Show the predictions in a grid
plt.figure(figsize=(8, 4))

for index, (image, prediction, label) in enumerate(
    zip(Data_test[0:10], predictions[0:10], Label_test[0:10])
):
    plt.subplot(2, 5, index + 1)
    plt.imshow(np.reshape(image, (28, 28)), cmap=plt.cm.gray)

    # Green if correct, red if incorrect
    fontcolor = "g" if prediction == label else "r"
    plt.title(
        "Prediction: %i\n Label: %i" % (prediction, label), fontsize=10, color=fontcolor
    )

    plt.axis("off")  # hide axes

plt.show()