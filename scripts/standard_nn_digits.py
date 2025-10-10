import torch
import torch.nn as nn
import torch.optim as optim

from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt



def svm_classification():
    # Load the images
    digits = datasets.load_digits()

    # # Take a look at the data
    # _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    # for ax, image, label in zip(axes, digits.images, digits.target):
    #     ax.set_axis_off()
    #     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    #     ax.set_title("Training: %i" % label)
    # plt.show()

    # Flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )

    # Learn the digits in the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digits in the test subset
    predicted = clf.predict(X_test)

    # Show predicted digits
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
    plt.show()

    # Classification report
    print(
        f"Classification report for classifier {clf}:\n"
        f"{classification_report(y_test, predicted)}\n"
    )

    # Confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    # print(f"Confusion matrix:\n{disp.confusion_matrix}")
    plt.show()


class DigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # local receptive fields
            nn.ReLU(),
            nn.MaxPool2d(2)  # reduces 8×8 → 4×4
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # reduces 4×4 → 2×2
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2),  # reduces 2×2 → 1×1
            nn.ReLU()
        )
        self.output = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # now shape (batch, 64, 1, 1)
        x = x.view(x.size(0), -1)  # flatten
        x = self.output(x)
        return x


def nn_classification():
    # Load dataset
    digits = datasets.load_digits()
    X = digits.images  # shape: (n_samples, 8, 8)
    y = digits.target

    # Normalize pixel values (0–16 → 0–1)
    X = X / 16.0

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, shuffle=False
    )

    # Convert to torch tensors with channel dim
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # (N, 1, 8, 8)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Initialize nn
    model = DigitCNN()

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        acc = (predicted == y_train).float().mean()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Train Acc: {acc:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)

    print("Classification report:")
    print(classification_report(y_test, predicted.numpy()))


if __name__ == '__main__':
    nn_classification()
