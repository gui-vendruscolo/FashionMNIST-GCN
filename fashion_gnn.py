import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the transformation to convert the dataset into tensors and normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))  # Normalize the data with mean=0.5 and std=0.5
])

# Load the Fashion-MNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Define the desired sizes for training and validation
train_size = 5000  # Number of samples for training
val_size = 1000   # Number of samples for validation

# Calculate the desired number of samples per class
num_samples_per_class_train = train_size // 10
num_samples_per_class_val = val_size // 10

# Create an empty list to store the selected samples
balanced_train_dataset = []
balanced_val_dataset = []

# Iterate over the classes in the dataset
for class_label in range(10):
    # Filter the samples based on the class label
    class_samples = [sample for sample in train_dataset if sample[1] == class_label]
    
    # Randomly select the desired number of samples from the class for training
    selected_samples_train = torch.utils.data.random_split(class_samples, [num_samples_per_class_train, len(class_samples) - num_samples_per_class_train])[0]
    
    # Randomly select the desired number of samples from the class for validation
    selected_samples_val = torch.utils.data.random_split(class_samples, [num_samples_per_class_val, len(class_samples) - num_samples_per_class_val])[0]
    
    # Add the selected samples to the balanced training and validation datasets
    balanced_train_dataset.extend(selected_samples_train)
    balanced_val_dataset.extend(selected_samples_val)

# Create data loaders for training, validation, and testing
train_loader = torch.utils.data.DataLoader(balanced_train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(balanced_val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

print("loaders set")


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gcn_layer = nn.Linear(input_dim, hidden_dim)
        self.fc_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adjacency_matrix):
        x = self.gcn_layer(torch.matmul(adjacency_matrix, x))
        x = torch.relu(x)
        x = self.fc_layer(x)
        return x
    
def create_adjacency_matrix(images):
    adjacency_matrix = torch.zeros((images.shape[0], images.shape[0]))
    for i in range(images.shape[0]):
        for j in range(images.shape[0]):
            if i != j:
                adjacency_matrix[i][j] = torch.cosine_similarity(images[i].flatten(), images[j].flatten(), dim=0)
    return adjacency_matrix

input_dim = 784  # Input size (28x28 flattened images)
hidden_dim = 64  # Hidden layer size
output_dim = 10  # Number of output classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GCN(input_dim, hidden_dim, output_dim).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


# Training loop
epochs = 10

for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images = images.view(-1, input_dim).to(device)
        adjacency_matrix = create_adjacency_matrix(images).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, adjacency_matrix)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
    
model.eval()
val_loss = 0.0
val_correct = 0
val_total = 0

for images, labels in val_loader:
    images = images.view(-1, input_dim).to(device)
    adjacency_matrix = create_adjacency_matrix(images).to(device)
    labels = labels.to(device)

    outputs = model(images, adjacency_matrix)
    _, predicted = torch.max(outputs.data, 1)

    val_loss += criterion(outputs, labels).item()
    val_total += labels.size(0)
    val_correct += (predicted == labels).sum().item()

val_loss /= len(val_loader)
val_accuracy = 100 * val_correct / val_total
print(f"Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_accuracy:.2f}%")

model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

for images, labels in test_loader:
    images = images.view(-1, input_dim).to(device)
    adjacency_matrix = create_adjacency_matrix(images).to(device)
    labels = labels.to(device)

    outputs = model(images, adjacency_matrix)
    _, predicted = torch.max(outputs.data, 1)

    test_loss += criterion(outputs, labels).item()
    test_total += labels.size(0)
    test_correct += (predicted == labels).sum().item()

test_loss /= len(test_loader)
test_accuracy = 100 * test_correct / test_total
print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%")


# Function to visualize the Fashion-MNIST images and their predicted labels
def visualize_predictions(images, labels, predicted_labels):
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    axs = axs.ravel()
    
    for i in range(25):
        img = images[i].reshape(28, 28)  # Reshape the image to (28, 28)
        axs[i].imshow(img, cmap='gray')
        axs[i].set_title(f"True: {classes[labels[i]]}\nPredicted: {classes[predicted_labels[i]]}")
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Evaluation on the testing dataset and visualization of predictions
model.eval()
predicted_labels = []
images = []
true_labels = []

for images_batch, labels_batch in test_loader:
    images_batch = images_batch.view(-1, input_dim).to(device)
    adjacency_matrix_batch = create_adjacency_matrix(images_batch).to(device)
    labels_batch = labels_batch.to(device)

    outputs_batch = model(images_batch, adjacency_matrix_batch)
    _, predicted_batch = torch.max(outputs_batch.data, 1)

    predicted_labels.extend(predicted_batch.cpu().numpy())
    images.extend(images_batch.cpu().numpy())
    true_labels.extend(labels_batch.cpu().numpy())

predicted_labels = np.array(predicted_labels)
images = np.array(images)
true_labels = np.array(true_labels)

visualize_predictions(images, true_labels, predicted_labels)