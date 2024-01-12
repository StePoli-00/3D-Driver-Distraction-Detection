from Utils import GCNUtils as gcn_op
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from GCNModel import GATNet
from FocalLoss import FocalLoss

train_path = "./Data/csv/train/"
val_path = "./Data/csv/val/"

batch_size = 64

data = gcn_op.load_gnn_data(train_path, val_path, batch_size, 0, 1)

train_loader = data[0]
validation_loader = data[1]
train_counts = data[2]
train_weights = data[3]
val_counts = data[4]
val_weights = data[5]

num_classes = 5

model = GATNet(input_dim=15, hidden_dim=256, output_dim=5)

# Focal Loss [better for imbalanced datasets]
criterion = FocalLoss(weight=train_weights)
criterion_val = FocalLoss(weight=val_weights)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-7)

accuracies = []
losses = []

# Create a dictionary to store accuracy and loss for each class
class_metrics = {0: {'accuracy': [], 'loss': []},
                 1: {'accuracy': [], 'loss': []},
                 2: {'accuracy': [], 'loss': []},
                 3: {'accuracy': [], 'loss': []},
                 4: {'accuracy': [], 'loss': []}}

epoch_losses = []
epoch_accuracies = []
class_losses = [[] for _ in range(5)]
class_accuracies = [[] for _ in range(5)]
class_precisions = [[] for _ in range(5)]
class_recalls = [[] for _ in range(5)]
class_f1_scores = [[] for _ in range(5)]

num_epochs = 1000

patience = 10
best_val_loss = float('inf')
consecutive_no_improvement = 0
stop_training = False

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    correct_preds = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    total_samples = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for data in tqdm(train_loader):
        optimizer.zero_grad()
        target = data.label.view(-1).long()

        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy and loss for each class
        for class_idx in range(5):
            class_mask = (target == class_idx)
            predicted_labels = torch.argmax(output[class_mask], dim=1)
            correct_preds[class_idx] += (predicted_labels == class_idx).sum().item()

            class_mask_sum = class_mask.sum().item()
            if class_mask_sum > 0:
                class_accuracy = (predicted_labels == class_idx).sum().item() / class_mask.sum().item()
            else:
                class_accuracy = 0.0

            total_samples[class_idx] += class_mask_sum
            class_accuracies[class_idx].append(class_accuracy)
            class_loss = loss.item()
            class_losses[class_idx].append(class_loss)

            # Calculate precision, recall, and F1-score
            if class_mask_sum > 0:
                y_true = (target[class_mask] == class_idx).cpu().numpy()
                y_pred = (predicted_labels == class_idx).cpu().numpy()
                precision = precision_score(y_true, y_pred, zero_division=1.0)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
            else:
                precision = recall = f1 = 0.0

            class_precisions[class_idx].append(precision)
            class_recalls[class_idx].append(recall)
            class_f1_scores[class_idx].append(f1)

    # Update epoch-level metrics
    epoch_losses.append(running_loss / len(train_loader))
    epoch_accuracy = sum(correct_preds.values()) / sum(total_samples.values())
    epoch_accuracies.append(epoch_accuracy)

    # After the epoch is complete, print the results for each class
    for class_idx in range(5):
        if len(class_losses[class_idx]) > 0:
            avg_class_loss = sum(class_losses[class_idx]) / len(class_losses[class_idx])
        else:
            avg_class_loss = 0.0  # Set to 0 if there are no losses for the class

        if len(class_accuracies[class_idx]) > 0:
            avg_class_accuracy = sum(class_accuracies[class_idx]) / len(class_accuracies[class_idx])
        else:
            avg_class_accuracy = 0.0

        print(f"Class {class_idx} - Avg Loss: {avg_class_loss:.4f}, Avg Accuracy: {avg_class_accuracy:.4f}")

    # Calculate the average epoch loss
    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
    avg_epoch_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)
    print(
        f"Epoch {epoch + 1}/{num_epochs} - Avg Epoch Loss: {avg_epoch_loss:.4f}, Avg Epoch Accuracy: {avg_epoch_accuracy:.4f}")

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for data in validation_loader:
            output = model(data)
            target = data.label.view(-1).long()
            val_loss += criterion_val(output, target).item()

        avg_val_loss = val_loss / len(validation_loader)

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1
            if consecutive_no_improvement >= patience:
                print(f"Early stopping after {epoch + 1} epochs with no improvement.")
                stop_training = True
                break

    if stop_training:
        break

    accuracies.append(epoch_accuracy)
    losses.append(running_loss / len(train_loader))

# Save the trained model
model_path = './Model/GCN.pth'
torch.save(model.state_dict(), model_path)
print(f"Trained model saved to '{model_path}'")

# Calculate the average loss and accuracy for each class
for class_idx in range(len(train_weights)):
    if len(class_losses[class_idx]) > 0:
        avg_class_loss = sum(class_losses[class_idx]) / len(class_losses[class_idx])
    else:
        avg_class_loss = 0.0

    if len(class_accuracies[class_idx]) > 0:
        avg_class_accuracy = sum(class_accuracies[class_idx]) / len(class_accuracies[class_idx])
    else:
        avg_class_accuracy = 0.0

    print(f"Class {class_idx} - Avg Loss: {avg_class_loss:.4f}, Avg Accuracy: {avg_class_accuracy:.4f}")

# Calculate the average epoch loss
if len(epoch_losses) > 0:
    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
else:
    avg_epoch_loss = 0.0  # Set to 0 if there are no losses in the epoch
print(f"Avg Epoch Loss: {avg_epoch_loss:.4f}")


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(accuracies) + 1), accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Over Time')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()

plt.tight_layout()
plt.show()

# Plot class-specific metrics
for class_idx in range(5):
    plt.figure(figsize=(10, 5))
    plt.plot(class_precisions[class_idx], label='Precision')
    plt.plot(class_recalls[class_idx], label='Recall')
    plt.plot(class_f1_scores[class_idx], label='F1 Score')
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.legend()
    plt.title(f'Class {class_idx} Metrics Over Epochs')
    plt.show()

# Initialize lists to store the true labels and predicted labels
true_labels = []
predicted_labels = []

model.eval()
with torch.no_grad():
    for data in validation_loader:
        output = model(data)
        target = data.label.view(-1).long()

        # Append true and predicted labels to the lists
        true_labels.extend(target.tolist())
        predicted_labels.extend(torch.argmax(output, dim=1).tolist())

# Calculate the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Initialize lists to store the final metrics
final_precisions = []
final_recalls = []
final_f1_scores = []

for class_idx in range(5):
    if len(class_precisions[class_idx]) > 0:
        final_precision = sum(class_precisions[class_idx]) / len(class_precisions[class_idx])
    else:
        final_precision = 0.0

    if len(class_recalls[class_idx]) > 0:
        final_recall = sum(class_recalls[class_idx]) / len(class_recalls[class_idx])
    else:
        final_recall = 0.0

    if len(class_f1_scores[class_idx]) > 0:
        final_f1_score = sum(class_f1_scores[class_idx]) / len(class_f1_scores[class_idx])
    else:
        final_f1_score = 0.0

    final_precisions.append(final_precision)
    final_recalls.append(final_recall)
    final_f1_scores.append(final_f1_score)

# Print or use the final metrics as needed
for class_idx in range(5):
    print(
        f"Class {class_idx} - Precision: {final_precisions[class_idx]:.4f}, Recall: {final_recalls[class_idx]:.4f}, F1 Score: {final_f1_scores[class_idx]:.4f}")
