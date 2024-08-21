import re
import matplotlib.pyplot as plt

def extract_data_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    epochs = []
    accuracies = []
    losses = []
    val_accuracies = []
    val_losses = []

    for line in lines:
        if 'Epoch' in line:
            # Extract epoch number
            epoch_match = re.search(r'Epoch (\d+)/50', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                epochs.append(epoch)
        elif 'accuracy:' in line and 'loss:' in line and 'val_accuracy:' in line and 'val_loss:' in line:
            # Extract metrics
            accuracy_match = re.search(r'accuracy: (\d+\.\d+)', line)
            loss_match = re.search(r'loss: (\d+\.\d+)', line)
            val_accuracy_match = re.search(r'val_accuracy: (\d+\.\d+)', line)
            val_loss_match = re.search(r'val_loss: (\d+\.\d+)', line)

            if accuracy_match and loss_match and val_accuracy_match and val_loss_match:
                accuracies.append(float(accuracy_match.group(1)))
                losses.append(float(loss_match.group(1)))
                val_accuracies.append(float(val_accuracy_match.group(1)))
                val_losses.append(float(val_loss_match.group(1)))

    return epochs, accuracies, losses, val_accuracies, val_losses

def plot_metrics(epochs, accuracies, losses, val_accuracies, val_losses):
    plt.figure(figsize=(14, 7))

    # Plot Accuracy and Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(accuracies, marker='o', markersize=0.1, color='b', linestyle='-', label='Accuracy')
    plt.plot(val_accuracies, marker='o', markersize=0.1, color='g', linestyle='--', label='Validation Accuracy')
    plt.title('Accuracy and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0.9, 1.011)  # Assuming accuracy ranges from 0 to 1
    plt.grid(True)
    plt.legend()

    # Plot Loss and Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(losses, marker='o', markersize=0.1, color='r', linestyle='-', label='Loss')
    plt.plot(val_losses, marker='o', markersize=0.1, color='m', linestyle='--', label='Validation Loss')
    plt.title('Loss and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(-0.01, 0.1)  # Adjust based on your data
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    filename = 'train_terminal_100x100'  # Replace with your actual file name
    epochs, accuracies, losses, val_accuracies, val_losses = extract_data_from_file(filename)
    plot_metrics(epochs, accuracies, losses, val_accuracies, val_losses)

if __name__ == '__main__':
    main()
