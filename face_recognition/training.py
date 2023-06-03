import torch
import torch.nn as nn
import torch.optim as optim
from model import SiameseNetwork


def round(data):
    for i, value in enumerate(data):
        data[i] = 0 if value < 0.5 else 1
    return data


def training(train_loader, val_loader, epoch,save_model=True):
    torch.manual_seed(34)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create an instance of the SiameseNetwork
    siamese_net = SiameseNetwork().to(device)

    # Define the loss function (e.g., Binary Cross Entropy)
    criterion = nn.BCELoss()

    # Define the optimizer (e.g., Stochastic Gradient Descent)
    optimizer = optim.Adam(siamese_net.parameters(), lr=1e-4)

    # Training loop
    num_epochs = epoch

    for epoch in range(num_epochs):
        train_loss = 0.0
        total_samples = 0

        # Iterate over the training dataset
        for i, (input_image, validation_image, label) in enumerate(train_loader):
            optimizer.zero_grad()  # Zero the gradients

            input_image = input_image.to(device)
            validation_image = validation_image.to(device)
            label = label.to(device)
            # Forward pass
            output = siamese_net(input_image, validation_image)

            # Compute the loss
            loss = criterion(output, label)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update the running loss
            train_loss += loss.item()
            total_samples += label.size(0)  # Increment total_samples by the batch size

        avg_train_loss = train_loss / total_samples

        print(f"Epoch [{epoch + 1}/{num_epochs}]", flush=True)

        # Evaluate the model on the validation set

        acc_batches = 0
        val_loss = 0.0
        val_total = 0
        total_samples = 0
        correct_val = 0
        val_correct = 0
        with torch.no_grad():
            for i, (input_image, validation_image, label) in enumerate(val_loader):
                input_image = input_image.to(device)
                validation_image = validation_image.to(device)
                label = label.to(device)

                # Move inputs and labels to the GPU
                output = siamese_net(input_image, validation_image)
                loss = criterion(output, label)
                result = round(output)

                val_loss += loss.item()
                total_samples += label.size(0)  # Increment total_samples by the batch size
                val_correct += (result == label).sum().item()

            avg_val_loss = val_loss / total_samples

            print(f"Val Total Loss %f :" % (val_loss), flush=True)
            print(f"Val Average Loss %f :" % (avg_val_loss), flush=True)
            print(f"Val Accuracy %.2f :" % (100 * (val_correct / total_samples)), flush=True)

    if save_model:
        model_scripted = torch.jit.script(siamese_net)
        model_scripted.save('model.pt')
