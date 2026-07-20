# train.py
import torch


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate=0.001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(self, num_epochs):
        for epoch in range(num_epochs):

            # --- training phase ---
            self.model.train()  # enables dropout + batchnorm's training behavior
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()          # clear old gradients
                outputs = self.model(images)        # forward pass
                loss = self.criterion(outputs, labels)
                loss.backward()                     # backward pass (compute gradients)
                self.optimizer.step()               # update weights

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

            # --- validation phase ---
            self.model.eval()  # disables dropout, freezes batchnorm running stats
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():  # no gradients needed, saves memory + time
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

            val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        return self.model