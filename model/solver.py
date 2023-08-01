import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.model import CNNModel, FC2Layer
from dataloader.custom_dataloader import CustomDataset 
import matplotlib.pyplot as plt
import torch.utils.data as data
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler


class Solver(object):
    """Solver for training and testing."""

    def __init__(self, train_loader, valid_loader, test_loader, device, args):
        """Initialize configurations."""
        self.args = args
        self.model_name = 'marker_detector_{}.pth'.format(self.args.model_name)

        self.model = CNNModel().to(device)

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.device = device
        if self.args.criterion == "mse":
            self.criterion = nn.MSELoss()
        elif self.args.criterion == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()

        # load a pretrained model
        if self.args.resume_train or self.args.mode in ['test','evaluate']:
            self.load_model()
        
        if(self.args.mode == "train"):
            if self.args.opt == "SGD":
                self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr)
            elif self.args.opt == "Adam":
                self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
            if self.args.scheduler:
                gamma = args.gamma     
                step_size = args.step_size
                self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
            self.epochs = self.args.epochs
        if self.args.mode == 'evaluate':
            self.evaluate()
            self.plot_loss()

    def save_model(self, epoch):
        # if you want to save the model
        checkpoint_name = "epoch" + str(epoch) + "_" + self.model_name
        check_path = os.path.join(self.args.checkpoint_path, checkpoint_name)
        torch.save(self.model.state_dict(), check_path)
        print("Model saved!")

    def load_model(self):
        # function to load the model
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        self.model.load_state_dict(torch.load(check_path, map_location=torch.device(self.device)))
        print("Model loaded!", flush=True)
    
    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            prog_bar = tqdm(self.train_loader, total=len(self.train_loader))
            for i,data in enumerate(prog_bar):
                # Transfer data to the GPU if available
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Calculate the loss
                loss = self.criterion(outputs, labels)

                # Backpropagation and optimization
                loss.backward()
                self.optimizer.step()

                # Update statistics
                running_loss += loss.item() * inputs.size(0)
            if self.args.scheduler:
                self.scheduler.step()
            epoch_loss = running_loss / len(self.train_loader.dataset) 
            val_loss = self.validate()
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            if (epoch + 1) % self.args.print_every == 0:
                self.evaluate()
        self.save_model(epoch+1)
        print("Training finished!")
        self.evaluate()

    def validate(self):
        print('Validating')
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in self.valid_loader:
                # Transfer data to the GPU if available
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Calculate the loss
                loss = self.criterion(outputs, labels)

                # Update validation loss
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(self.valid_loader.dataset)
        self.model.train()
        return val_loss
            
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        predictions = []
        criterion = nn.L1Loss()
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                # for i in range(outputs.size()[0]):
                #     print(f"x_pred: {outputs[i][0]:.3f}, x_gt: {labels[i][0]:.3f}")
                #     print(f"y_pred: {outputs[i][1]:.3f}, y_gt: {labels[i][1]:.3f}")
                #     print("\n")
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                predictions.append(outputs)

        avg_loss = total_loss / len(self.test_loader.dataset)
        print("MAE: ", avg_loss)
        return avg_loss
    

    def debug(self):
        print("Debug")

    def plot_loss(self):
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'yellow']  # List of colors for different pairs
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                # Create a scatter plot
                for i, (target_x, target_y) in enumerate(outputs):
                    pred_x, pred_y = labels[i][0], labels[i][1]
                    color = colors[i % len(colors)]  # Get the color from the list based on the pair index
                    
                    plt.plot(target_x, target_y, color=color, label=f'Target {i+1}', marker='o')
                    plt.plot(pred_x, pred_y, color=color, label=f'Prediction {i+1}', marker='o')
                    plt.plot([target_x, pred_x], [target_y, pred_y], color=color, linestyle='-')
                break
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.xticks([i/10 for i in range(150, 171, 1)])
        plt.yticks([i/10 for i in range(150, 171, 1)])
        plt.title('Ground Truth vs. Predicted Coordinates')
        #plt.legend()
        plt.grid(True)
        plt.show()




