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
from torch.utils.tensorboard import SummaryWriter

class Solver(object):
    """Solver for training and testing."""

    def __init__(self, train_loader, valid_loader, test_loader, device, args):
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
        
        if self.args.use_tensorboard:
            self.writer = SummaryWriter(self.args.checkpoint_path + '/runs/' + self.args.model_name + self.args.opt)

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
        if self.args.mode == 'test':
            self.test()

    def save_model(self, epoch):
        checkpoint_name = "epoch" + str(epoch) + "_" + self.model_name
        check_path = os.path.join(self.args.checkpoint_path, checkpoint_name)
        torch.save(self.model.state_dict(), check_path)
        print("Model saved!")

    def load_model(self):
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        self.model.load_state_dict(torch.load(check_path, map_location=torch.device(self.device)))
        print("Model loaded!", flush=True)
    
    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            prog_bar = tqdm(self.train_loader, total=len(self.train_loader))
            for i,data in enumerate(prog_bar):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                if i % self.args.print_every == self.args.print_every - 1:  
                    self.writer.add_scalar('training loss',
                        running_loss / self.args.print_every,
                        epoch * len(self.train_loader) + i)
                    
            if self.args.scheduler:
                self.scheduler.step()

            epoch_loss = running_loss / len(self.train_loader.dataset) 
            val_loss = self.validate()
            self.writer.add_scalar('validation loss',
                        val_loss,epoch)
            
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

            if (epoch + 1) % self.args.save_every == 0:
                self.evaluate()
                self.save_model(epoch)
        self.save_model(epoch+1)
        print("Training finished!")
        self.evaluate()

    def validate(self):
        print('Validating')
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)

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
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                predictions.append(outputs)

        avg_loss = total_loss / len(self.test_loader.dataset)
        print("MAE: ", avg_loss)
        return avg_loss
    
    def test(self, img_count=10):   
        i=0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                predictions = self.model(inputs)
                plt.figure(figsize=(16,8))
                for i in range(4):
                    plt.subplot(1, 4, i + 1)
                    center_x, center_y = predictions[i][0], predictions[i][1]
                    gt_x, gt_y = labels[i][0], labels[i][1]
                    plt.plot(gt_x, gt_y, 'go', markersize=5, color='blue')
                    plt.plot(center_x, center_y, 'go', markersize=5, color='green')

                    plt.text(0,-5, f"Immagine {i}", color='black', fontsize=10, ha='left', va='top')
                    plt.text(0,-3, f"x_pred: {center_x:.6f}, y_pred: {center_y:.6f}", color='green', fontsize=8, ha='left', va='top')
                    plt.text(0,-2, f"x_real: {gt_x:.6f}, y_real: {gt_y:.6f}", color='blue', fontsize=8, ha='left', va='top')
        
                    plt.imshow(inputs[i].squeeze().numpy(), cmap='gray')
                    plt.xticks([])
                    plt.yticks([])
                
                plt.tight_layout()
                plt.show()   
                i+=1
                if i%img_count==0:
                    break      

    def debug(self):
        print("Debug")

    def plot_loss(self):
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'yellow']  # List of colors for different pairs
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                for i, (target_x, target_y) in enumerate(outputs):
                    pred_x, pred_y = labels[i][0], labels[i][1]
                    color = colors[i % len(colors)] 
                    
                    plt.plot(target_x, target_y, color=color, label=f'Target {i+1}', marker='o')
                    plt.plot(pred_x, pred_y, color=color, label=f'Prediction {i+1}', marker='o')
                    plt.plot([target_x, pred_x], [target_y, pred_y], color=color, linestyle='-')
                break
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.xticks([i/10 for i in range(140, 161, 1)])
        plt.yticks([i/10 for i in range(140, 161, 1)])
        plt.title('Ground Truth vs. Predicted Coordinates')
        plt.grid(True)
        plt.show()




