import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset




######################### MLPs #########################

# Define the MLP Model
class ReLUMLP(nn.Module):
    def __init__(self, input_size=1, hidden_sizes=[128, 128], output_size=1):
        super(ReLUMLP, self).__init__()
        
        # Define layers
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        
        # Combine layers in a Sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    

# Define the MLP model with sine activation function
class SineActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class SineMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SineMLP, self).__init__()
        
        # Define layers
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(SineActivation())  # Use sine activation instead of ReLU
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        
        # Combine layers in a Sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    

class TanhMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(TanhMLP, self).__init__()
        
        # Define layers
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.Tanh()) 
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        
        # Combine layers in a Sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    

# Define the MLP model with snake activation function
class SnakeActivation(nn.Module):
    def forward(self, x):
        return x + torch.sin(x)**2

class SnakeMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SnakeMLP, self).__init__()
        
        # Define layers
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(SnakeActivation())  # Use sine activation instead of ReLU
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        
        # Combine layers in a Sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
    

######################### Data Generation #########################

def generate_data(num_points, noise_std = 0.3, n_waves = 15, shifted=False):

    train_x = np.linspace(-n_waves * np.pi, n_waves * np.pi, num_points)        # E1
    #train_x = np.linspace(0, 2*n_waves * np.pi, num_points)                     # E2
    train_y = np.cos(train_x)+ np.random.normal(0, noise_std, num_points)
    

    if shifted:
        test_x = np.linspace(n_waves * np.pi, 3 * n_waves * np.pi, 100)        # E1
        #test_x = np.linspace(2 * n_waves * np.pi, 4 * n_waves * np.pi, 1000)   # E2
        test_y = np.cos(test_x) + np.random.normal(0, noise_std, 100)

        

    else:
        #val_x = np.linspace(-n_waves * np.pi, n_waves * np.pi, num_points)
        #val_y = np.sin(val_x) #+ np.random.normal(0, noise_std, num_points)

        test_x = np.linspace(-n_waves * np.pi, n_waves * np.pi, 100)   # E1
        #test_y = np.sin(test_x)
        test_y = np.cos(test_x)


    return train_x, train_y, test_x, test_y #, val_x, val_y, test_x, test_y

    




def plot_data_example(train_x, train_y, test_x, test_y):
    plt.figure(figsize=(6, 3))
    plt.title('Noisy Cosine Data')
    plt.plot(train_x, train_y, label='Train data', marker='o', ls='', color='grey', ms=5)
    plt.plot(test_x, test_y, label='Test data', marker='o', ls='-', color='black', ms=3)
    plt.legend(loc='upper left', frameon=False, markerfirst=True) 
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    
    # xticks from -3 pi to 3 pi
    plt.xticks([-3*np.pi, -2*np.pi, -np.pi, 0, np.pi, 2*np.pi, 3*np.pi], ['-3$\pi$', '-2$\pi$', '-$\pi$', '0', '$\pi$', '2$\pi$', '3$\pi$']) # E1

    # xticks from -6 to 18 pi
    #plt.xticks([-6*np.pi, -4*np.pi, -2*np.pi, 0, 2*np.pi, 4*np.pi, 6*np.pi, 8*np.pi, 10*np.pi, 12*np.pi, 14*np.pi, 16*np.pi, 18*np.pi], ['-6$\pi$', '-4$\pi$', '-2$\pi$', '0', '2$\pi$', '4$\pi$', '6$\pi$', '8$\pi$', '10$\pi$', '12$\pi$', '14$\pi$', '16$\pi$', '18$\pi$']) # E2



######################### DATA PREPROCESSING #########################

# Convert each part of the dataset to tensors
def convert_to_tensor(train_x_array, train_y_array, test_x_array, test_y_array):
    train_x = torch.tensor(train_x_array, dtype=torch.float32).view(-1, 1)
    train_y = torch.tensor(train_y_array, dtype=torch.float32).view(-1, 1)
    test_x = torch.tensor(test_x_array, dtype=torch.float32).view(-1, 1)
    test_y = torch.tensor(test_y_array, dtype=torch.float32).view(-1, 1)

    return train_x, train_y, test_x, test_y


# Create dataloaders
def create_dataloader(train_x, train_y, batch_size=25):
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    #val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)
    #test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size)
    
    return train_loader #, val_loader, test_loader




######################### TRAINING #########################

def train_model(model, train_dataloader, test_x, test_y, lr = 0.001, num_epochs=20, print_every=100, val=True):
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Store loss values for plotting
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs): #tqdm(, desc="Training Progress"):
        
        # Training phase
        model.train()
        train_loss = 0.0

        for inputs, targets in train_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        if val:
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                    outputs = model(test_x)
                    loss = criterion(outputs, test_y)
                    val_loss += loss.item()
            val_losses.append(val_loss)

            # Print losses
            if (epoch + 1) % print_every == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        else:
            val_losses = None
            # Print losses
            if (epoch + 1) % print_every == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")

    return model, train_losses, val_losses


def plot_losses(train_losses, val_losses, model_name):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name}: Training and Validation Loss')
    plt.xscale('log')
    plt.yscale('log')   
    plt.legend()
    plt.show()




######################## EVALUATION #########################


def evaluate_model(model, test_x, test_y):
    model.eval()
    criterion = nn.MSELoss()
    test_loss = []

    with torch.no_grad():
            outputs = model(test_x)
            loss = criterion(outputs, test_y)
            test_loss.append(loss.item())
    
    
    return test_loss[0]

def plot_predictions(model, train_x, train_y, test_x, test_y, model_name):
    model.eval()

    # Predict on the chosen test sample
    with torch.no_grad():
        predictions = model(test_x).squeeze().numpy()  # Squeeze to get a 1D array

    # Plot true vs predicted values
    plt.figure(figsize=(8,4))
    #plt.plot(train_x.numpy(), train_y.numpy(), 'o', label='True Data')
    plt.plot(test_x, test_y, label = "True function", linestyle='dashed', color='black')
    plt.plot(test_x, predictions, label='Predictions', color='black')
    plt.title(f'{model_name}: Predictions vs Data')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()




import matplotlib.pyplot as plt

def plot_test_loss_vs_parameter(df, parameter, parameter_name, log = None, fig = None, ax=None, legend=False):
    # Calculate mean and standard deviation of test loss for each model and epoch
    mean_std_df = df.groupby(['Model', parameter]).agg(
        Test_Mean_Loss=('Test_Mean_Loss', 'mean'),
        Test_Std_Loss=('Test_Mean_Loss', 'std')  # Calculate std deviation from multiple repetitions
    ).reset_index()

    # Separate data for plotting
    ReLUMLP_data = mean_std_df[mean_std_df['Model'] == 'ReLUMLP']
    SineMLP_data = mean_std_df[mean_std_df['Model'] == 'SineMLP']
    TanhMLP_dtata = mean_std_df[mean_std_df['Model'] == 'TanhMLP']
    SnakeMLP_data = mean_std_df[mean_std_df['Model'] == 'SnakeMLP']

    # Create a figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # ReLUMLP plot
    ax.plot(ReLUMLP_data[parameter], ReLUMLP_data['Test_Mean_Loss'], '-o', label='ReLU')
    ax.fill_between(
        ReLUMLP_data[parameter],
        ReLUMLP_data['Test_Mean_Loss'] - ReLUMLP_data['Test_Std_Loss'],
        ReLUMLP_data['Test_Mean_Loss'] + ReLUMLP_data['Test_Std_Loss'],
        alpha=0.2
    )

    # TanhMLP plot
    ax.plot(TanhMLP_dtata[parameter], TanhMLP_dtata['Test_Mean_Loss'], '-o', label='Tanh')
    ax.fill_between(
        TanhMLP_dtata[parameter],
        TanhMLP_dtata['Test_Mean_Loss'] - TanhMLP_dtata['Test_Std_Loss'],
        TanhMLP_dtata['Test_Mean_Loss'] + TanhMLP_dtata['Test_Std_Loss'],
        alpha=0.2
    )

    # SineMLP plot
    ax.plot(SineMLP_data[parameter], SineMLP_data['Test_Mean_Loss'], '-o', label='Sine')
    ax.fill_between(
        SineMLP_data[parameter],
        SineMLP_data['Test_Mean_Loss'] - SineMLP_data['Test_Std_Loss'],
        SineMLP_data['Test_Mean_Loss'] + SineMLP_data['Test_Std_Loss'],
        alpha=0.2
    )

    # SnakeMLP plot
    ax.plot(SnakeMLP_data[parameter], SnakeMLP_data['Test_Mean_Loss'], '-o', label='Snake')
    ax.fill_between(
        SnakeMLP_data[parameter],
        SnakeMLP_data['Test_Mean_Loss'] - SnakeMLP_data['Test_Std_Loss'],
        SnakeMLP_data['Test_Mean_Loss'] + SnakeMLP_data['Test_Std_Loss'],
        alpha=0.2
    )

    


    if log:
        ax.set_xscale('log')
    ax.set_yscale('log')

    # Labels and plot formatting
    ax.set_xlabel(f'{parameter_name}')
    ax.set_ylabel('Test Loss (MSE)')
    ax.set_title(f'Test Loss vs. {parameter_name}')
    ax.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)

    if legend:
        ax.legend(frameon=False, markerfirst=False, ncols=2, loc='upper right')

    # Adjust layout only if figure was created in this function
    if ax is None:
        plt.tight_layout()
        plt.show()

    return fig, ax
