import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from framestripper import *
import numpy as np
import pandas as pd
from datetime import datetime

if torch.cuda.is_available():
    print("CUDA is available. PyTorch can use GPU.")
else:
    print("CUDA is not available. PyTorch is using CPU.")


# Example data
feature_size = 156
num_frames = 4000 # this is the number

print(f"loading frames from file")
frame_np_array = strip_frame("data/Take_EugeniaPhrase_001.frames",num_frames, feature_size, 4000)
frame_tensor = torch.tensor(frame_np_array)


#print(f"{frame_tensor}")


# Generate random data


#print(f"{data1.size()} {data.size()}")
# Create DataLoader

print(f"Creating TensorDataset")
dataset = TensorDataset(frame_tensor)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define LSTM model
class GenerativeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GenerativeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

# Hyperparameters
input_size = feature_size
hidden_size = 100
num_layers = 2
output_size = feature_size  # Output size should match feature size
num_epochs = 4000
learning_rate = 0.001

# Instantiate the model, define the loss function and the optimizer
print(f"Setting up model")
model = GenerativeLSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()  # Mean Squared Error for sequence generation
optimizer = optim.Adam(model.parameters(), lr=learning_rate )

print(f"training")
# Training loop
model.train()
for epoch in range(num_epochs):

    start_time = time.time()
    for inputs, in data_loader:
        # Initialize hidden state
        hidden = model.init_hidden(inputs.size(0))
        
        # Forward pass
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs, inputs)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    
    end_time = time.time()
    time_taken = end_time - start_time
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f} Time Taken: {time_taken}')

print("Training finished!")

# Sequence generation
def generate_sequence(model, start_sequence, sequence_length):
    model.eval()
    generated_sequence = start_sequence
    hidden = model.init_hidden(1)
    input_seq = start_sequence

    for _ in range(sequence_length):
        output, hidden = model(input_seq, hidden)
        generated_sequence = torch.cat((generated_sequence, output[:, -1:, :]), dim=1)
        input_seq = output[:, -1:, :]

    return generated_sequence

# Generate a new sequence
start_sequence = torch.randn(1, 1, feature_size)

num_generated = 10
generated_sequence = generate_sequence(model, start_sequence, num_generated)
print("Generated sequence:", generated_sequence)
now = datetime.now()
numeric_timestamp = now.strftime("%Y%m%d%H%M%S")

torch.save(model, "models/output_model" + str(numeric_timestamp) + ".model")
