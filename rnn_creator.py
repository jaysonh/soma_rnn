import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from framestripper import *
import numpy as np
import pandas as pd
from datetime import datetime


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

# Sequence generation
def generate_tensor(model, start_sequence, sequence_length):
    model.eval()
    generated_sequence = start_sequence
    hidden = model.init_hidden(1)
    input_seq = start_sequence

    for _ in range(sequence_length):
        output, hidden = model(input_seq, hidden)
        #print(f"output: {output}")
        generated_sequence = torch.cat((generated_sequence, output[:, -1:, :]), dim=1)
        input_seq = output[:, -1:, :]

    return generated_sequence

def load_header(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    return lines

def generate_list(model, start_sequence, sequence_length):
    result = []
    model.eval()
    generated_sequence = start_sequence
    hidden = model.init_hidden(1)
    input_seq = start_sequence

    for _ in range(sequence_length):
        output, hidden = model(input_seq, hidden)
        new_line = output[:, -1:, :].tolist()[0][0]
        result.append( new_line )
        input_seq = output[:, -1:, :]

    return result, sequence_length

def create_frames( generated_sequence ):
	num_frames = 0
	txt = ""
	return txt

model = torch.load('models/output_model2500.model')

feature_size = 156
num_generated_frames = 3000

start_sequence = torch.randn(1, 1, feature_size)

# Read the contents of the file
with open('data/Take_EugeniaPhrase_001.bodymodel', 'r') as file:
    header_contents = file.read()

#header_lines = load_header('data/Take_EugeniaPhrase_001.bodymodel')
generated_sequence, num_frames = generate_list(model, start_sequence, num_generated_frames)
movement_data = create_frames( generated_sequence )

result_txt = header_contents + "\n" + "MOTION\n" + "Frames:    " + str(num_frames) + "\n" + "Frame Time:    0.008333\n" #+ movement_data +  "\n"

for i in range(len(generated_sequence)):
    frame_line = str(generated_sequence[i]).replace("[","").replace("]","").replace(",","   ")
    num_line_verts = frame_line.split("    ")
    print(f"Line{i} contains: {num_line_verts} points")
    result_txt += frame_line + "\n"

with open("output.bvh", "w") as text_file:
    text_file.write(result_txt)

