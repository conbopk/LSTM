import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm

# load and preprocessing data
with open('Data/Sarcasm data/sarcasm.json', 'r') as f:
    datastore = json.load(f)

dataset = [item['headline'] for item in datastore]
label_dataset = [item['is_sarcastic'] for item in datastore]

# tokenization and vocabulary building
tokenizer = word_tokenize
tokenized_data = [['<bos>'] + tokenizer(text.lower()) + ['<eos>'] for text in dataset]

counter = Counter([token for sublist in tokenized_data for token in sublist])
vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common())}
vocab['<unk>'] = 0
vocab['<pad>'] = 1

# Transform text to tensor
max_length = 25


def text_to_tensor(text):
    tokens = ['<bos>'] + tokenizer(text) + ['<eos>']
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    if len(indices) < max_length:
        indices += [vocab['<pad>']] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    return torch.tensor(indices, dtype=torch.long)


# Prepare Datasets
class SarcasmDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = [text_to_tensor(seq) for seq in sequences]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences_padded = torch.stack(sequences)
    labels = torch.tensor(labels, dtype=torch.float32)
    return sequences_padded, labels


# Split data
train_sequences, test_sequences, train_labels, test_labels = train_test_split(dataset, label_dataset, test_size=0.2,
                                                                              random_state=42)

train_dataset = SarcasmDataset(train_sequences, train_labels)
test_dataset = SarcasmDataset(test_sequences, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


# Define the custom LSTM model
class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.forget_gate = nn.Linear(in_features=input_size + hidden_size,
                                     out_features=hidden_size)  # matrix with size (in_features x out_features)
        self.input_gate = nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size)
        self.output = nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size)
        self.cell_gate = nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden

        combined = torch.cat((x, h_prev), 1)
        f_t = torch.sigmoid(self.forget_gate(combined))
        i_t = torch.sigmoid(self.forget_gate(combined))
        o_t = torch.sigmoid(self.forget_gate(combined))
        c_tilde = torch.tanh(self.cell_gate(combined))

        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.lstm_cell = CustomLSTMCell(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, seq_len = x.size()
        embedded = self.embedding(x)  # ---> (batch_size x seq_len x embedding_dim)

        h_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        for t in range(seq_len):
            h_t, c_t = self.lstm_cell(embedded[:, t, :], (h_t, c_t))

        out = self.fc(h_t)
        return self.sigmoid(out)


# Hyperparameter
embedding_size = 100
hidden_dim = 128
output_dim = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTMClassifier(len(vocab), embedding_size, hidden_dim, output_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training model
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for sequences, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', colour='blue'):
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_train_loss:.4f}')

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()

            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(test_sequences)
    accuracy = correct / total
    print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')
