import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==== CONFIGURATION ====
input_file = "all_sequences.txt"
seq_length = 30 ## average 17.45, maximum 690
batch_size = 32
embedding_dim = 64
hidden_dim = 128
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ==== LOAD SEQUENCES ====
def load_sequences(path):
    sequences = []
    with open(path) as f:
        for line in f:
            tokens = [int(x) for x in line.strip().split()]
            if len(tokens) > 1:
                sequences.append(tokens)
    return sequences

data = load_sequences(input_file)
vocab_size = max(max(seq) for seq in data) + 1
print(f"Loaded {len(data)} sequences. Vocab size = {vocab_size}")

# ==== DATASET PREPARATION ====
class SequenceDataset(Dataset):
    def __init__(self, data, seq_len):
        self.samples = []
        for seq in data:
            for i in range(len(seq) - seq_len):
                x = seq[i:i + seq_len]
                y = seq[i + seq_len]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

dataset = SequenceDataset(data, seq_length)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==== MODEL DEFINITION ====
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel(vocab_size, embedding_dim, hidden_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==== TRAINING LOOP ====
for epoch in range(epochs):
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "lstm_model.pt")
print("✅ Training complete. Model saved as lstm_model.pt.")
