import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1. Load training dataset and separate data
df = pd.read_csv(r'D:\Study\digit_recognizer\data\train.csv')
y = df['label']                  
X = df.drop(columns=['label'])    

# 2. Normalize pixel intensities (0.0 to 1.0)
X_scaled = X / 255.0

# 3. Convert to appropriate classification tensors
X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)

# 4. Wrap tensors into a TensorDataset
train_dataset = TensorDataset(X_tensor, y_tensor)

# 5. Create DataLoader to yield mini-batches
# shuffle=True randomizes the dataset order at the start of every epoch
BATCH_SIZE = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Quick verification: Inspect the first mini-batch
features_batch, labels_batch = next(iter(train_loader))
print(f"Dataset total size: {len(train_dataset)} rows")
print(f"Total mini-batches generated: {len(train_loader)}")
print(f"Mini-batch features shape: {features_batch.shape}")  # torch.Size([64, 784])
print(f"Mini-batch labels shape:   {labels_batch.shape}")    # torch.Size([64])



# Define the Neural Network Architecture
class DigitRecognizerNN(nn.Module):
    def __init__(self):
        super(DigitRecognizerNN, self).__init__()
        
        # Hidden Layer 1: Linear 784 -> 128 + ReLU
        self.hidden1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        
        # Hidden Layer 2: Linear 128 -> 64 + ReLU
        self.hidden2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        
        # Output Layer: Linear 64 -> 10 (Raw Logits)
        self.output = nn.Linear(64, 10)
        
    def forward(self, x):
        # Pass data through the layers sequentially
        x = self.relu1(self.hidden1(x))
        x = self.relu2(self.hidden2(x))
        x = self.output(x)  # Returns raw logits (no activation needed before CrossEntropyLoss)
        return x

# Instantiate the Model
model = DigitRecognizerNN()

# Setup Loss Function and Optimizer as specified
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(model)



# Configuration parameters
epochs = 50  # Set how many times you want the model to see the entire dataset

# Set the model to training mode
model.train()

print("🚀 Starting training...")
for epoch in range(epochs):
    running_loss = 0.0
    
    # Loop over all mini-batches from your DataLoader
    for batch_idx, (features, labels) in enumerate(train_loader):
        
        # Clear old gradients from the previous step
        optimizer.zero_grad()
        
        # Run the forward pass (predict logits)
        outputs = model(features)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Call backward() to compute gradients via backpropagation
        loss.backward()
        
        # Update weights using the Adam optimizer
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        
    # Calculate and print average epoch loss
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] - Average Loss: {epoch_loss:.4f}")

print("✅ Training complete!")




# Define the file path
MODEL_PATH = 'digit_model.pth'

# Save the trained model weights
torch.save(model.state_dict(), MODEL_PATH)

print(f"💾 Model parameters successfully saved to {MODEL_PATH}")
