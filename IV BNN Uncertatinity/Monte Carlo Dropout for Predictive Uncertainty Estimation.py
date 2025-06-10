import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed
torch.manual_seed(0)
np.random.seed(0)

# Generate synthetic data (noisy sine wave)
x = np.linspace(-3, 3, 100).reshape(-1, 1)
y = np.sin(x) + 0.1 * np.random.randn(*x.shape)

# Convert to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Split into train and test
train_size = 80
x_train_tensor = x_tensor[:train_size]
y_train_tensor = y_tensor[:train_size]
x_test_tensor = x_tensor[train_size:]
y_test = y[train_size:]
x_test = x[train_size:]

# Define MC Dropout model
class MCDropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Initialize model, loss, optimizer
model = MCDropoutModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train model
model.train()
for epoch in range(3000):
    optimizer.zero_grad()
    output = model(x_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

# Enable dropout during inference
model.eval()

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

enable_dropout(model)

# Monte Carlo sampling
T = 100
predictions = []
with torch.no_grad():
    for _ in range(T):
        preds = model(x_test_tensor)
        predictions.append(preds.cpu().numpy())

# Compute mean and std
predictions = np.array(predictions).squeeze()
mean_pred = np.mean(predictions, axis=0)
std_pred = np.std(predictions, axis=0)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x_test, y_test, 'k.', label='True')
plt.plot(x_test, mean_pred, 'b-', label='Prediction')
plt.fill_between(x_test.squeeze(), mean_pred - 2 * std_pred, mean_pred + 2 * std_pred,
                 alpha=0.3, label='Uncertainty')
plt.legend()
plt.title('Monte Carlo Dropout Predictions')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
