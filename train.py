import numpy as np
import matplotlib.pyplot as plt
import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt
from model import EGCAutoencoder


windows = np.load("windows.npy")
windows_tensor = torch.tensor(windows, dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
windows_tensor = windows_tensor.to(device)

model = EGCAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(windows_tensor)
    loss = criterion(outputs, windows_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "egc_autoencoder.pth")
print("Model saved to egc_autoencoder.pth")

model.eval()
with torch.no_grad():
    reconstructed = model(windows_tensor[0:1])

plt.plot(windows_tensor[0].cpu().numpy(), label="Original")
plt.plot(reconstructed[0].cpu().numpy(), label="Reconstructed")
plt.legend()
plt.show()


# print("Windows shape: ", windows.shape)
# print("First window: ", windows[0][:5])

# plt.plot(windows[0])
# plt.show()