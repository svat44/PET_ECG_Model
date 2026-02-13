import numpy as np
import torch  # pyright: ignore[reportMissingImports]
import wfdb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from model import EGCAutoencoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

model = EGCAutoencoder(window_size=360).to(device)
model.load_state_dict(torch.load("egc_autoencoder.pth", map_location=device))
model.eval()

windows_normal = np.load("windows.npy")
windows_normal_tensor = torch.tensor(windows_normal, dtype=torch.float32).to(device)

with torch.no_grad():
    reconstructed = model(windows_normal_tensor)
    

errors_normal = ((windows_normal_tensor - reconstructed)**2).mean(dim=1).cpu().numpy()

threshold = errors_normal.mean() + 3 * errors_normal.std()
print(f"Threshold: {threshold:.4f}")

abnormal_windows = []
for i, error in enumerate(errors_normal):
    if error > threshold:
        abnormal_windows.append(windows_normal[i])

abnormal_windows = np.array(abnormal_windows) if abnormal_windows else np.array([])
print(f"Number of abnormal windows: {len(abnormal_windows)}")
if len(abnormal_windows) > 0:
    print(f"Abnormal windows shape: {abnormal_windows.shape}")
else:
    print("No abnormal windows detected in training data")


record_id = "119"
signal = wfdb.rdrecord(record_id, pn_dir="mitdb").p_signal[:, 0]  # pyright: ignore[reportAttributeAccessIssue, reportOptionalSubscript]  # pyright: ignore[reportAttributeAccessIssue]
annotation = wfdb.rdann(record_id, "atr", pn_dir="mitdb")

beat_samples = annotation.sample
beat_symbols = annotation.symbol

WINDOW_SIZE = 360
STEP = 50
WINDOW_BEFORE = 180
WINDOW_AFTER = 180

sliding_windows = []
for start in range(0, len(signal) - WINDOW_SIZE, STEP):
    window = signal[start:start+WINDOW_SIZE]
    window = (window - window.mean()) / (window.std() + 1e-8)
    sliding_windows.append(window)

sliding_windows = np.array(sliding_windows)
windows_slide_tensor = torch.tensor(sliding_windows, dtype=torch.float32).to(device)


windows_abnormal = []

for i, sample in enumerate(beat_samples):
    start = sample - WINDOW_BEFORE
    end = sample + WINDOW_AFTER
    if start < 0 or end > len(signal):
        continue
    window = signal[start:end]

    window = (window - np.mean(window)) / (np.std(window) + 1e-8)
    windows_abnormal.append(window)

windows_abnormal = np.array(windows_abnormal)
windows_abnormal_tensor = torch.tensor(windows_abnormal, dtype=torch.float32).to(device)

with torch.no_grad():
    reconstructed_abnormal = model(windows_abnormal_tensor)

errors_abnormal = ((windows_abnormal_tensor - reconstructed_abnormal)**2).mean(dim=1).cpu().numpy()

anomalies = errors_abnormal > threshold
num_anomalies = np.sum(anomalies)
total_beats = len(errors_abnormal)
print(f"Detected {num_anomalies}/{total_beats} anomalies out of beats")


N = 3
top_indices = errors_abnormal.argsort()[-N:][::-1]  # highest errors

for i in top_indices:
    plt.plot(windows_abnormal[i], label="Original")
    plt.plot(reconstructed_abnormal[i].cpu().numpy(), label="Reconstructed")
    plt.title(f"Top {N} Anomalous Heartbeats, Error={errors_abnormal[i]:.4f}")
    plt.legend()
    plt.show()

anomaly_score = anomalies.sum() / len(anomalies)
print(f"Anomaly score (fraction of abnormal beats flagged): {anomaly_score:.2f}")


plt.figure()
plt.hist(errors_normal, bins=50, alpha=0.7, label="Normal")
plt.hist(errors_abnormal, bins=50, alpha=0.7, label="Abnormal")
plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.title("Reconstruction Error Histogram")
plt.savefig("error_histogram.png")

for i in range(len(errors_abnormal)):
    if anomalies[i]:
        plt.plot(windows_abnormal[i], label="Original (Abnormal)")
        plt.plot(reconstructed_abnormal[i].cpu().numpy(), label="Reconstructed")
        plt.title(f"Abnormal heartbeat reconstruction, Error={errors_abnormal[i]:.4f}")
        plt.legend()
        plt.show()
        break


plt.hist(errors_normal, bins=50, alpha=0.7, label="Normal")
plt.hist(errors_abnormal, bins=50, alpha=0.7, label="Abnormal / Flagged")
plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.title("Reconstruction Error with Anomaly Threshold")
plt.show()

# --- ROC Curve & AUC ---
# Labels: 0 = normal (training data), 1 = abnormal (test record)
all_errors = np.concatenate([errors_normal, errors_abnormal])
labels = np.concatenate([np.zeros(len(errors_normal)), np.ones(len(errors_abnormal))])

fpr, tpr, thresholds = roc_curve(labels, all_errors)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Anomaly Detection')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

print(f"AUC: {roc_auc:.4f}")

def this_is_code_displaying_abnormality_graphed_and_reconstruction():
    """
    Pick an abnormal heartbeat with high error
    index = errors_abnormal.argmax()

    plt.plot(windows_abnormal[index], label="Original (Abnormal)")
    plt.plot(reconstructed_abnormal[index].cpu().numpy(), label="Reconstructed")
    plt.title(f"Abnormal heartbeat reconstruction, Error={errors_abnormal[index]:.4f}")
    plt.legend()
    plt.show()
    """
