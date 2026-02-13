import wfdb
import matplotlib.pyplot as plt
import numpy as np


WINDOW_BEFORE = 180  
WINDOW_AFTER  = 180  
WINDOW_SIZE   = WINDOW_BEFORE + WINDOW_AFTER 

def load_egc_record(record_id="100"):
    record = wfdb.rdrecord(record_id, pn_dir="mitdb")
    signal = record.p_signal    # pyright: ignore[reportAttributeAccessIssue]
    if signal is None:
        raise ValueError(f"No signal data found for record {record_id}")
    ecg = signal[:, 0]
    return ecg

def load_annotations(record_id="100"):
    annotations = wfdb.rdann(record_id, "atr", pn_dir="mitdb")
    if annotations is None:
        raise ValueError(f"No annotations found for record {record_id}")
    samples = annotations.sample  # type: ignore[attr-defined]
    symbols = annotations.symbol  # type: ignore[attr-defined]
    if symbols is None:
        raise ValueError(f"No symbols found in annotations for record {record_id}")
    return samples, symbols

def normalize_signal(ecg):
    mean = np.mean(ecg)
    std = np.std(ecg)
    return (ecg - mean) / std

def create_windows(ecg, window_size=256):
    windows = []
    for i in range(0, len(ecg), window_size):
        window = ecg[i:i+window_size]
        if len(window) < window_size:
            padding = np.zeros(window_size - len(window))
            window = np.concatenate([window, padding])
        windows.append(window)
    
    return np.array(windows)

def extract_normal_windows(signal, beat_samples, beat_symbols):
    windows = []

    for i, sample in enumerate(beat_samples):
        label = beat_symbols[i]
        
        if label != 'N':
            continue

        start = sample - WINDOW_BEFORE
        end = sample + WINDOW_AFTER

        if start < 0 or end > len(signal):
            continue
            
        window = signal[start:end]

        windows.append(window)
    
    return np.array(windows)


if __name__ == "__main__":
    record_id = "100"
    ecg = load_egc_record(record_id)
    print("EGC length: ", len(ecg))
    signal = normalize_signal(ecg)
    # windows = create_windows(signal)

    samples, symbols = load_annotations("100")
    beat_samples = samples
    beat_symbols = symbols
    
    beat_samples, beat_symbols = load_annotations("100")
    windows = extract_normal_windows(signal, beat_samples, beat_symbols)
    # plt.figure(figsize=(10, 4))
    # plt.plot(ecg[:1000])
    # plt.title("Raw ECG Signal (Record 100)")
    # plt.xlabel("Time")
    # plt.ylabel("Voltage")
    # plt.show()
  
    # samples, symbols = load_annotations("100")
    # print("First 10 beat locations: ", samples[:10])
    # print("First 10 beat labels: ", symbols[:10] if len(symbols) >= 10 else symbols)


    print("Number of windows: ", len(windows))
    print("Windows shape ", windows.shape[1])

    np.save("windows.npy", windows)
    print("Windows saved to windows.npy")
    
    # plt.figure(figsize=(5,3))
    # plt.plot(windows[3])
    # plt.title("Window 1")
    # plt.show()

    # print("Number of windows: ", windows.shape[0])
    # print("Windows shape: ", windows.shape[1])
    # print("Original min/max:", np.min(ecg), np.max(ecg))
    # print("Normalized min/max:", np.min(norm_ecg), np.max(norm_ecg))
