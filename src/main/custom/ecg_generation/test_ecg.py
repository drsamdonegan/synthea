import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from generate_ecg_from_vitals import ecg_simulate
import numpy as np
import os

# Mapping from numeric state label to descriptive state label
STATE_LABELS = {
    0: "Neutral",
    1: "Cardiac Ischaemia",
    2: "Sepsis",
    3: "Acute Anxiety/Panic",
    4: "Breathing Difficulty",
    5: "Hypovolaemia",
    6: "Arrhythmic Flare",
    7: "Hypoglycemia",
    8: "TIA",
    9: "Bathroom (harmless)",
    10: "White Coat (harmless)",
    11: "STEMI (crisis)",
    12: "Septic Shock (crisis)",
    13: "Compromised Airway (crisis)",
    14: "Haemorrhagic Shock (crisis)",
    15: "Stroke (crisis)",
    16: "Death"
}

def create_ecg_params_from_vitals_single_row(row, base_params):
    """
    Wrapper for create_ecg_params_from_vitals to process a single row.
    Converts a single row (Series) into a one-row DataFrame and extracts the state label.
    Handles NaN values by replacing them with defaults.
    """
    from generate_ecg_from_vitals import create_ecg_params_from_vitals
    
    # Create a copy of the row to avoid modifying the original
    row = row.copy()
    
    # Define default values for vital signs (using exact column names from vitals.csv)
    defaults = {
        "heart_rate": 70,
        "diastolic_bp": 80,
        "systolic_bp": 120,
        "respiratory_rate": 12,
        "oxygen_saturation": 98
    }
    
    # Replace NaN values and invalid values with defaults
    for col in defaults:
        if col in row and (pd.isna(row[col]) or np.isinf(row[col]) or row[col] < 0):
            print(f"Warning: Invalid {col} value ({row.get(col, 'NaN')}) for patient {row.get('patient_id', 'unknown')}, using default value of {defaults[col]}")
            row[col] = defaults[col]
        elif col not in row:
            print(f"Warning: Missing {col} column for patient {row.get('patient_id', 'unknown')}, using default value of {defaults[col]}")
            row[col] = defaults[col]
    
    df_row = pd.DataFrame([row])
    state_label = row["state_label"]
    return create_ecg_params_from_vitals(df_row, base_params, state_label)

def test_ecg_states_demo(vitals_csv="vitals.csv"):
    # 1. Load the vitals
    df = pd.read_csv(vitals_csv)
    
    # Print column names to verify what's available
    print("Available columns in vitals.csv:")
    print(df.columns.tolist())
    
    # Exclude death state
    df = df[df["state_label"] != 16].copy()
    df.sort_values(["patient_id", "timestamp"], inplace=True)
    
    # 2. Find unique states from 0..15
    all_states = df["state_label"].unique()
    all_states = sorted(s for s in all_states if s >= 0 and s <= 15)
    
    # Pick one random row per state (one patient per state)
    sample_entries = []
    for st in all_states:
        state_df = df[df["state_label"] == st]
        if state_df.empty: 
            continue
        pick = state_df.sample(n=1, random_state=42)
        for _, row in pick.iterrows():
            sample_entries.append(row)
    
    # 3. Generate a short 10-second ECG snippet for each sample entry
    base_params = {
        "duration": 4,
        "length": None,
        "sampling_rate": 250,
        "noise": 0.05,
        "heart_rate": 70,
        "heart_rate_std": 1,
        "lfhfratio": 0.5,
        "ti": (-85, -15, 0, 15, 125),
        "ai": (0.39, -5, 30, -7.5, 0.30),
        "bi": (0.29, 0.1, 0.1, 0.1, 0.44),
    }
    
    ecg_snippets = []
    for row in sample_entries:
        st_label = row["state_label"]
        pid = row["patient_id"]
        
        # Create ECG parameters
        p_out = create_ecg_params_from_vitals_single_row(row, base_params)
        # Run the simulation
        wave = ecg_simulate(**p_out)
        
        # Store the snippet and include descriptive state label
        ecg_snippets.append({
            "state_label": STATE_LABELS.get(st_label, st_label),
            "patient_id": pid,
            "ecg": wave
        })
    
    # 4. Interactive plot with slider
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)
    line, = ax.plot(ecg_snippets[0]["ecg"], lw=1.5)
    ax.set_title(f"State={ecg_snippets[0]['state_label']} | Patient={ecg_snippets[0]['patient_id']}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    
    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
    slider = Slider(ax_slider, 'Snippet idx',
                    0, len(ecg_snippets)-1, valinit=0, valstep=1)
    
    def update(val):
        idx = int(slider.val)
        snippet = ecg_snippets[idx]["ecg"]
        line.set_ydata(snippet)
        line.set_xdata(range(len(snippet)))
        ax.relim()
        ax.autoscale_view()
        state_lbl = ecg_snippets[idx]["state_label"]
        pid = ecg_snippets[idx]["patient_id"]
        ax.set_title(f"State={state_lbl} | Patient={pid}")
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    # Create output directory if it doesn't exist
    output_dir = "test_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual plots for each snippet with state labels included
    for idx, snippet in enumerate(ecg_snippets):
        plt.figure(figsize=(10, 6))
        plt.plot(snippet["ecg"], lw=1.5)
        plt.title(f"State={snippet['state_label']} | Patient={snippet['patient_id']}")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.savefig(os.path.join(output_dir, f"ecg_snippet_{idx}.png"))
        plt.close()
    
    plt.show()

# Run the test
if __name__ == "__main__":
    test_ecg_states_demo("vitals.csv")