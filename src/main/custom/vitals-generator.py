#!/usr/bin/env python
import os
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from datetime import datetime, timedelta
# Import the ECG simulation function (assumed available)
from ecg_simulate import ecg_simulate

##########################################################
# ECG GENERATION FUNCTIONS (from your ecg script)
##########################################################
def base_ecg_params():
    """
    Returns a dictionary of baseline parameters for ecg_simulate (ECGSYN).
    """
    return {
        "duration": 10,             # will be overridden per block/chunk
        "length": None,             # computed by ecg_simulate if None
        "sampling_rate": 250,
        "noise": 0.05,              # amplitude of Laplace noise
        "heart_rate": 70,           # BPM
        "heart_rate_std": 1,        # BPM standard deviation
        "lfhfratio": 0.5,           # ratio of low-frequency to high-frequency HR variability
        "ti": (-85, -15, 0, 15, 125),  # P, Q, R, S, T wave angles (degrees)
        "ai": (0.39, -5, 30, -7.5, 0.30),  # amplitudes of PQRST waves
        "bi": (0.29, 0.1, 0.1, 0.1, 0.44)  # Gaussian widths for each wave
    }

def apply_vitals_to_params(hr, diastolic_bp, systolic_bp, rr, spo2, a, b, t):
    """
    Modify ECG parameters (a, b, t) based on current vitals.
    (See your original comments for details.)
    """
    a = list(a)
    b = list(b)
    t = list(t)
    
    baseline_hr   = 70.0
    baseline_sbp  = 120.0
    baseline_dbp  = 80.0
    baseline_rr   = 16.0
    baseline_spo2 = 98.0

    # Adjust R-wave amplitude based on systolic BP
    if systolic_bp >= baseline_sbp:
        sbp_diff = systolic_bp - baseline_sbp
        factor = 1 + 0.05 * (sbp_diff / 10.0)
        factor = min(max(factor, 0.9), 1.1)
    else:
        sbp_diff = baseline_sbp - systolic_bp
        factor = 1 + 0.02 * (sbp_diff / 10.0)
        factor = min(factor, 1.05)
    a[2] = a[2] * factor

    # Adjust Q-wave amplitude based on diastolic BP
    dbp_diff = diastolic_bp - baseline_dbp
    q_factor = 1 + 0.03 * (dbp_diff / 10.0) if dbp_diff > 0 else 1
    q_factor = min(max(q_factor, 0.95), 1.05)
    a[1] = a[1] * q_factor

    # Adjust T-wave amplitude based on respiratory rate
    rr_diff = rr - baseline_rr
    if rr_diff > 0:
        t_factor = max(1 - 0.01 * rr_diff, 0.7)
        a[4] = a[4] * t_factor

    # Adjust P-wave amplitude based on SpO2
    spo2_diff = baseline_spo2 - spo2
    if spo2_diff > 0:
        new_a0 = a[0] - 0.01 * spo2_diff
        a[0] = np.clip(new_a0, 0.2, 0.3)
    
    # Additional T-wave reduction for severe hypoxemia
    if spo2 < 90:
        hypoxemia_diff = 90 - spo2
        hypoxemia_factor = max(1 - 0.02 * hypoxemia_diff, 0.85)
        a[4] = a[4] * hypoxemia_factor

    # Adjust widths based on heart rate
    if hr > baseline_hr:
        width_factor = 1 - 0.01 * (hr - baseline_hr)
        width_factor = max(width_factor, 0.8)
        b = [width * width_factor for width in b]

    # Adjust timing parameters inversely with heart rate
    if hr > 0:
        t = [ti * (baseline_hr / hr) for ti in t]

    return a, b, t

def create_ecg_params_from_vitals(block_df, base_params, state_label):
    """
    Create new ECG parameters from a block of vitals data.
    Applies state-specific morphological changes.
    """
    p_out = dict(base_params)
    defaults = {
        "heart_rate": 70,
        "diastolic_bp": 80,
        "systolic_bp": 120,
        "respiratory_rate": 12,
        "oxygen_saturation": 98
    }
    hr_avg = block_df["heart_rate"].mean() if not pd.isna(block_df["heart_rate"].mean()) else defaults["heart_rate"]
    hr_std = block_df["heart_rate"].std() if not pd.isna(block_df["heart_rate"].std()) else 1.0
    dbp_avg = block_df["diastolic_bp"].mean() if not pd.isna(block_df["diastolic_bp"].mean()) else defaults["diastolic_bp"]
    sbp_avg = block_df["systolic_bp"].mean() if not pd.isna(block_df["systolic_bp"].mean()) else defaults["systolic_bp"]
    rr_avg = block_df["respiratory_rate"].mean() if not pd.isna(block_df["respiratory_rate"].mean()) else defaults["respiratory_rate"]
    spo2_avg = block_df["oxygen_saturation"].mean() if not pd.isna(block_df["oxygen_saturation"].mean()) else defaults["oxygen_saturation"]

    p_out["heart_rate"] = max(min(hr_avg, 200), 40)
    p_out["heart_rate_std"] = max(min(hr_std, 10), 0)
    
    a_i_list = list(p_out["ai"])
    t_i_list = list(p_out["ti"])
    b_i_list = list(p_out["bi"])
    
    def angle_multiplier(current_angle, desired_degree_shift):
        if abs(current_angle) < 1e-6:
            base_val = 1.0
        else:
            base_val = current_angle
        new_angle = base_val + desired_degree_shift
        return new_angle / base_val

    # Example state-specific changes (you can expand these as needed)
    if state_label == 1:  # Cardiac Ischaemia
        a_i_list[4] = -abs(a_i_list[4]) * 0.8
        a_i_list[2] = a_i_list[2] * 0.9
        t_i_list[4] *= angle_multiplier(t_i_list[4], +3)
        p_out["lfhfratio"] *= 0.8
    elif state_label == 9:  # Bathroom: zero out the ECG
        a_i_list = [0, 0, 0, 0, 0]
        p_out["noise"] = 0.0
    elif state_label == 16:  # Death
        a_i_list = [0, 0, 0, 0, 0]
        p_out["noise"] = 0.0
    # (Include other states as needed...)

    p_out["ai"] = tuple(a_i_list)
    p_out["ti"] = tuple(t_i_list)
    p_out["bi"] = tuple(b_i_list)
    return p_out

def s_curve_ecg(alpha, steepness=3.0):
    """S-curve (logistic) function for ECG blending."""
    return 1.0 / (1.0 + np.exp(-steepness * (alpha - 0.5)))

def blend_ecg_params(old_params, new_params, alpha):
    """
    Interpolate numeric fields in old_params -> new_params with factor alpha.
    """
    out = dict(new_params)
    for key in ["ti", "ai", "bi"]:
        if key in old_params and key in new_params:
            old_array = old_params[key]
            new_array = new_params[key]
            if len(old_array) == len(new_array):
                blended = []
                for (ov, nv) in zip(old_array, new_array):
                    blended.append((1 - alpha) * ov + alpha * nv)
                out[key] = tuple(blended)
    for key2 in ["heart_rate", "noise", "heart_rate_std"]:
        if key2 in old_params and key2 in new_params:
            out[key2] = (1 - alpha) * old_params[key2] + alpha * new_params[key2]
    return out

def generate_ecg_for_block(block_df, old_params, new_params, sampling_rate=250):
    """
    Create one contiguous ECG snippet for the entire block.
    Uses sub-chunks to blend parameters from old_params to new_params.
    """
    block_duration_sec = 5.0 * len(block_df)
    if block_duration_sec <= 0:
        return np.array([]), np.array([])

    chunk_size = 5.0
    n_chunks = int(math.ceil(block_duration_sec / chunk_size))
    n_total_samples = int(block_duration_sec * sampling_rate)
    ecg_out = np.zeros(n_total_samples)
    time_out = np.linspace(0, block_duration_sec, n_total_samples, endpoint=False)
    idx_start = 0

    for chunk_i in range(n_chunks):
        chunk_start_sec = chunk_i * chunk_size
        chunk_end_sec = min(block_duration_sec, chunk_start_sec + chunk_size)
        this_chunk_duration = chunk_end_sec - chunk_start_sec
        if this_chunk_duration <= 0:
            break
        alpha_lin = chunk_i / float(max(n_chunks - 1, 1))
        alpha_s = s_curve_ecg(alpha_lin, steepness=5.0)
        chunk_params = blend_ecg_params(old_params, new_params, alpha_s)
        chunk_params["duration"] = this_chunk_duration
        chunk_params["length"] = None
        chunk_params["sampling_rate"] = sampling_rate
        ecg_chunk = ecg_simulate(**chunk_params)
        chunk_nsamples = len(ecg_chunk)
        idx_end = idx_start + chunk_nsamples
        if idx_end > len(ecg_out):
            ecg_chunk = ecg_chunk[:(len(ecg_out) - idx_start)]
            idx_end = len(ecg_out)
        ecg_out[idx_start:idx_end] = ecg_chunk
        idx_start = idx_end
    return time_out, ecg_out

def generate_ecg_postprocess(vitals_csv_path, ecg_csv_name):
    """
    Reads the vitals CSV, groups rows into blocks (per patient and state),
    generates ECG snippets for each block using the ecg_simulate function,
    and writes the output to ecg_csv_name.
    """
    import csv
    if not os.path.exists(vitals_csv_path):
        raise FileNotFoundError(f"Cannot find {vitals_csv_path}")
    df = pd.read_csv(vitals_csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.sort_values(["patient_id", "timestamp"], inplace=True)
    df["block_change"] = (df["patient_id"].shift(1) != df["patient_id"]) | (df["state_label"].shift(1) != df["state_label"])
    df["block_id"] = df["block_change"].cumsum()
    base_p = base_ecg_params()
    ecg_records = []
    prev_params_by_pid = {}
    grouped = df.groupby(["block_id", "patient_id"], as_index=False)
    for (block_id, pid), block_df in grouped:
        st_label = block_df["state_label"].iloc[0]
        new_params = create_ecg_params_from_vitals(block_df, base_p, st_label)
        old_params = prev_params_by_pid.get(pid, base_p)
        if st_label == 9:  # For Bathroom state, use flat parameters.
            old_params = new_params
        t_ecg, snippet = generate_ecg_for_block(block_df, old_params, new_params, sampling_rate=base_p["sampling_rate"])
        block_start_ts = block_df["timestamp"].iloc[0].isoformat() if len(block_df) > 0 else ""
        for i, amp in enumerate(snippet):
            ecg_records.append({
                "patient_id": pid,
                "block_id": block_id,
                "block_start_timestamp": block_start_ts,
                "time_in_block_sec": round(t_ecg[i], 3),
                "ecg_amplitude": round(float(amp), 5),
                "state_label": st_label
            })
        prev_params_by_pid[pid] = new_params
    out_df = pd.DataFrame(ecg_records)
    out_df.to_csv(ecg_csv_name, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"ECG data saved to {ecg_csv_name}")


def generate_ecg_for_patient(patient_df, patient_id, base_params, ecg_csv, prev_params=None):
    """
    Generate ECG data for a single patient's vital signs DataFrame and append
    the ECG records to ecg_csv.

    Parameters:
      patient_df   : DataFrame with columns including timestamp, heart_rate,
                     diastolic_bp, systolic_bp, respiratory_rate, oxygen_saturation,
                     and state_label.
      patient_id   : ID of the patient.
      base_params  : The base ECG parameters (from base_ecg_params()).
      ecg_csv      : Path to the ECG output CSV file.
      prev_params  : The previous ECG parameters (optional) for smooth blending.
                     If None, base_params is used.
    Returns:
      The final ECG parameters for this patient (useful if processing multiple blocks).
    """
    # Sort patient data by timestamp
    patient_df = patient_df.sort_values("timestamp")
    # Identify blocks by change in state_label
    patient_df["block_change"] = patient_df["state_label"].shift(1) != patient_df["state_label"]
    patient_df["block_id"] = patient_df["block_change"].cumsum()

    ecg_records = []
    if prev_params is None:
        prev_params = base_params

    # Process each block for this patient
    for block_id, block_df in patient_df.groupby("block_id"):
        st_label = block_df["state_label"].iloc[0]
        new_params = create_ecg_params_from_vitals(block_df, base_params, st_label)
        # For example, if in Bathroom state, override blending:
        if st_label == 9:
            prev_params = new_params
        # Generate the ECG snippet for this block:
        t_ecg, snippet = generate_ecg_for_block(block_df, prev_params, new_params,
                                                sampling_rate=base_params["sampling_rate"])
        block_start_ts = block_df["timestamp"].iloc[0].isoformat()
        for i, amp in enumerate(snippet):
            ecg_records.append({
                "patient_id": patient_id,
                "block_id": block_id,
                "block_start_timestamp": block_start_ts,
                "time_in_block_sec": round(t_ecg[i], 3),
                "ecg_amplitude": round(float(amp), 5),
                "state_label": st_label
            })
        prev_params = new_params

    # Convert to DataFrame and write/appended to ecg_csv
    df_ecg = pd.DataFrame(ecg_records)
    if not os.path.exists(ecg_csv):
        df_ecg.to_csv(ecg_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)
    else:
        df_ecg.to_csv(ecg_csv, mode='a', header=False, index=False)
    return prev_params

##########################################################
# VITAL SIGNS SIMULATION FUNCTIONS & DICTIONARIES
##########################################################
# (The following dictionaries and functions are taken from your vital sign generator script)

# Example dictionaries (adjust as needed)
CHARACTERISTICS = {
    "Age_Pediatric": {
        "vital_sign_modifiers": {
            "HR_factor_range": (1.10, 1.20),
            "RR_factor_range": (1.10, 1.20)
        },
        "risk_modifiers": {
            "Breathing_difficulty_warning": 1.3,
            "Cardiac_Ischaemia_warning": 0.5
        }
    },
    "Age_Elderly": {
        "vital_sign_modifiers": {
            "BP_offset_range": (5, 10)
        },
        "risk_modifiers": {
            "Cardiac_Ischaemia_warning": 1.5,
            "Sepsis_warning": 1.5,
            "TIA_warning": 1.5
        }
    },
    "Sex_Male": {
        "vital_sign_modifiers": {
            "BP_offset_range": (0, 5)
        },
        "risk_modifiers": {
            "Cardiac_Ischaemia_warning": 1.2
        }
    },
    "Sex_Female": {
        "vital_sign_modifiers": {},
        "risk_modifiers": {}
    },
    "BodyBuild_Obese": {
        "vital_sign_modifiers": {
            "BP_offset_range": (5, 10),
            "HR_offset_range": (5, 5)
        },
        "risk_modifiers": {
            "Cardiac_Ischaemia_warning": 2.0,
            "STEMI_crisis": 1.5
        }
    },
    "BodyBuild_Underweight": {
        "vital_sign_modifiers": {
            "BP_offset_range": (-5, -5)
        },
        "risk_modifiers": {
            "Sepsis_warning": 1.3
        }
    }
}

PREVIOUS_CONDITIONS = {
    "Hypertension": {
        "vital_sign_modifiers": {
            "BP_offset_range": (10, 20)
        },
        "risk_modifiers": {
            "Cardiac_Ischaemia_warning": 2.0,
            "STEMI_crisis_if_Cardiac_Ischaemia": 1.5
        }
    },
    "Atrial_Fibrillation": {
        "vital_sign_modifiers": {
            "HR_irregular_variation": (10, 20)
        },
        "risk_modifiers": {
            "Cardiac_Ischaemia_warning": 1.2,
            "Stroke_crisis": 2.0
        }
    },
    "Diabetes_Type2": {
        "vital_sign_modifiers": {
            "HR_offset_range": (5, 10)
        },
        "risk_modifiers": {
            "Hypoglycemia_warning": 1.5,
            "Compromised_Airway_crisis_if_seizure": 1.2
        }
    },
    "Seizure_Disorder": {
        "risk_modifiers": {
            "Seizure_warning": 1.5,
            "Compromised_Airway_crisis_if_seizure": 1.3
        }
    },
    "Depression_Anxiety": {
        "vital_sign_modifiers": {
            "HR_offset_range": (5, 5)
        },
        "risk_modifiers": {
            "Acute_Anxiety_warning": 2.0
        }
    },
    "COPD": {
        "vital_sign_modifiers": {
            "O2_sat_offset_range": (-3, -2),
            "RR_offset_range": (2, 3)
        },
        "risk_modifiers": {
            "Breathing_difficulty_warning": 1.5,
            "Compromised_Airway_crisis_if_Breathing_difficulty": 1.3
        }
    },
    "Hemophilia": {
        "risk_modifiers": {
            "Hypovolaemia_warning": 2.0,
            "Hemorrhagic_Shock_crisis_if_Hypovolaemia": 2.0
        }
    },
    "Anemia": {
        "vital_sign_modifiers": {
            "HR_offset_range": (5, 10)
        },
        "risk_modifiers": {
            "Hemorrhagic_Shock_crisis_if_Hypovolaemia": 1.5,
            "Breathing_difficulty_warning": 1.2
        }
    },
    "Chronic_Kidney_Disease": {
        "risk_modifiers": {
            "Sepsis_warning": 1.2,
            "Hemorrhagic_Shock_crisis_if_Hypovolaemia": 1.2
        }
    }
}

CURRENT_CONDITIONS = {
    "Mild_Injury": {
        "vital_sign_modifiers": {
            "HR_offset_range": (10, 20),
            "BP_offset_range": (5, 10),
            "RR_offset_range": (2, 5),
            "O2_sat_offset_range": (-1, 0)
        },
        "risk_modifiers": {}
    },
    "Weird_Infection": {
        "vital_sign_modifiers": {
            "HR_factor_range": (1.3, 1.8),
            "RR_factor_range": (1.3, 1.6),
            "BP_offset_range": (-30, -10),
            "O2_sat_offset_range": (-5, -2)
        },
        "risk_modifiers": {
            "Sepsis_warning": 1.2
        }
    },
    "Appendicitis": {
        "vital_sign_modifiers": {
            "HR_offset_range": (10, 20),
            "RR_offset_range": (2, 5)
        },
        "risk_modifiers": {
            "Sepsis_warning": 1.3
        }
    },
    "Asthma": {
        "vital_sign_modifiers": {
            "RR_factor_range": (1.5, 2.0),
            "HR_offset_range": (5, 10),
            "O2_sat_offset_range": (-5, -3)
        },
        "risk_modifiers": {
            "Breathing_difficulty_warning": 1.2
        }
    },
    "Chest_Pain": {
        "vital_sign_modifiers": {
            "HR_factor_range": (1.2, 1.5),
            "O2_sat_offset_range": (-5, -1)
        },
        "risk_modifiers": {
            "Cardiac_Ischaemia_warning": 1.0
        }
    },
    "Drug_Overdose": {
        "vital_sign_modifiers": {
            "HR_offset_range": (-20, 20),
            "RR_offset_range": (-5, 10),
            "O2_sat_offset_range": (-10, 0)
        },
        "risk_modifiers": {
            "Breathing_difficulty_warning": 1.5
        }
    },
    "Normal_Pregnancy": {
        "vital_sign_modifiers": {
            "HR_offset_range": (10, 10),
            "BP_offset_range": (2, 5)
        },
        "risk_modifiers": {}
    }
}

STATE_LIST = [
    "Neutral",                     # 0
    "Cardiac Ischaemia",           # 1
    "Sepsis",                      # 2
    "Acute Anxiety/Panic",         # 3
    "Breathing Difficulty",        # 4
    "Hypovolaemia",                # 5
    "Arrhythmic Flare",            # 6
    "Hypoglycemia",                # 7
    "TIA",                         # 8
    "Bathroom (harmless)",         # 9
    "White Coat (harmless)",       # 10
    "STEMI (crisis)",              # 11
    "Septic Shock (crisis)",       # 12
    "Compromised Airway (crisis)", # 13
    "Haemorrhagic Shock (crisis)", # 14
    "Stroke (crisis)",             # 15
    "Death"                        # 16
]

basis_transition_matrix = np.array([
    [0.70, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.05, 0.05, 0,    0,    0,    0,    0,    0],
    [0.50, 0.30,  0,     0,     0,     0,     0,     0,     0,     0,    0,    0.20, 0,    0,    0,    0,    0],
    [0.15, 0,     0.45,  0,     0,     0,     0,     0,     0,     0,    0,    0,    0.40, 0,    0,    0,    0],
    [0.80, 0,     0,     0.20,  0,     0,     0,     0,     0,     0,    0,    0,    0,    0,    0,    0,    0],
    [0.50, 0,     0,     0,     0.35,  0,     0,     0,     0,     0,    0,    0,    0.15, 0,    0,    0,    0],
    [0.35, 0,     0,     0,     0,     0.40,  0,     0,     0,     0,    0,    0,    0,    0,    0.25, 0,    0],
    [0.65, 0,     0,     0,     0,     0,     0.20,  0,     0,     0,    0,    0.15, 0,    0,    0,    0,    0],
    [0.75, 0,     0,     0,     0,     0,     0,     0.17,  0,     0,    0,    0,    0,    0.08, 0,    0,    0],
    [0.70, 0,     0,     0,     0,     0,     0,     0,     0.10,  0,    0,    0,    0,    0,    0,    0.20, 0],
    [1.00, 0,     0,     0,     0,     0,     0,     0,     0,     0,    0,    0,    0,    0,    0,    0,    0],
    [1.00, 0,     0,     0,     0,     0,     0,     0,     0,     0,    0,    0,    0,    0,    0,    0,    0],
    [0,    0.80,  0,     0,     0,     0,     0,     0,     0,     0,    0,    0,    0,    0,    0,    0,    0.20],
    [0,    0,     0.90,  0,     0,     0,     0,     0,     0,     0,    0,    0,    0,    0,    0,    0,    0.10],
    [0,    0,     0,     0,     0,     0,     0,     0.50,  0,     0,    0,    0,    0,    0,    0,    0,    0.50],
    [0,    0,     0,     0,     0,     0,     0,     0,     0,     0,    0,    0,    0,    0,    0,    0,    0.50],
    [0,    0,     0,     0,     0,     0,     0,     0,     0,     0,    0,    0,    0,    0,    0,    0,    0.30],
    [0,    0,     0,     0,     0,     0,     0,     0,     0,     0,    0,    0,    0,    0,    0,    0,    1.00]
])

basis_transition_matrix_comb = {
    "states": STATE_LIST,
    "transition_matrix": basis_transition_matrix
}

WARNING_STATES = {
    "Cardiac Ischaemia": {
        "HR_factor_range": (1.3, 1.7),
        "BP_offset_range": (20, 40),
        "RR_offset_range": (3, 6),
        "O2_sat_offset_range": (-4, -2),
        "duration_range": (360,720), 
        "prob_escalate_crisis": 0.03,
        "maps_to_crisis": "STEMI"
    },
    "Sepsis": {
        "HR_factor_range": (1.2, 1.5),
        "BP_offset_range": (-20, -10),
        "RR_factor_range": (1.2, 1.4),
        "O2_sat_offset_range": (-4, -1),
        "duration_range": (360,720),
        "prob_escalate_crisis": 0.03,
        "maps_to_crisis": "Septic_Shock"
    },
    "Acute Anxiety/Panic": {
        "HR_factor_range": (1.3, 1.6),
        "RR_factor_range": (1.2, 1.5),
        "O2_sat_offset_range": (-2, 0),
        "duration_range": (120,240),
        "prob_escalate_crisis": 0.0,
        "maps_to_crisis": None
    },
    "Breathing Difficulty": {
        "HR_factor_range": (1.3, 1.6),
        "RR_factor_range": (1.3, 1.7),
        "O2_sat_offset_range": (-5, -2),
        "duration_range": (360,720),
        "prob_escalate_crisis": 0.03,
        "maps_to_crisis": "Compromised_Airway"
    },
    "Hypovolaemia": {
        "HR_factor_range": (1.3,1.6),
        "BP_offset_range": (-20,-10),
        "RR_offset_range": (3,6),
        "O2_sat_offset_range": (-1,0),
        "duration_range": (240,720),
        "prob_escalate_crisis": 0.03,
        "maps_to_crisis": "Hemorrhagic_Shock"
    },
    "Arrhythmic Flare": {
        "HR_irregular_variation": (30,50),
        "BP_offset_range": (-20,20),
        "duration_range": (240,720),
        "prob_escalate_crisis": 0.03,
        "maps_to_crisis": "STEMI"
    },
    "Hypoglycemia": {
        "HR_factor_range": (1.2,1.6),
        "BP_offset_range": (-5,0),
        "RR_offset_range": (3,5),
        "O2_sat_offset_range": (0,0),
        "duration_range": (240,600),
        "prob_escalate_crisis": 0.02,
        "maps_to_crisis": "Compromised_Airway"
    },
    "TIA": {
        "HR_factor_range": (1.1,1.3),
        "BP_offset_range": (30,50),
        "duration_range": (240,720),
        "prob_escalate_crisis": 0.04,
        "maps_to_crisis": "Stroke"
    }
}

CRISIS_STATES = {
    "STEMI (crisis)": {
        "HR_factor_range": (1.6,2.3),
        "BP_offset_range": (-40,-20),
        "O2_sat_offset_range": (-8,-3),
        "no_spontaneous_recovery": True
    },
    "Septic Shock (crisis)": {
        "HR_factor_range": (1.6,2.0),
        "BP_offset_range": (-80,-40),
        "RR_factor_range": (1.5,2.0),
        "O2_sat_offset_range": (-10,-6),
        "no_spontaneous_recovery": True
    },
    "Compromised Airway (crisis)": {
        "HR_factor_range": (1.5,2.0),
        "RR_factor_range": (2.0,3.0),
        "O2_sat_offset_range": (-40,-20),
        "no_spontaneous_recovery": True
    },
    "Haemorrhagic Shock (crisis)": {
        "HR_factor_range": (1.5,2.5),
        "BP_offset_range": (-80,-50),
        "RR_factor_range": (1.5,2.0),
        "O2_sat_offset_range": (-8,-4),
        "no_spontaneous_recovery": True
    },
    "Stroke (crisis)": {
        "HR_factor_range": (1.2,1.5),
        "BP_offset_range": (40,60),
        "RR_factor_range": (1.2,1.4),
        "no_spontaneous_recovery": True
    }
}

DEATH_STATE = "Death"

# Functions for patient encounter processing
def load_encounters(csv_path="test_final_ed_patients.csv"):
    """Load encounter data from a CSV file."""
    return pd.read_csv(csv_path)

def reason_modifiers(reason):
    if not isinstance(reason, str):
        reason = str(reason) if reason else ""
    reason = reason.lower()
    offsets = {'heart_rate': 0.0, 'systolic_bp': 0.0, 'diastolic_bp': 0.0,
               'respiratory_rate': 0.0, 'oxygen_saturation': 0.0}
    if any(x in reason for x in ["laceration", "sprain", "fracture of bone", "injury of neck", "injury of knee", "impacted molars", "minor burn"]):
        offsets['heart_rate'] += 10; offsets['systolic_bp'] += 5; offsets['diastolic_bp'] += 3; offsets['respiratory_rate'] += 2
    elif any(x in reason for x in ["myocardial infarction", "chest pain"]):
        offsets['heart_rate'] += 10; offsets['systolic_bp'] -= 5; offsets['diastolic_bp'] -= 3; offsets['oxygen_saturation'] -= 2
    elif any(x in reason for x in ["asthma", "difficulty breathing"]):
        offsets['respiratory_rate'] += 4; offsets['oxygen_saturation'] -= 3
    elif "seizure" in reason:
        offsets['heart_rate'] += 6
    elif "drug overdose" in reason:
        offsets['respiratory_rate'] -= 3; offsets['oxygen_saturation'] -= 5; offsets['heart_rate'] -= 5
    elif any(x in reason for x in ["sepsis", "appendicitis", "infection", "pyrexia of unknown origin"]):
        offsets['heart_rate'] += 5; offsets['systolic_bp'] -= 5; offsets['diastolic_bp'] -= 3; offsets['oxygen_saturation'] -= 2
    elif any(x in reason for x in ["burn injury", "major burn"]):
        offsets['heart_rate'] += 15; offsets['systolic_bp'] += 5; offsets['diastolic_bp'] += 3; offsets['respiratory_rate'] += 3
    return offsets

def pain_modifiers(pain):
    try:
        p = float(pain)
    except:
        p = 0
    return {'heart_rate': p * 1.0, 'systolic_bp': p * 0.5, 'diastolic_bp': p * 0.3,
            'respiratory_rate': p * 0.2, 'oxygen_saturation': 0.0}

def apply_modifiers(baseline_dict, reason_desc, pain_score, height, weight, temperature):
    def fallback(key):
        if key == "systolic_bp": return 120
        if key == "diastolic_bp": return 80
        if key == "heart_rate": return 75
        if key == "respiratory_rate": return 16
        if key == "oxygen_saturation": return 98
        return 100
    r_off = reason_modifiers(reason_desc)
    p_off = pain_modifiers(pain_score)
    out = {}
    for k in baseline_dict.keys():
        base = baseline_dict[k]
        if pd.isna(base) or base < 1:
            base = fallback(k)
        out[k] = base + r_off[k] + p_off[k]
    if out["systolic_bp"] < out["diastolic_bp"] + 20:
        out["systolic_bp"] = out["diastolic_bp"] + 40
    return out

def parse_conditions_from_history(hist, proc):
    if hist is None: hist = ""
    if proc is None: proc = ""
    txt = (str(hist) + " " + str(proc)).lower()
    conds = []
    if "hypertension" in txt or "high blood pressure" in txt: conds.append("Hypertension")
    if "atrial fibrillation" in txt or "af" in txt or "heart arrhythmia" in txt: conds.append("Atrial_Fibrillation")
    if "diabetes" in txt: conds.append("Diabetes_Type2")
    if "seizure disorder" in txt or "epilepsy" in txt: conds.append("Seizure_Disorder")
    if "severe anxiety" in txt or "panic" in txt or "depression" in txt: conds.append("Depression_Anxiety")
    if "copd" in txt: conds.append("COPD")
    if "hemophilia" in txt or "coagulopathy" in txt: conds.append("Hemophilia")
    if "anemia" in txt: conds.append("Anemia")
    if "chronic kidney disease" in txt or "ckd" in txt: conds.append("Chronic_Kidney_Disease")
    return list(set(conds))

def apply_condition_baseline(df, condition_list):
    if 'heart_rate_baseline' not in df.columns:
        for c in ['heart_rate','systolic_bp','diastolic_bp','respiratory_rate','oxygen_saturation']:
            df[f"{c}_baseline"] = df[c].copy()
    n = len(df)
    for cond in condition_list:
        vs_mod = PREVIOUS_CONDITIONS.get(cond, {}).get("vital_sign_modifiers", {})
        for key, val in vs_mod.items():
            if key.endswith("_range"):
                low, high = val
                offset = np.random.uniform(low, high)
                if "BP_offset_range" in key:
                    df['systolic_bp_baseline'] += offset; df['diastolic_bp_baseline'] += offset * 0.6
                elif "HR_offset_range" in key:
                    df['heart_rate_baseline'] += offset
                elif "RR_offset_range" in key:
                    df['respiratory_rate_baseline'] += offset
                elif "O2_sat_offset_range" in key:
                    df['oxygen_saturation_baseline'] += offset
                    df['oxygen_saturation_baseline'] = df['oxygen_saturation_baseline'].clip(0,100)
                elif "factor_range" in key:
                    factor = np.random.uniform(low, high)
                    df['heart_rate_baseline'] *= factor
            elif key == "HR_irregular_variation":
                mag = np.random.randint(*val)
                arr_events = np.random.choice([0,1], size=n, p=[0.98,0.02])
                for i in range(n):
                    if arr_events[i] == 1:
                        change = np.random.randint(-mag, mag + 1)
                        df.at[i, 'heart_rate_baseline'] += change
    for c in ['heart_rate','systolic_bp','diastolic_bp','respiratory_rate','oxygen_saturation']:
        df[c] = df[f"{c}_baseline"]
    return df

# Vital Signs Generator Class
class VitalSignsGenerator:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
    def add_natural_variation(self, base, tpoints, vs):
        var_sin = np.sin(np.linspace(0, 2*np.pi, len(tpoints))) * 2
        noise = np.random.normal(0, 0.5, size=len(tpoints))
        if vs in ['heart_rate']:
            return base + var_sin + noise
        elif vs in ['systolic_bp', 'diastolic_bp']:
            var = np.sin(np.linspace(0, np.pi, len(tpoints))) * 3
            return base + var + noise
        else:
            return np.array([base]*len(tpoints)) + noise
            
    def generate_patient_series(self, patient_baseline, duration_minutes=240, interval_seconds=5, start_time=None):
        if start_time is None:
            start_time = datetime.now()
        n_points = int((duration_minutes * 60) // interval_seconds)
        tpoints = np.arange(n_points)
        timestamps = [start_time + timedelta(seconds=int(i * interval_seconds)) for i in tpoints]
        data = {'timestamp': timestamps}
        for vs, base_val in patient_baseline.items():
            if pd.isna(base_val) or base_val <= 0:
                if vs == "systolic_bp": base_val = 120
                elif vs == "diastolic_bp": base_val = 80
                elif vs == "heart_rate": base_val = 75
                elif vs == "respiratory_rate": base_val = 16
                elif vs == "oxygen_saturation": base_val = 98
                else: base_val = 100
            arr = self.add_natural_variation(base_val, tpoints, vs)
            data[vs] = arr
        return pd.DataFrame(data)

# State Simulation Functions
def build_markov_states(duration_rows, transition_matrix, initial_idx=0):
    states_array = np.zeros(duration_rows, dtype=int)
    states_array[0] = initial_idx
    rows_per_min = 12
    for i in range(1, duration_rows):
        prev = states_array[i - 1]
        if prev == 16:
            states_array[i:] = 16
            break
        if i % rows_per_min == 0 and prev not in [9, 10, 16]:
            nxt = np.random.choice(range(len(STATE_LIST)), p=transition_matrix[prev])
            states_array[i] = nxt
        else:
            states_array[i] = prev
    return states_array

def inject_bathroom_breaks(state_array):
    n = len(state_array)
    rows_per_min = 12
    n_breaks = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
    for _ in range(n_breaks):
        st = np.random.randint(0, n - 1)
        dur = np.random.randint(rows_per_min * 5, rows_per_min * 10)
        end = min(st + dur, n - 1)
        if all(state_array[x] == 0 for x in range(st, end + 1)):
            for x in range(st, end + 1):
                state_array[x] = 9
    return state_array

def inject_whitecoat(state_array, enable_whitecoat):
    if not enable_whitecoat:
        return state_array
    n = len(state_array)
    rows_per_min = 12
    n_wc = np.random.choice([1, 2], p=[0.7, 0.3])
    for _ in range(n_wc):
        st = np.random.randint(0, n - 1)
        dur = np.random.randint(rows_per_min * 2, rows_per_min * 5)
        end = min(st + dur, n - 1)
        if all(state_array[x] == 0 for x in range(st, end + 1)):
            for x in range(st, end + 1):
                state_array[x] = 10
    return state_array

# Transition matrix functions
description_to_column = {
    'Body mass index 30+ - obesity (finding)': 'Obese',
    'Body mass index 40+ - severely obese (finding)': 'Obese',
    'Essential hypertension (disorder)': 'Hypertension',
    'Atrial fibrillation (disorder)': 'Atrial_Fibrillation',
    'Diabetes mellitus type 2 (disorder)': 'Diabetes_Type2',
    'Seizure disorder (disorder)': 'Seizure_Disorder',
    'Major depression single episode (disorder)': 'Depression_Anxiety',
    'Major depressive disorder (disorder)': 'Depression_Anxiety',
    'Pulmonary emphysema (disorder)': 'COPD',
    'Respiratory distress (finding)': 'COPD',
    'Anemia (disorder)': 'Anemia',
    'Chronic kidney disease stage 1 (disorder)': 'Chronic_Kidney_Disease',
    'Chronic kidney disease stage 2 (disorder)': 'Chronic_Kidney_Disease',
    'Chronic kidney disease stage 3 (disorder)': 'Chronic_Kidney_Disease',
    'Chronic kidney disease stage 4 (disorder)': 'Chronic_Kidney_Disease',
    'Chronic kidney disease stage 5 (disorder)': 'Chronic_Kidney_Disease',
    'Laceration - injury (disorder)': 'MildInjury_Pain',
    'Injury of knee (disorder)': 'MildInjury_Pain',
    'Injury of tendon of the rotator cuff of shoulder (disorder)': 'MildInjury_Pain',
    'Concussion injury of brain (disorder)': 'MildInjury_Pain',
    'Injury of neck (disorder)': 'MildInjury_Pain',
    'Childhood asthma (disorder)': 'Asthma',
    'Asthma (disorder)': 'Asthma',
    'Appendicitis (disorder)': 'Appendicitis',
    'Drug overdose': 'Drug_Overdose'
}

MASTER_SPEC = {
    "Age_Pediatric": {"risk_modifiers": {"sepsis_warning": 1.0, "pre_mi_warning": 0.5}},
    "Age_Elderly": {"risk_modifiers": {"pre_mi_warning": 1.5, "sepsis_warning": 1.5}},
    "Sex_Male": {"risk_modifiers": {"pre_mi_warning": 1.2}},
    "Sex_Female": {"risk_modifiers": {"pre_mi_warning": 1.0}},
    "Obese": {"risk_modifiers": {"pre_mi_warning": 2.0, "stemi_crisis": 1.5, "sepsis_warning": 1.0}},
    "Underweight": {"risk_modifiers": {"infection_warning": 1.3}},
    "Hypertension": {"risk_modifiers": {"pre_mi_warning": 2.0, "stemi_crisis_if_in_pre_mi_warning": 1.5}},
    "Atrial_Fibrillation": {"risk_modifiers": {"pre_mi_warning": 1.2, "stroke_crisis": 2.0}},
    "Diabetes_Type2": {"risk_modifiers": {"hypoglycemia_warning": 1.5}},
    "Seizure_Disorder": {"risk_modifiers": {"seizure_crisis": 1.5}},
    "Depression_Anxiety": {"risk_modifiers": {"panic_warning": 2.0}},
    "COPD": {"risk_modifiers": {"breathing_difficulty_warning": 1.5, "compromised_airway_crisis_if_in_breathing_difficulty": 1.3}},
    "Hemophilia": {"risk_modifiers": {"hypovolaemia_warning": 2.0, "hemorrhagic_crisis_if_in_hypovolaemia": 2.0}},
    "Anemia": {"risk_modifiers": {"hemorrhagic_crisis_if_in_hypovolaemia": 1.5, "breathing_difficulty_warning": 1.2}},
    "Chronic_Kidney_Disease": {"risk_modifiers": {"sepsis_warning": 1.2, "hemorrhagic_crisis_if_in_hypovolaemia": 1.2}},
    "MildInjury_Pain": {},
    "ChestPain": {"risk_modifiers": {"cardiac_ischaemia_warning": 0.8}},
    "Asthma": {"risk_modifiers": {"breathing_difficulty_warning": 1.0}},
    "Appendicitis": {"risk_modifiers": {"sepsis_warning": 1.3}},
    "Drug_Overdose": {"risk_modifiers": {"breathing_difficulty_warning": 1.5}},
}

state_mapping = {
    "sepsis_warning": "Sepsis",
    "pre_mi_warning": "Cardiac Ischaemia",
    "stemi_crisis": "STEMI (crisis)",
    "hypoglycemia_warning": "Hypoglycemia",
    "seizure_crisis": "TIA",
    "panic_warning": "Acute Anxiety/Panic",
    "breathing_difficulty_warning": "Breathing Difficulty",
    "compromised_airway_crisis_if_in_breathing_difficulty": "Compromised Airway (crisis)",
    "hypovolaemia_warning": "Hypovolaemia",
    "hemorrhagic_crisis_if_in_hypovolaemia": "Haemorrhagic Shock (crisis)",
    "stroke_crisis": "Stroke (crisis)"
}

# Define load_patient_master_data function
def load_patient_master_data(patients_csv='../data/train/patients.csv',
                             conditions_csv='../data/train/conditions.csv'):
    pt = pd.read_csv(patients_csv)
    co = pd.read_csv(conditions_csv)
    co['CONDITION_SIMPLE'] = co['DESCRIPTION'].map(description_to_column)
    co = co.dropna(subset=['CONDITION_SIMPLE'])
    conditions = pd.get_dummies(co['CONDITION_SIMPLE'])
    co_expanded = co[['PATIENT']].join(conditions)
    co_grouped = co_expanded.groupby('PATIENT').sum().clip(upper=1).reset_index()
    today = pd.Timestamp('2025-01-01')
    pt['Age'] = (today - pd.to_datetime(pt['BIRTHDATE'])).dt.days // 365
    pt['Age_Pediatric'] = (pt['Age'] < 18).astype(int)
    pt['Age_Elderly'] = (pt['Age'] > 65).astype(int)
    pt['Sex_Male'] = (pt['GENDER'] == 'M').astype(int)
    pt['Sex_Female'] = (pt['GENDER'] == 'F').astype(int)
    pt_renamed = pt.rename(columns={'Id': 'PATIENT'})
    patient_conditions = pt_renamed[['PATIENT', 'Age_Pediatric', 'Age_Elderly', 'Sex_Male', 'Sex_Female']]
    patient_data = patient_conditions.merge(co_grouped, on='PATIENT', how='left').fillna(0)
    final_columns = [
        'PATIENT', 'Age_Pediatric', 'Age_Elderly', 'Sex_Male', 'Sex_Female',
        'Obese', 'Hypertension', 'Atrial_Fibrillation', 'Diabetes_Type2',
        'Seizure_Disorder', 'Depression_Anxiety', 'COPD', 'Anemia',
        'Chronic_Kidney_Disease', 'MildInjury_Pain', 'Asthma', 'Appendicitis',
        'Drug_Overdose'
    ]
    patient_data = patient_data.reindex(columns=final_columns, fill_value=0)
    patient_data.rename(columns={'PATIENT': 'patient_id'}, inplace=True)
    return patient_data

def create_patient_specific_matrices(basis_transition_matrix, patient_data, master_spec, state_mapping):
    states = basis_transition_matrix['states']
    base_matrix = np.array(basis_transition_matrix['transition_matrix'])
    patient_matrices = {}
    for index, patient in patient_data.iterrows():
        patient_matrix = np.copy(base_matrix)
        for characteristic, details in master_spec.items():
            if patient.get(characteristic, 0) == 1 and "risk_modifiers" in details:
                for modified_state, modifier in details['risk_modifiers'].items():
                    if modified_state in state_mapping and state_mapping[modified_state] in states:
                        from_state_index = states.index(state_mapping[modified_state])
                        for to_state_index in range(len(states)):
                            patient_matrix[to_state_index, from_state_index] *= modifier
        patient_matrices[patient['patient_id']] = patient_matrix
    return patient_matrices

def normalize_matrix(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return matrix / row_sums

# Smoothing functions for vital sign transitions
def s_curve(x, steepness=5.0):
    return 1.0 / (1.0 + np.exp(-steepness*(x - 0.5)))

def noise_for_vital(vital):
    if vital in ["heart_rate", "systolic_bp", "diastolic_bp"]:
        return 1.2
    elif vital == "respiratory_rate":
        return 0.7
    elif vital == "oxygen_saturation":
        return 0.4
    else:
        return 0.8

def state_noise_multiplier(state_label):
    state_name = STATE_LIST[state_label]
    if state_name == "White Coat (harmless)":
        return 0.6
    elif state_name in ["Cardiac Ischaemia", "Sepsis", "Acute Anxiety/Panic",
                        "Breathing Difficulty", "Hypovolaemia", "Arrhythmic Flare",
                        "Hypoglycemia", "TIA"]:
        return 0.9
    elif state_name in ["STEMI (crisis)", "Septic Shock (crisis)",
                        "Compromised Airway (crisis)", "Haemorrhagic Shock (crisis)",
                        "Stroke (crisis)"]:
        return 2.0
    else:
        return 0.8

def get_noise_std(vital, state_label, phase=None):
    base = noise_for_vital(vital)
    if STATE_LIST[state_label] == "Death":
        return base
    else:
        multiplier = state_noise_multiplier(state_label)
        return base * multiplier

def smooth_vitals_transitions(df, min_window=60, max_window=240, debug=False):
    df = df.copy()
    original_df = df.copy()
    vital_signs = ["heart_rate", "systolic_bp", "diastolic_bp", "respiratory_rate", "oxygen_saturation"]
    transition_idxs = df.index[df["state_label"].diff().fillna(0) != 0].tolist()
    
    for t_idx in transition_idxs:
        if t_idx == 0:
            continue
        new_state = df.loc[t_idx, "state_label"]
        old_state = df.loc[t_idx - 1, "state_label"]
        if new_state == old_state or new_state == 9 or old_state == 9:
            continue
        
        subsequent = df.loc[t_idx:]
        next_change = subsequent.index[subsequent["state_label"].diff().fillna(0) != 0]
        if len(next_change) > 0:
            end_idx = next_change[0] - 1
        else:
            end_idx = df.index[-1]
        
        total_segment_length = end_idx - t_idx + 1
        if total_segment_length < 1:
            continue
        
        window_length = int(min(max(total_segment_length, min_window), max_window))
        
        for col in vital_signs:
            old_val = df.loc[t_idx - 1, col]
            old_std = get_noise_std(col, old_state)
            for offset in range(total_segment_length):
                idx = t_idx + offset
                if idx > end_idx or idx not in df.index:
                    break
                new_val = original_df.loc[idx, col]
                new_std = get_noise_std(col, new_state)
                alpha_lin = offset / float(window_length)
                if alpha_lin > 1.0:
                    alpha_lin = 1.0
                alpha_s = s_curve(alpha_lin, steepness=5.0)
                base_value = (1 - alpha_s)*old_val + alpha_s*new_val
                current_std = (1 - alpha_s)*old_std + alpha_s*new_std
                noisy_val = base_value + np.random.normal(0, current_std)
                df.at[idx, col] = noisy_val
                old_val = noisy_val
    return df

def apply_continuous_noise(df):
    df = df.copy()
    vital_signs = ["heart_rate", "systolic_bp", "diastolic_bp", "respiratory_rate", "oxygen_saturation"]
    for idx, row in df.iterrows():
        st = row["state_label"]
        if st in [9, 16]:
            continue
        for col in vital_signs:
            base_val = df.at[idx, col]
            extra_std = get_noise_std(col, st) * 0.5
            df.at[idx, col] = base_val + np.random.normal(0, extra_std)
    return df

def fade_out_death(df, rows_to_fade=240):
    df = df.copy()
    vital_signs = ["heart_rate", "systolic_bp", "diastolic_bp", "respiratory_rate", "oxygen_saturation"]
    death_idxs = df.index[df["state_label"] == 16].tolist()
    if not death_idxs:
        return df
    dstart = death_idxs[0]
    if dstart > df.index[0]:
        old_idx = dstart - 1
    else:
        old_idx = dstart
    for col in vital_signs:
        old_val = df.loc[old_idx, col]
        old_std = get_noise_std(col, 16)
        for offset in range(rows_to_fade):
            idx = dstart + offset
            if idx not in df.index:
                break
            alpha_lin = offset / float(rows_to_fade)
            alpha_s = s_curve(alpha_lin, steepness=5.0)
            base_value = (1 - alpha_s)*old_val
            current_std = (1 - alpha_s)*old_std
            val = base_value + np.random.normal(0, current_std)
            df.at[idx, col] = max(val, 0.0)
        clamp_start = dstart + rows_to_fade
        for idx2 in range(clamp_start, df.index[-1] + 1):
            if idx2 in df.index:
                if df.loc[idx2, "state_label"] == 16:
                    df.at[idx2, col] = 0.0
                else:
                    break
    return df

def simplify_state_labels(df):
    mapping = {
        0: 0,
        9: 0,
        10: 0,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
        11: 2,
        12: 2,
        13: 2,
        14: 2,
        15: 2,
        16: 3
    }
    df["state_label"] = df["state_label"].map(mapping)
    return df

def interactive_patient_plot(df):
    unique_patients = np.sort(df['patient_id'].unique())[:10]
    initial_patient = unique_patients[0]
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)
    patient_df = df[df['patient_id'] == initial_patient].copy()
    patient_df['timestamp'] = pd.to_datetime(patient_df['timestamp'])
    lines = {}
    for vital in ["heart_rate", "systolic_bp", "diastolic_bp", "respiratory_rate", "oxygen_saturation"]:
        line, = ax.plot(patient_df['timestamp'], patient_df[vital], label=vital)
        lines[vital] = line
    ax.set_title(f'Patient {initial_patient}')
    ax.legend(loc='upper right')
    ax.grid(True)
    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
    slider = Slider(ax_slider, 'Patient Index', 0, len(unique_patients)-1, valinit=0, valfmt='%0.0f')
    def update(val):
        idx = int(slider.val)
        pid = unique_patients[idx]
        patient_df = df[df['patient_id'] == pid].copy()
        patient_df['timestamp'] = pd.to_datetime(patient_df['timestamp'])
        for vital in lines:
            lines[vital].set_xdata(patient_df['timestamp'])
            lines[vital].set_ydata(patient_df[vital])
        ax.relim()
        ax.autoscale_view()
        ax.set_title(f'Patient {pid}')
        fig.canvas.draw_idle()
    slider.on_changed(update)
    plt.show()

##########################################################
# MAIN FUNCTION: VITAL SIGNS & ECG SIMULATION
##########################################################
def main():
    print("Starting vital signs simulation and ECG generation...")
    
    # 1) Generate vital sign data for each patient
    df_enc = load_encounters("test_final_ed_patients.csv")
    base_time = datetime(2025, 1, 1, 19, 0, 0)
    output_csv = "vitals.csv"
    if os.path.exists(output_csv):
        os.remove(output_csv)
    generator = VitalSignsGenerator(seed=42)
    patient_master = load_patient_master_data()
    patient_matrices_raw = create_patient_specific_matrices(basis_transition_matrix_comb, patient_master, MASTER_SPEC, state_mapping)
    patient_matrices = {pid: normalize_matrix(mat) for pid, mat in patient_matrices_raw.items()}
    
    # Process each patient
    for i, row in df_enc.iterrows():
        pid = row.get("PATIENT", f"Unknown_{i}")
        reason = row.get("REASONDESCRIPTION", "")
        pain = row.get("Pain severity - 0-10 verbal numeric rating [Score]", "0")
        hist = row.get("PREVIOUS_MEDICAL_HISTORY", "")
        proc = row.get("PREVIOUS_MEDICAL_PROCEDURES", "")
    
        def sfloat(x):
            try:
                return float(x)
            except:
                return np.nan
    
        pbaseline = {
            "diastolic_bp": sfloat(row.get("Diastolic Blood Pressure", np.nan)),
            "systolic_bp":  sfloat(row.get("Systolic Blood Pressure", np.nan)),
            "heart_rate":   sfloat(row.get("Heart rate", np.nan)),
            "respiratory_rate": sfloat(row.get("Respiratory rate", np.nan)),
            "oxygen_saturation": sfloat(row.get("Oxygen saturation in Arterial blood", np.nan))
        }
    
        modded = apply_modifiers(pbaseline, reason, pain, None, None, None)
        start_t = base_time + timedelta(hours=i)
        df = generator.generate_patient_series(patient_baseline=modded, duration_minutes=240, interval_seconds=5, start_time=start_t)
    
        pconds = parse_conditions_from_history(hist, proc)
        df = apply_condition_baseline(df, pconds)
    
        trans_matrix = patient_matrices.get(pid, normalize_matrix(basis_transition_matrix))
        # Generate state sequence (unsimplified labels are needed for ECG)
        states_seq_str = build_markov_states(len(df), trans_matrix, initial_idx=0)
        df["state_label"] = states_seq_str
        print(f"Patient {pid} state sequence: {states_seq_str}")
    
        # Apply warning/crisis modifications
        npts = len(df)
        for idx in range(npts):
            st = states_seq_str[idx]
            if 1 <= st <= 8:
                wname = STATE_LIST[st]
                wdict = WARNING_STATES.get(wname, {})
                if "HR_factor_range" in wdict:
                    fmin, fmax = wdict["HR_factor_range"]
                    factor = np.random.uniform(fmin, fmax)
                    df.at[idx, "heart_rate"] *= factor
                if "BP_offset_range" in wdict:
                    omin, omax = wdict["BP_offset_range"]
                    off = np.random.uniform(omin, omax)
                    df.at[idx, "systolic_bp"] += off
                    df.at[idx, "diastolic_bp"] += off * 0.6
                if "RR_factor_range" in wdict:
                    rmin, rmax = wdict["RR_factor_range"]
                    rr_fact = np.random.uniform(rmin, rmax)
                    df.at[idx, "respiratory_rate"] *= rr_fact
                if "RR_offset_range" in wdict:
                    rmin, rmax = wdict["RR_offset_range"]
                    rr_off = np.random.uniform(rmin, rmax)
                    df.at[idx, "respiratory_rate"] += rr_off
                if "O2_sat_offset_range" in wdict:
                    omin2, omax2 = wdict["O2_sat_offset_range"]
                    o2off = np.random.uniform(omin2, omax2)
                    df.at[idx, "oxygen_saturation"] += o2off
            elif 11 <= st <= 15:
                cname = STATE_LIST[st]
                cdict = CRISIS_STATES.get(cname, {})
                if "HR_factor_range" in cdict:
                    fmin, fmax = cdict["HR_factor_range"]
                    factor = np.random.uniform(fmin, fmax)
                    df.at[idx, "heart_rate"] *= factor
                if "BP_offset_range" in cdict:
                    omin, omax = cdict["BP_offset_range"]
                    off = np.random.uniform(omin, omax)
                    df.at[idx, "systolic_bp"] += off
                    df.at[idx, "diastolic_bp"] += off * 0.6
                if "RR_factor_range" in cdict:
                    rrmin, rrmax = cdict["RR_factor_range"]
                    fac2 = np.random.uniform(rrmin, rrmax)
                    df.at[idx, "respiratory_rate"] *= fac2
                if "O2_sat_offset_range" in cdict:
                    omin2, omax2 = cdict["O2_sat_offset_range"]
                    o2off = np.random.uniform(omin2, omax2)
                    df.at[idx, "oxygen_saturation"] += o2off
            elif st == 9:
                df.at[idx, "heart_rate"] = 0
                df.at[idx, "systolic_bp"] = 0
                df.at[idx, "diastolic_bp"] = 0
                df.at[idx, "respiratory_rate"] = 0
                df.at[idx, "oxygen_saturation"] = 0
            elif st == 10:
                df.at[idx, "heart_rate"] += 15
                df.at[idx, "systolic_bp"] += 10
                df.at[idx, "diastolic_bp"] += 6
    
        if 16 in states_seq_str:
            died_idx = [i for i, state in enumerate(states_seq_str) if state == 16]
            if died_idx:
                dstart = died_idx[0]
                df.loc[dstart:, ["heart_rate", "systolic_bp", "diastolic_bp", "respiratory_rate", "oxygen_saturation"]] = 0
    
        for col in ["heart_rate", "systolic_bp", "diastolic_bp", "respiratory_rate", "oxygen_saturation"]:
            df[col] = df[col].clip(lower=0, upper=999).round(1)
    
        df["patient_id"] = pid
        df = df[["timestamp", "patient_id", "diastolic_bp", "systolic_bp", "heart_rate",
                 "respiratory_rate", "oxygen_saturation", "state_label"]]
    
        # Append unsmoothed, unsimplified data (needed for ECG generation)
        if not os.path.exists(output_csv):
            df.to_csv(output_csv, index=False)
        else:
            df.to_csv(output_csv, mode='a', header=False, index=False)

        # Get ECG base parameters from your existing function
        base_p = base_ecg_params()

        # Now, immediately generate ECG data for this patient and append it:
        ecg_csv = "ecg_data.csv"
        generate_ecg_for_patient(df, pid, base_p, ecg_csv, prev_params=None)
        
    print("Vital signs simulation complete.")

    # (Optional) Smooth and simplify the vitals data for plotting.
    df_vitals = pd.read_csv(output_csv)
    df_vitals['timestamp'] = pd.to_datetime(df_vitals['timestamp'])
    df_smoothed = smooth_vitals_transitions(df_vitals)
    df_smoothed = apply_continuous_noise(df_smoothed)
    df_smoothed = fade_out_death(df_smoothed, rows_to_fade=240)
    df_smoothed = simplify_state_labels(df_smoothed)
    df_smoothed.to_csv(output_csv, index=False)
    
    print("Done! Vitals and ECG data generated.")
    interactive_patient_plot(df_smoothed)
    
if __name__ == "__main__":
    main()
