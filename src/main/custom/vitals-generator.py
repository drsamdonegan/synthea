#!/usr/bin/env python
"""
Merged Script for Generating Patient Vital Signs and Patient‐Specific State Transitions

This script combines:
  1. A vital signs generator that uses an initial CSV file of encounters to create
     a series of vital signs (sampled at 5-second intervals) and simulates state transitions
     using a Markov chain.
  2. A patient-specific transition matrix generator that reads patient master data and conditions,
     applies risk modifiers, and then simulates unique state sequences for each patient.
  3. A smoothing step that applies state- and vital-specific noise (via separate noise functions)
     at the moment of state transitions. In the smoothing, the noise amplitude is interpolated
     with a shallower S-curve than the value interpolation so that the noise “fades” in/out gradually.
  4. An interactive plotting function that lets you scroll through the vital signs for each patient.
     
The output includes:
  - A CSV file "vitals.csv" with the simulated vital sign data.
  - An interactive plot window to view each patient’s time series.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

#############################################
# 1. DICTIONARIES (Complete from Spec Sheet)
#############################################

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
            "Hypovolemia_warning": 2.0,
            "Hemorrhagic_Shock_crisis_if_Hypovolemia": 2.0
        }
    },
    "Anemia": {
        "vital_sign_modifiers": {
            "HR_offset_range": (5, 10)
        },
        "risk_modifiers": {
            "Hemorrhagic_Shock_crisis_if_Hypovolemia": 1.5,
            "Breathing_difficulty_warning": 1.2
        }
    },
    "Chronic_Kidney_Disease": {
        "risk_modifiers": {
            "Sepsis_warning": 1.2,
            "Hemorrhagic_Shock_crisis_if_Hypovolemia": 1.2
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

#############################################
# 2. STATE LIST & BASE TRANSITION MATRIX
#############################################

STATE_LIST = [
    "Neutral",                     # 0
    "Cardiac Ischaemia",           # 1 (warning)
    "Sepsis",                      # 2 (warning)
    "Acute Anxiety/Panic",         # 3 (warning)
    "Breathing Difficulty",        # 4 (warning)
    "Hypovolaemia",                # 5 (warning)
    "Arrhythmic Flare",            # 6 (warning)
    "Hypoglycemia",                # 7 (warning)
    "TIA",                         # 8 (warning)
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

#########################################
# 3. PATIENT ENCOUNTER PROCESSING FUNCTIONS
#########################################

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

#########################################
# 4. VITAL SIGNS GENERATOR CLASS
#########################################

class VitalSignsGenerator:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
    def add_natural_variation(self, base, tpoints, vs):
        # Possibly add a small random noise to each row to avoid perfect sine wave:
        var_sin = np.sin(np.linspace(0, 2*np.pi, len(tpoints))) * 2
        # Add small random noise
        noise = np.random.normal(0, 0.5, size=len(tpoints))  # extra
        if vs in ['heart_rate']:
            return base + var_sin + noise
        elif vs in ['systolic_bp', 'diastolic_bp']:
            var = np.sin(np.linspace(0, np.pi, len(tpoints))) * 3
            return base + var + noise
        else:
            # for respiratory_rate, oxygen_saturation, we just add noise
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

#########################################
# 5. STATE SIMULATION FUNCTIONS
#########################################

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
            nxt = np.random.choice(len(STATE_LIST), p=transition_matrix[prev])
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

#########################################
# 6. PATIENT-SPECIFIC TRANSITION MATRIX FUNCTIONS
#########################################

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
    "Hemophilia": {"risk_modifiers": {"hypovolemia_warning": 2.0, "hemorrhagic_crisis_if_in_hypovolemia": 2.0}},
    "Anemia": {"risk_modifiers": {"hemorrhagic_crisis_if_in_hypovolemia": 1.5, "breathing_difficulty_warning": 1.2}},
    "Chronic_Kidney_Disease": {"risk_modifiers": {"sepsis_warning": 1.2, "hemorrhagic_crisis_if_in_hypovolemia": 1.2}},
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
    "hypovolemia_warning": "Hypovolaemia",
    "hemorrhagic_crisis_if_in_hypovolemia": "Haemorrhagic Shock (crisis)",
    "stroke_crisis": "Stroke (crisis)"
}

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

#########################################
# 7. SIMULATION & PLOTTING OF STATE SEQUENCES
#########################################

def simulate_patient_states(patient_id, transition_matrix, debug=False):
    total_time = 14400
    time_step = 5
    current_state = 'Neutral'
    states_sequence = []
    current_time = 0
    state_index = STATE_LIST.index(current_state)
    next_state_index = np.random.choice(range(len(STATE_LIST)), p=transition_matrix[state_index])
    current_state = STATE_LIST[next_state_index]
    if debug:
        print(f"Initial state transition from 'Neutral' to '{current_state}'")
    while current_time < total_time:
        duration_range = state_durations.get(current_state, [10, 30])
        possible_durations = list(range(duration_range[0], duration_range[1] + 1, 5))
        state_duration = np.random.choice(possible_durations)
        end_time = min(current_time + state_duration * 60, total_time)
        if debug:
            print(f"At second {current_time}, '{current_state}' for {state_duration} minutes")
        while current_time < end_time:
            states_sequence.append(current_state)
            current_time += time_step
        if current_time < total_time:
            state_index = STATE_LIST.index(current_state)
            next_state_index = np.random.choice(range(len(STATE_LIST)), p=transition_matrix[state_index])
            current_state = STATE_LIST[next_state_index]
            if debug:
                print(f"At second {current_time}, transitioning to '{current_state}'", end=' | ')
    return states_sequence

state_durations = {
    "Neutral": [10, 60],
    "Cardiac Ischaemia": [15, 45],
    "Sepsis": [30, 60],
    "Acute Anxiety/Panic": [10, 30],
    "Breathing Difficulty": [15, 40],
    "Hypovolaemia": [20, 50],
    "Arrhythmic Flare": [10, 30],
    "Hypoglycemia": [10, 25],
    "TIA": [15, 35],
    "Bathroom (harmless)": [5, 10],
    "White Coat (harmless)": [5, 20],
    "STEMI (crisis)": [20, 60],
    "Septic Shock (crisis)": [30, 60],
    "Compromised Airway (crisis)": [20, 50],
    "Haemorrhagic Shock (crisis)": [25, 55],
    "Stroke (crisis)": [20, 60],
    "Death": [5, 60]
}

#########################################
# 8. PATIENT MASTER DATA PROCESSING
#########################################

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

#########################################
# Noise Functions
#########################################
def s_curve(x, steepness=5.0):
    """Logistic S-curve from 0..1 as x goes 0..1."""
    return 1.0 / (1.0 + np.exp(-steepness*(x - 0.5)))

def noise_for_vital(vital):
    """
    Increase these to ensure overall more noise:
    """
    if vital in ["heart_rate", "systolic_bp", "diastolic_bp"]:
        return 1.2  # was 0.8
    elif vital == "respiratory_rate":
        return 0.7  # was 0.4
    elif vital == "oxygen_saturation":
        return 0.4  # was 0.2
    else:
        return 0.8

def state_noise_multiplier(state_label):
    """
    If you want even bigger differences, tweak these multipliers, 
    but now the base is already raised.
    """
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
        return 0.8  # e.g. Neutral

def get_noise_std(vital, state_label, phase=None):
    """
    Overall standard deviation for the given vital & state.
    """
    base = noise_for_vital(vital)
    if STATE_LIST[state_label] == "Death":
        # We'll fade to 0 noise in fade_out_death, but for early Death rows we can keep normal noise
        return base
    else:
        multiplier = state_noise_multiplier(state_label)
        return base * multiplier

#########################################
# 9. SMOOTHING STATE TRANSITIONS
#########################################

def smooth_vitals_transitions(df, min_window=60, max_window=240, debug=False):
    """
    S-curve interpolation of old->new state for the first portion 
    of each new state block. Also blends old noise -> new noise.
    """
    df = df.copy()
    original_df = df.copy()
    vital_signs = ["heart_rate", "systolic_bp", "diastolic_bp",
                   "respiratory_rate", "oxygen_saturation"]
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
                
                # blend old_val->new_val
                base_value = (1 - alpha_s)*old_val + alpha_s*new_val
                # blend old_std->new_std
                current_std = (1 - alpha_s)*old_std + alpha_s*new_std

                noisy_val = base_value + np.random.normal(0, current_std)
                df.at[idx, col] = noisy_val
                old_val = noisy_val  # for continuity
    
    return df

#########################################
# 9B. Ensure continuous noise in non-Death states
#########################################

def apply_continuous_noise(df):
    """
    After smoothing, ensure we never have a truly flat line in any state except 
    Death or Bathroom. We add a small random perturbation to each row based on 
    the state's normal noise level.
    """
    df = df.copy()
    vital_signs = ["heart_rate", "systolic_bp", "diastolic_bp",
                   "respiratory_rate", "oxygen_saturation"]
    for idx, row in df.iterrows():
        st = row["state_label"]
        # skip Death (16) and Bathroom(9)
        if st in [9, 16]:
            continue
        for col in vital_signs:
            base_val = df.at[idx, col]
            # Use maybe half the typical std so we don't overshadow the smoothing:
            extra_std = get_noise_std(col, st) * 0.5
            df.at[idx, col] = base_val + np.random.normal(0, extra_std)
    return df

#########################################
# 9C. Fade Death to zero after ~20min
#########################################

def fade_out_death(df, rows_to_fade=240):
    """
    For any patient that hits Death state, from the first row of Death onward,
    do a 20-min ramp (rows_to_fade) from the current value/noise to zero. 
    After that, clamp to zero.
    """
    df = df.copy()
    vital_signs = ["heart_rate", "systolic_bp", "diastolic_bp",
                   "respiratory_rate", "oxygen_saturation"]
    
    # Find rows where state_label == 16 (Death)
    death_idxs = df.index[df["state_label"] == 16].tolist()
    if not death_idxs:
        return df  # no death state
    
    dstart = death_idxs[0]  # first row of Death
    # Because the patient might remain in Death for many rows,
    # we fade from [dstart..dstart+rows_to_fade) to zero with an S-curve
    # then clamp subsequent rows at zero.
    
    # store the old baseline at dstart-1 if possible
    if dstart > df.index[0]:
        old_idx = dstart - 1
    else:
        old_idx = dstart
    for col in vital_signs:
        old_val = df.loc[old_idx, col]
        # We'll keep some noise at the beginning (like you said),
        # then fade to zero amplitude as well.
        old_std = get_noise_std(col, 16)
        
        # Phase 1: ramp to zero
        for offset in range(rows_to_fade):
            idx = dstart + offset
            if idx not in df.index:
                break
            alpha_lin = offset / float(rows_to_fade)
            alpha_s = s_curve(alpha_lin, steepness=5.0)
            base_value = (1 - alpha_s)*old_val  # new_val=0
            # also fade noise from old_std -> 0
            current_std = (1 - alpha_s)*old_std
            val = base_value + np.random.normal(0, current_std)
            df.at[idx, col] = max(val, 0.0)
        
        # Phase 2: clamp to zero after the fade
        clamp_start = dstart + rows_to_fade
        for idx2 in range(clamp_start, df.index[-1] + 1):
            if idx2 in df.index:
                if df.loc[idx2, "state_label"] == 16:
                    df.at[idx2, col] = 0.0
                else:
                    # if state changed from Death to something else (rare?), break
                    break
    return df

#########################################
# 10. INTERACTIVE PLOTTING FUNCTION
#########################################
def interactive_patient_plot(df):
    unique_patients = np.sort(df['patient_id'].unique())
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
    slider = Slider(ax_slider, 'Patient Index', 0, len(unique_patients)-1,
                    valinit=0, valfmt='%0.0f')
    
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

#########################################
# 11. MAIN FUNCTION: SIMULATION, OUTPUT, AND PLOTTING
#########################################

def main():
    print("Starting vital signs simulation...")
    
    # 1) Generate the raw data
    df_enc = load_encounters("test_final_ed_patients.csv")
    base_time = datetime(2025, 1, 1, 19, 0, 0)
    all_rows = []
    generator = VitalSignsGenerator(seed=42)

    patient_master = load_patient_master_data()
    patient_matrices_raw = create_patient_specific_matrices(
        basis_transition_matrix_comb, patient_master, MASTER_SPEC, state_mapping
    )
    patient_matrices = {pid: normalize_matrix(mat) for pid, mat in patient_matrices_raw.items()}

    for i, row in df_enc.iterrows():
        pid = row.get("PATIENT", f"Unknown_{i}")
        reason = row.get("REASONDESCRIPTION", "")
        pain = row.get("Pain severity - 0-10 verbal numeric rating [Score] - Reported", "0")
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
        df = generator.generate_patient_series(
            patient_baseline=modded,
            duration_minutes=240,
            interval_seconds=5,
            start_time=start_t
        )

        pconds = parse_conditions_from_history(hist, proc)
        df = apply_condition_baseline(df, pconds)
        
        trans_matrix = patient_matrices.get(pid, normalize_matrix(basis_transition_matrix))
        states_seq_str = simulate_patient_states(pid, trans_matrix, debug=True)
        states_seq = [STATE_LIST.index(state) for state in states_seq_str]
        df["state_label"] = states_seq
        print(f"Patient {pid} state sequence: {states_seq}")

        # Apply warning/crisis offsets row-by-row:
        npts = len(df)
        for idx in range(npts):
            st = states_seq[idx]
            # Warning states
            if 1 <= st <= 8:
                wname = STATE_LIST[st]
                wdict = WARNING_STATES[wname]
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
            
            # Crisis states
            elif 11 <= st <= 15:
                cname = STATE_LIST[st]
                cdict = CRISIS_STATES[cname]
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
            
            # Bathroom: forced to zero
            elif st == 9:
                df.at[idx, "heart_rate"] = 0
                df.at[idx, "systolic_bp"] = 0
                df.at[idx, "diastolic_bp"] = 0
                df.at[idx, "respiratory_rate"] = 0
                df.at[idx, "oxygen_saturation"] = 0
            
            # White Coat: quick offset
            elif st == 10:
                df.at[idx, "heart_rate"] += 15
                df.at[idx, "systolic_bp"] += 10
                df.at[idx, "diastolic_bp"] += 6

        # Just clip and round for safety
        for col in ["heart_rate", "systolic_bp", "diastolic_bp", "respiratory_rate", "oxygen_saturation"]:
            df[col] = df[col].clip(lower=0, upper=999).round(1)

        df["patient_id"] = pid
        df = df[[
            "timestamp", "patient_id", "diastolic_bp", "systolic_bp",
            "heart_rate", "respiratory_rate", "oxygen_saturation", "state_label"
        ]]
        all_rows.append(df)

    final = pd.concat(all_rows, ignore_index=True)
    final["timestamp"] = pd.to_datetime(final["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")

    # 2) Smooth transitions
    final_smoothed = smooth_vitals_transitions(final)

    # 3) Ensure we have continuous noise in non-Death states
    final_smoothed = apply_continuous_noise(final_smoothed)

    # 4) Fade Death to zero after ~20 min
    final_smoothed = fade_out_death(final_smoothed, rows_to_fade=240)

    # 5) Write to CSV
    final_smoothed.to_csv("vitals.csv", index=False)
    print("Done! Wrote 'vitals.csv' with patient-specific state transitions.")

    # 6) Interactive plot
    interactive_patient_plot(final_smoothed)
    
if __name__ == "__main__":
    main()
    
if __name__ == "__main__":
    main()
