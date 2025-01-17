import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os


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
# 0. STATE LIST & UPDATED TRANSITION MATRIX
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

# Updated so baseline -> warning is smaller,
# and warnings/crises last longer (high diagonal).
TRANSITION_MATRIX = [
    # 0. Neutral -> ...
    [0.90, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,   0.01, 0,  0,  0,    0,    0,    0   ],
    # 1. Cardiac Ischaemia -> ...
    [0.02, 0.95, 0,    0,    0,    0,    0,    0,    0,    0,   0,   0.03, 0,    0,    0,    0,    0   ],
    # 2. Sepsis -> ...
    [0.02, 0,    0.95, 0,    0,    0,    0,    0,    0,    0,   0,   0,    0.03, 0,    0,    0,    0   ],
    # 3. Anxiety/Panic -> ...
    [0.05, 0,    0,    0.95, 0,    0,    0,    0,    0,    0,   0,   0,    0,    0,    0,    0,    0   ],
    # 4. Breathing Difficulty -> ...
    [0.02, 0,    0,    0,    0.95, 0,    0,    0,    0,    0,   0,   0,    0.03, 0,    0,    0,    0   ],
    # 5. Hypovolaemia -> ...
    [0.02, 0,    0,    0,    0,    0.95, 0,    0,    0,    0,   0,   0,    0,    0,    0.03, 0,    0   ],
    # 6. Arrhythmic Flare -> ...
    [0.02, 0,    0,    0,    0,    0,    0.95, 0,    0,    0,   0,   0.03, 0,    0,    0,    0,    0   ],
    # 7. Hypoglycemia -> ...
    [0.02, 0,    0,    0,    0,    0,    0,    0.95, 0,    0,   0,   0,    0,    0.03, 0,    0,    0   ],
    # 8. TIA -> ...
    [0.02, 0,    0,    0,    0,    0,    0,    0,    0.93, 0,   0,   0,    0,    0,    0,    0.05, 0   ],
    # 9. Bathroom -> ...
    [1.00, 0,    0,    0,    0,    0,    0,    0,    0,    0,   0,   0,    0,    0,    0,    0,    0   ],
    # 10. White Coat -> ...
    [1.00, 0,    0,    0,    0,    0,    0,    0,    0,    0,   0,   0,    0,    0,    0,    0,    0   ],
    # 11. STEMI -> ...
    [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,   0.98, 0,    0,    0,    0,    0.02],
    # 12. Septic Shock -> ...
    [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,   0,    0.98, 0,    0,    0,    0.02],
    # 13. Compromised Airway -> ...
    [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,   0,    0,    0.98, 0,    0,    0.02],
    # 14. Haemorrhagic Shock -> ...
    [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,   0,    0,    0,    0.98, 0,    0.02],
    # 15. Stroke -> ...
    [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,   0,    0,    0,    0,    0.98, 0.02],
    # 16. Death -> ...
    [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,   0,    0,    0,    0,    0,    1.00]
]

#############################################
# 1. WARNING & CRISIS STATES (AMPLIFIED)
#############################################
WARNING_STATES = {
    "Cardiac_Ischaemia": {
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
    "Acute_Anxiety": {
        "HR_factor_range": (1.3, 1.6),
        "RR_factor_range": (1.2, 1.5),
        "O2_sat_offset_range": (-2, 0),
        "duration_range": (120,240),
        "prob_escalate_crisis": 0.0,
        "maps_to_crisis": None
    },
    "Breathing_difficulty": {
        "HR_factor_range": (1.3, 1.6),
        "RR_factor_range": (1.3, 1.7),
        "O2_sat_offset_range": (-5, -2),
        "duration_range": (360,720),
        "prob_escalate_crisis": 0.03,
        "maps_to_crisis": "Compromised_Airway"
    },
    "Hypovolemia": {
        "HR_factor_range": (1.3,1.6),
        "BP_offset_range": (-20,-10),
        "RR_offset_range": (3,6),
        "O2_sat_offset_range": (-1,0),
        "duration_range": (240,720),
        "prob_escalate_crisis": 0.03,
        "maps_to_crisis": "Hemorrhagic_Shock"
    },
    "Arrhythmic_Flare": {
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
    "STEMI": {
        "HR_factor_range": (1.6,2.3),
        "BP_offset_range": (-40,-20),
        "O2_sat_offset_range": (-8,-3),
        "no_spontaneous_recovery": True
    },
    "Septic_Shock": {
        "HR_factor_range": (1.6,2.0),
        "BP_offset_range": (-80,-40),
        "RR_factor_range": (1.5,2.0),
        "O2_sat_offset_range": (-10,-6),
        "no_spontaneous_recovery": True
    },
    "Compromised_Airway": {
        "HR_factor_range": (1.5,2.0),
        "RR_factor_range": (2.0,3.0),
        "O2_sat_offset_range": (-40,-20),
        "no_spontaneous_recovery": True
    },
    "Hemorrhagic_Shock": {
        "HR_factor_range": (1.5,2.5),
        "BP_offset_range": (-80,-50),
        "RR_factor_range": (1.5,2.0),
        "O2_sat_offset_range": (-8,-4),
        "no_spontaneous_recovery": True
    },
    "Stroke": {
        "HR_factor_range": (1.2,1.5),
        "BP_offset_range": (40,60),
        "RR_factor_range": (1.2,1.4),
        "no_spontaneous_recovery": True
    }
}


DEATH_STATE = "Death"


#########################################
# 2. LOADING ENCOUNTERS
#########################################
def load_encounters(csv_path="train_final_ed_patients.csv"):
    return pd.read_csv(csv_path)


#########################################
# 3. LEGACY REASON & PAIN OFFSET
#########################################
def reason_modifiers(reason):
    if not isinstance(reason, str):
        reason = str(reason) if reason else ""
    reason = reason.lower()

    offsets = {
        'heart_rate': 0.0,
        'systolic_bp': 0.0,
        'diastolic_bp': 0.0,
        'respiratory_rate': 0.0,
        'oxygen_saturation': 0.0
    }
    
    if any(x in reason for x in ["laceration", "sprain", "fracture of bone", "injury of neck", "injury of knee", "impacted molars", "minor burn"]):
        offsets['heart_rate'] += 10
        offsets['systolic_bp'] += 5
        offsets['diastolic_bp'] += 3
        offsets['respiratory_rate'] += 2
    elif any(x in reason for x in ["myocardial infarction", "chest pain"]):
        offsets['heart_rate'] += 10
        offsets['systolic_bp'] -= 5
        offsets['diastolic_bp'] -= 3
        offsets['oxygen_saturation'] -= 2
    elif any(x in reason for x in ["asthma", "difficulty breathing"]):
        offsets['respiratory_rate'] += 4
        offsets['oxygen_saturation'] -= 3
    elif "seizure" in reason:
        offsets['heart_rate'] += 6
    elif "drug overdose" in reason:
        offsets['respiratory_rate'] -= 3
        offsets['oxygen_saturation'] -= 5
        offsets['heart_rate'] -= 5
    elif any(x in reason for x in ["sepsis", "appendicitis", "infection", "pyrexia of unknown origin"]):
        offsets['heart_rate'] += 5
        offsets['systolic_bp'] -= 5
        offsets['diastolic_bp'] -= 3
        offsets['oxygen_saturation'] -= 2
    elif any(x in reason for x in ["burn injury", "major burn"]):
        offsets['heart_rate'] += 15
        offsets['systolic_bp'] += 5
        offsets['diastolic_bp'] += 3
        offsets['respiratory_rate'] += 3
    
    return offsets

def pain_modifiers(pain):
    try:
        p=float(pain)
    except:
        p=0
    offsets={
        'heart_rate': p*1.0,
        'systolic_bp': p*0.5,
        'diastolic_bp': p*0.3,
        'respiratory_rate': p*0.2,
        'oxygen_saturation': 0.0
    }
    return offsets

def apply_modifiers(baseline_dict, reason_desc, pain_score, height, weight, temperature):
    """
    If something is missing, choose a realistic fallback:
      - systolic_bp => 120
      - diastolic_bp => 80
      - heart_rate => 75
      - respiratory_rate => 16
      - oxygen_saturation => 98
    Also ensure systolic>diastolic by ~40 mmHg if needed.
    """
    # fallback
    def fallback(key):
        if key=="systolic_bp": return 120
        if key=="diastolic_bp": return 80
        if key=="heart_rate": return 75
        if key=="respiratory_rate": return 16
        if key=="oxygen_saturation": return 98
        return 100

    r_off= reason_modifiers(reason_desc)
    p_off= pain_modifiers(pain_score)
    out={}
    for k in baseline_dict.keys():
        base= baseline_dict[k]
        if pd.isna(base) or base<1:
            base= fallback(k)
        out[k]= base + r_off[k]+ p_off[k]

    # ensure difference between sys & dia
    if out["systolic_bp"] < out["diastolic_bp"]+20:
        # push systolic up
        out["systolic_bp"]= out["diastolic_bp"]+ 40

    return out


#########################################
# 4. PARSE CONDITIONS FROM HISTORY
#########################################
def parse_conditions_from_history(hist, proc):
    if hist is None: hist = ""
    if proc is None: proc = ""
    txt= (str(hist)+" "+str(proc)).lower()
    conds=[]
    if "hypertension" in txt or "high blood pressure" in txt:
        conds.append("Hypertension")
    if "atrial fibrillation" in txt or "af" in txt or "heart arrhythmia" in txt:
        conds.append("Atrial_Fibrillation")
    if "diabetes" in txt:
        conds.append("Diabetes_Type2")
    if "seizure disorder" in txt or "epilepsy" in txt:
        conds.append("Seizure_Disorder")
    if "severe anxiety" in txt or "panic" in txt or "depression" in txt:
        conds.append("Depression_Anxiety")
    if "copd" in txt:
        conds.append("COPD")
    if "hemophilia" in txt or "coagulopathy" in txt:
        conds.append("Hemophilia")
    if "anemia" in txt:
        conds.append("Anemia")
    if "chronic kidney disease" in txt or "ckd" in txt:
        conds.append("Chronic_Kidney_Disease")
    return list(set(conds))


#########################################
# 5. APPLY PREVIOUS CONDITIONS
#########################################
def apply_condition_baseline(df, condition_list):
    if 'heart_rate_baseline' not in df.columns:
        for c in ['heart_rate','systolic_bp','diastolic_bp','respiratory_rate','oxygen_saturation']:
            df[f"{c}_baseline"]= df[c].copy()
    n= len(df)
    for cond in condition_list:
        vs_mod= PREVIOUS_CONDITIONS.get(cond,{}).get("vital_sign_modifiers",{})
        for key,val in vs_mod.items():
            if key.endswith("_range"):
                low,high= val
                offset= np.random.uniform(low,high)
                if "BP_offset_range" in key:
                    df['systolic_bp_baseline']+= offset
                    df['diastolic_bp_baseline']+= offset*0.6
                elif "HR_offset_range" in key:
                    df['heart_rate_baseline']+= offset
                elif "RR_offset_range" in key:
                    df['respiratory_rate_baseline']+= offset
                elif "O2_sat_offset_range" in key:
                    df['oxygen_saturation_baseline']+= offset
                    df['oxygen_saturation_baseline']= df['oxygen_saturation_baseline'].clip(0,100)
                elif "factor_range" in key:
                    factor= np.random.uniform(low,high)
                    df['heart_rate_baseline']*= factor
            elif key=="HR_irregular_variation":
                mag= np.random.randint(*val)
                arr_events= np.random.choice([0,1], size=n, p=[0.98,0.02])
                for i in range(n):
                    if arr_events[i]==1:
                        change= np.random.randint(-mag, mag+1)
                        df.at[i,'heart_rate_baseline']+= change

    # Overwrite
    for c in ['heart_rate','systolic_bp','diastolic_bp','respiratory_rate','oxygen_saturation']:
        df[c]= df[f"{c}_baseline"]
    return df


#########################################
# 6. VITAL SIGNS GENERATOR
#########################################
class VitalSignsGenerator:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
    def add_natural_variation(self, base, tpoints, vs):
        if vs=='heart_rate':
            var= np.sin(np.linspace(0,2*np.pi,len(tpoints)))*2
            noise= np.random.normal(0,1,len(tpoints))
            return base+ var+ noise
        elif vs in ['systolic_bp','diastolic_bp']:
            var= np.sin(np.linspace(0,np.pi,len(tpoints)))*3
            noise= np.random.normal(0,0.5,len(tpoints))
            return base+ var+ noise
        elif vs=='respiratory_rate':
            noise= np.random.normal(0,0.3,len(tpoints))  # smaller => no 100 RR
            return base+ noise
        elif vs=='oxygen_saturation':
            # keep a small stdev so it stays near baseline
            noise= np.random.normal(0,0.05,len(tpoints))
            return base+ noise
        else:
            return np.array([base]*len(tpoints))
    
    def generate_patient_series(self,
                                patient_baseline,
                                duration_minutes=240,
                                interval_seconds=5,
                                start_time=None):
        if start_time is None:
            start_time= datetime.now()
        n_points= int((duration_minutes*60)//interval_seconds)
        tpoints= np.arange(n_points)
        timestamps=[
            start_time+ timedelta(seconds=int(i*interval_seconds))
            for i in tpoints
        ]
        data= {'timestamp': timestamps}
        for vs,base_val in patient_baseline.items():
            if pd.isna(base_val) or base_val<=0:
                if vs=="systolic_bp": base_val=120
                elif vs=="diastolic_bp": base_val=80
                elif vs=="heart_rate": base_val=75
                elif vs=="respiratory_rate": base_val=16
                elif vs=="oxygen_saturation": base_val=98
                else: base_val=100
            arr= self.add_natural_variation(base_val, tpoints, vs)
            data[vs]= arr
        return pd.DataFrame(data)


#########################################
# 7. EPHEMERAL STATES
#########################################

def get_next_state(current_state_idx):
    row= TRANSITION_MATRIX[current_state_idx]
    nxt= np.random.choice(len(STATE_LIST), p=row)
    return nxt

def build_markov_states(duration_rows, initial_idx=0):
    """
    We'll do once/min transitions if the state is not ephemeral(9/10) or death(16).
    """
    states= np.zeros(duration_rows,dtype=int)
    states[0]= initial_idx
    rows_per_min=12
    for i in range(1,duration_rows):
        prev= states[i-1]
        if prev==16:
            states[i:]=16
            break
        # if ephemeral => remain ephemeral for that minute
        if i%rows_per_min==0 and prev not in [9,10,16]:
            # normal Markov
            nxt= get_next_state(prev)
            states[i]= nxt
        else:
            states[i]= prev
    return states

def inject_bathroom_breaks(state_array):
    """
    If entire sub-interval is Neutral => set to 9 (Bathroom).
    Then revert to Neutral after that sub-interval.
    """
    n= len(state_array)
    rows_per_min=12
    n_breaks= np.random.choice([0,1,2], p=[0.3,0.4,0.3])
    for _ in range(n_breaks):
        st= np.random.randint(0,n-1)
        dur= np.random.randint(rows_per_min*5, rows_per_min*10)
        end= min(st+dur,n-1)
        if all(state_array[x]==0 for x in range(st,end+1)):
            for x in range(st,end+1):
                state_array[x]= 9
    return state_array

def inject_whitecoat(state_array, enable_whitecoat):
    if not enable_whitecoat: return state_array
    n= len(state_array)
    rows_per_min=12
    n_wc= np.random.choice([1,2], p=[0.7,0.3]) 
    for _ in range(n_wc):
        st= np.random.randint(0,n-1)
        dur= np.random.randint(rows_per_min*2, rows_per_min*5)
        end= min(st+dur,n-1)
        if all(state_array[x]==0 for x in range(st,end+1)):
            for x in range(st,end+1):
                state_array[x]= 10
    return state_array

#########################################
# 8. MAIN
#########################################
def main():
    df_enc= load_encounters("train_final_ed_patients.csv")
    base_time= datetime(2025,1,1,19,0,0)
    all_rows=[]
    generator= VitalSignsGenerator(seed=42)

    for i,row in df_enc.iterrows():
        pid= row.get("PATIENT", f"Unknown_{i}")
        reason= row.get("REASONDESCRIPTION","")
        pain= row.get("Pain severity - 0-10 verbal numeric rating [Score] - Reported","0")
        hist= row.get("PREVIOUS_MEDICAL_HISTORY","")
        proc= row.get("PREVIOUS_MEDICAL_PROCEDURES","")

        def sfloat(x):
            try: return float(x)
            except: return np.nan
        pbaseline= {
            "diastolic_bp": sfloat(row.get("Diastolic Blood Pressure", np.nan)),
            "systolic_bp":  sfloat(row.get("Systolic Blood Pressure", np.nan)),
            "heart_rate":   sfloat(row.get("Heart rate", np.nan)),
            "respiratory_rate": sfloat(row.get("Respiratory rate", np.nan)),
            "oxygen_saturation": sfloat(row.get("Oxygen saturation in Arterial blood", np.nan))
        }
        modded= apply_modifiers(pbaseline, reason, pain, None, None, None)

        start_t= base_time+ timedelta(hours=i)
        df= generator.generate_patient_series(
            patient_baseline= modded,
            duration_minutes=240,
            interval_seconds=5,
            start_time= start_t
        )

        # apply conditions
        pconds= parse_conditions_from_history(hist,proc)
        df= apply_condition_baseline(df, pconds)

        # build markov
        npts= len(df)
        states= build_markov_states(npts, initial_idx=0)
        # ephemeral
        states= inject_bathroom_breaks(states)
        if np.random.rand()<0.1:
            states= inject_whitecoat(states, True)

        df["state_label"]= states  # store as int

        # apply big changes if it's a warning or crisis
        for idx in range(npts):
            st= states[idx]
            if 1<= st <=8:
                wname= STATE_LIST[st] 
                wdict= WARNING_STATES.get(wname, {})
                # apply offsets
                if "HR_factor_range" in wdict:
                    fmin,fmax= wdict["HR_factor_range"]
                    factor= np.random.uniform(fmin,fmax)
                    df.at[idx,"heart_rate"]*= factor
                if "BP_offset_range" in wdict:
                    omin,omax= wdict["BP_offset_range"]
                    off= np.random.uniform(omin,omax)
                    df.at[idx,"systolic_bp"]+= off
                    df.at[idx,"diastolic_bp"]+= off*0.6
                if "RR_factor_range" in wdict:
                    # if we have factor
                    rmin,rmax= wdict["RR_factor_range"]
                    rr_fact= np.random.uniform(rmin,rmax)
                    df.at[idx,"respiratory_rate"]*= rr_fact
                if "RR_offset_range" in wdict:
                    rmin,rmax= wdict["RR_offset_range"]
                    rr_off= np.random.uniform(rmin,rmax)
                    df.at[idx,"respiratory_rate"]+= rr_off
                if "O2_sat_offset_range" in wdict:
                    omin2,omax2= wdict["O2_sat_offset_range"]
                    o2off= np.random.uniform(omin2,omax2)
                    df.at[idx,"oxygen_saturation"]+= o2off

            elif 11<= st <=15:
                cname= STATE_LIST[st]
                cdict= CRISIS_STATES.get(cname,{})
                if "HR_factor_range" in cdict:
                    fmin,fmax= cdict["HR_factor_range"]
                    factor= np.random.uniform(fmin,fmax)
                    df.at[idx,"heart_rate"]*= factor
                if "BP_offset_range" in cdict:
                    omin,omax= cdict["BP_offset_range"]
                    off= np.random.uniform(omin,omax)
                    df.at[idx,"systolic_bp"]+= off
                    df.at[idx,"diastolic_bp"]+= off*0.6
                if "RR_factor_range" in cdict:
                    rrmin,rrmax= cdict["RR_factor_range"]
                    fac2= np.random.uniform(rrmin,rrmax)
                    df.at[idx,"respiratory_rate"]*= fac2
                if "O2_sat_offset_range" in cdict:
                    omin2,omax2= cdict["O2_sat_offset_range"]
                    o2off= np.random.uniform(omin2,omax2)
                    df.at[idx,"oxygen_saturation"]+= o2off
            elif st==9:
                # bathroom => all zero
                df.at[idx,"heart_rate"]=0
                df.at[idx,"systolic_bp"]=0
                df.at[idx,"diastolic_bp"]=0
                df.at[idx,"respiratory_rate"]=0
                df.at[idx,"oxygen_saturation"]=0
            elif st==10:
                # white coat => moderate
                df.at[idx,"heart_rate"]+= 15
                df.at[idx,"systolic_bp"]+= 10
                df.at[idx,"diastolic_bp"]+= 6

        # death => zero out vitals
        died_idx= np.where(states==16)[0]
        if len(died_idx)>0:
            dstart= died_idx[0]
            df.loc[dstart:, ["heart_rate","systolic_bp","diastolic_bp","respiratory_rate","oxygen_saturation"]]= 0

        # final clamp
        for ccol in ["heart_rate","systolic_bp","diastolic_bp","respiratory_rate","oxygen_saturation"]:
            df[ccol]= df[ccol].clip(lower=0, upper=999).round(1)

        df["patient_id"]= pid
        df= df[[
            "timestamp","patient_id",
            "diastolic_bp","systolic_bp","heart_rate",
            "respiratory_rate","oxygen_saturation","state_label"
        ]]
        all_rows.append(df)

    final= pd.concat(all_rows, ignore_index=True)
    final["timestamp"]= pd.to_datetime(final["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    final.to_csv("vitals.csv", index=False)
    print("Done! Wrote 'vitals.csv' with fixed O2, normal baseline splits, numeric states, extended warnings/crises, etc.")

if __name__=="__main__":
    main()


