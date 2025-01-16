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


WARNING_STATES = {
    "Cardiac_Ischaemia": {
        "HR_factor_range": (1.2, 1.5),
        "BP_offset_range": (5, 10),
        "RR_offset_range": (2, 5),
        "O2_sat_offset_range": (-2, -1),
        "duration_range": (5,30),
        "prob_escalate_crisis": 0.3,
        "maps_to_crisis": "STEMI"
    },
    "Sepsis": {
        "HR_factor_range": (1.1, 1.3),
        "BP_offset_range": (-10, -5),
        "RR_factor_range": (1.1, 1.2),
        "O2_sat_offset_range": (-2, 0),
        "duration_range": (60,360),
        "prob_escalate_crisis": 0.4,
        "maps_to_crisis": "Septic_Shock"
    },
    "Acute_Anxiety": {
        "HR_factor_range": (1.2, 1.4),
        "RR_factor_range": (1.2, 1.5),
        "O2_sat_offset_range": (-1, 0),
        "duration_range": (10,60),
        "prob_escalate_crisis": 0.0,
        "maps_to_crisis": None
    },
    "Breathing_difficulty": {
        "HR_factor_range": (1.2, 1.4),
        "RR_factor_range": (1.2, 1.5),
        "O2_sat_offset_range": (-3, -1),
        "duration_range": (60,120),
        "prob_escalate_crisis": 0.2,
        "maps_to_crisis": "Compromised_Airway"
    },
    "Hypovolemia": {
        "HR_factor_range": (1.2,1.4),
        "BP_offset_range": (-10,-5),
        "RR_offset_range": (2,5),
        "O2_sat_offset_range": (0,0),
        "duration_range": (10,60),
        "prob_escalate_crisis": 0.25,
        "maps_to_crisis": "Hemorrhagic_Shock"
    },
    "Arrhythmic_Flare": {
        "HR_irregular_variation": (20,20),
        "BP_offset_range": (-10,10),
        "duration_range": (5,30),
        "prob_escalate_crisis": 0.15,
        "maps_to_crisis": "STEMI"
    },
    "Hypoglycemia": {
        "HR_factor_range": (1.1,1.3),
        "BP_offset_range": (0,0),
        "RR_offset_range": (2,3),
        "O2_sat_offset_range": (0,0),
        "duration_range": (15,60),
        "prob_escalate_crisis": 0.1,
        "maps_to_crisis": "Compromised_Airway"
    },
    "TIA": {
        "HR_factor_range": (1.0,1.2),
        "BP_offset_range": (10,20),
        "duration_range": (5,120),
        "prob_escalate_crisis": 0.2,
        "maps_to_crisis": "Stroke"
    }
}

CRISIS_STATES = {
    "STEMI": {
        "HR_factor_range": (1.3,2.0),
        "BP_offset_range": (-30,-10),
        "O2_sat_offset_range": (-5,-2),
        "no_spontaneous_recovery": True
    },
    "Septic_Shock": {
        "HR_factor_range": (1.3,1.8),
        "BP_offset_range": (-50,-20),
        "RR_factor_range": (1.3,1.6),
        "O2_sat_offset_range": (-10,-5),
        "no_spontaneous_recovery": True
    },
    "Compromised_Airway": {
        "HR_factor_range": (1.2,1.6),
        "RR_factor_range": (1.5,2.0),
        "O2_sat_offset_range": (-30,-20),
        "no_spontaneous_recovery": True
    },
    "Hemorrhagic_Shock": {
        "HR_factor_range": (1.3,2.0),
        "BP_offset_range": (-50,-20),
        "RR_factor_range": (1.2,1.5),
        "O2_sat_offset_range": (-5,-2),
        "no_spontaneous_recovery": True
    },
    "Stroke": {
        "HR_factor_range": (1.1,1.3),
        "BP_offset_range": (20,40),
        "RR_factor_range": (1.1,1.2),
        "no_spontaneous_recovery": True
    }
}

DEATH_STATE = "Death"


#############################################
# 0. STATE LIST & UPDATED TRANSITION MATRIX
#############################################

# We'll keep the same list of 17 states with integer indices 0..16.
# The user wants a LONGER stay in warnings & crises, so let's adjust:
#  - Warnings => ~0.95 remain in that warning, ~0.03 escalate to crisis, ~0.02 revert to neutral.
#  - Crises => ~0.98 remain in crisis, 0.02 => death.
#  - Neutral => lower chance to go to warning.

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

#  We'll define a matrix so that once in a warning, you usually stay in it (0.95),
#  0.03 chance of going to the relevant crisis, and 0.02 chance returning to neutral or some distribution.
#  For crises, 0.98 remain, 0.02 => death.
#  For neutral, we give a small chance to jump to a warning (0.01 each).
TRANSITION_MATRIX = [
    # 0. Neutral -> ...
    [0.90, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0,  0,    0,    0,    0,    0   ],
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
    # 11. STEMI (crisis) -> ...
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
# 1. DICTIONARIES WITH BIGGER WARNING/CRISIS MODIFIERS
#############################################

# We'll just amplify the multipliers. For instance, if "Cardiac_Ischaemia" used to do +5..10,
# let's do +20..40. We only show changes in the relevant dictionaries:

WARNING_STATES = {
    "Cardiac_Ischaemia": {
        "HR_factor_range": (1.3, 1.7),       # bigger
        "BP_offset_range": (20, 40),         # bigger
        "RR_offset_range": (3, 6),           # bigger
        "O2_sat_offset_range": (-4, -2),
        "duration_range": (360,720),         # 30–60 min in 5-sec intervals => 360–720 rows
        "prob_escalate_crisis": 0.03,        # smaller chance
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
        "duration_range": (120,240),   # 10–20 min => let's do bigger for demonstration
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
# 2. LOADING THE ENCOUNTERS CSV
#########################################
def load_encounters(csv_path="test_final_ed_patients.csv"):
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
    r_off= reason_modifiers(reason_desc)
    p_off= pain_modifiers(pain_score)
    out={}
    for k in baseline_dict.keys():
        base= baseline_dict[k]
        if pd.isna(base):
            base= 75 if k=="oxygen_saturation" else 100  # or some fallback
        out[k]= base + r_off[k]+ p_off[k]
    return out


#########################################
# 4. PARSE CONDITIONS FROM HISTORY
#########################################
def parse_conditions_from_history(hist, proc):
    if hist is None: hist = ""
    if proc is None: proc = ""
    txt = (str(hist)+" "+str(proc)).lower()
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
    n=len(df)
    for cond in condition_list:
        if cond not in PREVIOUS_CONDITIONS:
            continue
        cdict=PREVIOUS_CONDITIONS[cond]
        vs_mod= cdict.get("vital_sign_modifiers",{})
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
                    df['heart_rate_baseline']*=factor
            elif key=="HR_irregular_variation":
                magnitude= np.random.randint(*val)
                arr_events= np.random.choice([0,1],size=n,p=[0.98,0.02])
                for i in range(n):
                    if arr_events[i]==1:
                        change= np.random.randint(-magnitude,magnitude+1)
                        df.at[i,'heart_rate_baseline']+= change
    # Overwrite columns
    for c in ['heart_rate','systolic_bp','diastolic_bp','respiratory_rate','oxygen_saturation']:
        df[c]= df[f"{c}_baseline"]  # no final clamp except maybe 0..999
    return df


#########################################
# 6. VITAL SIGNS GENERATOR
#########################################
class VitalSignsGenerator:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
    def add_natural_variation(self, base, tpoints, vs):
        # remove strict clamp except a large range
        if vs=='heart_rate':
            var= np.sin(np.linspace(0,2*np.pi,len(tpoints)))*2
            noise= np.random.normal(0,1,len(tpoints))
            return base+ var+ noise
        elif vs in ['systolic_bp','diastolic_bp']:
            var= np.sin(np.linspace(0,np.pi,len(tpoints)))*3
            noise= np.random.normal(0,0.5,len(tpoints))
            return base+ var+ noise
        elif vs=='respiratory_rate':
            noise= np.random.normal(0,0.5,len(tpoints))
            return base+ noise
        elif vs=='oxygen_saturation':
            noise= np.random.normal(0,0.2,len(tpoints))
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
            if pd.isna(base_val): base_val= 100
            arr= self.add_natural_variation(base_val,tpoints,vs)
            data[vs]= arr
        df= pd.DataFrame(data)
        return df


#########################################
# 7. EPHEMERAL STATES
#########################################

def inject_bathroom_breaks(state_array):
    """
    Only from Neutral(0) => go to Bathroom(9).
    Then row: remain bathroom for 5-10 min, then back to neutral
    We won't combine with warnings/crises.
    """
    n= len(state_array)
    rows_per_min=12
    n_breaks= np.random.choice([0,1,2], p=[0.3,0.4,0.3])
    for _ in range(n_breaks):
        start= np.random.randint(0,n-1)
        dur= np.random.randint(rows_per_min*5, rows_per_min*10)
        end= min(start+dur,n-1)
        # check if the entire sub-interval is neutral
        if all(state_array[i]==0 for i in range(start,end+1)):
            for i in range(start,end+1):
                state_array[i]=9
    return state_array

def inject_whitecoat(state_array, enable_whitecoat):
    """
    10% of patients => white coat from neutral only
    For 2-5 min each hour?
    We'll keep it simpler: maybe 1-2 intervals in the 4 hours
    If we see the sub-interval is neutral, set WhiteCoat(10).
    """
    if not enable_whitecoat: return state_array
    n= len(state_array)
    rows_per_min=12
    n_wc= np.random.choice([1,2], p=[0.7,0.3])  # 1 or 2 visits
    for _ in range(n_wc):
        start= np.random.randint(0,n-1)
        dur= np.random.randint(rows_per_min*2, rows_per_min*5)
        end= min(start+dur,n-1)
        # only if sub-interval is neutral
        if all(state_array[i]==0 for i in range(start,end+1)):
            for i in range(start,end+1):
                state_array[i]=10
    return state_array


#########################################
# 8. MARKOV TRANSITIONS PER ROW
#########################################
def get_next_state(current_state_idx):
    row= TRANSITION_MATRIX[current_state_idx]
    next_idx= np.random.choice(len(STATE_LIST), p=row)
    return next_idx

def build_markov_states(duration_rows, initial_idx=0):
    """
    We'll track the primary state transitions. Once we are in
    Bathroom(9) or WhiteCoat(10), we remain there for that sub-interval,
    then revert to neutral. We'll do once per minute transitions for 
    the main Markov chain if not in ephemeral or crisis states.
    """
    states= np.zeros(duration_rows,dtype=int)
    states[0]= initial_idx
    rows_per_min=12
    for i in range(1,duration_rows):
        prev= states[i-1]
        # If death => remain
        if prev==16:
            states[i:]=16
            break
        # If ephemeral => remain ephemeral until end of that minute sub-interval
        # But simpler: we do the main Markov once per minute if we are in a "normal" state
        if i%rows_per_min==0 and prev not in [9,10] and (11>prev>0 or prev==0 or prev>=11 and prev<16):
            # do normal Markov
            nxt= get_next_state(prev)
            states[i]= nxt
        else:
            states[i]= prev
    return states


#########################################
# 9. MAIN
#########################################
def main():
    enc_df= load_encounters("test_final_ed_patients.csv")
    base_time= datetime(2025,1,1,19,0,0)
    all_rows=[]
    generator= VitalSignsGenerator(seed=42)

    for i,row in enc_df.iterrows():
        pid= row.get("PATIENT",f"Unknown_{i}")
        reason= row.get("REASONDESCRIPTION","")
        pain= row.get("Pain severity - 0-10 verbal numeric rating [Score] - Reported","0")
        hist= row.get("PREVIOUS_MEDICAL_HISTORY","")
        proc= row.get("PREVIOUS_MEDICAL_PROCEDURES","")

        def sfloat(x):
            try:
                return float(x)
            except:
                return np.nan
        pbaseline={
            'diastolic_bp': sfloat(row.get("Diastolic Blood Pressure",np.nan)),
            'systolic_bp':  sfloat(row.get("Systolic Blood Pressure",np.nan)),
            'heart_rate':   sfloat(row.get("Heart rate",np.nan)),
            'respiratory_rate': sfloat(row.get("Respiratory rate",np.nan)),
            'oxygen_saturation': sfloat(row.get("Oxygen saturation in Arterial blood",np.nan))
        }
        # apply reason/pain
        modded= apply_modifiers(pbaseline, reason, pain, None, None, None)

        # generate 4h timeseries
        start_t= base_time+ timedelta(hours=i)
        df= generator.generate_patient_series(
            patient_baseline= modded,
            duration_minutes=240,
            interval_seconds=5,
            start_time= start_t
        )

        # apply prev conditions
        pconds= parse_conditions_from_history(hist, proc)
        df= apply_condition_baseline(df, pconds)

        # Build Markov chain
        npts= len(df)
        markov_states= build_markov_states(npts, initial_idx=0) # 0 => Neutral

        # ephemeral states
        # first bathroom
        markov_states= inject_bathroom_breaks(markov_states)
        # 10% chance for White Coat
        enable_whitecoat= (np.random.rand()<0.1)
        markov_states= inject_whitecoat(markov_states, enable_whitecoat)

        # store numeric state_label_idx
        df["state_label"]= markov_states

        # If crisis => degrade vitals strongly, if warning => degrade vitals significantly
        # We'll do a pass to amplify
        for idx in range(npts):
            st= markov_states[idx]
            if st in range(1,9): # it's a warning
                warn_name= STATE_LIST[st]  # e.g. "Cardiac Ischaemia"
                wdict= WARNING_STATES.get(warn_name,{})
                # scale vitals
                # e.g. HR_factor_range
                if "HR_factor_range" in wdict:
                    fmin,fmax= wdict["HR_factor_range"]
                    factor= np.random.uniform(fmin,fmax)
                    df.at[idx,"heart_rate"]*= factor
                if "BP_offset_range" in wdict:
                    offmin,offmax= wdict["BP_offset_range"]
                    offset= np.random.uniform(offmin, offmax)
                    df.at[idx,"systolic_bp"]+= offset
                    df.at[idx,"diastolic_bp"]+= offset*0.6
                if "RR_offset_range" in wdict:
                    rrmin,rrmax= wdict["RR_offset_range"]
                    rr_off= np.random.uniform(rrmin,rrmax)
                    df.at[idx,"respiratory_rate"]+= rr_off
                if "O2_sat_offset_range" in wdict:
                    o2min,o2max= wdict["O2_sat_offset_range"]
                    o2_off= np.random.uniform(o2min,o2max)
                    df.at[idx,"oxygen_saturation"]+= o2_off
            elif st in range(11,16): # crisis
                cname= STATE_LIST[st]
                cdict= CRISIS_STATES.get(cname,{})
                if "HR_factor_range" in cdict:
                    fmin,fmax= cdict["HR_factor_range"]
                    factor= np.random.uniform(fmin,fmax)
                    df.at[idx,"heart_rate"]*= factor
                if "BP_offset_range" in cdict:
                    offmin,offmax= cdict["BP_offset_range"]
                    offset= np.random.uniform(offmin,offmax)
                    df.at[idx,"systolic_bp"]+= offset
                    df.at[idx,"diastolic_bp"]+= offset*0.6
                if "RR_factor_range" in cdict:
                    rrmin,rrmax= cdict["RR_factor_range"]
                    factor2= np.random.uniform(rrmin,rrmax)
                    df.at[idx,"respiratory_rate"]*= factor2
                if "O2_sat_offset_range" in cdict:
                    o2min,o2max= cdict["O2_sat_offset_range"]
                    off2= np.random.uniform(o2min,o2max)
                    df.at[idx,"oxygen_saturation"]+= off2
            elif st==9:
                # Bathroom => set vitals=0
                df.at[idx,"heart_rate"]=0
                df.at[idx,"systolic_bp"]=0
                df.at[idx,"diastolic_bp"]=0
                df.at[idx,"respiratory_rate"]=0
                df.at[idx,"oxygen_saturation"]=0
            elif st==10:
                # White Coat => bigger HR, BP
                # We'll do a modest approach
                # no time for gradual ramp here, but possible
                df.at[idx,"heart_rate"]+= 15
                df.at[idx,"systolic_bp"]+= 10
                df.at[idx,"diastolic_bp"]+= 6

        # Death => vitals=0 from that row onward
        died_idx= np.where(markov_states==16)[0]
        if len(died_idx)>0:
            dstart= died_idx[0]
            df.loc[dstart:,["heart_rate","systolic_bp","diastolic_bp","respiratory_rate","oxygen_saturation"]]=0

        # round + clamp
        for c3 in ["heart_rate","systolic_bp","diastolic_bp","respiratory_rate","oxygen_saturation"]:
            df[c3]= df[c3].clip(lower=0,upper=999).round(1)

        df["patient_id"]= pid
        # reorder
        df= df[[
            "timestamp","patient_id",
            "diastolic_bp","systolic_bp","heart_rate",
            "respiratory_rate","oxygen_saturation","state_label"
        ]]
        all_rows.append(df)

    final_df= pd.concat(all_rows, ignore_index=True)
    final_df["timestamp"]= pd.to_datetime(final_df["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    final_df.to_csv("vitals.csv", index=False)
    print("All done. Output => vitals.csv with bigger warning/crisis modifiers, numeric state labels, no overlap ephemeral, longer stays, etc.")

if __name__=="__main__":
    main()

