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
            "BP_offset_range": (5, 10)  # +5–10 mmHg
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
            "BP_offset_range": (10, 20)  # +10–20 mmHg
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


# We won't detail every single attribute in each warning or crisis,
# as we might apply them at runtime to the patient's vitals.
WARNING_STATES = {
    "Cardiac_Ischaemia": {
        "HR_factor_range": (1.2, 1.5),
        "BP_offset_range": (5, 10),
        "RR_offset_range": (2, 5),
        "O2_sat_offset_range": (-2, -1),
        "duration_range": (5,30),  # minutes
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
        "prob_escalate_crisis": 0.0,  # won't escalate
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


#########################################
# 2. LOADING THE ENCOUNTERS CSV
#########################################
def load_encounters(csv_path="test_final_ed_patients.csv"):
    return pd.read_csv(csv_path)


#########################################
# 3. LEGACY REASON & PAIN OFFSET
#########################################
def reason_modifiers(reason):
    # Make sure reason is a string
    if not isinstance(reason, str):
        reason = str(reason) if reason is not None else ""
    reason = reason.lower()

    offsets = {
        'heart_rate': 0.0,
        'systolic_bp': 0.0,
        'diastolic_bp': 0.0,
        'respiratory_rate': 0.0,
        'oxygen_saturation': 0.0
    }
    
    # Example expansions based on your spec sheet:
    if any(x in reason for x in ["laceration", "sprain", "fracture of bone", "injury of neck", "injury of knee", "impacted molars", "minor burn"]):
        # Mild/Moderate injuries => pain offsets
        # (heart_rate + 10–20 BPM, BP + 5–10 mmHg, etc.)
        offsets['heart_rate'] += 10
        offsets['systolic_bp'] += 5
        offsets['diastolic_bp'] += 3
        offsets['respiratory_rate'] += 2
    
    elif any(x in reason for x in ["myocardial infarction", "chest pain"]):
        # MI => bigger effect
        offsets['heart_rate'] += 10
        offsets['systolic_bp'] -= 5
        offsets['diastolic_bp'] -= 3
        offsets['oxygen_saturation'] -= 2
    
    elif any(x in reason for x in ["asthma", "difficulty breathing"]):
        # Asthma => +RR, -O2
        offsets['respiratory_rate'] += 4
        offsets['oxygen_saturation'] -= 3
    
    elif "seizure" in reason:
        offsets['heart_rate'] += 6
    
    elif "drug overdose" in reason:
        offsets['respiratory_rate'] -= 3
        offsets['oxygen_saturation'] -= 5
        offsets['heart_rate'] -= 5
    
    elif any(x in reason for x in ["sepsis", "appendicitis", "infection", "pyrexia of unknown origin"]):
        # Infectious => tachy, possible hypotension
        offsets['heart_rate'] += 5
        offsets['systolic_bp'] -= 5
        offsets['diastolic_bp'] -= 3
        offsets['oxygen_saturation'] -= 2
    
    elif any(x in reason for x in ["burn injury", "major burn"]):
        # more severe burn => bigger pain effect
        offsets['heart_rate'] += 15
        offsets['systolic_bp'] += 5
        offsets['diastolic_bp'] += 3
        offsets['respiratory_rate'] += 3
    
    # etc... add more if needed

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
    # trivial approach
    r_off= reason_modifiers(reason_desc)
    p_off= pain_modifiers(pain_score)
    out={}
    for k in baseline_dict.keys():
        base= baseline_dict[k]
        if pd.isna(base):
            base=0
        out[k]= base + r_off[k]+ p_off[k] # ignoring height/weight for brevity
    return out


#########################################
# 4. PARSE CONDITIONS FROM HISTORY
#########################################
def parse_conditions_from_history(hist, proc):
    if hist is None: hist = ""
    if proc is None: proc = ""
    txt = (str(hist) + " " + str(proc)).lower()

    conds = []

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
    
    # etc... add any others from your final list

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
        if cond not in PREVIOUS_CONDITIONS: continue
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
                elif "factor_range" in key:
                    factor= np.random.uniform(low,high)
                    df['heart_rate_baseline']*=factor
            elif key=="HR_irregular_variation":
                magnitude= np.random.randint(*val)
                arr_events= np.random.choice([0,1], size=n, p=[0.98,0.02])
                for i in range(n):
                    if arr_events[i]==1:
                        change= np.random.randint(-magnitude, magnitude+1)
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
        self.normal_ranges={
            'heart_rate': (60,100),
            'systolic_bp': (90,120),
            'diastolic_bp': (60,80),
            'respiratory_rate': (12,20),
            'oxygen_saturation': (95,100)
        }
    def add_natural_variation(self, base, tpoints, vs):
        if vs=='heart_rate':
            var= np.sin(np.linspace(0,2*np.pi,len(tpoints)))*2
            noise= np.random.normal(0,1,len(tpoints))
            return base+var+noise
        elif vs in ['systolic_bp','diastolic_bp']:
            var= np.sin(np.linspace(0,np.pi,len(tpoints)))*3
            noise= np.random.normal(0,0.5,len(tpoints))
            return base+var+noise
        elif vs=='respiratory_rate':
            noise= np.random.normal(0,0.5,len(tpoints))
            return base+ noise
        elif vs=='oxygen_saturation':
            noise= np.random.normal(0,0.2,len(tpoints))
            return np.minimum(100, base+ noise)
        else:
            return np.array([base]*len(tpoints))
    
    def generate_patient_series(self, patient_baseline,
                                duration_minutes=240,
                                interval_seconds=5,
                                start_time=None):
        if start_time is None:
            start_time= datetime.now()
        n_points= int((duration_minutes*60)/interval_seconds)
        tpoints= np.arange(n_points)
        timestamps=[
            start_time+ timedelta(seconds=int(i*interval_seconds))
            for i in tpoints
        ]
        data={'timestamp': timestamps}
        for vs,base_val in patient_baseline.items():
            if pd.isna(base_val) or base_val==0:
                low,high=self.normal_ranges.get(vs,(60,100))
                base_val=(low+high)/2.0
            arr= self.add_natural_variation(base_val,tpoints,vs)
            data[vs]= arr
        return pd.DataFrame(data)


#########################################
# 7. EPHEMERAL STATES
#########################################
def inject_bathroom_breaks(ephemeral_arr, main_state_arr):
    """
    Insert 0-2 bathroom breaks of 5-10min each, 
    skipping if main_state is Warning/Crisis
    """
    n=len(ephemeral_arr)
    n_breaks= np.random.choice([0,1,2], p=[0.3,0.4,0.3])
    rows_per_min=12 # 5s intervals =>12/min
    for _ in range(n_breaks):
        start= np.random.randint(0,n-1)
        dur= np.random.randint(rows_per_min*5, rows_per_min*10)
        end= min(start+dur,n-1)
        # skip if main_state is "Warning_*" or "Crisis_*" or "Death"
        # We'll only apply if the entire interval is baseline
        # to keep it simpler, check a subset
        if any([ms.startswith("Warning_") or ms.startswith("Crisis_") or ms=="Death"
                for ms in main_state_arr[start:end+1]]):
            continue
        for i in range(start,end+1):
            ephemeral_arr[i].add("Bathroom")
    return ephemeral_arr

def inject_whitecoat(ephemeral_arr):
    """
    White Coat checks each hour for 2-5 min, plus some extras in first hour.
    White coat can stack on top of anything (including warning, crisis).
    """
    n=len(ephemeral_arr)
    rows_per_min=12
    hours= [0,1,2,3]
    for h in hours:
        start= h*60*rows_per_min + np.random.randint(0,5*rows_per_min)
        if start>=n: break
        dur= np.random.randint(2*rows_per_min,5*rows_per_min)
        end= min(start+dur, n-1)
        for i in range(start,end+1):
            ephemeral_arr[i].add("WhiteCoat")
    # extras in first hour
    extras= np.random.randint(1,3) # 1-2
    for _ in range(extras):
        st= np.random.randint(0,60*rows_per_min-1)
        dur= np.random.randint(2*rows_per_min,5*rows_per_min)
        end= min(st+dur, n-1)
        for i in range(st,end+1):
            ephemeral_arr[i].add("WhiteCoat")
    return ephemeral_arr


#########################################
# 8. DYNAMIC STATE MACHINE
#########################################
def pick_warning_states(characteristics, prev_conditions, current_condition):
    """
    We'll compute a weighted probability for each possible WARNING state
    based on:
      - risk_modifiers from characteristics + conditions + current_condition
      - a base chance if the current condition suggests that warning
    """
    # 1) Base suggestions from CURRENT_CONDITION
    base_prob = {}
    if current_condition == "Chest_Pain":
        base_prob["Cardiac_Ischaemia"] = 1.0
    if current_condition == "Asthma":
        base_prob["Breathing_difficulty"] = 1.0
    if current_condition == "Appendicitis":
        base_prob["Sepsis"] = 1.0
    if current_condition == "Drug_Overdose":
        base_prob["Breathing_difficulty"] = 0.8  # slightly less sure

    # 2) Incorporate PREVIOUS_CONDITIONS risk_modifiers
    # e.g. if "Hypertension" => "Cardiac_Ischaemia_warning": x2
    final_scores = {}
    for wstate, base_p in base_prob.items():
        score = base_p
        # apply from previous conditions
        for c in prev_conditions:
            if c not in PREVIOUS_CONDITIONS:
                continue
            rm = PREVIOUS_CONDITIONS[c].get("risk_modifiers", {})
            # e.g. "Cardiac_Ischaemia_warning": 2.0
            if f"{wstate}_warning" in rm:
                factor = rm[f"{wstate}_warning"]
                score *= factor
        
        # apply from characteristics if needed
        if characteristics:
            for char in characteristics:
                if char in CHARACTERISTICS:
                    rmc = CHARACTERISTICS[char].get("risk_modifiers", {})
                    if f"{wstate}_warning" in rmc:
                        score *= rmc[f"{wstate}_warning"]

        final_scores[wstate] = score

    # 3) pick the best if any
    if not final_scores:
        return None  # no relevant states
    best_w = max(final_scores, key=final_scores.get)
    if final_scores[best_w] < 0.5:
        # if the top state has <0.5 probability => skip
        return None
    return best_w



def apply_warning_offsets(df, wstate):
    """Given a warning state, apply its offsets to the df's vitals 
       for the duration of the warning (like 1 block)."""
    if wstate not in WARNING_STATES:
        return df
    st=WARNING_STATES[wstate]
    # We'll do a random start, random duration
    n=len(df)
    rows_per_min=12
    start= np.random.randint(0,n-30)
    dur_min= np.random.randint(st["duration_range"][0], st["duration_range"][1]+1)
    dur= dur_min*rows_per_min
    end= min(start+dur,n-1)
    
    # Mark state_label
    for i in range(start,end+1):
        if df.at[i,"state_label"]=="Baseline": # only override baseline
            df.at[i,"state_label"]= f"Warning_{wstate}"
    # chance to escalate
    if np.random.rand()< st["prob_escalate_crisis"]:
        cstart=end+1
        if cstart<n:
            crisis_name= st["maps_to_crisis"]
            for j in range(cstart,n):
                # if we are still Baseline or Warning => escalate
                if df.at[j,"state_label"].startswith("Warning_") or df.at[j,"state_label"]=="Baseline":
                    df.at[j,"state_label"]= f"Crisis_{crisis_name}"
    return df


def apply_crisis_offsets(df):
    """
    If 'Crisis_X' in state_label, apply crisis offsets for the rest of the timeline
    Also handle Death => once in Death, vitals=0
    """
    n=len(df)
    # find transitions to crisis
    for i in range(n):
        label= df.at[i,"state_label"]
        if label.startswith("Crisis_"):
            crisis_name= label.split("_",1)[1]
            # apply crisis offset
            apply_crisis_effect(df, i, crisis_name)

    return df

def apply_crisis_effect(df, start_idx, crisis_name):
    """
    From start_idx onward or until Death
    """
    n=len(df)
    cdict= CRISIS_STATES.get(crisis_name,{})
    if not cdict: return
    rows_per_min=12
    # We'll do a simple approach: once in crisis, remain in crisis or die
    # chance of death each minute, e.g. 5% => let's code that
    idx=start_idx
    while idx<n:
        if df.at[idx,"state_label"]=="Death":
            break
        if not df.at[idx,"state_label"].startswith("Crisis_"):
            break
        # roll for death each minute
        if idx%rows_per_min==0: # once per minute
            # pick a probability, e.g. 5% if you define it
            p_death= np.random.uniform(0.02,0.08) # placeholder or from cdict
            if np.random.rand()< p_death:
                # mark the rest as Death
                for k in range(idx,n):
                    df.at[k,"state_label"]="Death"
                break
        idx+=1

def combine_ephemeral_effects(df, ephemeral_arr):
    """
    WhiteCoat can *add* to the existing state's vitals.
    Bathroom => override if not in warning/crisis.
    We'll do a pass that merges ephemeral states with main state
    in a 'final_state_label' if needed.
    Also adjust vitals if ephemeral states have modifiers.
    """
    n= len(df)
    # baseline columns
    for col in ['heart_rate','systolic_bp','diastolic_bp','respiratory_rate','oxygen_saturation']:
        if f"{col}_pre_ephem" not in df.columns:
            df[f"{col}_pre_ephem"]= df[col].copy()

    # define ephemeral offsets
    ephemeral_mods={
        "Bathroom":{
            "override_to_zero":True
        },
        "WhiteCoat":{
            "HR_offset_range":(10,20),
            "BP_offset_range":(10,20)
        }
    }

    for i in range(n):
        e_states= ephemeral_arr[i]
        if "Bathroom" in e_states:
            # if main state is Warning or Crisis => ignore bathroom
            main_label= df.at[i,"state_label"]
            if main_label.startswith("Warning_") or main_label.startswith("Crisis_"):
                e_states.remove("Bathroom")
        # now apply ephemeral mods
        # e.g. WhiteCoat can stack
        # we store final label as state_label + ephemeral tags
    for i in range(n):
        main_label= df.at[i,"state_label"]
        e_states= ephemeral_arr[i]
        # if "Death", do nothing
        if main_label=="Death":
            continue
        # apply ephemeral changes
        for e in e_states:
            # get ephemeral_mods[e]
            em= ephemeral_mods.get(e,{})
            if "override_to_zero" in em:
                # bathroom => set vitals=0
                for c in ["heart_rate","systolic_bp","diastolic_bp","respiratory_rate","oxygen_saturation"]:
                    df.at[i,c]=0
                df.at[i,"state_label"]= f"{main_label}+Bathroom"
            else:
                # e.g. WhiteCoat => add offset
                for key,val in em.items():
                    if key.endswith("_range"):
                        low,high= val
                        offset= np.random.uniform(low,high)
                        if "BP_offset_range" in key:
                            df.at[i,"systolic_bp"]+= offset
                            df.at[i,"diastolic_bp"]+= offset*0.6
                        elif "HR_offset_range" in key:
                            df.at[i,"heart_rate"]+= offset
        # rename state_label if WhiteCoat present
        if len(e_states)>0 and "Bathroom" not in e_states:
            for e in e_states:
                if e!="Bathroom":
                    df.at[i,"state_label"]= f"{main_label}+{e}"

    return df


#########################################
# 9. MAIN
#########################################
def main():
    enc_df= load_encounters("test_final_ed_patients.csv")
    from datetime import datetime
    base_time= datetime(2025,1,1,19,0,0)
    all_rows=[]
    gen= VitalSignsGenerator(seed=42)

    for i,row in enc_df.iterrows():
        pid= row.get("PATIENT",f"Unknown_{i}")
        reason= row.get("REASONDESCRIPTION","")
        pain= row.get("Pain severity - 0-10 verbal numeric rating [Score] - Reported","0")
        hist= row.get("PREVIOUS_MEDICAL_HISTORY","")
        proc= row.get("PREVIOUS_MEDICAL_PROCEDURES","")
        height= row.get("Body Height","")
        weight= row.get("Body Weight","")
        temp=   row.get("Body temperature","")

        # baseline vitals
        def sfloat(x):
            try: return float(x)
            except: return np.nan
        pbaseline={
            'diastolic_bp': sfloat(row.get("Diastolic Blood Pressure",np.nan)),
            'systolic_bp':  sfloat(row.get("Systolic Blood Pressure",np.nan)),
            'heart_rate':   sfloat(row.get("Heart rate",np.nan)),
            'respiratory_rate': sfloat(row.get("Respiratory rate",np.nan)),
            'oxygen_saturation': sfloat(row.get("Oxygen saturation in Arterial blood",np.nan))
        }
        # apply reason/pain
        modded= apply_modifiers(pbaseline, reason,pain,height,weight,temp)

        start_t= base_time+ timedelta(hours=i)
        df= gen.generate_patient_series(
            patient_baseline=modded,
            duration_minutes=240,
            interval_seconds=5,
            start_time=start_t
        )

        # parse prev cond
        pconds= parse_conditions_from_history(hist,proc)
        df= apply_condition_baseline(df,pconds)

        # pick a "current condition" from reason or random
        ccond=None
        if not isinstance(reason, str):
            reason = str(reason) if reason is not None else ""
        reason_l = reason.lower()

        if "chest pain" in reason_l:
            ccond="Chest_Pain"
        elif "asthma" in reason_l:
            ccond="Asthma"
        elif "appendicitis" in reason_l:
            ccond="Appendicitis"
        elif "drug overdose" in reason_l:
            ccond="Drug_Overdose"
        # else random ?

        # We'll store final state_label init as "Baseline"
        df["state_label"]=["Baseline"]*len(df)

        # 1) We figure out which warning state to pick
        wstate= pick_warning_states(characteristics=None, # not implemented
                                   prev_conditions=pconds,
                                   current_condition=ccond)
        if wstate:
            df= apply_warning_offsets(df,wstate)
        
        # 2) apply crisis offsets (some patients might have escalated)
        df= apply_crisis_offsets(df)

        # 3) ephemeral states array -> a set for each row
        ephemeral_arr=[ set() for _ in range(len(df))]
        # inject bathroom if baseline
        ephemeral_arr= inject_bathroom_breaks(ephemeral_arr, list(df["state_label"]))
        # random chance for WhiteCoat
        # if np.random.rand()<0.5 => we do white coat
        ephemeral_arr= inject_whitecoat(ephemeral_arr)

        # 4) combine ephemeral with main
        df= combine_ephemeral_effects(df, ephemeral_arr)

        # 5) if Death => vitals=0 rest
        died_idx= np.where(df["state_label"]=="Death")[0]
        if len(died_idx)>0:
            dstart= died_idx[0]
            for c2 in ["heart_rate","systolic_bp","diastolic_bp","respiratory_rate","oxygen_saturation"]:
                df.loc[dstart:, c2]=0

        # round
        for c3 in ["heart_rate","systolic_bp","diastolic_bp","respiratory_rate","oxygen_saturation"]:
            df[c3]= df[c3].round(1)

        df["patient_id"]= pid
        df= df[["timestamp","patient_id","diastolic_bp","systolic_bp","heart_rate","respiratory_rate","oxygen_saturation","state_label"]]
        all_rows.append(df)

    final_df= pd.concat(all_rows, ignore_index=True)
    final_df["timestamp"]= pd.to_datetime(final_df["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    final_df.to_csv("vitals.csv", index=False)
    print("All done. Output => vitals.csv with advanced states + ephemeral layering.")

if __name__=="__main__":
    main()
