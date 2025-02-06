CHARACTERISTICS = {
    "Age_Pediatric": {
        "vital_sign_modifiers": {
            "HR_factor_range": (1.10, 1.20),   # +10–20%
            "RR_factor_range": (1.10, 1.20),
            "BP_offset_range": (0, 0)          # no direct offset, or 0–0 mmHg
        },
        "risk_modifiers": {
            # e.g. for sepsis, children might have same base or slightly higher
            "sepsis_warning": 1.0,
            "pre_mi_warning": 0.5
        }
    },
    "Age_Elderly": {
        "vital_sign_modifiers": {
            "BP_offset_range": (5, 10),  # +5–10 mmHg systolic
        },
        "risk_modifiers": {
            "pre_mi_warning": 1.5,   # ×1.5 chance
            "sepsis_warning": 1.5,
        }
    },
    "Sex_Male": {
        "vital_sign_modifiers": {
            "BP_offset_range": (0, 5),
        },
        "risk_modifiers": {
            "pre_mi_warning": 1.2,
        }
    },
    "Sex_Female": {
        # minimal direct effect in your sim
        "vital_sign_modifiers": {},
        "risk_modifiers": {
            "pre_mi_warning": 1.0
        }
    },
    "Obese": {
        "vital_sign_modifiers": {
            "BP_offset_range": (5, 10),   # +5–10 mmHg
            "HR_offset_range": (5, 5)     # +5 BPM
        },
        "risk_modifiers": {
            "pre_mi_warning": 2.0,   # 2× chance for angina
            "stemi_crisis": 1.5,
            "sepsis_warning": 1.0
        }
    },
    "Underweight": {
        "vital_sign_modifiers": {
            "BP_offset_range": (-5, -5)
        },
        "risk_modifiers": {
            "infection_warning": 1.3,  # if you have an infection path
        }
    }
    # etc. for other characteristics
}


PREVIOUS_CONDITIONS = {
    "Hypertension": {
        "vital_sign_modifiers": {
            "BP_offset_range": (10, 20)  # +10–20 mmHg
        },
        "risk_modifiers": {
            "pre_mi_warning": 2.0,
            "stemi_crisis_if_in_pre_mi_warning": 1.5
        }
    },
    "Atrial_Fibrillation": {
        "vital_sign_modifiers": {
            "HR_irregular_variation": (10, 20)  # ±10–20 BPM fluctuations
        },
        "risk_modifiers": {
            "pre_mi_warning": 1.2,
            "stroke_crisis": 2.0  # if you had a stroke crisis
        }
    },
    "Diabetes_Type2": {
        "vital_sign_modifiers": {
            "HR_offset_range": (5, 10) # +5–10 BPM
        },
        "risk_modifiers": {
            "hypoglycemia_warning": 1.5  # if you add mild hypoglycemia
        }
    },
    "Seizure_Disorder": {
        "risk_modifiers": {
            "seizure_crisis": 1.5
        }
    },
    "Depression_Anxiety": {
        "risk_modifiers": {
            "panic_warning": 2.0
        }
    },
    "COPD": {
        "vital_sign_modifiers": {
            "O2_sat_offset_range": (-3, -2),  # might keep sat ~2–3% lower
            "RR_offset_range": (2, 3)
        },
        "risk_modifiers": {
            "breathing_difficulty_warning": 1.5,
            "compromised_airway_crisis_if_in_breathing_difficulty": 1.3
        }
    },
    "Hemophilia": {
        "risk_modifiers": {
            "hypovolemia_warning": 2.0,  
            "hemorrhagic_crisis_if_in_hypovolemia": 2.0
        }
    },
    "Anemia": {
        "vital_sign_modifiers": {
            "HR_offset_range": (5, 10)
        },
        "risk_modifiers": {
            "hemorrhagic_crisis_if_in_hypovolemia": 1.5,
            "breathing_difficulty_warning": 1.2
        }
    },
    "Chronic_Kidney_Disease": {
        "risk_modifiers": {
            "sepsis_warning": 1.2,
            "hemorrhagic_crisis_if_in_hypovolemia": 1.2
        }
    }
}


CURRENT_CONDITIONS = {
    "MildInjury_Pain": {
        "vital_sign_modifiers": {
            "HR_offset_range": (10, 20),  # +10–20 BPM
            "BP_offset_range": (5, 10),   # +5–10 mmHg
            "RR_offset_range": (2, 5),
            "O2_sat_offset_range": (-1, 0)
        },
        "risk_modifiers": {}
    },
    "ChestPain": {
        "vital_sign_modifiers": {
            "HR_factor_range": (1.2, 1.5),
            "BP_factor_range": (0.9, 1.1),  # can rise or fall
            "O2_sat_offset_range": (-5, -1),
        },
        "risk_modifiers": {
            "cardiac_ischaemia_warning": 0.8, # baseline chance to be in that warning now
            # or if they're already in a warning state, higher chance of stemi?
        }
    },
    "Asthma": {
        "vital_sign_modifiers": {
            "RR_factor_range": (1.5, 2.0),
            "HR_offset_range": (5, 10),
            "O2_sat_offset_range": (-3, -5)
        },
        "risk_modifiers": {
            "breathing_difficulty_warning": 1.0
        }
    },
    "Appendicitis": {
        "vital_sign_modifiers": {
            "HR_offset_range": (10, 20),
            "RR_offset_range": (2, 5)
        },
        "risk_modifiers": {
            "sepsis_warning": 1.3 # moderate chance
        }
    },
    "Drug_Overdose": {
        "vital_sign_modifiers": {
            # can be wide
        },
        "risk_modifiers": {
            "breathing_difficulty_warning": 1.5
        }
    }
}


WARNING_STATES = {
    "Cardiac_Ischaemia": {
        "vital_sign_effects": {
            "HR_factor_range": (1.2, 1.5),
            "BP_offset_range": (5, 10),    # mild ↑
            "RR_offset_range": (2, 5),
            "O2_sat_offset_range": (-2, -1)
        },
        "duration_range_minutes": (5, 30),
        "prob_return_baseline": 0.4,  # 40% average
        "prob_progress_crisis": {
            "STEMI": 0.2  # base 20% 
        }
    },
    "Sepsis": {
        "vital_sign_effects": {
            "HR_factor_range": (1.1, 1.3),
            "BP_offset_range": (-5, -10),
            "RR_factor_range": (1.1, 1.2),
            "O2_sat_offset_range": (-2, 0)
        },
        "duration_range_minutes": (60, 360),  # 1–6 hours
        "prob_return_baseline": 0.15,
        "prob_progress_crisis": {
            "Septic_Shock": 0.3
        }
    },
    "Acute_Anxiety": {
        "vital_sign_effects": {
            "HR_factor_range": (1.2, 1.4),
            "RR_factor_range": (1.2, 1.5),
            "O2_sat_offset_range": (-1, 0)
        },
        "duration_range_minutes": (10, 60),
        "prob_return_baseline": 0.8,
        "prob_progress_crisis": {}  # none
    },
    "Breathing_Difficulty": {
        "vital_sign_effects": {
            "HR_factor_range": (1.2, 1.4),
            "RR_factor_range": (1.2, 1.5),
            "O2_sat_offset_range": (-3, -1)
        },
        "duration_range_minutes": (60, 120),
        "prob_return_baseline": 0.5,
        "prob_progress_crisis": {
            "Compromised_Airway": 0.2  # 10–20% base
        }
    },
    "Bathroom_Leads_Off": {
        "vital_sign_effects": {
            "override_to_zero": True
        },
        "duration_range_minutes": (5, 10),
        "prob_return_baseline": 1.0,
        "prob_progress_crisis": {}
    },
    "WhiteCoat": {
        "vital_sign_effects": {
            "HR_offset_range": (10, 20),
            "BP_offset_range": (10, 20)
        },
        "duration_range_minutes": (5, 20),
        "prob_return_baseline": 1.0,
        "prob_progress_crisis": {}
    },
    "Hypovolemia": {
        "vital_sign_effects": {
            "HR_factor_range": (1.2, 1.4),
            "BP_offset_range": (-5, -10),
            "RR_offset_range": (2, 5)
        },
        "duration_range_minutes": (10, 60),
        "prob_return_baseline": 0.35,
        "prob_progress_crisis": {
            "Hemorrhagic_Crisis": 0.25  # ~25% base
        }
    },
    "Arrhythmic_Flare": {
        "vital_sign_effects": {
            "HR_irregular_variation": (20, 20),  # e.g. ±20 BPM
            "BP_offset_range": (-10, 10)
        },
        "duration_range_minutes": (5, 30),
        "prob_return_baseline": 0.65,
        "prob_progress_crisis": {
            "STEMI": 0.15
        }
    },
    "Mild_Hypoglycemia": {
        "vital_sign_effects": {
            "HR_factor_range": (1.1, 1.3),
            "BP_offset_range": (0, 0),
            "RR_offset_range": (2, 3)
        },
        "duration_range_minutes": (15, 60),
        "prob_return_baseline": 0.75,
        "prob_progress_crisis": {
            # Possibly 'Compromised_Airway' if patient passes out
            "Compromised_Airway": 0.1
        }
    },
    "TIA": {
        "vital_sign_effects": {
            "HR_factor_range": (1.0, 1.2),
            "BP_offset_range": (10, 20),
        },
        "duration_range_minutes": (10, 120),
        "prob_return_baseline": 0.80,
        "prob_progress_crisis": {
            # If you define stroke crisis
            # "Stroke_Crisis": 0.20
        }
    }
}

CRISIS_STATES = {
    "STEMI": {
        "vital_sign_effects": {
            "HR_factor_range": (1.3, 2.0),
            "BP_offset_range": (-10, -30),
            "O2_sat_offset_range": (-5, -2)
        },
        "no_spontaneous_recovery": True,
        "prob_death_per_minute": 0.50  # or 0.05, depending on your timescale
    },
    "Septic_Shock": {
        "vital_sign_effects": {
            "HR_factor_range": (1.3, 1.8),
            "BP_offset_range": (-20, -50),
            "RR_factor_range": (1.3, 1.6),
            "O2_sat_offset_range": (-10, -5)
        },
        "no_spontaneous_recovery": True,
        "prob_death_per_minute": 0.10
    },
    "Compromised_Airway": {
        "vital_sign_effects": {
            "RR_factor_range": (1.5, 2.0),
            "HR_factor_range": (1.2, 1.6),
            "O2_sat_offset_range": (-20, -30)
        },
        "no_spontaneous_recovery": True,
        "prob_death_per_minute": 0.70
    },
    "Hemorrhagic_Crisis": {
        "vital_sign_effects": {
            "HR_factor_range": (1.3, 2.0),
            "BP_offset_range": (-20, -50),
            "RR_factor_range": (1.2, 1.5),
            "O2_sat_offset_range": (-5, -2)
        },
        "no_spontaneous_recovery": True,
        "prob_death_per_minute": 0.20
    }
}

DEATH_STATE = {
    "vital_sign_effects": {
        "override_to_zero": True
    },
    "no_spontaneous_recovery": True
}
