import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ecg_simulate import ecg_simulate
import os
import math




##########################################################
# 1. Base ECG parameters (ECGSYN defaults for normal sinus)
##########################################################
def base_ecg_params():
    """
    Returns a dictionary of baseline parameters for ecg_simulate (ECGSYN).
    You can tweak duration, sampling_rate, or amplitude arrays as defaults.
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


# 2a. Adjust ecg params based on vitals
import numpy as np

def apply_vitals_to_params(hr, diastolic_bp, systolic_bp, rr, spo2, a, b, t):
    """
    Modify ECG parameters (a, b, t) based on current vitals.
    
    Parameters:
      hr : float
          Heart rate in beats per minute.
      diastolic_bp : float
          Diastolic blood pressure in mmHg.
      systolic_bp : float
          Systolic blood pressure in mmHg.
      rr : float
          Respiratory rate in breaths per minute.
      spo2 : float
          Oxygen saturation percentage.
      a : list or np.array
          Amplitudes for the P, Q, R, S, and T waves.
      b : list or np.array
          Widths (standard deviations) for the Gaussian shapes of each wave.
      t : list or np.array
          Timing (phase) values for each wave.
          
    Returns:
      a, b, t : Modified parameters.
      
    Relationships implemented:
      - Heart rate (HR): For HR > 70, widths (b) are reduced by 1% per bpm above baseline.
        Additionally, the timing parameters (t) are scaled by (70/hr) to shorten the QT interval.
      - Systolic BP (SBP): R-wave amplitude (a[2]) is increased by 5% per 10 mmHg above 120 mmHg,
        bounded between 0.9 and 1.1. For SBP below 120, a modest increase of 2% per 10 mmHg is applied,
        capped at 1.05.
      - Diastolic BP (DBP): Q-wave amplitude (a[1]) is increased by 3% per 10 mmHg above 80 mmHg,
        bounded between 0.95 and 1.05.
      - Respiratory rate (RR): For RR above 16 bpm, T-wave amplitude (a[4]) is reduced by 1% per bpm above baseline,
        with a minimum scaling factor of 0.7.
      - Oxygen saturation (SpO₂): For SpO₂ below 98%, P-wave amplitude (a[0]) is decreased by 0.01 per percentage point drop,
        clipped between 0.2 and 0.3. Additionally, for severe hypoxemia (SpO₂ < 90%), T-wave amplitude (a[4])
        is further reduced by 2% per percentage point below 90, with a lower bound factor of 0.85.
    """
    # Define baseline values
    a = list(a)
    b = list(b)
    t = list(t)
    
    # Define baseline values
    baseline_hr   = 70.0
    baseline_sbp  = 120.0
    baseline_dbp  = 80.0
    baseline_rr   = 16.0
    baseline_spo2 = 98.0

    # Adjust R-wave amplitude with systolic BP (a[2])
    if systolic_bp >= baseline_sbp:
        sbp_diff = systolic_bp - baseline_sbp
        factor = 1 + 0.05 * (sbp_diff / 10.0)
        factor = min(max(factor, 0.9), 1.1)
    else:
        sbp_diff = baseline_sbp - systolic_bp
        # Increase R-wave amplitude modestly when SBP is low (simulating mechanical axis shift)
        factor = 1 + 0.02 * (sbp_diff / 10.0)
        factor = min(factor, 1.05)
    a[2] = a[2] * factor

    # Adjust Q-wave amplitude with diastolic BP (a[1])
    dbp_diff = diastolic_bp - baseline_dbp
    q_factor = 1 + 0.03 * (dbp_diff / 10.0) if dbp_diff > 0 else 1
    q_factor = min(max(q_factor, 0.95), 1.05)
    a[1] = a[1] * q_factor

    # Adjust T-wave amplitude with respiratory rate (a[4])
    rr_diff = rr - baseline_rr
    if rr_diff > 0:
        t_factor = max(1 - 0.01 * rr_diff, 0.7)
        a[4] = a[4] * t_factor

    # Adjust P-wave amplitude based on SpO2 (a[0])
    spo2_diff = baseline_spo2 - spo2
    if spo2_diff > 0:
        new_a0 = a[0] - 0.01 * spo2_diff
        a[0] = np.clip(new_a0, 0.2, 0.3)
    
    # Additional T-wave reduction for severe hypoxemia (SpO2 < 90)
    if spo2 < 90:
        hypoxemia_diff = 90 - spo2
        hypoxemia_factor = max(1 - 0.02 * hypoxemia_diff, 0.85)
        a[4] = a[4] * hypoxemia_factor

    # Adjust widths (b) based on heart rate
    if hr > baseline_hr:
        width_factor = 1 - 0.01 * (hr - baseline_hr)
        width_factor = max(width_factor, 0.8)
        b = [width * width_factor for width in b]

    # Adjust timing (t) based on heart rate to simulate QT and beat interval changes.
    # This scales the timing parameters inversely with the heart rate.
    if hr > 0:
        t = [ti * (baseline_hr / hr) for ti in t]

    return a, b, t



################################################################################
# 2b. Create or adjust ECG parameters from the block's average vitals + state
################################################################################
def create_ecg_params_from_vitals(block_df, base_params, state_label):
    """
    Given a contiguous block of vitals and a state_label (integer),
    produce a new parameter dictionary for ecg_simulate.

    States with unique morphological changes:
      - 9 (Bathroom): zeroed-out ECG
      - 1 (Cardiac Ischaemia): mild ST depression, T wave inversion, HR +10
      - 11 (STEMI (crisis)): stronger ST elevation, remove T wave inversion, HR +15
      (expand as needed for other states).
    """
    p_out = dict(base_params)  # shallow copy
    # print(f"base_params in create_ecg_params_from_vitals: {base_params}")
    # Define default values for vital signs
    defaults = {
        "heart_rate": 70,
        "diastolic_bp": 80,
        "systolic_bp": 120,
        "respiratory_rate": 12,
        "oxygen_saturation": 98
    }

    # average vitals with NaN handling
    hr_avg = block_df["heart_rate"].mean() if not pd.isna(block_df["heart_rate"].mean()) else defaults["heart_rate"]
    hr_std = block_df["heart_rate"].std() if not pd.isna(block_df["heart_rate"].std()) else 1.0
    dbp_avg = block_df["diastolic_bp"].mean() if not pd.isna(block_df["diastolic_bp"].mean()) else defaults["diastolic_bp"]
    sbp_avg = block_df["systolic_bp"].mean() if not pd.isna(block_df["systolic_bp"].mean()) else defaults["systolic_bp"]
    rr_avg = block_df["respiratory_rate"].mean() if not pd.isna(block_df["respiratory_rate"].mean()) else defaults["respiratory_rate"]
    spo2_avg = block_df["oxygen_saturation"].mean() if not pd.isna(block_df["oxygen_saturation"].mean()) else defaults["oxygen_saturation"]

    # set basic HR with validation
    p_out["heart_rate"] = max(min(hr_avg, 200), 40)  # Limit HR between 40 and 200
    p_out["heart_rate_std"] = max(min(hr_std, 10), 0)  # Limit HR std between 0 and 10

    # Some convenient references
    a_i_list = list(p_out["ai"])
    t_i_list = list(p_out["ti"])
    b_i_list = list(p_out["bi"])

    # adjust ecg params based on vitals, can comment this out as this introduces additional variability
    # a_i_list, b_i_list, t_i_list = apply_vitals_to_params(hr_avg, dbp_avg, sbp_avg, rr_avg, spo2_avg, p_out["ai"], p_out["bi"], p_out["ti"])
    def angle_multiplier(current_angle, desired_degree_shift):
        # new_angle = current_angle + desired_degree_shift
        # factor = new_angle / current_angle (but watch for zero or near-zero)
        if abs(current_angle) < 1e-6:
            # If the baseline angle is 0, we can't do an exact ratio. 
            # We'll just pick a small nonzero baseline so the ratio is stable.
            # E.g. set baseline to 1 to form the ratio. This is approximate.
            base_val = 1.0 if current_angle == 0 else current_angle
        else:
            base_val = current_angle
        new_angle = base_val + desired_degree_shift
        return new_angle / base_val
    
    # Switch on the state_label
    if state_label == 0:
        # Neutral (baseline). No morphological changes: keep everything as in base_params
        pass

    elif state_label == 1:  # Cardiac Ischaemia
        # mild ST depression or T‐wave inversion
        # e.g., reduce T wave amplitude, invert the T wave
        a_i_list[4] = -abs(a_i_list[4]) * 0.8
        # maybe reduce R wave amplitude slightly
        a_i_list[2] = a_i_list[2] * 0.9
        
        # - Maybe shift T wave angle by ~+3 deg from baseline 125 => 128 => factor ~ 128/125 = 1.024
        t_i_list[4] *= angle_multiplier(t_i_list[4], +3)
        # optionally add slight noise or reduce lfhfratio
        p_out["lfhfratio"] *= 0.8

    elif state_label == 2:  # Sepsis
        
        # - QRS amplitude lower ~15% => multiply by 0.85
        a_i_list[2] *= 0.85

        # - T wave often inverted ~70% => multiply by -0.7
        a_i_list[4] *= -0.7

        # - T wave width ~ +6% to reflect mild QTc prolongation => 0.308 => 0.326 => factor ~1.06
        b_i_list[4] *= 1.06
        
        # moderate changes => lower T wave amplitude or flatten T wave
        a_i_list[4] = a_i_list[4] * 0.99
        # maybe raise noise a bit for more variation
        # p_out["noise"] = 0.05
        # optional widths => slight broadening of QRS or T wave
        # b_i_list[2] += 0.02  # R wave slightly broader

    elif state_label == 3:  # Acute Anxiety/Panic
        # possibly tall P waves if hyperventilation => P wave index=0
        a_i_list[0] = abs(a_i_list[0]) * 1.4
        # can also slightly raise T wave amplitude
        a_i_list[4] *= 1.1
        # - T wave angle shift: say we want ~ -2 deg from 125 => 123 => factor ~ 123/125=0.984
        t_i_list[4] *= angle_multiplier(t_i_list[4], -2)
        # reduce lfhfratio => more high‐frequency variation
        p_out["lfhfratio"] = 0.4

    elif state_label == 4:  # Breathing Difficulty
        # minimal changes => maybe small T wave decrease
        a_i_list[4] *= 0.9
        # - T wave amplitude ~ -10%
        a_i_list[4] *= 0.9
        # or shift angles slightly => e.g., T wave peak delayed
        # - T wave angle: +5 deg => factor ~ 130/125=1.04 if baseline is 125. If the baseline is 125 indeed:
        #   but if we have a different angle from prior changes, we do a ratio approach:
        t_i_list[4] *= angle_multiplier(t_i_list[4], +5)
        # slightly increase noise
        p_out["noise"] = 0.06

    elif state_label == 5:  # Hypovolaemia
        # Possibly lower QRS amplitude => reduce R wave
        a_i_list[2] *= 0.8
        # T wave may flatten
        a_i_list[4] *= 0.8
        # - T wave angle: +3 deg => factor ~128/125=1.024 if baseline was 125
        t_i_list[4] *= angle_multiplier(t_i_list[4], +3)
        # slight increase in noise
        p_out["noise"] = 0.065

    elif state_label == 6:  # Arrhythmic Flare
        # Possibly random angle shifts => e.g., Q wave (index=1) shift
        t_i_list[1] += np.random.uniform(-5, 5)
        t_i_list[2] += np.random.uniform(-3, 3)
        # moderate changes to R wave amplitude
        a_i_list[2] *= 0.85
        # T wave can vary widely if conduction is abnormal in some beats
        a_i_list[4] *= np.random.uniform(0.7, 1.0)
        # lfhfratio => more irregular HR => we can reduce ratio to add HF noise
        p_out["lfhfratio"] = 0.3


    elif state_label == 7:  # Hypoglycemia
        # T wave flattening => reduce T wave amplitude
        a_i_list[4] = abs(a_i_list[4]) * 0.5
        # slight ST depression => reduce baseline after QRS
        a_i_list[2] *= 0.95
        # Slight T wave broadening to reflect QTc prolongation
        b_i_list[4] += 0.03  
        p_out["noise"] = 0.055

    elif state_label == 8:  # TIA
        # typically minor => very small changes or no changes
        a_i_list[4] = a_i_list[4] * 0.75  # slightly lower T wave
        # maybe a small shift in T wave angle
        t_i_list[4] *= angle_multiplier(t_i_list[4], +3)
        # no big noise changes
        pass

    elif state_label == 9:  # Bathroom (harmless)
        # zero out. Means amplitude=0 for ai, or skip generating ECG
        a_i_list = [0,0,0,0,0]
        # also can reduce noise
        p_out["noise"] = 0.0

    elif state_label == 10:  # White Coat (harmless)
        # mild changes => slight overall amplitude boost? e.g. R wave up
        a_i_list[2] *= 1.1
        # Very mild T wave flattening from brief anxiety
        a_i_list[4] *= 0.9
        # maybe a small noise bump
        p_out["noise"] = 0.055

    elif state_label == 11:  # STEMI (crisis)

        # pathologic Q wave
        a_i_list[1] = -abs(a_i_list[1]) * 1.2
        # significant ST elevation => big R wave offset
        # a_i_list[2] += 0.0
        # if T wave was inverted, revert it
        a_i_list[3] = abs(a_i_list[3]) * 0.3
        
        # T wave hyperacute => big initial amplitude
        a_i_list[4] = abs(a_i_list[4]) * 1.2
        # Widen T wave 
        b_i_list[4] += 0.08

        t_i_list[0] = t_i_list[0] * 0.55
        b_i_list[4] = b_i_list[4] * 0.7
        # higher noise
        p_out["noise"] = 0.06

    elif state_label == 12:  # Septic Shock (crisis)
        # Possibly low QRS amplitude => reduce R wave
        a_i_list[2] *= 0.7
        # Flatten T wave more
        # a_i_list[4] *= 0.5
        # T wave strongly flattened or inverted
        a_i_list[4] = -abs(a_i_list[4]) * 0.4
        # raise noise
        p_out["noise"] = 0.065
        # maybe adjust widths => broad QRS
        b_i_list[2] += 0.05

    elif state_label == 13:  # Compromised Airway (crisis)
        # extreme tachyarrhythmias => we can't do direct beat‐to‐beat here,
        # so maybe raise noise or random angle changes
        a_i_list[2] *= 0.8
        # T wave partial inversion from severe hypoxia
        a_i_list[4] = abs(a_i_list[4]) * 0.8
        # more noise
        p_out["noise"] = 0.1
        # or shift T wave angle
        t_i_list[4] *= angle_multiplier(t_i_list[4], +8)

    elif state_label == 14:  # Haemorrhagic Shock (crisis)
        # Marked tachycardia => not changing HR here
        # Possibly low QRS amplitude => reduce R wave more
        a_i_list[2] *= 0.6
        # ST depression if myocardial perfusion compromised
        a_i_list[4] = -abs(a_i_list[4]) * 0.5
        p_out["noise"] = 0.09

    elif state_label == 15:  # Stroke (crisis)
        # Possibly T wave inversions
        a_i_list[4] = -abs(a_i_list[4]) * 1.7
        
        # U waves or prolonged QT => we simulate a T width increase
        b_i_list[4] += 0.03
        # no big R wave changes
        p_out["noise"] = 0.07

    elif state_label == 16:  # Death
        # zero amplitude
        a_i_list = [0,0,0,0,0]
        p_out["noise"] = 0.0

    # Reassign the morphological arrays
    p_out["ai"] = tuple(a_i_list)
    p_out["ti"] = tuple(t_i_list)
    p_out["bi"] = tuple(b_i_list)

    return p_out


###################################################################
# 3. S-curve blending function for wave arrays, HR, noise, etc.
###################################################################
def s_curve(alpha, steepness=3.0):
    return 1.0 / (1.0 + np.exp(-steepness*(alpha - 0.5)))


def blend_ecg_params(old_params, new_params, alpha):
    """
    Interpolate numeric fields in old_params -> new_params with factor alpha in [0..1].
    Typically: (ti, ai, bi) plus heart_rate, noise, heart_rate_std, etc.
    """
    out = dict(new_params)  # final param set
    # morphological arrays
    for key in ["ti", "ai", "bi"]:
        if key in old_params and key in new_params:
            old_array = old_params[key]
            new_array = new_params[key]
            if len(old_array) == len(new_array):
                blended = []
                for (ov, nv) in zip(old_array, new_array):
                    val = (1 - alpha)*ov + alpha*nv
                    blended.append(val)
                out[key] = tuple(blended)

    # also blend heart_rate, noise, heart_rate_std if present
    for key2 in ["heart_rate", "noise", "heart_rate_std"]:
        if key2 in old_params and key2 in new_params:
            ov = old_params[key2]
            nv = new_params[key2]
            out[key2] = (1-alpha)*ov + alpha*nv

    return out


#############################################################
# 4. Generate ECG for a block with multiple sub-chunk morphs
#############################################################
def generate_ecg_for_block(block_df, old_params, new_params, sampling_rate=250):
    """
    Create one contiguous ECG snippet for the entire block 
    (which is N rows * 5 seconds/row => block_duration_sec).
    We break it into sub-chunks to smoothly transition from 
    old_params -> new_params using an S-curve blend.
    
    We'll rely on an ecg_simulate(**params) function to create
    each sub-chunk, then concatenate.
    """

    block_duration_sec = 5.0 * len(block_df)
    if block_duration_sec <= 0:
        return np.array([]), np.array([])

    chunk_size = 5.0  # each chunk 5s
    n_chunks = int(math.ceil(block_duration_sec / chunk_size))

    # the final ECG container
    n_total_samples = int(block_duration_sec * sampling_rate)
    ecg_out = np.zeros(n_total_samples)
    time_out= np.linspace(0, block_duration_sec, n_total_samples, endpoint=False)

    idx_start = 0

    for chunk_i in range(n_chunks):
        chunk_start_sec = chunk_i * chunk_size
        chunk_end_sec   = min(block_duration_sec, chunk_start_sec + chunk_size)
        this_chunk_duration = chunk_end_sec - chunk_start_sec
        if this_chunk_duration <= 0:
            break

        alpha_lin = chunk_i / float(max(n_chunks - 1, 1))
        alpha_s   = s_curve(alpha_lin, steepness=5.0)

        # blend morphological parameters
        chunk_params = blend_ecg_params(old_params, new_params, alpha_s)
        chunk_params["duration"] = this_chunk_duration
        chunk_params["length"]   = None
        chunk_params["sampling_rate"] = sampling_rate

        # call the ecg_simulate function
        ecg_chunk = ecg_simulate(**chunk_params)
        chunk_nsamples = len(ecg_chunk)

        idx_end = idx_start + chunk_nsamples
        if idx_end > len(ecg_out):
            idx_end = len(ecg_out)
            ecg_chunk = ecg_chunk[:(idx_end - idx_start)]
        ecg_out[idx_start:idx_end] = ecg_chunk
        idx_start = idx_end

    return time_out, ecg_out


##########################################################
# 5. Main post-processing driver function
##########################################################
def generate_ecg_postprocess(vitals_csv_path, ecg_csv_name):
    """

    1) Reads the final vitals.csv with 5-second rows.
    2) Groups consecutive rows by (patient_id, state_label) => blocks.
    3) For each block, obtains new ecg parameters from vitals + state 
       (including morphological changes).
    4) Blends from old to new params so that wave shape changes smoothly.
    5) Writes an 'ecg_data.csv' with columns:
         patient_id, block_id, block_start_timestamp, time_in_block_sec, ecg_amplitude
    """
    import csv
    
    if not os.path.exists(vitals_csv_path):
        raise FileNotFoundError(f"Cannot find {vitals_csv_path}")

    df = pd.read_csv(vitals_csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.sort_values(["patient_id", "timestamp"], inplace=True)


    # Identify new blocks whenever patient_id or state_label changes
    df["block_change"] = (df["patient_id"].shift(1) != df["patient_id"]) \
                         | (df["state_label"].shift(1) != df["state_label"])
    df["block_id"] = df["block_change"].cumsum()

    # Baseline ECG config
    base_p = base_ecg_params()

    ecg_records = []
    prev_params_by_pid = {}  # track the last block's final params per patient

    grouped = df.groupby(["block_id", "patient_id"], as_index=False)
    for (block_id, pid), block_df in grouped:
        # The new block's first row sets the state_label
        st_label = block_df["state_label"].iloc[0]

        # compute new ecg params from average vitals + morphological changes
        new_params = create_ecg_params_from_vitals(block_df, base_p, st_label)

        # old_params is what we had from the last block for this patient
        old_params = prev_params_by_pid.get(pid, base_p)
        
        # For the bathroom state, override blending so that parameters are flat (all zeros)
        if st_label == 9:
            old_params = new_params
        # generate the snippet
        t_ecg, snippet = generate_ecg_for_block(block_df, old_params, new_params,
                                                sampling_rate=base_p["sampling_rate"])

        # store snippet
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

        # update old_params
        prev_params_by_pid[pid] = new_params

    out_df = pd.DataFrame(ecg_records)
    out_df.to_csv(ecg_csv_name, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"ECG data saved to {ecg_csv_name}")



#####################################
# 6. CLI entry point
#####################################
if __name__ == "__main__":
    generate_ecg_postprocess(vitals_csv_path="vitals.csv", ecg_csv_name="ecg_data.csv")
    print("Done generating ECG waveforms from vitals.csv.")
