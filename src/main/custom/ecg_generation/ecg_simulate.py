# -*- coding: utf-8 -*-
import math

import numpy as np
import pandas as pd
import scipy

from neurokit2.misc import check_random_state, check_random_state_children
from neurokit2.signal import signal_distort, signal_resample



def ecg_simulate(
    duration=10,
    length=None,
    sampling_rate=1000,
    noise=0.05,
    heart_rate=70,
    heart_rate_std=1,
    method="ecgsyn",  # This parameter is retained only for backwards compatibility.
    random_state=None,
    random_state_distort="spawn",
    **kwargs,
):
    """Simulate a single-lead (Lead II) ECG/EKG signal using the McSharry model (ECGSYN).

    This function generates an artificial ECG signal of a given duration and sampling rate
    using the dynamical model by McSharry et al. (2003). The simulation always produces a single lead,
    representing Lead II.

    Parameters
    ----------
    duration : int
        Desired recording length in seconds.
    length : int, optional
        Desired length of the signal in samples. If not provided, it defaults to duration * sampling_rate.
    sampling_rate : int
        The desired sampling rate (Hz).
    noise : float
        Noise level (amplitude of the Laplace noise).
    heart_rate : int
        Desired simulated heart rate (beats per minute).
    heart_rate_std : int
        Desired heart rate standard deviation (beats per minute).
    method : str
        For compatibility; this simulation always uses ECGSYN (McSharry) and outputs Lead II.
    random_state : None, int, numpy.random.RandomState, or numpy.random.Generator
        Seed for the random number generator.
    random_state_distort : {'legacy', 'spawn'}, None, int, numpy.random.RandomState, or numpy.random.Generator
        Random state to be used to distort the signal.
    **kwargs
        Other keyword parameters for the ECGSYN algorithm.

    Returns
    -------
    numpy.ndarray
        A one-dimensional array containing the simulated ECG signal for Lead II.
    """
    # Seed the random state for reproducibility.
    rng = check_random_state(random_state)

    # Determine signal length (in samples) and capture duration if necessary.
    if length is None:
        length = duration * sampling_rate
    if duration is None:
        duration = length / sampling_rate

    # Calculate approximate number of heart beats.
    approx_number_beats = int(np.round(duration * (heart_rate / 60)))

    # Always simulate using ECGSYN with Lead II parameters.
    # The gamma parameter corresponding to Lead II is taken as: [2, 0.2, 0.2, 0.2, 3].
    signals, results = _ecg_simulate_ecgsyn(
        sfecg=sampling_rate,
        N=approx_number_beats,
        hrmean=heart_rate,
        hrstd=heart_rate_std,
        sfint=sampling_rate,
        # gamma=np.array([[2, 0.2, 0.2, 0.2, 3]]),  # Original Lead II parameters
        gamma = np.ones(5),  # Modified to use uniform scaling
        rng=rng,
        **kwargs,

    )

    # Truncate signals to match the desired signal length.
    for i in range(len(signals)):
        signals[i] = signals[i][0:int(length)]

    # Add random noise if specified.
    if noise > 0:
        random_state_distort = check_random_state_children(random_state, random_state_distort, n_children=len(signals))
        for i in range(len(signals)):
            signals[i] = signal_distort(
                signals[i],
                sampling_rate=sampling_rate,
                noise_amplitude=noise,
                noise_frequency=[5, 10, 100],  # Frequencies of noise components in Hz
                noise_shape="laplace",  # Laplace distribution for more realistic ECG noise
                random_state=random_state_distort[i],
                silent=True,
            )

    # Always return a single channel (Lead II).
    return signals[0]


# ECGSYN - McSharry's ECG model implementation
def _ecg_simulate_ecgsyn(
    sfecg=256,  # ECG sampling frequency in Hz
    N=256,  # Approximate number of heart beats to simulate
    hrmean=60,  # Mean heart rate in BPM
    hrstd=1,  # Standard deviation of heart rate in BPM
    lfhfratio=0.5,  # Low-frequency to high-frequency ratio for heart rate variability
    sfint=512,  # Internal sampling frequency for accurate wave morphology
    ti=(-85, -15, 0, 15, 125),  # Angular positions of PQRST waves in degrees
    ai=(0.39, -5, 30, -7.5, 0.30),  # Wave amplitudes for PQRST respectively
    bi=(0.29, 0.1, 0.1, 0.1, 0.44),  # Gaussian width parameters for PQRST waves
    gamma=np.array([2, 0.2, 0.2, 0.2, 3]),  # Lead-specific scaling factors for wave components
    rng=None,
    **kwargs,
):
    """
    Generates synthetic ECG signals using McSharry's dynamical model.

    The model creates realistic ECG morphology by combining Gaussian functions
    for each wave (P, Q, R, S, T) and modulating them with heart rate variability.

    Parameters
    ----------
    ti : tuple
        Angular positions (in degrees) for PQRST waves. These determine the timing
        of each wave component relative to the R peak (at 0 degrees).
    ai : tuple 
        Amplitudes of PQRST waves. These control the height/depth of each wave component.
        Positive values create upward deflections, negative values create downward deflections.
    bi : tuple
        Gaussian width parameters for PQRST waves. Larger values create wider wave components,
        while smaller values create narrower, more peaked waves.
    gamma : array
        Lead-specific scaling factors that modify wave amplitudes to simulate different ECG leads.
        Default values [2, 0.2, 0.2, 0.2, 3] correspond to typical Lead II morphology.

    Returns
    -------
    tuple
        (signals, results) where signals contains the ECG waveforms and results contains
        the intermediate simulation data.
    """

    # Convert angular positions from degrees to radians
    if not isinstance(ti, np.ndarray):
        ti = np.array(ti)
    if not isinstance(ai, np.ndarray):
        ai = np.array(ai)
    if not isinstance(bi, np.ndarray):
        bi = np.array(bi)

    ti = ti * np.pi / 180  # Convert degrees to radians

    # Adjust wave parameters based on heart rate
    # Higher heart rates lead to compressed waves
    hrfact = np.sqrt(hrmean / 60)  # Scaling factor based on heart rate
    hrfact2 = np.sqrt(hrfact)
    bi = hrfact * bi  # Adjust wave widths
    # Scale timing of waves - P and T waves scale more than QRS complex
    ti = np.array([hrfact2, hrfact, 1, hrfact, hrfact2]) * ti

    # Verify sampling rate relationship
    q = np.round(sfint / sfecg)
    qd = sfint / sfecg
    if q != qd:
        raise ValueError(
            "Internal sampling frequency (sfint) must be an integer multiple of the ECG sampling frequency"
            " (sfecg). Your current choices are: sfecg = "
            + str(sfecg)
            + " and sfint = "
            + str(sfint)
            + "."
        )

    # Define frequency parameters for heart rate variability
    flo = 0.1  # Mayer waves frequency (low-frequency oscillations)
    fhi = 0.25  # Respiratory frequency (high-frequency oscillations)
    flostd = 0.01  # Standard deviation of low-frequency component
    fhistd = 0.01  # Standard deviation of high-frequency component

    # Calculate time scales for RR intervals and total output
    sfrr = 1  # Sampling rate for RR series
    trr = 1 / sfrr
    rrmean = 60 / hrmean  # Mean RR interval in seconds
    n = 2 ** (np.ceil(np.log2(N * rrmean / trr)))

    rr0 = _ecg_simulate_rrprocess(flo, fhi, flostd, fhistd, lfhfratio, hrmean, hrstd, sfrr, n, rng)

    # Upsample RR intervals to match internal sampling rate
    rr = signal_resample(rr0, sampling_rate=1, desired_sampling_rate=sfint)

    # Generate time series of RR intervals
    dt = 1 / sfint
    rrn = np.zeros(len(rr))
    tecg = 0
    i = 0
    while i < len(rr):
        tecg += rr[i]
        ip = int(np.round(tecg / dt))
        rrn[i:ip] = rr[i]
        i = ip
    Nt = ip

    # Initial conditions for the dynamical system
    x0 = np.array([1, 0, 0.04])  # [x, y, z] state variables

    # Time points for numerical integration
    Tspan = [0, (Nt - 1) * dt]
    t_eval = np.linspace(0, (Nt - 1) * dt, Nt)

    # Initialize results containers
    results = []
    signals = []

    # Generate ECG for each lead
    for lead in range(len(gamma)):
        # Solve differential equations using Runge-Kutta method
        result = scipy.integrate.solve_ivp(
            lambda t, x: _ecg_simulate_derivsecgsyn(t, x, rrn, ti, sfint, gamma[lead] * ai, bi),
            Tspan,
            x0,
            t_eval=t_eval,
        )
        results.append(result)
        X0 = result.y

        # Downsample to desired ECG sampling rate
        X = X0[:, np.arange(0, X0.shape[1], q).astype(int)]

        # Scale signal to physiological range (-0.4 to 1.2 mV)
        z = X[2, :].copy()
        zmin = np.min(z)
        zmax = np.max(z)
        zrange = zmax - zmin
        z = (z - zmin) * 1.6 / zrange - 0.4

        signals.append(z)
        
        # signals.append(X[2, :])

    return signals, results


def _ecg_simulate_derivsecgsyn(t, x, rr, ti, sfint, ai, bi):
    """
    Compute derivatives for the dynamical ECG model.
    
    This function implements the core differential equations that generate
    the ECG morphology through a quasi-periodic trajectory in state space.
    """
    # Calculate angular position in the trajectory
    ta = math.atan2(x[1], x[0])
    r0 = 1  # Unit circle radius
    # Radial component that maintains trajectory near unit circle
    a0 = 1.0 - np.sqrt(x[0] ** 2 + x[1] ** 2) / r0

    # Calculate instantaneous angular velocity from RR interval
    ip = np.floor(t * sfint).astype(int)
    w0 = 2 * np.pi / rr[min(ip, len(rr) - 1)]

    # Add respiratory modulation to baseline
    fresp = 0.25  # Respiratory frequency
    zbase = 0.001 * np.sin(2 * np.pi * fresp * t)

    # Equations of motion
    dx1dt = a0 * x[0] - w0 * x[1]  # x component
    dx2dt = a0 * x[1] + w0 * x[0]  # y component

    # Calculate angular difference to each wave component
    dti = (ta - ti) - np.round((ta - ti) / 2 / np.pi) * 2 * np.pi
    # z component - sum of Gaussian-shaped deflections for each wave
    dx3dt = -np.sum(ai * dti * np.exp(-0.5 * (dti / bi) ** 2)) - 1 * (x[2] - zbase)

    dxdt = np.array([dx1dt, dx2dt, dx3dt])
    return dxdt


def _ecg_simulate_rrprocess(
    flo=0.1,  # Mayer wave frequency (low)
    fhi=0.25,  # Respiratory frequency (high)
    flostd=0.01,  # Low-frequency variation
    fhistd=0.01,  # High-frequency variation
    lfhfratio=0.5,  # Ratio of low to high frequency power
    hrmean=60,  # Mean heart rate
    hrstd=1,  # Heart rate standard deviation
    sfrr=1,  # RR sampling frequency
    n=256,  # Number of points
    rng=None,
):
    """
    Generate RR intervals with realistic heart rate variability.
    
    Creates a time series of RR intervals that exhibits both low-frequency
    (Mayer waves) and high-frequency (respiratory) oscillations.
    """
    w1 = 2 * np.pi * flo
    w2 = 2 * np.pi * fhi
    c1 = 2 * np.pi * flostd
    c2 = 2 * np.pi * fhistd
    sig2 = 1
    sig1 = lfhfratio
    rrmean = 60 / hrmean
    rrstd = 60 * hrstd / (hrmean * hrmean)

    df = sfrr / n
    w = np.arange(n) * 2 * np.pi * df
    dw1 = w - w1
    dw2 = w - w2

    # Create power spectrum with two Gaussian peaks
    Hw1 = sig1 * np.exp(-0.5 * (dw1 / c1) ** 2) / np.sqrt(2 * np.pi * c1 ** 2)
    Hw2 = sig2 * np.exp(-0.5 * (dw2 / c2) ** 2) / np.sqrt(2 * np.pi * c2 ** 2)
    Hw = Hw1 + Hw2
    Hw0 = np.concatenate((Hw[0 : int(n / 2)], Hw[int(n / 2) - 1 :: -1]))
    Sw = (sfrr / 2) * np.sqrt(Hw0)

    # Generate random phases
    ph0 = 2 * np.pi * rng.uniform(size=int(n / 2 - 1))
    ph = np.concatenate([[0], ph0, [0], -np.flipud(ph0)])
    SwC = Sw * np.exp(1j * ph)
    x = (1 / n) * np.real(np.fft.ifft(SwC))

    # Scale to desired RR interval variability
    xstd = np.std(x)
    ratio = rrstd / xstd
    return rrmean + x * ratio  # Return RR intervals