import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from matplotlib.animation import FuncAnimation, PillowWriter

# --- PARAMETERS ---
VibAmp = 1.0
InstVibAmpRate = 10.0
VibExtent = 40.0
VibRate = 6.0
f0 = 100.0
fs = 10000
T = 4.0
c = 343.0
n_reflections = 5
attenuation = 0.5  # reflection amplitude factor

# Room dimensions (X, Y, Z)
room_dims = np.array([4.05, 4.31, 1.66])

# --- TIME VECTOR ---
dt = 1.0/fs
t = np.arange(0, T+dt, dt)

# --- VIBRATO SIGNAL ---
VibSignalFreq = f0 + VibExtent * np.sin(2*np.pi*VibRate*t)
VibSignalPhase = 2*np.pi*np.cumsum(VibSignalFreq)/fs
VibAmpInst = np.abs(VibAmp * np.cos(2*np.pi*InstVibAmpRate*t))
VibratoSignal = VibAmpInst * np.sin(VibSignalPhase)
sig_len = len(VibratoSignal)

# --- ORIGINAL HILBERT ENVELOPE ---
env_original = np.abs(hilbert(VibratoSignal))

# --- RMS FUNCTION ---
def rolling_rms(signal, win_s, fs):
    win_samples = int(win_s*fs)
    step = max(1, win_samples//2)
    rms = np.array([np.sqrt(np.mean(signal[i:i+win_samples]**2))
                    for i in range(0, len(signal)-win_samples, step)])
    times = np.arange(0, len(rms))*step/fs
    return times, rms

# --- PANEL SETUP ---
fig, ax = plt.subplots(2, 1, figsize=(10, 6))

# Panel 1: original Hilbert envelope
ax[0].plot(t, env_original, color='b', label='Original Hilbert Envelope')
ax[0].set_title("Original Signal Envelope")
ax[0].set_ylabel("Amplitude (a.u.)")
ax[0].set_xlim(0, 0.5)
ax[0].set_ylim(0, 3*VibAmp)
ax[0].legend()

# Panel 2: echoed signal (Hilbert + RMS)
line_hilb, = ax[1].plot([], [], color='r', label='Hilbert Amplitude')
line_rms, = ax[1].plot([], [], color='g', label='Rolling RMS')
ax[1].set_title("Echoed Signal (Hilbert vs RMS)")
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel("Amplitude (a.u.)")
ax[1].set_xlim(0, 0.5)
ax[1].set_ylim(0, 3*VibAmp)
ax[1].legend()

plt.tight_layout()

# --- WALL POSITIONS ---

# room geometry
room_length = 4.05   # m
room_width  = 4.31   # m
ceiling_height = 4.48  # m
singer_height  = 1.66  # m

# singer assumed centered horizontally
dist_x = room_length / 2     # 2.025 m
dist_y = room_width / 2      # 2.155 m
dist_floor   = singer_height # 1.66 m
dist_ceiling = ceiling_height - singer_height  # 2.82 m

wall_positions = [
    (dist_x, dist_x),                 # +X wall and -X wall
    (dist_y, dist_y),                 # +Y wall and -Y wall
    (dist_floor, dist_ceiling)        # floor and ceiling
]

# --- ANIMATION FUNCTION ---
def animate(frame):
    echo_signal = VibratoSignal.copy()
    for ref in range(frame+1):
        for dim, (d0, d1) in enumerate(wall_positions):
            for distance in [d0, d1]:
                delay_samples = int(round(distance / c * fs * (ref+1)))
                if delay_samples < sig_len:
                    N = sig_len - delay_samples
                    echo_signal[delay_samples:delay_samples+N] += attenuation * VibratoSignal[:N]

    # Instantaneous amplitude (Hilbert)
    env_echo = np.abs(hilbert(echo_signal))
    line_hilb.set_data(t, env_echo)

    # Rolling RMS
    win_s = 0.03  # 10 ms window
    rms_times, rms_vals = rolling_rms(echo_signal, win_s, fs)
    line_rms.set_data(rms_times, rms_vals)

    ax[1].set_title(f"Echoed Signal (after {frame+1} reflections)")
    return line_hilb, line_rms

# --- CREATE ANIMATION ---
anim = FuncAnimation(fig, animate, frames=n_reflections, interval=700, blit=True)

# --- SAVE GIF ---
writer = PillowWriter(fps=2)  # slower playback
anim.save("D:\\vibrato_echo_rms_vs_hilbert.gif", writer=writer)

plt.show()


def generate_vibrato_signal(
    f0=100.0,
    fs=10000,
    T=4.0,
    VibRate_FM=6.0,
    VibExtent_cents=40.0,
    baseAmp=1,
    InstVibAmpRate=10.0,
    AM_depth=0.1,  # fraction 0 → 1
    fm_phase = 0,
    am_phase = 0
):
    """
    Vibrato signal with controllable AM depth.

    AM_depth = 1.0 → full modulation (0 → VibAmp)
    AM_depth = 0.3 → 30% modulation (0.7*VibAmp → VibAmp)
    """
    dt = 1.0 / fs
    t = np.arange(0, T + dt, dt)

    # Frequency modulation (linear Hz)
    cents_deviation = VibExtent_cents * np.sin(2.0 * np.pi * VibRate_FM * t + fm_phase)
    VibSignalFreq = f0 * (2.0 ** (cents_deviation / 1200.0))
    VibSignalPhase = 2.0 * np.pi * np.cumsum(VibSignalFreq) / fs

    # Amplitude envelope with controlled depth
    VibAmpInst = baseAmp * ((1 - AM_depth) + AM_depth * np.cos(2.0 * np.pi * InstVibAmpRate * t + am_phase))

    # Output
    VibratoSignal = VibAmpInst * np.sin(VibSignalPhase)

    return VibratoSignal, t







import itertools
import numpy as np
import pandas as pd

# --- Sweep ranges ---
f0s = [294, 659]
vib_rates = np.arange(3.0, 8.5, 1)        # Hz
vib_extents = np.arange(15.0, 80.0, 20.0)    # cents (or Hz if that's your definition)
am_depth_percents = np.linspace(0.005, 0.025, 5)
phase_offsets = np.array([0, 0.5*np.pi, np.pi, 1.5*np.pi])

###Simple
# f0s = [100]#, 659]

# vib_rates = np.arange(3.0, 8.5, 0.5)                # Hz
# vib_extents = np.arange(15.0, 75.0, 5.0)#20.0, 40.0, 60.0]         # cents
# am_depth_percents = [0.01]#, 0.01, 0.025]   # fraction of VibAmp

# phase_offsets = [0.0]#, np.pi]             # radians


# Optional: test FM-only, AM-only, and combined
conditions = ['FM+AM']#,''FM', 'AM']


import numpy as np
import itertools
import pandas as pd

results = []

for f0, vib_rate_fm, vib_extent_fm, vib_rate_am, am_pct, condition in itertools.product(f0s,
        vib_rates, vib_extents, vib_rates, am_depth_percents, conditions):

    # Select phase grids based on condition
    if condition == 'FM':
        fm_phases = phase_offsets
        am_phases = [None]

    elif condition == 'AM':
        fm_phases = [None]
        am_phases = phase_offsets

    elif condition == 'FM+AM':
        fm_phases = phase_offsets
        am_phases = phase_offsets
        

    for fm_phase, am_phase in itertools.product(fm_phases, am_phases):
        if fm_phase == None:
            fm_phase = 0
        if am_phase == None:
            am_phase = 0
        phaseDiff_gt = am_phase - fm_phase
        # --- Generate signal ---
        f_s = 10000
        VibratoSignal, ____ = generate_vibrato_signal(
                                f0=f0,
                                fs=f_s,
                                T=2,
                                VibRate_FM=vib_rate_fm,
                                VibExtent_cents=vib_extent_fm,
                                baseAmp=1,
                                InstVibAmpRate=vib_rate_am,
                                AM_depth=am_pct,
                                fm_phase=fm_phase,
                                am_phase=am_phase
                            )
        
            # --- Analyze ---
            ###Technically don't need a rolling window for an artificial signal, but let's see what happens
        pitch_contour, f_s_contour = returnContour(VibratoSignal, f_s, model)
        meanFreq = np.mean(pitch_contour)
        # signal_norm = returnNormalized(highestPitch, samplerate)
        # signal_downsample = downsample_audio(highestPitch, samplerate, f_s_contour)
        envelope, filtered, f_s = returnEnvelope(VibratoSignal, f_s, meanFreq)
        # vibRate_f0 = autocorrVib3HzLocal(pitch_contour, f_s_contour)
        vibRate_f0 = fftVib3HzLocal(pitch_contour, f_s_contour)
        vibExtent_f0 = vibAmp(pitch_contour, f_s_contour, vibRate_f0, 0.75)#window factor of 0.75 the wavelength
        # meanCentLogEnv = np.log(envelope + 1e-12)
        # meanCentLogEnv -= np.mean(meanCentLogEnv)
        # vibRate_amp = autocorrVib3HzLocal(meanCentLogEnv, f_s)
        vibRate_amp = fftVib3HzLocal(envelope, f_s)
        vibExtent_amp = vibAmpPerc(envelope, f_s, vibRate_amp, 0.75)
        refRMS = np.nan
        # result = apply_vibTremorDecision_rolling_harmonics(VibratoSignal, f_s, model, refRMS, meanFreq,
                                              # window_duration=2, step_duration=2,
                                              # max_freq=2.5*meanFreq, calibrated=False)
        # vibRate_f0, vibExtent_f0, vibRate_amp, vibExtent_amp, vibExtent_SPL, vibExtent_dB, vibPercent, vibExtentPa_roll, vibExtentSPL_roll, harmonicSPL_mean  = result



        if ((vibRate_f0 - vibRate_amp < 0.5) & ~np.isnan([vibRate_f0,vibRate_amp]).any()):
            phaseDiff = extract_phase_difference(envelope, pitch_contour, f_s, f_s_contour, vibRate_f0, vibRate_amp)
            phaseDiff_gt = am_phase - fm_phase
        if np.isnan([vibRate_f0,vibRate_amp]).any():
            phaseDiff = np.nan
        else: 
            phaseDiff = False
        
        # df_vibrato = process_file_amp(VibratoSignal, f_s, meanFreq, file_id='unknown', vibRate=vibRate_f0, vibExtent=vibExtent_f0)
        df_vibrato = analyze_vibrato_amp(envelope, f_s, f0, vibRate_f0=6, vibExtent_f0=100, file_id="unknown", max_freq=2.5*meanFreq)

        # print(df_vibrato)
        # origMask = ((df_vibrato['type'] =='original') & (df_vibrato['metric'] == 'RMS Vibrato Extent'))
        # normMask0 = ((df_vibrato['type'] =='normalized') & (df_vibrato['metric'] == 'RMS Vibrato Extent'))
        harmMask = ((df_vibrato['type'] =='original') & (df_vibrato['metric'] == 'Instantaneous Amplitude'))
        # normMask1 = ((df_vibrato['type'] =='normalized') & (df_vibrato['metric'] == 'Instantaneous Amplitude'))
        
        
        results.append({
            'f0': f0,
            'vibRate_fm_gt': vib_rate_fm,
            'vibExtent_fm_gt': vib_extent_fm,
            'vibRate_am_gt':vib_rate_am,
            'amDepth_gt': am_pct,
            'condition': condition,
            'fmPhase_gt': fm_phase,
            'amPhase_gt': am_phase,
            'phaseDiff_gt': phaseDiff_gt,
            'vibRate_f0':vibRate_f0,
            'vibExtent_f0':vibExtent_f0,
            'vibRate_amp':vibRate_amp,
            'vibExtent_amp':vibExtent_amp,
            'vibRate_amp_harm':df_vibrato.loc[harmMask,'vibRate_amp'],
            'vibExtent_harm':df_vibrato.loc[harmMask,'extent_pa'],
            'vibExtent_ampPerc':df_vibrato.loc[harmMask,'extent_pa'].iloc[0]/np.mean(envelope),
            'phaseDiff': phaseDiff
            
        })

sine_sweep_df = pd.DataFrame(results)
df = sine_sweep_df.copy()


import matplotlib.pyplot as plt

mask = df['condition'].str.contains('FM')

plt.figure()
plt.scatter(
    df.loc[mask, 'vibRate_fm_gt'],
    df.loc[mask, 'vibRate_f0'],
    alpha=0.6
)
lims = [df['vibRate_fm_gt'].min(), df['vibRate_fm_gt'].max()]
plt.plot(lims, lims)
plt.xlabel('Ground-truth FM rate (Hz)')
plt.ylabel('Estimated FM rate (Hz)')
plt.title('FM Vibrato Rate: Estimated vs Ground Truth')
plt.savefig('fm_rate_est_vs_gt.png', dpi=300)
plt.show()
plt.close()

plt.figure()
plt.scatter(
    df.loc[mask, 'vibExtent_fm_gt'],
    df.loc[mask, 'vibExtent_f0'],
    alpha=0.6
)
lims = [df['vibExtent_fm_gt'].min(), df['vibExtent_fm_gt'].max()]
plt.plot(lims, lims)
plt.xlabel('Ground-truth vibrato extent (cents)')
plt.ylabel('Estimated vibrato extent (cents)')
plt.title('FM Vibrato Extent: Estimated vs Ground Truth')
plt.savefig('fm_extent_est_vs_gt.png', dpi=300)
plt.show()
plt.close()

mask = df['condition'].str.contains('AM')

plt.figure()
plt.scatter(
    df.loc[mask, 'vibRate_am_gt'],
    df.loc[mask, 'vibRate_amp'],
    alpha=0.6
)
lims = [df['vibRate_am_gt'].min(), df['vibRate_am_gt'].max()]
plt.plot(lims, lims)
plt.xlabel('Ground-truth AM rate (Hz)')
plt.ylabel('Estimated AM rate (Hz)')
plt.title('AM Rate: Estimated vs Ground Truth')
plt.show()
plt.savefig('am_rate_est_vs_gt.png', dpi=300)
plt.close()

plt.figure()
plt.scatter(
    df.loc[mask, 'amDepth_gt'],
    df.loc[mask, 'vibExtent_amp'],
    alpha=0.6
)
plt.xlabel('Ground-truth AM depth')
plt.ylabel('Estimated AM extent')
plt.title('AM Depth: Estimated vs Ground Truth')
plt.savefig('am_extent_est_vs_gt.png', dpi=300)
plt.show()
plt.close()

mask = df['condition'].str.contains('FM')

rate_error = df.loc[mask, 'vibRate_f0'] - df.loc[mask, 'vibRate_fm_gt']

plt.figure()
plt.scatter(
    df.loc[mask, 'phaseDiff_gt'],
    rate_error,
    alpha=0.6
)
plt.axhline(0)
plt.xlabel('AM–FM phase difference (rad)')
plt.ylabel('FM rate error (Hz)')
plt.title('FM Rate Error vs Phase Difference')
plt.savefig('fm_rate_error_vs_phaseDiff.png', dpi=300)
plt.show()
plt.close()

# df['vibExtent_ampPercent'] = df['vibExtent_ampPerc'].apply(lambda x: x.iloc[0])
plt.figure()
plt.scatter(
    df.loc[mask, 'amDepth_gt'],
    df.loc[mask, 'vibExtent_ampPerc'],
    alpha=0.6
)

plt.xlabel('Ground-truth vibrato extent (AmpPercent)')
plt.ylabel('Harmonic envelope extent (AmpPercent)')
plt.title('Harmonic Envelope AM Extent (AmpPercent) vs Vibrato Extent (AM, percent)')
plt.savefig('harmonic_extent_vs_vibExtentAmp.png', dpi=300)
plt.show()
plt.close()


### MADDE PARAMETER SWEEP
#I have Madde files in the form:
# f0_vibRateFM_vibExtentFM_vibRateAM_vibExtentFM
# The vibRateAM and vibExtentAM are performed differently from the artificial signal
import os
import glob
import pandas as pd
path = os.getcwd()
wav_files = glob.glob(os.path.join(path, "*.wav*"))#"*.xlsx"))
df = pd.DataFrame({})
for filename in wav_files:
    f0, vibRateFM, vibExtentFM, vibRateAM, vibExtentAM = filename.split('\\')[-1].split('.')[0].split('_')
    resultDict = {'filename':filename, 
              'f0':f0,
              'vibRateFM':vibRateFM,
              'vibExtentFM':vibExtentFM,
              'vibRateAM':vibRateAM,
              'vibExtentAM':vibExtentAM,
              }
    df = pd.concat([df, pd.DataFrame.from_records([resultDict])])
    
import pandas as pd
import soundfile as sf

results = []

for idx, row in df.iterrows():
    filename = row['filename']
    f_s = None

    # --- Load audio ---
    signal, f_s = sf.read(filename)
    # Avoid initial onset artifact
    begin25 = int(len(signal)*0.25)
    signal = signal[begin25:]

       ###Technically don't need a rolling window for an artificial signal, but let's see what happens
    pitch_contour, f_s_contour = returnContour(signal, f_s, model)
    meanFreq = np.mean(pitch_contour)
    # signal_norm = returnNormalized(highestPitch, samplerate)
    # signal_downsample = downsample_audio(highestPitch, samplerate, f_s_contour)
    envelope, filtered, f_s = returnEnvelope(signal, f_s, meanFreq)
    vibRate_f0 = fftVib3HzLocal(pitch_contour, f_s_contour)
    vibExtent_f0, __ , __ = compute_vibrato_extent_cents(pitch_contour, f_s_contour, vibRate_f0, windowFactor=0.75)#window factor of 0.75 the wavelength

    vibRate_amp = fftVib3HzLocal(envelope, f_s)
    vibExtent_amp = vibAmpPerc(envelope, f_s, vibRate_amp, 0.75)
    refRMS = np.nan
    # result = apply_vibTremorDecision_rolling_harmonics(VibratoSignal, f_s, model, refRMS, meanFreq,
                                          # window_duration=2, step_duration=2,
                                          # max_freq=2.5*meanFreq, calibrated=False)
    # vibRate_f0, vibExtent_f0, vibRate_amp, vibExtent_amp, vibExtent_SPL, vibExtent_dB, vibPercent, vibExtentPa_roll, vibExtentSPL_roll, harmonicSPL_mean  = result



    if ((vibRate_f0 - vibRate_amp < 0.5) & ~np.isnan([vibRate_f0,vibRate_amp]).any()):
        phaseDiff = extract_phase_difference(envelope, pitch_contour, f_s, f_s_contour, vibRate_f0, vibRate_amp)
        # Don't have a ground truth phase difference
        # phaseDiff_gt = am_phase - fm_phase
    if np.isnan([vibRate_f0,vibRate_amp]).any():
        phaseDiff = np.nan
    else: 
        phaseDiff = False
    
    # df_vibrato = process_file_amp(VibratoSignal, f_s, meanFreq, file_id='unknown', vibRate=vibRate_f0, vibExtent=vibExtent_f0)
    df_vibrato = analyze_vibrato_amp(envelope, f_s, f0, vibRate_f0=vibRate_f0, vibExtent_f0=vibExtent_f0, file_id="unknown", max_freq=2.5*meanFreq)

    # print(df_vibrato)
    # origMask = ((df_vibrato['type'] =='original') & (df_vibrato['metric'] == 'RMS Vibrato Extent'))
    # normMask0 = ((df_vibrato['type'] =='normalized') & (df_vibrato['metric'] == 'RMS Vibrato Extent'))
    harmMask = ((df_vibrato['type'] =='original') & (df_vibrato['metric'] == 'Instantaneous Amplitude'))
    # normMask1 = ((df_vibrato['type'] =='normalized') & (df_vibrato['metric'] == 'Instantaneous Amplitude'))
    
    
    result = {
        'filename': filename,
        'f0': row['f0'],
        'vibRateFM_gt': row['vibRateFM'],
        'vibExtentFM_gt': row['vibExtentFM'],
        'vibRateAM_gt': row['vibRateAM'],
        'vibExtentAM_gt': row['vibExtentAM'],
        'vibRate_f0':vibRate_f0,
        'vibExtent_f0':vibExtent_f0,
        'vibRate_amp':vibRate_amp,
        'vibExtent_amp':vibExtent_amp,
        'vibRate_amp_harm':df_vibrato.loc[harmMask,'vibRate_amp'],
        'vibExtent_harm':df_vibrato.loc[harmMask,'extent_pa'],
        'vibExtent_ampPerc':df_vibrato.loc[harmMask,'ampDepthPerc'].iloc[0],
        'phaseDiff': phaseDiff
    }

    results.append(result)

results_df = pd.DataFrame(results)

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# --- FM Vibrato Rate ---
plt.figure()
plt.scatter(
    results_df['vibRateFM_gt'],
    results_df['vibRate_f0'],
    alpha=0.6
)
lims = [results_df['vibRateFM_gt'].min(), results_df['vibRateFM_gt'].max()]
# plt.plot(lims, lims)
plt.xlabel('Ground-truth FM rate (Hz)')
plt.ylabel('Estimated FM rate (Hz)')
plt.title('FM Vibrato Rate: Estimated vs Ground Truth')
plt.savefig('fm_rate_est_vs_gt.png', dpi=300)
plt.show()
plt.close()

# --- FM Vibrato Extent ---
plt.figure()
plt.scatter(
    results_df['vibExtentFM_gt'],
    results_df['vibExtent_f0'],
    alpha=0.6
)
lims = [results_df['vibExtentFM_gt'].min(), results_df['vibExtentFM_gt'].max()]
# plt.plot(lims, lims)
plt.xlabel('Ground-truth vibrato extent (cents)')
plt.ylabel('Estimated vibrato extent (cents)')
plt.title('FM Vibrato Extent: Estimated vs Ground Truth')
plt.savefig('fm_extent_est_vs_gt.png', dpi=300)
plt.show()
plt.close()

# --- AM Rate ---
plt.figure()
plt.scatter(
    results_df['vibRateAM_gt'],
    results_df['vibRate_amp'],
    alpha=0.6
)
lims = [results_df['vibRateAM_gt'].min(), results_df['vibRateAM_gt'].max()]
# plt.plot(lims, lims)
plt.xlabel('Ground-truth AM rate (Hz)')
plt.ylabel('Estimated AM rate (Hz)')
plt.title('AM Rate: Estimated vs Ground Truth')
plt.savefig('am_rate_est_vs_gt.png', dpi=300)
plt.show()
plt.close()

# --- AM Depth / Extent ---
plt.figure()
plt.scatter(
    results_df['vibExtentAM_gt'],
    results_df['vibExtent_amp'],
    alpha=0.6
)
plt.xlabel('Ground-truth AM depth')
plt.ylabel('Estimated AM extent')
plt.title('AM Depth: Estimated vs Ground Truth')
plt.savefig('am_extent_est_vs_gt.png', dpi=300)
plt.show()
plt.close()

# --- FM Rate Error vs Phase Difference ---
# rate_error = results_df['vibRate_f0'] - results_df['vibRate_fm_gt']

# plt.figure()
# plt.scatter(
    # results_df['phaseDiff_gt'],
    # rate_error,
    # alpha=0.6
# )
# plt.axhline(0)
# plt.xlabel('AM–FM phase difference (rad)')
# plt.ylabel('FM rate error (Hz)')
# plt.title('FM Rate Error vs Phase Difference')
# plt.savefig('fm_rate_error_vs_phaseDiff.png', dpi=300)
# plt.show()
# plt.close()

# --- Harmonic Envelope AM Extent vs Vibrato Extent ---
# results_df['vibExtent_ampPercent'] = results_df['vibExtent_ampPerc'].apply(lambda x: x.iloc[0])

plt.figure()
plt.scatter(
    results_df['vibExtentAM_gt'],
    results_df['vibExtent_ampPerc'],
    alpha=0.6
)
plt.xlabel('Ground-truth vibrato extent (AmpPercent)')
plt.ylabel('Harmonic envelope extent (AmpPercent)')
plt.title('Harmonic Envelope AM Extent (AmpPercent) vs Vibrato Extent (AM, percent)')
plt.savefig('harmonic_extent_vs_vibExtentAmp.png', dpi=300)
plt.show()
plt.close()
