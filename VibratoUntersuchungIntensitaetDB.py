#PSEUDOCODE
#Load DataFrame
#Run through Code Sandbox for all Test2 files
#Calculate:
    #FINAL Trial 
        #Beginning Frame
        #End Frame
    #Maximum Frequency
    #Frames within a minor third of this frequency
        #Beginning Frame
        #End Frame
    #Middle 50% of sustained pitch
        #Beginning Frame
        #End Frame
    #Rolling Mean/Std Vibrato Frequency
    #Rolling Mean/Std Vibrato Amplitude
#Save all four measures

#Train K Nearest neighbors algorithm on stabil, labil, ohne
#Rerun the classifier on 90% of sustained pitch. Only record values if they return as "stable"


###Let's clean this up.

#Necessary Libraries:
from parselmouth.praat import call
from parselmouth import Sound 
import scipy
from scipy.signal import hilbert, butter, sosfilt, find_peaks, resample, find_peaks_cwt, resample_poly
from scipy.io.wavfile import read
import statsmodels.api as sm 
from statsmodels.tsa.stattools import acovf, acf
import matplotlib
import pandas as pd
from scipy.signal import savgol_filter
import pickle
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
from scipy.signal import butter, filtfilt
import seaborn as sns
import sys
import os
import glob
from scipy.io import wavfile


def autocorr(x):
    n = x.size
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm, mode='same')
    acorr = result[n//2 + 1:] / (x.var() * np.arange(n-1, n//2, -1))
    lag = np.abs(acorr).argmax() + 1
    r = acorr[lag-1]        
    # if np.abs(r) > 0.5:
      # print('Appears to be autocorrelated with r = {}, lag = {}'. format(r, lag))
    # else: 
      # print('Appears to be not autocorrelated')
    # fig, ax = subplots(4)
    # ax[0].plot(norm)
    # ax[1].plot(result)
    # ax[2].plot(acorr)
    # ax[3].plot(r)
    return r, lag

def tremorRateCalc(wavData, samplerate):
    analytic_signal = hilbert(wavData)
    amplitude_envelope = np.abs(analytic_signal)
    #resample takes number of samples. 
    #50 Hz would be number of samples at samplerate * 50/samplerate rounded to the nearest integer
    samples = round(len(amplitude_envelope)*50/samplerate)
    amplitude_env_downsampled = resample(amplitude_envelope, samples)
    # Local maxima of the envelopes were detected
    # using the peak-finding function (https://terpconnect.umd.edu
    # /~toh/spectrum/). 
    peaks = find_peaks(amplitude_env_downsampled, distance=round(50/10))[0]
    plt.close('all')
    plot(np.arange(len(amplitude_env_downsampled))/50,amplitude_env_downsampled)
    plot(peaks/50, amplitude_env_downsampled[peaks], 'x')
    # Peaks were assessed by downward zerocrossings in the smoothed first derivative using a pseudogaussian smoothing algorithm. A peak was classified as a group
    # of points with amplitude that exceeded the amplitude of
    # neighboring points on either side. The rate of tremor was
    # calculated by the total number of peaks divided by the duration
    # of the audio sample (Figure 1).
    sampleDuration = len(wavData)/samplerate
    tremorRate = len(peaks)/sampleDuration
    #Both samples show lower tremorRate than the Madde simulation.
    #Cut sample duration from first peak to last peak, 
        #then subtract the peaks by one.
    peakDuration = len(amplitude_env_downsampled[peaks[0]:peaks[-1]])/len(amplitude_env_downsampled)*sampleDuration
    tremorRate_final = len(peaks[:-1])/peakDuration
    return tremorRate_final

def vibratoCalc8(stableWavArray, samplerate):
#0.5 s window, 3Hz autoCorr
    sound = Sound(stableWavArray, samplerate)
    #Creates PRAAT sound file from .wav array, default 44100 Hz sampling frequency?
    pitch = call(sound, "To Pitch", 0.0, 60, 1000) #create a praat pitch object
    pitch_contour = pitch.selected_array['frequency']
    #Calculate the contour's sample rate from the differing array sizes
    f_s_contour = pitch.selected_array['frequency'].size/sound.values.size*sound.sampling_frequency
    #r, lag = autocorr(pitch_contour)
    #vibrato_Frequency = 1/lag*f_s_contour
    #if (r < 0.5) | (vibrato_Frequency > 12):
    #    vibrato_Frequency = np.nan
    window = math.ceil(0.5*f_s_contour) # 0.5 s * sampling frequency of pitch contour
    pandasContour = pd.Series(pitch_contour)
    rollingVib = pandasContour.rolling(window).apply(lambda x: autocorrVib3Hz(x, f_s_contour))
    vibrato_Frequency = rollingVib.mean()
    vibratoStd = rollingVib.std()
    vibratoPercentage = len(rollingVib[window:][rollingVib[window:].notna()])/len(rollingVib[window:])
    return vibrato_Frequency, vibratoPercentage, vibratoStd
    
###Massive Calculation
def vibratoCalcMF(stableWavArray, samplerate, gender=np.nan, windowSecs=0.5):
#0.5 s window, 3Hz autoCorr
    sound = Sound(stableWavArray, samplerate)
    #Creates PRAAT sound file from .wav array, default 44100 Hz sampling frequency?
    #We actually just need the single pitches for this calculation, between C-E
        #Let's go A-G
    if gender == 'männl.':
        pitch = call(sound, "To Pitch", 0.0, 100, 390) #c3-g4
    elif gender == 'weibl.':
        pitch = call(sound, "To Pitch", 0.0, 261, 784) #c4-g5
    else:
        pitch = call(sound, "To Pitch", 0.0, 60, 784) #c4-g5
    pitch_contour = pitch.selected_array['frequency']
    #Calculate the contour's sample rate from the differing array sizes
    f_s_contour = pitch.selected_array['frequency'].size/sound.values.size*sound.sampling_frequency
    #r, lag = autocorr(pitch_contour)
    #vibrato_Frequency = 1/lag*f_s_contour
    #if (r < 0.5) | (vibrato_Frequency > 12):
    #    vibrato_Frequency = np.nan
    window = math.ceil(windowSecs*f_s_contour) # 0.5 s * sampling frequency of pitch contour
    if window > pitch_contour.shape[0]:
        window = pitch_contour.shape[0]-1
    pandasContour = pd.Series(pitch_contour)
    rollingVib = pandasContour.rolling(window).apply(lambda x: autocorrVib3Hz(x, f_s_contour))
    rollingAmp = pandasContour.rolling(window).apply(lambda x: vibAmpRoll(x, f_s_contour,rollingVib, windowFactor=0.75))
    vibrato_Frequency = rollingVib.mean()
    vibratoStd = rollingVib.std()
    vibratoAmplitude = rollingAmp.mean()
    vibAmpStd = rollingAmp.std()
    vibFreqTotal = autocorrVib3Hz(pandasContour, f_s_contour)
    vibAmpTotal, vibAmpStdTotal = vibAmp(pandasContour, f_s_contour, vibFreqTotal, windowFactor=0.75)
    
    # vibFreq = vibrato_Frequency
    # wavLengthWindow = 0.75*1/vibFreq*f_s_contour
    # meanFreq = pitch_contour.mean()
    # maxPeaks = find_peaks(pitch_contour, distance=wavLengthWindow)[0]
    # prominences = scipy.signal.peak_prominences(pitch_contour, maxPeaks)[0]/2
    # contour_heights = pitch_contour[maxPeaks] - prominences
    # plt.close()
    # plt.plot(pitch_contour)
    # plt.plot(maxPeaks, pitch_contour[maxPeaks], "x")
    # plt.ylabel('Frequency (Hz)')
    # plt.title('Amplitude Calculation')
    # plt.vlines(x=maxPeaks, ymin=contour_heights, ymax=pitch_contour[maxPeaks],color='r')
    # plt.show()
    # print(str(vibratoAmplitude) + ' cents')
    # prompt = input("Press Enter to continue...")
    # plt.close()
    
    vibratoPercentage = len(rollingVib[window:][rollingVib[window:].notna()])/len(rollingVib[window:])
    #vibratoAmplitude, vibAmpStd = vibAmp(pitch_contour, f_s_contour, vibrato_Frequency, windowFactor=0.33)
    #Amplitude Calculations
    intensityAmplitude = np.nan#intensityAmp(stableWavArray, samplerate, vibrato_Frequency, windowFactor=0.33)
    #ampWindow = math.ceil(windowSecs*samplerate)
    #pandasContourAmp = pd.Series(stableWavArray)
    #rollingVibAmp = pandasContourAmp.rolling(ampWindow).apply(lambda x: autocorrVib3HzAmp(x, samplerate))
    #amplitudeFreq_rolling = rollingVibAmp.mean()
    #amplitudeFreq_simple = np.nan#autocorrVib3HzAmp(stableWavArray, samplerate)
    
    return vibrato_Frequency, vibratoPercentage, vibratoStd, vibratoAmplitude, vibAmpStd, vibFreqTotal, vibAmpTotal, vibAmpStdTotal

def predictVibFrame(rate, extent, non_normedVibArray, classifier):
    frameArray = np.array([rate, extent]).reshape(1,-1)
    #print(frameArray)
    if np.isnan(frameArray).any():
        return 0
    vibState = classifier.predict(preprocessing.StandardScaler().fit(non_normedVibArray).transform(frameArray))[0]
    return vibState

###Training Data Calculation
def vibratoCalcTraining(stableWavArray, samplerate, normedTrainingSet, classifier, gender=np.nan, windowSecs=0.5):
    ###Training Data Calculation
    #0.5 s window, 3Hz autoCorr
    sound = Sound(stableWavArray, samplerate)
    #sound = Sound(highestPitch, samplerate)
    #gender=geschlecht
    #windowSecs = 1#0.5, let's try one second.
    #Creates PRAAT sound file from .wav array, default 44100 Hz sampling frequency?
    #We actually just need the single pitches for this calculation, between C-E
        #Let's go A-G
    if gender == 'männl.':
        pitch = call(sound, "To Pitch", 0.0, 100, 390) #c3-g4
    elif gender == 'weibl.':
        pitch = call(sound, "To Pitch", 0.0, 261, 784) #c4-g5
    else:
        pitch = call(sound, "To Pitch", 0.0, 60, 784) #c4-g5
    pitch_contour = pitch.selected_array['frequency']
    #Calculate the contour's sample rate from the differing array sizes
    f_s_contour = pitch.selected_array['frequency'].size/sound.values.size*sound.sampling_frequency
    #r, lag = autocorr(pitch_contour)
    #vibrato_Frequency = 1/lag*f_s_contour
    #if (r < 0.5) | (vibrato_Frequency > 12):
    #    vibrato_Frequency = np.nan
    window = math.ceil(windowSecs*f_s_contour) # 0.5 s * sampling frequency of pitch contour
    if window > pitch_contour.shape[0]:
        window = pitch_contour.shape[0]-1
    pandasContour = pd.Series(pitch_contour)
    rollingDF = pd.DataFrame({})
    #This only does one way
    rollingDF['rollingVib'] = pandasContour.rolling(window).apply(lambda x: autocorrVibNoThresh(x, f_s_contour))
    #rollingDF['rollingVib'] = pandasContour.rolling(window).apply(lambda x: autocorrVibNoThresh(x, f_s_contour))
    #Calculate the initial window backwards by taking final "window" frames of pitch contour calculated in reverse
    rollingDF['rollingVib'].iloc[:window] = pandasContour[::-1].rolling(window).apply(lambda x: autocorrVibNoThresh(x, f_s_contour)).iloc[(len(pandasContour)-window):]
    rollingDF['rollingVib'].iloc[:window] = rollingDF['rollingVib'].iloc[:window][::-1]

    rollingDF['rollingAmp'] = pandasContour.rolling(window).apply(lambda x: vibAmpRollNoThresh(x, f_s_contour,rollingDF['rollingVib'], windowFactor=0.75))
    rollingDF['rollingAmp'].iloc[:window] = pandasContour[::-1].rolling(window).apply(lambda x: vibAmpRollNoThresh(x, f_s_contour,rollingDF['rollingVib'], windowFactor=0.75)).iloc[(len(pandasContour)-window):]
    rollingDF['rollingAmp'].iloc[:window] = rollingDF['rollingAmp'].iloc[:window][::-1]

    #Check Non-Vibrato:
    rollingDF['vibState'] = rollingDF[['rollingVib', 'rollingAmp']].apply(lambda x: predictVibFrame(x.rollingVib, x.rollingAmp, X_0[:,:2], classifier), axis=1)
    
    #Let's only calculate values for windows classified as vibrato:
    vibrato_Frequency = rollingDF[rollingDF['vibState'] == 1]['rollingVib'].mean()
    vibratoStd = rollingDF[rollingDF['vibState'] == 1]['rollingVib'].std()
    vibratoAmplitude = rollingDF[rollingDF['vibState'] == 1]['rollingAmp'].mean()
    vibAmpStd = rollingDF[rollingDF['vibState'] == 1]['rollingAmp'].std()
    #vibFreqTotal = autocorrVib3Hz(pandasContour, f_s_contour)
    #vibAmpTotal, vibAmpStdTotal = vibAmp(pandasContour, f_s_contour, vibFreqTotal, windowFactor=0.75)
    
    # vibFreq = vibrato_Frequency
    # wavLengthWindow = 0.75*1/vibFreq*f_s_contour
    # meanFreq = pitch_contour.mean()
    # maxPeaks = find_peaks(pitch_contour, distance=wavLengthWindow)[0]
    # prominences = scipy.signal.peak_prominences(pitch_contour, maxPeaks)[0]/2
    # contour_heights = pitch_contour[maxPeaks] - prominences
    # plt.close()
    # plt.plot(pitch_contour)
    # plt.plot(maxPeaks, pitch_contour[maxPeaks], "x")
    # plt.ylabel('Frequency (Hz)')
    # plt.title('Amplitude Calculation')
    # plt.vlines(x=maxPeaks, ymin=contour_heights, ymax=pitch_contour[maxPeaks],color='r')
    # plt.show()
    # print(str(vibratoAmplitude) + ' cents')
    # prompt = input("Press Enter to continue...")
    # plt.close()
    
    #vibratoPercentage = len(rollingVib[window:][rollingVib[window:].notna()])/len(rollingVib[window:])
    vibratoPercentage = len(rollingDF[rollingDF['vibState'] == 1])/len(rollingDF)
    #vibratoAmplitude, vibAmpStd = vibAmp(pitch_contour, f_s_contour, vibrato_Frequency, windowFactor=0.33)
    #Amplitude Calculations
    #intensityAmplitude = np.nan#intensityAmp(stableWavArray, samplerate, vibrato_Frequency, windowFactor=0.33)
    #ampWindow = math.ceil(windowSecs*samplerate)
    #pandasContourAmp = pd.Series(stableWavArray)
    #rollingVibAmp = pandasContourAmp.rolling(ampWindow).apply(lambda x: autocorrVib3HzAmp(x, samplerate))
    #amplitudeFreq_rolling = rollingVibAmp.mean()
    #amplitudeFreq_simple = np.nan#autocorrVib3HzAmp(stableWavArray, samplerate)
    
    return vibrato_Frequency, vibratoStd, vibratoAmplitude, vibAmpStd, vibratoPercentage


###Training Data Calculation
def vibratoCalcTrainingSimple(stableWavArray, samplerate, normedTrainingSet, classifier, gender=np.nan, windowSecs=0.5):
    ###Training Data Calculation
    #0.5 s window, 3Hz autoCorr
    sound = Sound(stableWavArray, samplerate)
    #sound = Sound(highestPitch, samplerate)
    #gender=geschlecht
    #windowSecs = 1#0.5, let's try one second.
    #Creates PRAAT sound file from .wav array, default 44100 Hz sampling frequency?
    #We actually just need the single pitches for this calculation, between C-E
        #Let's go A-G
    if gender == 'männl.':
        pitch = call(sound, "To Pitch", 0.0, 100, 390) #c3-g4
    elif gender == 'weibl.':
        pitch = call(sound, "To Pitch", 0.0, 261, 784) #c4-g5
    else:
        pitch = call(sound, "To Pitch", 0.0, 60, 784) #c4-g5
    pitch_contour = pitch.selected_array['frequency']
    #Calculate the contour's sample rate from the differing array sizes
    f_s_contour = pitch.selected_array['frequency'].size/sound.values.size*sound.sampling_frequency
    #r, lag = autocorr(pitch_contour)
    #vibrato_Frequency = 1/lag*f_s_contour
    #if (r < 0.5) | (vibrato_Frequency > 12):
    #    vibrato_Frequency = np.nan
    window = math.ceil(windowSecs*f_s_contour) # 0.5 s * sampling frequency of pitch contour
    if window > pitch_contour.shape[0]:
        window = pitch_contour.shape[0]-1
    
    vibrato_frequency =  autocorrVibNoThresh(pitch_contour, f_s_contour)
    vibratoAmplitude = vibAmpRollNoThresh(x, f_s_contour,rollingDF['rollingVib'], windowFactor=0.75).iloc[(len(pandasContour)-window):]
    
    pandasContour = pd.Series(pitch_contour)
    rollingDF = pd.DataFrame({})
    #This only does one way
    rollingDF['rollingVib'] = pandasContour.rolling(window).apply(lambda x: autocorrVibNoThresh(x, f_s_contour))
    #rollingDF['rollingVib'] = pandasContour.rolling(window).apply(lambda x: autocorrVibNoThresh(x, f_s_contour))
    #Calculate the initial window backwards by taking final "window" frames of pitch contour calculated in reverse
    rollingDF['rollingVib'].iloc[:window] = pandasContour[::-1].rolling(window).apply(lambda x: autocorrVibNoThresh(x, f_s_contour)).iloc[(len(pandasContour)-window):]
    rollingDF['rollingVib'].iloc[:window] = rollingDF['rollingVib'].iloc[:window][::-1]

    rollingDF['rollingAmp'] = pandasContour.rolling(window).apply(lambda x: vibAmpRollNoThresh(x, f_s_contour,rollingDF['rollingVib'], windowFactor=0.75))
    rollingDF['rollingAmp'].iloc[:window] = pandasContour[::-1].rolling(window).apply(lambda x: vibAmpRollNoThresh(x, f_s_contour,rollingDF['rollingVib'], windowFactor=0.75)).iloc[(len(pandasContour)-window):]
    rollingDF['rollingAmp'].iloc[:window] = rollingDF['rollingAmp'].iloc[:window][::-1]

    #Check Non-Vibrato:
    rollingDF['vibState'] = rollingDF[['rollingVib', 'rollingAmp']].apply(lambda x: predictVibFrame(x.rollingVib, x.rollingAmp, X_0[:,:2], classifier), axis=1)
    
    #Let's only calculate values for windows classified as vibrato:
    vibrato_Frequency = rollingDF[rollingDF['vibState'] == 1]['rollingVib'].mean()
    vibratoStd = rollingDF[rollingDF['vibState'] == 1]['rollingVib'].std()
    vibratoAmplitude = rollingDF[rollingDF['vibState'] == 1]['rollingAmp'].mean()
    vibAmpStd = rollingDF[rollingDF['vibState'] == 1]['rollingAmp'].std()
    #vibFreqTotal = autocorrVib3Hz(pandasContour, f_s_contour)
    #vibAmpTotal, vibAmpStdTotal = vibAmp(pandasContour, f_s_contour, vibFreqTotal, windowFactor=0.75)
    
    # vibFreq = vibrato_Frequency
    # wavLengthWindow = 0.75*1/vibFreq*f_s_contour
    # meanFreq = pitch_contour.mean()
    # maxPeaks = find_peaks(pitch_contour, distance=wavLengthWindow)[0]
    # prominences = scipy.signal.peak_prominences(pitch_contour, maxPeaks)[0]/2
    # contour_heights = pitch_contour[maxPeaks] - prominences
    # plt.close()
    # plt.plot(pitch_contour)
    # plt.plot(maxPeaks, pitch_contour[maxPeaks], "x")
    # plt.ylabel('Frequency (Hz)')
    # plt.title('Amplitude Calculation')
    # plt.vlines(x=maxPeaks, ymin=contour_heights, ymax=pitch_contour[maxPeaks],color='r')
    # plt.show()
    # print(str(vibratoAmplitude) + ' cents')
    # prompt = input("Press Enter to continue...")
    # plt.close()
    
    #vibratoPercentage = len(rollingVib[window:][rollingVib[window:].notna()])/len(rollingVib[window:])
    vibratoPercentage = len(rollingDF[rollingDF['vibState'] == 1])/len(rollingDF)
    #vibratoAmplitude, vibAmpStd = vibAmp(pitch_contour, f_s_contour, vibrato_Frequency, windowFactor=0.33)
    #Amplitude Calculations
    #intensityAmplitude = np.nan#intensityAmp(stableWavArray, samplerate, vibrato_Frequency, windowFactor=0.33)
    #ampWindow = math.ceil(windowSecs*samplerate)
    #pandasContourAmp = pd.Series(stableWavArray)
    #rollingVibAmp = pandasContourAmp.rolling(ampWindow).apply(lambda x: autocorrVib3HzAmp(x, samplerate))
    #amplitudeFreq_rolling = rollingVibAmp.mean()
    #amplitudeFreq_simple = np.nan#autocorrVib3HzAmp(stableWavArray, samplerate)
    
    return vibrato_Frequency, vibratoStd, vibratoAmplitude, vibAmpStd, vibratoPercentage




def autocorrVibNoThresh(pitch_contour, f_s_contour):
    x = pitch_contour
    n = len(x)
    acorr = sm.tsa.acf(x, nlags = n-1)
    #95% Confidence interval is +- 1.96/math.sqrt(n)
    ###DON'T NEED FOR TRAINING DATA
    #highCI = 1.96/math.sqrt(n)
    #lowCI = -highCI
    #Desired range is 3 Hz - 10 Hz
        #=> Desired period is 1/10 s - 1/4 s
        #=> Desired lag times in frames are:
            #{1/10*f_s_contour:1/4*f_s_contour}
    frame10Hz = math.floor(1/10*f_s_contour)
    frame3Hz = math.floor(1/3*f_s_contour)
    #lag = np.abs(acorr)[frame12Hz:frame3Hz].argmax() + 1 + frame12Hz
    #maxLag = covariance[frame10Hz:frame4Hz].argmax() + frame10Hz
    maxLag = acorr[frame10Hz:frame3Hz].argmax() + frame10Hz
    vibratoFreq = 1/maxLag*f_s_contour
    #if (acorr[:maxLag].min() < lowCI) & (acorr[maxLag] > highCI):
    #    vibratoFreq = 1/maxLag*f_s_contour
    #else:
    #    vibratoFreq = np.nan
    #r = acorr[lag-1]        
    # if np.abs(r) > 0.5:
      # print('Appears to be autocorrelated with r = {}, lag = {}'. format(r, lag))
    # else: 
      # print('Appears to be not autocorrelated')
    #fig, ax = subplots(2)
    #ax.plot(1/np.arange(len(covariance[frame12Hz:frame3Hz]))*f_s_contour,covariance[frame12Hz:frame3Hz])
    #ax[0].plot(np.arange(len(acorr[frame10Hz:frame4Hz]))+frame10Hz, acorr[frame10Hz:frame4Hz])
    #ax[0].plot(np.arange(len(acorr)), acorr)
    #pd.plotting.autocorrelation_plot(pitch_contour, ax=ax[1])
    #plt.vline
    #prompt = input("Press Enter to continue...")
    #plt.close()
    # ax[1].plot(result)
    # ax[2].plot(acorr)
    # ax[3].plot(r)
    return vibratoFreq

def autocorrVib3Hz(pitch_contour, f_s_contour):
    x = pitch_contour
    n = len(x)
    acorr = sm.tsa.acf(x, nlags = n-1)
    #95% Confidence interval is +- 1.96/math.sqrt(n)
    highCI = 1.96/math.sqrt(n)
    lowCI = -highCI
    #Desired range is 3 Hz - 10 Hz
        #=> Desired period is 1/10 s - 1/4 s
        #=> Desired lag times in frames are:
            #{1/10*f_s_contour:1/4*f_s_contour}
    frame10Hz = math.floor(1/10*f_s_contour)
    frame3Hz = math.floor(1/3*f_s_contour)
    #lag = np.abs(acorr)[frame12Hz:frame3Hz].argmax() + 1 + frame12Hz
    #maxLag = covariance[frame10Hz:frame4Hz].argmax() + frame10Hz
    maxLag = acorr[frame10Hz:frame3Hz].argmax() + frame10Hz
    if (acorr[:maxLag].min() < lowCI) & (acorr[maxLag] > highCI):
        vibratoFreq = 1/maxLag*f_s_contour
    else:
        vibratoFreq = np.nan
    #r = acorr[lag-1]        
    # if np.abs(r) > 0.5:
      # print('Appears to be autocorrelated with r = {}, lag = {}'. format(r, lag))
    # else: 
      # print('Appears to be not autocorrelated')
    #fig, ax = subplots(2)
    #ax.plot(1/np.arange(len(covariance[frame12Hz:frame3Hz]))*f_s_contour,covariance[frame12Hz:frame3Hz])
    #ax[0].plot(np.arange(len(acorr[frame10Hz:frame4Hz]))+frame10Hz, acorr[frame10Hz:frame4Hz])
    #ax[0].plot(np.arange(len(acorr)), acorr)
    #pd.plotting.autocorrelation_plot(pitch_contour, ax=ax[1])
    #plt.vline
    #prompt = input("Press Enter to continue...")
    #plt.close()
    # ax[1].plot(result)
    # ax[2].plot(acorr)
    # ax[3].plot(r)
    return vibratoFreq
    
def autocorrVib3HzLocal(pitch_contour, f_s_contour):
    x = pitch_contour
    n = len(x)
    acorr = sm.tsa.acf(x, nlags=n-1)
    
    # 95% Confidence interval
    highCI = 1.96 / math.sqrt(n)
    lowCI = -highCI

    # Desired frequency range: 3–10 Hz
    frame10Hz = math.floor(f_s_contour / 10)
    frame3Hz = math.floor(f_s_contour / 3)

    # Find local maxima in desired range
    acorr_segment = acorr[frame10Hz:frame3Hz]
    peaks, _ = find_peaks(acorr_segment)

    vibratoFreq = np.nan
    if len(peaks) > 0:
        # Get correlation values for each peak
        peak_values = acorr_segment[peaks]
        
        # Filter only those above the confidence interval
        valid_indices = np.where(peak_values > highCI)[0]
        
        if len(valid_indices) > 0:
            # Find the *highest* local maximum
            best_peak_idx = valid_indices[np.argmax(peak_values[valid_indices])]
            maxLag = peaks[best_peak_idx] + frame10Hz
            vibratoFreq = f_s_contour / maxLag
        # else: vibratoFreq remains np.nan

    return vibratoFreq



def vibratoCalcAmplitude(stableWavArray, samplerate):
#0.5 s window, 3Hz autoCorr
    sample = stableWavArray
    window = math.ceil(0.5*samplerate) # 0.5 s * sampling frequency of pitch contour
    pandasContour = pd.Series(sample)
    rollingVib = pandasContour.rolling(window).apply(lambda x: autocorrVib3HzAmp(x, samplerate))
    vibrato_Frequency = rollingVib.mean()
    vibratoStd = rollingVib.std()
    vibratoPercentage = len(rollingVib[window:][rollingVib[window:].notna()])/len(rollingVib[window:])
    return vibrato_Frequency, vibratoPercentage, vibratoStd

def autocorrVib3HzAmp(sustainedAudio, f_s):
    x = np.abs(hilbert(sustainedAudio))
    n = len(x)
    acorr = sm.tsa.acf(x, nlags = n-1)
    #95% Confidence interval is +- 1.96/math.sqrt(n)
    highCI = 1.96/math.sqrt(n)
    lowCI = -highCI
    #Desired range is 3 Hz - 10 Hz
        #=> Desired period is 1/10 s - 1/4 s
        #=> Desired lag times in frames are:
            #{1/10*f_s_contour:1/4*f_s_contour}
    frame10Hz = math.floor(1/10*f_s)
    frame3Hz = math.floor(1/3*f_s)
    #lag = np.abs(acorr)[frame12Hz:frame3Hz].argmax() + 1 + frame12Hz
    #maxLag = covariance[frame10Hz:frame4Hz].argmax() + frame10Hz
    maxLag = acorr[frame10Hz:frame3Hz].argmax() + frame10Hz
    if (acorr[:maxLag].min() < lowCI) & (acorr[maxLag] > highCI):
        vibratoFreq = 1/maxLag*f_s
    else:
        vibratoFreq = np.nan
    #r = acorr[lag-1]        
    # if np.abs(r) > 0.5:
      # print('Appears to be autocorrelated with r = {}, lag = {}'. format(r, lag))
    # else: 
      # print('Appears to be not autocorrelated')
    #fig, ax = subplots(2)
    #ax.plot(1/np.arange(len(covariance[frame12Hz:frame3Hz]))*f_s_contour,covariance[frame12Hz:frame3Hz])
    #ax[0].plot(np.arange(len(acorr[frame10Hz:frame4Hz]))+frame10Hz, acorr[frame10Hz:frame4Hz])
    #ax[0].plot(np.arange(len(acorr)), acorr)
    #pd.plotting.autocorrelation_plot(pitch_contour, ax=ax[1])
    #plt.vline
    #prompt = input("Press Enter to continue...")
    #plt.close()
    # ax[1].plot(result)
    # ax[2].plot(acorr)
    # ax[3].plot(r)
    return vibratoFreq

def vibAmpOld(pitch_contour, f_s_contour, vibFreq, windowFactor=0):
    if np.isnan(vibFreq):
        #vibFreq = 5.5
        ampCents = np.nan
        ampStd = np.nan
        #print(str(ampCents))
        return ampCents, ampStd
    wavelength = 1.0/vibFreq*f_s_contour # in frames
    print('wavelength: ', str(wavelength))
    try:
        window = math.floor(wavelength*windowFactor)
    except ValueError:
        window = math.floor(1.0/5.5*f_s_contour*0.75)
    #
    if window == 0:
        window = 1
    print('Window: ', str(window))
    maxPeaks = find_peaks(pitch_contour, distance=window)[0]
    prominences = scipy.signal.peak_prominences(pitch_contour, maxPeaks)[0]/2
    #minPeaks = find_peaks(pitch_contour*-1,distance=window)[0]
    #maxMean = pitch_contour[maxPeaks].mean()
    #minMean = pitch_contour[minPeaks].mean()
    #ampEstimate = (maxMean - minMean)/2 # in Hz
    ampEstimate = prominences.mean()
    meanFreq = pitch_contour.mean()
    ampStd = prominences.std()
    #Now we need that in cents. How?
    #100 cent is a HALF step(?)
    #(x)^n*f_0 = 2*f_0, for the full octave where n is 12*100
    #25 cent is 33/32
    #x^25*f_0 = 33/32*f_0
    #=> x^1200 = 2
        #=> 1200*ln(x) = 2
        #=> x = e^(2/1200)
    cent = np.power(math.e,(np.log(2)/1200)) # Ok, so this is an even tempered cent for the octave
    #cent = 1.0005777895065548
    #ampEstimate in Hz
    #meanFreq + ampEstimate = meanFreq*(cent)^n where n is the number of cents
    #cent^n = (meanFreq + ampEstimate)/meanFreq
    #n*ln(cent) = ln(1 + ampEstimate/meanFreq)
    #n = ln(1 + ampEstimate/meanFreq)/ln(cent)
    ampCents = 1200*np.log(1 + ampEstimate/meanFreq)/np.log(2)
    ampStd = 1200*np.log(1 + ampStd/meanFreq)/np.log(2)
    return ampCents, ampStd # amplitude in cents

def vibAmpRoll(pitch_contour, f_s_contour, rollingVib, windowFactor=0):
    vibFreq = rollingVib[pitch_contour.index.max()]
    if math.isnan(vibFreq):
        #vibFreq = 5.5
        ampCents = np.nan
        #print(str(ampCents))
        return ampCents
    wavelength = 1.0/vibFreq*f_s_contour # in frames
    try:
        window = math.floor(wavelength*windowFactor)
    except ValueError:
        window = math.floor(1.0/5.5*f_s_contour*0.75)
    #
    if window == 0:
        window = 1
    maxPeaks = find_peaks(pitch_contour, distance=window)[0]
    prominences = scipy.signal.peak_prominences(pitch_contour, maxPeaks)[0]/2
    #minPeaks = find_peaks(pitch_contour*-1,distance=window)[0]
    #maxMean = pitch_contour[maxPeaks].mean()
    #minMean = pitch_contour[minPeaks].mean()
    #ampEstimate = (maxMean - minMean)/2 # in Hz
    #Do not calculate with the first and final peaks.
    ampEstimate = prominences[1:-1].mean()
    meanFreq = pitch_contour.mean()
    ampStd = prominences[1:-2].std()
    #
    ampCents = 1200*np.log(1 + ampEstimate/meanFreq)/np.log(2)
    #vibStd = 1200*np.log(1 + ampStd/meanFreq)/np.log(2)
    #if vibFreq == np.nan:
    #    ampCents = np.nan
    return ampCents#, vibStd # amplitude in cents
    

def vibAmpRollNoThresh(pitch_contour, f_s_contour, rollingVib, windowFactor=0):
    #Let's take the highest vibrato rate possible to then have the smallest wavelength threshold possible.
    vibFreq = rollingVib.max()#[pitch_contour.index.max()]
    #if math.isnan(vibFreq):
    #    #vibFreq = 5.5
    #    ampCents = np.nan
    #    #print(str(ampCents))
    #    return ampCents
    wavelength = 1.0/vibFreq*f_s_contour # in frames
    try:
        window = math.floor(wavelength*windowFactor)
    except ValueError:
        window = math.floor(1.0/5.5*f_s_contour*0.75)
    #
    if window == 0:
        window = 1
    maxPeaks = find_peaks(pitch_contour, distance=window)[0]
    prominences = scipy.signal.peak_prominences(pitch_contour, maxPeaks)[0]/2
    #minPeaks = find_peaks(pitch_contour*-1,distance=window)[0]
    #maxMean = pitch_contour[maxPeaks].mean()
    #minMean = pitch_contour[minPeaks].mean()
    #ampEstimate = (maxMean - minMean)/2 # in Hz
    #Do not calculate with the first and final peaks.
    ampEstimate = prominences[1:-1].mean()
    meanFreq = pitch_contour.mean()
    ampStd = prominences[1:-2].std()
    #
    ampCents = 1200*np.log(1 + ampEstimate/meanFreq)/np.log(2)
    #vibStd = 1200*np.log(1 + ampStd/meanFreq)/np.log(2)
    #if vibFreq == np.nan:
    #    ampCents = np.nan
    return ampCents#, vibStd # amplitude in cents

def intensityAmp(stableWav, samplerate, vibFreq, windowFactor=0):
    ampEnv = np.abs(stableWav)
    if vibFreq == np.nan:
        vibFreq = 5.5
    wavelength = 1/vibFreq*samplerate # in frames
    try:
        window = math.floor(wavelength*windowFactor)
    except ValueError:
        window = math.floor(1.0/5.5*f_s_contour*0.33)
    if window == 0:
        window = 1
    maxPeaks = find_peaks(ampEnv, distance=window)[0]
    minPeaks = find_peaks(ampEnv*-1,distance=window)[0]
    maxMean = ampEnv[maxPeaks].mean()
    minMean = ampEnv[minPeaks].mean()
    ampEstimate = (maxMean - minMean)/2 # in Hz
    meanAmplitude = ampEnv.mean()
    dB = np.np.log10(ampEstimate/meanAmplitude + 1) # np.np.log10((ampEstimate+meanAmplitude)/meanAmplitude)
    return dB # amplitude in cents

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
    
#wavFilename = wavFilePath
#samplerate, data = read(wavFilename)
def selectMiddleTrial(wavFilename):
    #load wav file.
    #wavFilename = wavFilename
    samplerate, data = read(wavFilename)
    #Load audio file
    #If audio is stereo, take the left channel
    try:
        if data.shape[1] > 1:
            data = data[:,0]
    except:
        pass
    #There are some local zeros. Take rolling average and find max closest to midpoint
    analytic_signal = hilbert(data)
    amplitude_envelope = np.abs(analytic_signal)
    #Need a big moving average to not cut out mid-trial lows.
    #Let's use a half-second moving average (samplerate/2)
    a = moving_average(amplitude_envelope, n=round(samplerate/2))
    #Try to find larger peaks
    #Set all values where the amplitude_envelope is a factor of 2^3 lower to zero (-3dB)
    #Why -3 dB? Had one error, so let's lower the threshold to -4 dB
    a1 = np.where(a < a.max()/math.pow(2,3), 0, a) 
    #Could be a problem with clipping^
    
    ###Find the amplitude envelope closest to the midpoint.
        #Could be a nonzero section with dB >-3 of max or a section closest to it.
        #If you find the sections where amplitude envelope != 0, you have an array of x tests
        #Choose the middle one.
    peaks = find_peaks(a1)
    #Calculate the distance from each to the midpoint
    distance_from_midpoint = abs(peaks[0] - round(len(a1)/2))
    minIndex = peaks[0][np.argmin(distance_from_midpoint)]
    #Ok, that gives us the local max closest to the midpoint.
    #Set zeros of audio intensity to -4 dB below this point.
    a = np.where(a < a[minIndex]/math.pow(2,4), 0, a)
    #How do we find the section of the signal around that local max?
    #First, find zero prior to that point.
    if np.where(a[:minIndex] == 0)[0].size != 0:
        startMiddleAttempt = np.where(a[:minIndex]==0)[0][-1]
    else:
        startMiddleAttempt = 0  
    
    #Then find zero after that point. 
        #In one case, the recording ends before a zero.
    if np.where(a[minIndex:] == 0)[0].size != 0:
        finishMiddleAttempt = minIndex + np.where(a[minIndex:]==0)[0][0]
    else:
        finishMiddleAttempt = len(data)
    selectedMiddleTrial = data[startMiddleAttempt:finishMiddleAttempt]
    #visualCheckSelection(data, startMiddleAttempt, finishMiddleAttempt)
    #prompt = input("Press Enter to continue...")
    #if #prompt == 'q':
    #    break
    return samplerate, selectedMiddleTrial
    
  
#wavFilename = wavFilePath
#samplerate, data = read(wavFilename)
def selectMiddleTrial(wavFilename):
    #load wav file.
    #wavFilename = wavFilename
    samplerate, data = read(wavFilename)
    #Load audio file
    #If audio is stereo, take the left channel
    try:
        if data.shape[1] > 1:
            data = data[:,0]
    except:
        pass
    #There are some local zeros. Take rolling average and find max closest to midpoint
    analytic_signal = hilbert(data)
    amplitude_envelope = np.abs(analytic_signal)
    #Need a big moving average to not cut out mid-trial lows.
    #Let's use a half-second moving average (samplerate/2)
    a = moving_average(amplitude_envelope, n=round(samplerate/2))
    #Try to find larger peaks
    #Set all values where the amplitude_envelope is a factor of 2^3 lower to zero (-3dB)
    #Why -3 dB? Had one error, so let's lower the threshold to -4 dB
    a1 = np.where(a < a.max()/math.pow(2,3), 0, a) 
    #Could be a problem with clipping^
    
    ###Find the amplitude envelope closest to the midpoint.
        #Could be a nonzero section with dB >-3 of max or a section closest to it.
        #If you find the sections where amplitude envelope != 0, you have an array of x tests
        #Choose the middle one.
    peaks = find_peaks(a1)
    #Calculate the distance from each to the midpoint
    distance_from_midpoint = abs(peaks[0] - round(len(a1)/2))
    minIndex = peaks[0][np.argmin(distance_from_midpoint)]
    #Ok, that gives us the local max closest to the midpoint.
    #Set zeros of audio intensity to -4 dB below this point.
    a = np.where(a < a[minIndex]/math.pow(2,4), 0, a)
    #How do we find the section of the signal around that local max?
    #First, find zero prior to that point.
    if np.where(a[:minIndex] == 0)[0].size != 0:
        startMiddleAttempt = np.where(a[:minIndex]==0)[0][-1]
    else:
        startMiddleAttempt = 0  
    
    #Then find zero after that point. 
        #In one case, the recording ends before a zero.
    if np.where(a[minIndex:] == 0)[0].size != 0:
        finishMiddleAttempt = minIndex + np.where(a[minIndex:]==0)[0][0]
    else:
        finishMiddleAttempt = len(data)
    selectedMiddleTrial = data[startMiddleAttempt:finishMiddleAttempt]
    #visualCheckSelection(data, startMiddleAttempt, finishMiddleAttempt)
    #prompt = input("Press Enter to continue...")
    #if #prompt == 'q':
    #    break
    return samplerate, selectedMiddleTrial    
    
def selectFinalTrial(wavFilename):
    #load wav file.
    #wavFilename = wavFilename
    samplerate, data = read(wavFilename)
    #Load audio file
    #If audio is stereo, take the left channel
    try:
        if data.shape[1] > 1:
            data = data[:,0]
    except:
        pass
    #There are some local zeros. Take rolling average and find max closest to midpoint
    analytic_signal = hilbert(data)
    amplitude_envelope = np.abs(analytic_signal)
    #Need a big moving average to not cut out mid-trial lows.
    #Let's use a half-second moving average (samplerate/2)
    a = moving_average(amplitude_envelope, n=round(samplerate/2))
    #Try to find larger peaks
    #Set all values where the amplitude_envelope is a factor of 2^3 lower to zero (-3dB)
    #Why -3 dB? Had one error, so let's lower the threshold to -4 dB
    a1 = np.where(a < a.max()/math.pow(2,3), 0, a) 
    #Could be a problem with clipping^
    
    ###Find the amplitude envelope closest to the midpoint.
        #Could be a nonzero section with dB >-3 of max or a section closest to it.
        #If you find the sections where amplitude envelope != 0, you have an array of x tests
        #Choose the middle one.
    peaks = find_peaks(a1)
    #Calculate the distance from each to the end of the file
    distance_from_midpoint = abs(peaks[0] - len(a1))
    minIndex = peaks[0][np.argmin(distance_from_midpoint)]
    #Ok, that gives us the local max closest to the midpoint.
    #Set zeros of audio intensity to -4 dB below this point.
    a = np.where(a < a[minIndex]/math.pow(2,4), 0, a)
    #How do we find the section of the signal around that local max?
    #First, find zero prior to that point.
    if np.where(a[:minIndex] == 0)[0].size != 0:
        startFinalAttempt = np.where(a[:minIndex]==0)[0][-1]
    else:
        startFinalAttempt = 0  
    
    #Then find zero after that point. 
        #In one case, the recording ends before a zero.
    if np.where(a[minIndex:] == 0)[0].size != 0:
        finishFinalAttempt = minIndex + np.where(a[minIndex:]==0)[0][0]
    else:
        finishFinalAttempt = len(data)
    selectedFinalTrial = data[startFinalAttempt:finishFinalAttempt]
    #visualCheckSelection(data, startMiddleAttempt, finishMiddleAttempt)
    #prompt = input("Press Enter to continue...")
    #if #prompt == 'q':
    #    break
    return samplerate, selectedFinalTrial

def isolateHighestPitch50MF(samplerate, selectedMiddleTrial, gender=np.nan):
    #Can we get the pitch contour?
    sound = Sound(selectedMiddleTrial, samplerate)
    #Create a praat pitch object,
    #Probably need upper frequency bound 2x potential sung frequency
    #Piano key frequencies:
    #g5: 784
    #g4: 392
    #c4: 261
    #c3: 131
    if gender == 'männl.':
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c3-g4
    elif gender == 'weibl.':
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c4-g5
    else:
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c4-g5
    #This provides the frequencies of the sample.
    pitch_contour = pitch.selected_array['frequency']
    #What is the new samplingrate?
    f_s_Audio = sound.sampling_frequency
    wavLength = sound.values.size
    pitchContLength = pitch.selected_array['frequency'].size
    f_s_contour = pitchContLength/wavLength*f_s_Audio
    #So we have an interval of a minor third between the highest note and the middle note. 
    #Yeah!
    ###Ok, now we want to find the corresponding interval in the pitch_contour array    
        #That are within a minor third of the maximum pitch. 
    #This is a little sensitive to pitch artifacts.
    #maxFreq = max(pitch_contour)
    #maxIndex = argmax(pitch_contour)
    #Let's just grab the middle value of the selection and hope.
    maxIndex = round(len(pitch_contour)/2)
    maxFreq = pitch_contour[maxIndex]
    #Minor 3rd ratio is 6:5
    thresholdFreq = maxFreq*5/6
    beginInterval = np.where(pitch_contour[:maxIndex] < thresholdFreq)[0][-1]
    if np.where(pitch_contour[maxIndex:] < thresholdFreq)[0].size != 0:
        endInterval = maxIndex + np.where(pitch_contour[maxIndex:] < thresholdFreq)[0][0]
    else:
        endInterval = len(pitch_contour)  
    #Let's take the middle fifty percent of this interval.
    #If you save the audio file here, you could use it for all data analysis.
    #close('all')
    begin50 = beginInterval + round((endInterval - beginInterval)*.25)
    #print(str(begin50))
    end50 =  beginInterval + round((endInterval - beginInterval)*.75)
    #print(str(end50))
    #visualCheckSelection(pitch_contour, begin50, end50)
    #prompt = input("Press Enter to continue...")
    #beginAudioInterval = startMiddleAttempt + round(begin50*f_s_Audio/f_s_contour)
    beginAudioInterval = round(begin50*f_s_Audio/f_s_contour) #+ startMiddleAttempt
    #endAudioInterval = startMiddleAttempt + round(end50*f_s_Audio/f_s_contour)
    endAudioInterval = round(end50*f_s_Audio/f_s_contour) #+ startMiddleAttempt  
    middleFiftyPercentHighestPitch = selectedMiddleTrial[beginAudioInterval:endAudioInterval]
    #Let's get the mean pitch of this interval
    meanFreq = pitch_contour[begin50:end50].mean()
    return samplerate, middleFiftyPercentHighestPitch, maxFreq, meanFreq

def isolateHighestPitch90(samplerate, selectedMiddleTrial, gender=np.nan):
    #Can we get the pitch contour?
    sound = Sound(selectedMiddleTrial, samplerate)
    #Create a praat pitch object,
    #Probably need upper frequency bound 2x potential sung frequency
    #Piano key frequencies:
    #g5: 784
    #g4: 392
    #c4: 261
    #c3: 131
    if gender == 'männl.':
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c3-g4
    elif gender == 'weibl.':
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c4-g5
    else:
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c4-g5
    #This provides the frequencies of the sample.
    pitch_contour = pitch.selected_array['frequency']
    #What is the new samplingrate?
    f_s_Audio = sound.sampling_frequency
    wavLength = sound.values.size
    pitchContLength = pitch.selected_array['frequency'].size
    f_s_contour = pitchContLength/wavLength*f_s_Audio
    #So we have an interval of a minor third between the highest note and the middle note. 
    #Yeah!
    ###Ok, now we want to find the corresponding interval in the pitch_contour array    
        #That are within a minor third of the maximum pitch. 
    #This is a little sensitive to pitch artifacts.
    #maxFreq = max(pitch_contour)
    #maxIndex = argmax(pitch_contour)
    #Let's just grab the middle value of the selection and hope.
    maxIndex = round(len(pitch_contour)/2)
    maxFreq = pitch_contour[maxIndex]
    #Minor 3rd ratio is 6:5
    thresholdFreq = maxFreq*5/6
    beginInterval = np.where(pitch_contour[:maxIndex] < thresholdFreq)[0][-1]
    if np.where(pitch_contour[maxIndex:] < thresholdFreq)[0].size != 0:
        endInterval = maxIndex + np.where(pitch_contour[maxIndex:] < thresholdFreq)[0][0]
    else:
        endInterval = len(pitch_contour)  
    #Let's take the middle fifty percent of this interval.
    #If you save the audio file here, you could use it for all data analysis.
    #close('all')
    begin50 = beginInterval + round((endInterval - beginInterval)*.3)
    #print(str(begin50))
    end50 =  beginInterval + round((endInterval - beginInterval)*.8)
    #print(str(end50))
    #visualCheckSelection(pitch_contour, begin50, end50)
    #prompt = input("Press Enter to continue...")
    #beginAudioInterval = startMiddleAttempt + round(begin50*f_s_Audio/f_s_contour)
    beginAudioInterval = round(begin50*f_s_Audio/f_s_contour) #+ startMiddleAttempt
    #endAudioInterval = startMiddleAttempt + round(end50*f_s_Audio/f_s_contour)
    endAudioInterval = round(end50*f_s_Audio/f_s_contour) #+ startMiddleAttempt  
    middleFiftyPercentHighestPitch = selectedMiddleTrial[beginAudioInterval:endAudioInterval]
    #Let's get the mean pitch of this interval
    meanFreq = pitch_contour[begin50:end50].mean()
    return samplerate, middleFiftyPercentHighestPitch, maxFreq, meanFreq

def visualizeResults(wavFilename, middleTrial, isolatedHighestPitch, samplerate, gender=np.nan):
    samplerate, data = read(wavFilename)
    #Visualize results:ipy
    fig, ax = plt.subplots(4)
    plt.rcParams['font.size'] = '14'
    ax[0].plot(np.arange(len(data))/samplerate,data)
    ax[0].set_title('Original Audio Waveform')
    ax[0].axes.xaxis.set_visible(False)
    ax[0].axes.yaxis.set_visible(False)
    ax[1].plot(np.arange(len(middleTrial))/samplerate,middleTrial)
    ax[1].set_title('Middle Trial')
    ax[1].axes.xaxis.set_visible(False)
    ax[1].axes.yaxis.set_visible(False)
    sound = Sound(middleTrial, samplerate)
    #Create a praat pitch object,
    if gender == 'männl.':
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c3-g4
    elif gender == 'weibl.':
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c4-g5
    else:
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c4-g5 
    #This provides the frequencies of the sample.
    pitch_contour = pitch.selected_array['frequency']
    pitchContLength = pitch.selected_array['frequency'].size
    wavLength = len(middleTrial)
    f_s_contour = pitchContLength/wavLength*samplerate
    ax[2].plot(np.arange(len(pitch_contour))/f_s_contour,pitch_contour)
    ax[2].set_title('Pitch Contour Middle Task')
    ax[2].set_ylabel('Freq (Hz)', fontsize=14)
    #ax[2].axes.xaxis.set_visible(False)
    sound2 = Sound(isolatedHighestPitch, samplerate)
    #Create a praat pitch object,
    if gender == 'männl.':
        pitch2 = call(sound2, "To Pitch", 0.0, 100, 390) #c3-g4
    elif gender == 'weibl.':
        pitch2 = call(sound2, "To Pitch", 0.0, 261, 784) #c4-g5
    else:
        pitch2 = call(sound2, "To Pitch", 0.0, 60, 784) #c4-g5
    #This provides the frequencies of the sample.
    pitch_contour2 = pitch2.selected_array['frequency']
    pitchContLength2 = pitch2.selected_array['frequency'].size
    wavLength2 = len(isolatedHighestPitch)
    f_s_contour2 = pitchContLength2/wavLength2*samplerate
    ax[3].plot(np.arange(len(pitch_contour2))/f_s_contour2,pitch_contour2)
    ax[3].set_title('Pitch Contour Highest Pitch')
    ax[3].set_ylabel('Freq (Hz)', fontsize=14)
    plt.show()
    #ax[3].axes.xaxis.set_visible(False)



#This for a single array check.
def showPitchContour(wavArray, samplerate):
    sound = Sound(wavArray, samplerate)
    #Create a praat pitch object,
    #Probably need upper frequency bound 2x potential sung frequency
    pitch = call(sound, "To Pitch", 0.0, 60, 1000) 
    pitch_contour = pitch.selected_array['frequency']
    plt.subplots(1)
    plot(pitch_contour)

#Visually check middle sample against full sample
def visualCheckSelection(sample0, beginSample, endSample):
    plt.close('all')
    plt.subplots(1)
    plot(sample0, color='b')
    plot(np.arange(beginSample,endSample),sample0[beginSample:endSample], color='r')

def vibAmp(pitch_contour, f_s_contour, vibFreq, windowFactor=0):
    if np.isnan(vibFreq):
        #vibFreq = 5.5
        ampCents = np.nan
        #print(str(ampCents))
        return ampCents
    wavelength = 1.0/vibFreq*f_s_contour # in frames
    # print('wavelength: ', str(wavelength))
    try:
        window = math.floor(wavelength*windowFactor)
    except ValueError:
        window = math.floor(1.0/5.5*f_s_contour*0.75)
    #
    if window == 0:
        window = 1
    # print('Window: ', str(window))
    maxPeaks = find_peaks(pitch_contour, distance=window)[0]
    prominences = scipy.signal.peak_prominences(pitch_contour, maxPeaks)[0]/2
    #minPeaks = find_peaks(pitch_contour*-1,distance=window)[0]
    #maxMean = pitch_contour[maxPeaks].mean()
    #minMean = pitch_contour[minPeaks].mean()
    #ampEstimate = (maxMean - minMean)/2 # in Hz
    #Do not calculate with the first and final peaks.
    # ampEstimate = prominences[1:-1].mean() # With a 1/2 sec window, we might have 2 peaks.
    ampEstimate = prominences.mean()
    meanFreq = pitch_contour.mean()
    # ampStd = prominences[1:-2].std()
    ampStd = prominences.std()
    #
    ampCents = 1200*np.log(1 + ampEstimate/meanFreq)/np.log(2)
    #vibStd = 1200*np.log(1 + ampStd/meanFreq)/np.log(2)
    #if vibFreq == np.nan:
    #    ampCents = np.nan
    return ampCents#, vibStd # amplitude in cents


def vibAmpDB(amplitude_envelope, f_s_contour, vibFreq, windowFactor=0):
    if np.isnan(vibFreq):
        #vibFreq = 5.5
        extentDB = np.nan
        #print(str(extentDB))
        return extentDB
    wavelength = 1.0/vibFreq*f_s_contour # in frames
    try:
        window = math.floor(wavelength*windowFactor)
    except ValueError:
        window = math.floor(1.0/5.5*f_s_contour*0.75)
    #
    if window == 0:
        window = 1
    maxPeaks = find_peaks(amplitude_envelope, distance=window)[0]
    prominences = scipy.signal.peak_prominences(amplitude_envelope, maxPeaks)[0]/2
    #minPeaks = find_peaks(amplitude_envelope*-1,distance=window)[0]
    #maxMean = amplitude_envelope[maxPeaks].mean()
    #minMean = amplitude_envelope[minPeaks].mean()
    #ampEstimate = (maxMean - minMean)/2 # in Hz
    #Do not calculate with the first and final peaks.
    extentEstimate = prominences[1:-1].mean()
    meanAmp = amplitude_envelope.mean()
    extentStd = prominences[1:-2].std()
    #
    #ampCents = 1200*np.log(1 + ampEstimate/meanFreq)/np.log(2)
    extentDB = 20 * np.log10(extentEstimate / meanAmp)
    #vibStd = 1200*np.log(1 + ampStd/meanFreq)/np.log(2)
    #if vibFreq == np.nan:
    #    extentDB = np.nan
    return extentDB#, vibStd # amplitude in cents

def vibTremor(wavFile, samplerate):
    ###CALCULATE vibRate
    sound = Sound(wavFile, samplerate)
    #Creates PRAAT sound file from .wav array, default 44100 Hz sampling frequency?
    pitch = call(sound, "To Pitch", 0.0, 60, 1000) #create a praat pitch object
    pitch_contour = pitch.selected_array['frequency']
    #Calculate the contour's sample rate from the differing array sizes
    f_s_contour = pitch.selected_array['frequency'].size/sound.values.size*sound.sampling_frequency
    vibRate = autocorrVib3Hz(pitch_contour, f_s_contour)
    
    ###CALCULATE vibExtent
    vibExtent = vibAmp(pitch_contour, f_s_contour, vibRate, 0.75)#window factor of 0.75 the wavelength
    
    ###CALCULATE ampRate
    wavData = wavFile/max(max(wavFile),abs(min(wavFile)))
    analytic_signal = hilbert(wavData)
    amplitude_envelope = np.abs(analytic_signal)
    vibAmpRate = autocorrVib3Hz(amplitude_envelope, samplerate)
    
    ###Calculate ampExtent
    vibAmpExtent = vibAmpDB(amplitude_envelope, samplerate, vibAmpRate, 0.75)
    #vibAmpExtent = vibAmpPercent(amplitude_envelope, samplerate, vibAmpRate, 0.75)
    return vibRate, vibExtent, vibAmpRate, vibAmpExtent

def vibTremorDecision(wavFile, samplerate,model):
    ###CALCULATE vibRate
    sound = Sound(wavFile, samplerate)
    #Creates PRAAT sound file from .wav array, default 44100 Hz sampling frequency?
    # pitch = call(sound, "To Pitch", 0.001, 60, 1000) #Use time steps of 0.001 for 1000 Hz f_s
    pitch = call(sound, "To Pitch", 0.001, 260, 1000) # Raising the low_frequency will reduce the time step and increase resultant fs_contour
    pitch_contour = pitch.selected_array['frequency']
    # plt.plot(pitch_contour)
    # plt.show()
    # prompt = input("Press Enter to continue, q to quit...")
    # if prompt == 'q':
       # sys.exit()
    #Calculate the contour's sample rate from the differing array sizes
    f_s_contour = pitch.selected_array['frequency'].size/sound.values.size*sound.sampling_frequency
    vibRate = autocorrVib3Hz(pitch_contour, f_s_contour)
    
    ###CALCULATE vibExtent
    vibExtent = vibAmp(pitch_contour, f_s_contour, vibRate, 0.75)#window factor of 0.75 the wavelength
    if pd.isna(vibRate):
        return pd.NA, pd.NA, pd.NA, pd.NA

    label = model.predict([[vibRate, vibExtent]])[0]
    if label == 0:
        return pd.NA, pd.NA, pd.NA, pd.NA
    
    # visualize_amplitude_vibrato_spec(wavFile, samplerate, vibRate=vibRate)
    # prompt = input("Press Enter to continue, q to quit...")
    # if prompt == 'q':
       # sys.exit()
    ###CALCULATE ampRate
    wavData = wavFile / max(abs(wavFile))
    env_ds, target_fs = calc_amplitude_envelope(wavData, samplerate, target_fs=200)
    filtered_env = bandpass_filter(env_ds, target_fs, vibRate)

    time_raw = np.arange(len(wavData)) / samplerate
    time_env = np.arange(len(filtered_env)) / target_fs

    # --- Calculate amplitude vibrato frequency ---
    vibAmpRate = autocorrVib3Hz(filtered_env, target_fs)
    vibAmpGuess = vibAmpRate
    if np.isnan(vibAmpRate):
        vibAmpGuess = vibRate

    wavelength = 1.0 / vibAmpGuess * target_fs
    window = int(np.floor(wavelength * 0.75))
    if window <= 0:
        window = 1

    maxPeaks = find_peaks(filtered_env, distance=window)[0]
    prominences = peak_prominences(filtered_env, maxPeaks)[0] / 2
    meanAmp = env_ds.mean()
    # extentEstimate = prominences[1:-1].mean()
    extentEstimate = prominences.mean()
    vibAmpExtent = 20 * np.log10(extentEstimate / meanAmp)

    return vibRate, vibExtent, vibAmpRate, vibAmpExtent

def vibTremorDecision2(wavFile, samplerate,model): #rolling RMS
    ###CALCULATE vibRate
    sound = Sound(wavFile, samplerate)
    #Creates PRAAT sound file from .wav array, default 44100 Hz sampling frequency?
    # pitch = call(sound, "To Pitch", 0.001, 60, 1000) #Use time steps of 0.001 for 1000 Hz f_s
    pitch = call(sound, "To Pitch", 0.001, 260, 1000) # Raising the low_frequency will reduce the time step and increase resultant fs_contour
    pitch_contour = pitch.selected_array['frequency']
    # plt.plot(pitch_contour)
    # plt.show()
    # prompt = input("Press Enter to continue, q to quit...")
    # if prompt == 'q':
       # sys.exit()
    #Calculate the contour's sample rate from the differing array sizes
    f_s_contour = pitch.selected_array['frequency'].size/sound.values.size*sound.sampling_frequency
    vibRate = autocorrVib3Hz(pitch_contour, f_s_contour)
    
    ###CALCULATE vibExtent
    vibExtent = vibAmp(pitch_contour, f_s_contour, vibRate, 0.75)#window factor of 0.75 the wavelength
    if pd.isna(vibRate):
        return pd.NA, pd.NA, pd.NA, pd.NA

    label = model.predict([[vibRate, vibExtent]])[0]
    if label == 0:
        return pd.NA, pd.NA, pd.NA, pd.NA
    
    # visualize_amplitude_vibrato_spec(wavFile, samplerate, vibRate=vibRate)
    # prompt = input("Press Enter to continue, q to quit...")
    # if prompt == 'q':
       # sys.exit()
    ###CALCULATE ampRate
    wavData = wavFile / max(abs(wavFile))
    env_rms, time_rms = rolling_rms(wavData, samplerate, max_vibrato_hz=vibRate, window_fraction=0.5)
    # env_ds, target_fs = calc_amplitude_envelope(wavData, samplerate, target_fs=200)
    # filtered_env = bandpass_filter(env_ds, target_fs, vibRate)

    # time_raw = np.arange(len(wavData)) / samplerate
    # time_env = np.arange(len(filtered_env)) / target_fs

    # --- Calculate amplitude vibrato frequency ---
    vibAmpRate = autocorrVib3Hz(env_rms, samplerate)
    vibAmpGuess = vibAmpRate
    if np.isnan(vibAmpRate):
        vibAmpGuess = vibRate

    wavelength = 1.0 / vibAmpGuess * samplerate
    window = int(np.floor(wavelength * 0.75))
    if window <= 0:
        window = 1

    maxPeaks = find_peaks(env_rms, distance=window)[0]
    prominences = peak_prominences(env_rms, maxPeaks)[0] / 2
    meanAmp = env_rms.mean()
    # extentEstimate = prominences[1:-1].mean()
    extentEstimate = prominences.mean()
    vibAmpExtent = 20 * np.log10(extentEstimate / meanAmp)
    
    # === 1️⃣ Find peaks and valleys in the rolling RMS ===
    peaks, _ = find_peaks(env_rms, distance=window)
    valleys, _ = find_peaks(-env_rms, distance=window)

    # Sort to be safe
    peaks = np.sort(peaks)
    valleys = np.sort(valleys)

    # === 2️⃣ Pair each peak with the nearest preceding valley ===
    pairs = []
    for p in peaks:
        # find the most recent valley before the peak
        v_candidates = valleys[valleys < p]
        if len(v_candidates) > 0:
            v = v_candidates[-1]
            pairs.append((p, v))

    # === 3️⃣ Compute the local dB differences ===
    db_diffs = []
    for p, v in pairs:
        A_peak = env_rms[p]
        A_valley = env_rms[v]
        if A_peak > 0 and A_valley > 0:  # avoid log(0)
            db_diffs.append(20 * np.log10(A_peak / A_valley))

    # === 4️⃣ Average the extent across all cycles ===
    vibAmpExtent_dB = np.mean(db_diffs) if len(db_diffs) > 0 else np.nan
    

    return vibRate, vibExtent, vibAmpRate, vibAmpExtent, vibAmpExtent_dB

def vibTremorDecision3(wavFile, samplerate, model, refRMS): #rolling RMS
    ###CALCULATE vibRate
    sound = Sound(wavFile, samplerate)
    #Creates PRAAT sound file from .wav array, default 44100 Hz sampling frequency?
    # pitch = call(sound, "To Pitch", 0.001, 60, 1000) #Use time steps of 0.001 for 1000 Hz f_s
    pitch = call(sound, "To Pitch", 0.001, 260, 1000) # Raising the low_frequency will reduce the time step and increase resultant fs_contour
    pitch_contour = pitch.selected_array['frequency']
    # plt.plot(pitch_contour)
    # plt.show()
    # prompt = input("Press Enter to continue, q to quit...")
    # if prompt == 'q':
       # sys.exit()
    #Calculate the contour's sample rate from the differing array sizes
    f_s_contour = pitch.selected_array['frequency'].size/sound.values.size*sound.sampling_frequency
    vibRate = autocorrVib3Hz(pitch_contour, f_s_contour)
    
    ###CALCULATE vibExtent
    vibExtent = vibAmp(pitch_contour, f_s_contour, vibRate, 0.75)#window factor of 0.75 the wavelength
    if pd.isna(vibRate):
        return pd.NA, pd.NA, pd.NA, pd.NA

    label = model.predict([[vibRate, vibExtent]])[0]
    if label == 0:
        return pd.NA, pd.NA, pd.NA, pd.NA
    
    # visualize_amplitude_vibrato_spec(wavFile, samplerate, vibRate=vibRate)
    # prompt = input("Press Enter to continue, q to quit...")
    # if prompt == 'q':
       # sys.exit()
    ###CALCULATE ampRate
    wavData = wavFile / max(abs(wavFile))
    env_rms, time_rms = rolling_rms(wavData, samplerate, max_vibrato_hz=vibRate, window_fraction=0.5)
    # env_ds, target_fs = calc_amplitude_envelope(wavData, samplerate, target_fs=200)
    # filtered_env = bandpass_filter(env_ds, target_fs, vibRate)

    # time_raw = np.arange(len(wavData)) / samplerate
    # time_env = np.arange(len(filtered_env)) / target_fs

    # --- Calculate amplitude vibrato frequency ---
    vibAmpRate = autocorrVib3HzLocal(env_rms, samplerate)
    vibAmpGuess = vibAmpRate
    if np.isnan(vibAmpRate):
        vibAmpGuess = vibRate

    wavelength = 1.0 / vibAmpGuess * samplerate
    window = int(np.floor(wavelength * 0.75))
    if window <= 0:
        window = 1

    maxPeaks = find_peaks(env_rms, distance=window)[0]
    prominences = peak_prominences(env_rms, maxPeaks)[0] / 2
    meanAmp = env_rms.mean()
    # extentEstimate = prominences[1:-1].mean()
    extentEstimate = prominences.mean()
    vibAmpExtent = 20 * np.log10(extentEstimate / meanAmp)
    
    if refRMS != np.nan:
        env_spl = 20 * np.log10(env_rms / refRMS)
        maxPeaks = find_peaks(env_spl, distance=window)[0]
        prominences = peak_prominences(env_spl, maxPeaks)[0] / 2
        # meanAmp = env_spl.mean()
        # extentEstimate = prominences[1:-1].mean()
        vibAmpExtent_SPL = prominences.mean()
    else:
        vibAmpExtent_SPL = np.nan
    # vibAmpExtent_SPL = 94 + 20 * np.log10(extentEstimate / refRMS)
    
    # === 1️⃣ Find peaks and valleys in the rolling RMS ===
    peaks, _ = find_peaks(env_rms, distance=window)
    valleys, _ = find_peaks(-env_rms, distance=window)

    # Sort to be safe
    peaks = np.sort(peaks)
    valleys = np.sort(valleys)

    # === 2️⃣ Pair each peak with the nearest preceding valley ===
    pairs = []
    for p in peaks:
        # find the most recent valley before the peak
        v_candidates = valleys[valleys < p]
        if len(v_candidates) > 0:
            v = v_candidates[-1]
            pairs.append((p, v))

    # === 3️⃣ Compute the local dB differences ===
    db_diffs = []
    for p, v in pairs:
        A_peak = env_rms[p]
        A_valley = env_rms[v]
        if A_peak > 0 and A_valley > 0:  # avoid log(0)
            db_diffs.append(20 * np.log10(A_peak / A_valley))

    # === 4️⃣ Average the extent across all cycles ===
    vibAmpExtent_dB = np.mean(db_diffs) if len(db_diffs) > 0 else np.nan
    

    return vibRate, vibExtent, vibAmpRate, vibAmpExtent, vibAmpExtent_SPL, vibAmpExtent_dB

def apply_vibTremorDecision_rolling(wavFile, samplerate, model, refRMS, window_duration=1, step_duration=0.01): #f_s_contour makes step less than 0.1 unnecessary
    """
    Apply vibTremorDecision over a rolling window and return median values of each measure.
    
    Parameters:
        wavFile: np.ndarray - 1D array of audio samples
        samplerate: int - sampling rate of audio
        model: sklearn-like classifier with .predict()
        window_duration: float - window size in seconds
        step_duration: float - step size in seconds
        
    Returns:
        median_vibRate, median_vibExtent, median_vibAmpRate, median_vibAmpExtent(, vibperc?)
    """
    window_size = int(window_duration * samplerate)
    step_size = int(step_duration * samplerate)

    vibRates = []
    vibExtents = []
    vibAmpRates = []
    vibAmpExtents = []
    vibAmpExtents_SPL = []
    vibAmpExtents_dB = []
    vibPercent = []

    for start in range(0, len(wavFile) - window_size + 1, step_size):
        segment = wavFile[start:start + window_size]
        result = vibTremorDecision3(segment, samplerate, model, refRMS)

        if pd.notna(result[0]):
            vibRate, vibExtent, vibAmpRate, vibAmpExtent, vibAmpExtent_SPL, vibAmpExtent_dB = result
            vibRates.append(vibRate)
            vibExtents.append(vibExtent)
            vibAmpRates.append(vibAmpRate)
            vibAmpExtents.append(vibAmpExtent)
            vibAmpExtents_SPL.append(vibAmpExtent_SPL)
            vibAmpExtents_dB.append(vibAmpExtent_dB)
            vibPercent.append(1)
        else:
            vibRates.append(np.nan)
            vibExtents.append(np.nan)
            vibAmpRates.append(np.nan)
            vibAmpExtents.append(np.nan)
            vibAmpExtents_SPL.append(np.nan)
            vibAmpExtents_dB.append(np.nan)
            vibPercent.append(np.nan)

    if not vibRates:  # If no valid windows
        return pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA
    #print('Mean: ', str(np.mean(vibRates)),' Std: ', str(np.std(vibRates)))
    #print('Mean: ', str(np.mean(vibAmpRates)),' Std: ', str(np.std(vibAmpRates)))
    # print(vibRates)
    # Compute medians
    vibPerc = np.nansum(vibPercent)/len(vibPercent)
    return (
        np.nanmedian(vibRates),
        np.nanmedian(vibExtents),
        np.nanmedian(vibAmpRates),
        np.nanmedian(vibAmpExtents),
        np.nanmedian(vibAmpExtents_SPL),
        np.nanmedian(vibAmpExtents_dB),
        vibPerc
    )
    # return (
        # vibRates,
        # vibExtents,
        # vibAmpRates,
        # vibAmpExtents,
        # vibPercent
    # )

def apply_vibTremorDecision_rolling_harmonics(wavFile, samplerate, model, refRMS, meanFreq,
                                              window_duration=1.0, step_duration=0.01,
                                              max_freq=8000):
    """
    Apply vibTremorDecision and harmonic vibrato analysis over rolling windows.
    Returns median values for each measure across the windows.
    
    Parameters:
        wavFile: np.ndarray - 1D audio samples
        samplerate: int
        model: sklearn-like classifier with .predict()
        refRMS: reference RMS for normalization
        window_duration: float - window size in seconds
        step_duration: float - step size in seconds
        max_freq: float - maximum frequency to analyze (Hz)
    
    Returns:
        median_vibRate, median_vibExtent, median_vibAmpRate, median_vibAmpExtent,
        median_vibAmpExtent_SPL, median_vibAmpExtent_dB, median_vibPerc,
        median_harmonic_extent_pa, median_harmonic_extent_spl, median_harmonic_mean_spl
    """
    window_size = int(window_duration * samplerate)
    step_size = int(step_duration * samplerate)

    vibRates, vibExtents = [], []
    vibAmpRates, vibAmpExtents = [], []
    vibAmpExtents_SPL, vibAmpExtents_dB = [], []
    vibPercent = []

    harmonic_extents_pa = []
    harmonic_extents_spl = []
    harmonic_mean_spl = []

    for start in range(0, len(wavFile) - window_size + 1, step_size):
        segment = wavFile[start:start + window_size]

        # --- Original vibTremorDecision
        result = vibTremorDecision3(segment, samplerate, model, refRMS)
        if pd.notna(result[0]):
            vibRate, vibExtent, vibAmpRate, vibAmpExtent, vibAmpExtent_SPL_val, vibAmpExtent_dB_val = result
            vibRates.append(vibRate)
            vibExtents.append(vibExtent)
            vibAmpRates.append(vibAmpRate)
            vibAmpExtents.append(vibAmpExtent)
            vibAmpExtents_SPL.append(vibAmpExtent_SPL_val)
            vibAmpExtents_dB.append(vibAmpExtent_dB_val)
            vibPercent.append(1)
        else:
            vibRates.append(np.nan)
            vibExtents.append(np.nan)
            vibAmpRates.append(np.nan)
            vibAmpExtents.append(np.nan)
            vibAmpExtents_SPL.append(np.nan)
            vibAmpExtents_dB.append(np.nan)
            vibPercent.append(np.nan)

        # --- Harmonic-level analysis
        df_harmonics = analyze_vibrato(segment, samplerate, f0=meanFreq, max_freq=max_freq)
        if not df_harmonics.empty:
            harmonic_extents_pa.append(df_harmonics['extent_pa'].median())
            harmonic_extents_spl.append(df_harmonics['extent_spl'].median())
            harmonic_mean_spl.append(df_harmonics['mean_spl'].median())
        else:
            harmonic_extents_pa.append(np.nan)
            harmonic_extents_spl.append(np.nan)
            harmonic_mean_spl.append(np.nan)

    # --- Compute medians across windows
    vibPerc = np.nansum(vibPercent) / len(vibPercent) if vibPercent else np.nan

    return (
        np.nanmedian(vibRates),
        np.nanmedian(vibExtents),
        np.nanmedian(vibAmpRates),
        np.nanmedian(vibAmpExtents),
        np.nanmedian(vibAmpExtents_SPL),
        np.nanmedian(vibAmpExtents_dB),
        vibPerc,
        np.nanmedian(harmonic_extents_pa),
        np.nanmedian(harmonic_extents_spl),
        np.nanmedian(harmonic_mean_spl)
    )


def bandpass_filter(signal, fs, f_center, bandwidth=1.0, order=4):
    nyquist = fs / 2
    low = (f_center - bandwidth/2) / nyquist
    high = (f_center + bandwidth/2) / nyquist
    if low <= 0:
        low = 1e-6  # just above 0
    if high >= 1:
        high = 0.999999
    if low >= high:
        raise ValueError(
            f"Invalid bandpass: f_center={f_center}, bandwidth={bandwidth}, fs={fs} "
            f"-> low={low:.3f}, high={high:.3f}"
        )

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)    
    
from scipy.signal import hilbert, resample

# def calc_amplitude_envelope(signal, fs, target_fs=200):
    # analytic = hilbert(signal)
    # env = np.abs(analytic)

    ###Downsample envelope for low-frequency filtering
    # n_samples = int(len(env) * target_fs / fs)
    # env_ds = resample(env, n_samples)
    # return env_ds, target_fs

from scipy.signal import hilbert, resample, butter, filtfilt

def calc_amplitude_envelope(signal, fs, target_fs=200, cutoff=20):
    """
    Calculate a smooth amplitude envelope using Hilbert transform.
    
    signal : 1D np.array audio
    fs : sampling rate of audio
    target_fs : desired downsampled rate for plotting
    cutoff : low-pass filter cutoff (Hz) for smoothing
    """
    # Hilbert transform
    analytic = hilbert(signal)
    env = np.abs(analytic)
    
    # Low-pass filter to smooth envelope
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(2, normal_cutoff, btype='low')
    env_smooth = filtfilt(b, a, env)
    
    # Downsample
    n_samples = int(len(env_smooth) * target_fs / fs)
    env_ds = resample(env_smooth, n_samples)
    
    return env_ds, target_fs


def visualize_amplitude_vibrato(wavFile, samplerate, vibRate=5.5, bandwidth=5):
    # --- Normalize and compute amplitude envelope ---
    wavData = wavFile / max(abs(wavFile))
    env_ds, target_fs = calc_amplitude_envelope(wavData, samplerate, target_fs=200)
    filtered_env = bandpass_filter(env_ds, target_fs, vibRate, bandwidth)

    time_raw = np.arange(len(wavData)) / samplerate
    time_env = np.arange(len(env_ds)) / target_fs

    # --- Calculate amplitude vibrato frequency ---
    vibAmpRate = autocorrVib3Hz(filtered_env, target_fs)
    if np.isnan(vibAmpRate):
        vibAmpRate = vibRate

    wavelength = 1.0 / vibAmpRate * target_fs
    window = int(np.floor(wavelength * 0.75))
    if window <= 0:
        window = 1

    maxPeaks = find_peaks(filtered_env, distance=window)[0]
    prominences = peak_prominences(filtered_env, maxPeaks)[0] / 2
    meanAmp = env_ds.mean()
    extentEstimate = prominences[1:-1].mean()
    extentDB = 20 * np.log10(extentEstimate / meanAmp)

    # --- Plotting: 3 subplots ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=False)

    # 1. Raw waveform
    axs[0].plot(time_raw, wavData, color="steelblue", linewidth=0.8)
    axs[0].set_title("Raw Signal")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)

    # 2. Filtered amplitude envelope
    axs[1].plot(time_env, filtered_env, label="Filtered Envelope", linewidth=1.2)
    axs[1].plot(time_env[maxPeaks], filtered_env[maxPeaks], 'rx', label="Peaks")
    for i in range(1, len(maxPeaks)-1):
        peak = maxPeaks[i]
        prom = prominences[i]
        axs[1].vlines(x=time_env[peak], ymin=filtered_env[peak] - prom, ymax=filtered_env[peak],
                      color='gray', linestyle='--', alpha=0.5)
    axs[1].set_title(f"Amplitude Vibrato Extent ≈ {extentDB:.2f} dB")
    axs[1].set_ylabel("Envelope Amplitude")
    axs[1].legend()
    axs[1].grid(True)

    # 3. Autocorrelation of filtered envelope
    acorr = acf(filtered_env, nlags=len(filtered_env)//10)
    lags = np.arange(len(acorr)) / target_fs
    highCI = 1.96 / np.sqrt(len(filtered_env))

    axs[2].plot(lags, acorr)
    axs[2].axhline(highCI, color='gray', linestyle='--', label="95% CI")
    axs[2].axhline(-highCI, color='gray', linestyle='--')
    axs[2].set_title(f"Amplitude Vibrato Frequency ≈ {vibAmpRate:.2f} Hz")
    axs[2].set_xlabel("Lag (s)")
    axs[2].set_ylabel("Autocorrelation")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def visualize_amplitude_vibrato_spec(wavFile, samplerate, vibRate=5.5, bandwidth=5):
    # --- Normalize and compute amplitude envelope ---
    wavData = wavFile / max(abs(wavFile))
    env_ds, target_fs = calc_amplitude_envelope(wavData, samplerate, target_fs=200)
    filtered_env = bandpass_filter(env_ds, target_fs, vibRate, bandwidth)

    time_raw = np.arange(len(wavData)) / samplerate
    time_env = np.arange(len(env_ds)) / target_fs  # <- envelope time axis

    # --- Calculate amplitude vibrato frequency ---
    vibAmpRate = autocorrVib3Hz(filtered_env, target_fs)
    if np.isnan(vibAmpRate):
        vibAmpRate = vibRate

    wavelength = 1.0 / vibAmpRate * target_fs
    window = int(np.floor(wavelength * 0.75))
    if window <= 0:
        window = 1

    maxPeaks = find_peaks(filtered_env, distance=window)[0]
    prominences = peak_prominences(filtered_env, maxPeaks)[0] / 2
    meanAmp = env_ds.mean()
    extentEstimate = prominences[1:-1].mean()
    extentDB = 20 * np.log10(extentEstimate / meanAmp)

    # --- Plotting: 3 subplots ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=False)

    # 1. Raw waveform + UNFILTERED envelope overlay
    axs[0].plot(time_raw, wavData, color="steelblue", linewidth=0.8, label="Raw Signal")
    env_scaled = env_ds / np.max(env_ds) * np.max(wavData)  # scale for overlay
    axs[0].plot(time_env, env_scaled, color="darkorange", linewidth=1.2, label="Downsampled Envelope (unfiltered)")
    axs[0].set_title("Raw Signal with Envelope Overlay")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()
    axs[0].grid(True)

    # 2. Filtered amplitude envelope (zoomed in view with peaks)
    axs[1].plot(time_env, filtered_env, label="Filtered Envelope", linewidth=1.2)
    axs[1].plot(time_env[maxPeaks], filtered_env[maxPeaks], 'rx', label="Peaks")
    for i in range(1, len(maxPeaks)-1):
        peak = maxPeaks[i]
        prom = prominences[i]
        axs[1].vlines(x=time_env[peak], ymin=filtered_env[peak] - prom, ymax=filtered_env[peak],
                      color='gray', linestyle='--', alpha=0.5)
    axs[1].set_title(f"Amplitude Vibrato Extent ≈ {extentDB:.2f} dB")
    axs[1].set_ylabel("Envelope Amplitude")
    axs[1].legend()
    axs[1].grid(True)

    # 3. Spectrogram of filtered envelope
    nperseg = min(64, len(env_ds))
    noverlap = max(0, nperseg // 2)

    f, t, Sxx = spectrogram(filtered_env, fs=target_fs, nperseg=nperseg, noverlap=noverlap)
    im = axs[2].pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud')
    axs[2].set_ylim(0, 15)  # focus on low Hz range
    axs[2].set_title(f"Spectrogram of Downsampled Envelope, Guessed Rate ≈ {vibRate:.2f} Hz")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Frequency [Hz]")

    fig.colorbar(im, ax=axs[2], label='Power [dB]')

    plt.tight_layout()
    plt.show()

def isolateHighestPitch1s(samplerate, selectedMiddleTrial, time_sec=1, gender=np.nan):
    """
    Isolate the middle 1 second of the sustained highest pitch region.

    Parameters
    ----------
    samplerate : int
        Sampling rate of the audio (Hz).
    selectedMiddleTrial : np.ndarray
        Audio waveform (1D numpy array).
    gender : str or np.nan
        Optional ('männl.' or 'weibl.') for pitch settings.

    Returns
    -------
    samplerate : int
        Same as input samplerate.
    middleOneSecond : np.ndarray
        1-second audio segment centered in the sustained highest pitch portion.
    maxFreq : float
        Frequency of the highest pitch (Hz).
    meanFreq : float
        Mean frequency within the 1-second window (Hz).
    begin50, end50 : int
        Pitch contour indices bounding the selection.
    pitch_contour : np.ndarray
        Full pitch contour (Hz).
    f_s_contour : float
        Effective sampling rate of the pitch contour (Hz).
    """

    # --- Compute pitch contour ---
    sound = Sound(selectedMiddleTrial, samplerate)
    if gender == 'männl.':
        pitch = call(sound, "To Pitch", 0.0, 60, 1000)
    elif gender == 'weibl.':
        pitch = call(sound, "To Pitch", 0.0, 60, 1000)
    else:
        pitch = call(sound, "To Pitch", 0.0, 60, 1000)

    pitch_contour = pitch.selected_array['frequency']
    f_s_Audio = sound.sampling_frequency
    wavLength = sound.values.size
    pitchContLength = pitch_contour.size
    f_s_contour = pitchContLength / wavLength * f_s_Audio

    # --- Find sustained interval around highest pitch ---
    maxIndex = round(len(pitch_contour) / 2)
    maxFreq = pitch_contour[maxIndex]
    thresholdFreq = maxFreq * 5 / 6

    beginInterval = np.where(pitch_contour[:maxIndex] < thresholdFreq)[0][-1]
    if np.where(pitch_contour[maxIndex:] < thresholdFreq)[0].size != 0:
        endInterval = maxIndex + np.where(pitch_contour[maxIndex:] < thresholdFreq)[0][0]
    else:
        endInterval = len(pitch_contour)

    # --- Use center of this sustained region ---
    centerIndex = (beginInterval + endInterval) // 2

    # Map pitch contour index to audio sample index
    centerSample = round(centerIndex * f_s_Audio / f_s_contour)

    # --- Extract exactly 1 second of audio ---
    half_window = int(samplerate * time_sec // 2)
    startSample = max(centerSample - half_window, 0)
    endSample = min(centerSample + half_window, wavLength)

    middleOneSecond = selectedMiddleTrial[startSample:endSample]

    # --- Mean frequency for that window ---
    begin50 = max(centerIndex - int(time_sec / 2* f_s_contour), 0)
    end50 = min(centerIndex + int(time_sec / 2 * 0.5 * f_s_contour), len(pitch_contour))
    # begin50 = max(centerIndex - int(1 * f_s_contour), 0)
    # end50 = min(centerIndex + int(1 * f_s_contour), len(pitch_contour))
    meanFreq = pitch_contour[begin50:end50].mean()

    return samplerate, middleOneSecond, maxFreq, meanFreq, begin50, end50, pitch_contour, f_s_contour

from statsmodels.tsa.stattools import acf
def plot_autocorr_with_peak_on_ax(env, target_fs, ax, fmin=2, fmax=20, f_peak_range=(3,10), alpha=0.05):
    """
    Plot autocorrelation of an envelope between fmin and fmax Hz,
    mark the maximum peak within f_peak_range, and add 95% CI as
    horizontal dashed lines centered on zero, directly onto the provided axis.
    """
    # --- Step 1. Compute autocorrelation ---
    max_lag = int(target_fs / fmin)   # lag corresponding to lowest freq of interest
    acf_vals = acf(env, nlags=max_lag, fft=True)
    lags = np.arange(len(acf_vals))

    # Convert lag -> frequency
    freqs = target_fs / lags
    freqs = freqs[1:]         # drop lag=0
    acf_vals = acf_vals[1:]

    # --- Step 2. Restrict to fmin–fmax Hz ---
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_masked = freqs[mask]
    acf_masked = acf_vals[mask]

    # --- Step 3. Find maximum peak in f_peak_range ---
    peak_mask = (freqs_masked >= f_peak_range[0]) & (freqs_masked <= f_peak_range[1])
    if np.any(peak_mask):
        idx_peak = np.argmax(acf_masked[peak_mask])
        idx_peak_global = np.where(peak_mask)[0][idx_peak]

        f_peak = freqs_masked[idx_peak_global]
        acf_peak = acf_masked[idx_peak_global]

        # --- Step 4. Plot autocorr and zero-centered CIs ---
        ax.plot(freqs_masked, acf_masked, label="Autocorrelation")

        # Compute 95% CI centered on 0
        N = len(env)
        ci_val = 1.96 / np.sqrt(N)  # standard approx for large N
        ax.axhline(y=ci_val, color="gray", linestyle="--", linewidth=1, label="95% CI")
        ax.axhline(y=-ci_val, color="gray", linestyle="--", linewidth=1)

        # Mark the peak
        ax.plot(f_peak, acf_peak, "x", color="red", markersize=10, 
                label=f"Peak {f_peak:.2f} Hz")

        ax.set_title(f"Autocorrelation ({fmin}–{fmax} Hz)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Autocorrelation")
        ax.legend(loc="upper right")
        ax.grid(True)

        return f_peak, acf_peak, (-ci_val, ci_val)
    else:
        ax.text(0.5, 0.5, "No peak in range", ha="center", va="center", transform=ax.transAxes)
        return None


import numpy as np
import pandas as pd

def rolling_rms(signal, fs, max_vibrato_hz=9, window_fraction=0.5):
    """
    Compute a rolling RMS of a signal that can resolve vibrato up to max_vibrato_hz.

    Parameters
    ----------
    signal : 1D np.array
        Raw amplitude signal (absolute waveform or Hilbert envelope)
    fs : int
        Sampling rate in Hz
    max_vibrato_hz : float
        Highest vibrato frequency to resolve
    window_fraction : float
        Fraction of the vibrato period to use as RMS window (0 < window_fraction <= 1)

    Returns
    -------
    rms_env : np.array
        Rolling RMS amplitude
    time : np.array
        Time vector corresponding to rms_env
    """

    # Vibrato period for the highest frequency we want to resolve
    T_min = 1.0 / max_vibrato_hz

    # RMS window in seconds
    window_sec = T_min * window_fraction
    window_samples = max(1, int(round(window_sec * fs)))

    # Rolling RMS (using mean of squares)
    df = pd.Series(signal)
    rms_env = np.sqrt(df.pow(2).rolling(window_samples, center=True, min_periods=1).mean())

    time = np.arange(len(signal)) / fs
    return rms_env.values, time


# --- main visualization ---
def visualizeResultsFull(wavFilename, refRMS, gender=np.nan, vibRate=5.5, bandwidth=5):
    samplerate, data = read(wavFilename)
    # Step 0: Normalize audio
    data = data/np.max(np.abs(data))
    # Step 1: isolate middle 50%
    samplerate, isolatedPitch50, maxFreq, meanFreq, begin50, end50, pitch_contour, f_s_contour = \
        isolateHighestPitch1s(samplerate, data, time_sec=2)
    pitch_contour[pitch_contour == 0] = np.nan
    fig, axs = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
    plt.rcParams['font.size'] = 14

    time_audio = np.arange(len(data)) / samplerate
    time_pitch = np.arange(len(pitch_contour)) / f_s_contour
    time_iso = np.arange(len(isolatedPitch50)) / samplerate

    # --- Row 0 ---
    # [0,0] Pitch contour full
    axs[0, 0].plot(time_pitch, pitch_contour, label="Pitch contour")
    axs[0, 0].plot(time_pitch[begin50:end50], pitch_contour[begin50:end50], color="r", label="2s Window")
    axs[0, 0].set_title("Pitch Contour with 2s Window Highlighted")
    axs[0, 0].set_ylabel("Frequency (Hz)")
    axs[0, 0].set_xlabel("Time (s)", fontsize=14)
    axs[0, 0].legend()

    # [0,1] Raw audio full
    axs[0, 1].plot(time_audio, data, color="steelblue")
    axs[0, 1].axvspan(time_audio[int(begin50 * samplerate / f_s_contour)],
                      time_audio[int(end50 * samplerate / f_s_contour)],
                      color="red", alpha=0.3, label="2s Window")
    axs[0, 1].set_title("Raw Audio with 2s Window Highlighted")
    axs[0, 1].set_ylabel("Amplitude")
    axs[0, 1].set_xlabel("Time (s)", fontsize=14)
    axs[0, 1].legend()

    # --- Row 1 ---
    # [1,0] Middle 50% with vibrato extent calculation
    sound2 = Sound(isolatedPitch50, samplerate)

    pitch2 = call(sound2, "To Pitch", 0.0, 60, 784)   # generic

    pitch_contour2 = pitch2.selected_array['frequency']
    pitchContLength2 = pitch_contour2.size
    wavLength2 = len(isolatedPitch50)
    f_s_contour2 = pitchContLength2 / wavLength2 * samplerate
    time_pitch2 = np.arange(len(pitch_contour2)) / f_s_contour2

    # estimate vibrato extent
    vibFreq = vibRate  # use function argument (instead of df.loc[i])
    wavLengthWindow = 0.75 * 1 / vibFreq * f_s_contour2
    maxPeaks = find_peaks(pitch_contour2, distance=wavLengthWindow)[0]
    prominences = scipy.signal.peak_prominences(pitch_contour2, maxPeaks)[0] / 2
    contour_heights = pitch_contour2[maxPeaks] - prominences

    # === Compute vibrato extent in cents ===
    meanFreq_Hz = np.nanmean(pitch_contour2)
    meanProm_Hz = np.nanmean(prominences)
    extent_cents = 1200 * np.log2((meanFreq_Hz + meanProm_Hz) / meanFreq_Hz)

    # === Plot ===
    axs[1, 0].plot(time_pitch2, pitch_contour2, label="Pitch contour (2s Window)")
    axs[1, 0].plot(time_pitch2[maxPeaks], pitch_contour2[maxPeaks], "x", label="Peaks")
    axs[1, 0].vlines(x=time_pitch2[maxPeaks],
                     ymin=contour_heights,
                     ymax=pitch_contour2[maxPeaks],
                     color='gray', linestyle='--', alpha=0.5)

    axs[1, 0].text(
        0.02, 0.95,
        f"Mean Extent ≈ {extent_cents:.1f} cents",
        transform=axs[1, 0].transAxes,
        fontsize=12,
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.7)
    )

    axs[1, 0].set_title("Middle 2s with Extent Calculation")
    axs[1, 0].set_ylabel("Freq (Hz)", fontsize=14)
    axs[1, 0].set_xlabel("Time (s)", fontsize=14)
    axs[1, 0].legend(loc="lower right")

    



    # [0, 1] Zoomed middle 50% audio
    # --- Panel [0, 1]: Downsampled amplitude envelope overlay ---
    # axs[0, 1].plot(time_iso, isolatedPitch50, color="steelblue", linewidth=0.8, label="Raw Signal")
    
    # --- Downsample amplitude envelope ---
    # env_ds, target_fs = calc_amplitude_envelope(isolatedPitch50, samplerate, target_fs=200)  # Hz
    # 2. Hilbert envelope (same length as audio)
    analytic = hilbert(isolatedPitch50)
    env = np.abs(analytic)
    f0_median = np.nanmedian(pitch_contour2)
    env_ds, target_fs = calc_amplitude_envelope(isolatedPitch50, samplerate, target_fs=200)
    env_rms, time_rms = rolling_rms(isolatedPitch50, samplerate, max_vibrato_hz=vibRate, window_fraction=0.5)
    # filtered_env = bandpass_filter(env_ds, target_fs, vibRate)
    # filtered_env = bandpass_filter(env, samplerate, vibRate)
    filtered_env = env
    # vibAmpRate = autocorrVib3Hz(filtered_env, target_fs)
    # 3. Scale envelope to match waveform max amplitude
    # env_scaled = env / np.max(env) * np.max(wavData)
    time_env = np.arange(len(filtered_env)) / samplerate


    # Scale the downsampled envelope so it overlays properly
    # env_scaled = env_ds / np.max(env_ds) * np.max(isolatedPitch50)
    # axs[0, 1].plot(time_iso, isolatedPitch50, color="blue", linewidth=1.2,
                   # label="Unfiltered Amplitude (Middle 2s)")

    # axs[0, 1].plot(time_iso, env, color="darkorange", linewidth=1.2, # + np.max(isolatedPitch50)
                   # label="Downsampled Amplitude Envelope (Middle 2s)")
    # axs[0, 1].set_title("Raw Signal with Envelope Overlay")
    # axs[0, 1].set_ylabel("Amplitude")
    # axs[0, 1].set_xlabel("Time (s)", fontsize=14)
    # axs[0, 1].legend(loc="lower right")
    # axs[0, 1].grid(True)

    wavelength = 1.0 / vibRate * samplerate
    window = int(np.floor(wavelength * 0.75))
    if window <= 0:
        window = 1

    maxPeaks = find_peaks(env, distance=window)[0]
    prominences = peak_prominences(env, maxPeaks)[0] #/ 2
    # meanAmp = env_ds.mean()
    meanAmp = env.mean()
    extentEstimate = prominences[1:-1].mean()
    extentDB = 20 * np.log10(extentEstimate / meanAmp)
    

    # 2. Filtered amplitude envelope (zoomed in view with peaks)
    # axs[0, 1].plot(time_env, env, label="Filtered Envelope", linewidth=1.2)
    # axs[0, 1].plot(time_env[maxPeaks], env[maxPeaks], 'rx', label="Peaks")

    
    # for i in range(1, len(maxPeaks)-1):
    # for i in range(0, len(maxPeaks)):
        # peak = maxPeaks[i]
        # prom = prominences[i]
        # axs[0, 1].vlines(x=time_env[peak],
                         # ymin=env[peak] - prom,
                         # ymax=env[peak],
                         # color='gray', linestyle='--', alpha=0.5
                        # )

    # axs[0, 1].axhline(0, color="darkorange",label="Mean)
    # axs[0, 1].set_title(f"Filtered Envelope With Extent ≈ {extentDB:.2f} dB")
    # axs[0, 1].set_ylabel("Envelope Amplitude")
    # axs[0, 1].set_xlabel("Time (s)")
    # axs[0, 1].legend(loc="lower right")
    # axs[0, 1].grid(True)
    
    
    
    time_ds = np.arange(len(env_ds)) / target_fs
    wavelength = 1.0 / vibRate * target_fs
    window = int(np.floor(wavelength * 0.75))
    if window <= 0:
        window = 1

    maxPeaks = find_peaks(env_ds, distance=window)[0]
    prominences = peak_prominences(env_ds, maxPeaks)[0] #/ 2
    # meanAmp = env_ds.mean()
    meanAmp = env.mean()
    extentEstimate = prominences[1:-1].mean()
    extentDB = 20 * np.log10(extentEstimate / meanAmp)
    

    # 2. Filtered amplitude envelope (zoomed in view with peaks)
    # axs[0, 1].plot(time_ds, env_ds, label="Filtered Envelope", linewidth=1.2)
    # axs[0, 1].plot(time_ds[maxPeaks], env_ds[maxPeaks], 'rx', label="Peaks")

    
    # for i in range(1, len(maxPeaks)-1):
    for i in range(0, len(maxPeaks)):
        peak = maxPeaks[i]
        prom = prominences[i]
        # axs[0, 1].vlines(x=time_ds[peak],
                         # ymin=env_ds[peak] - prom,
                         # ymax=env_ds[peak],
                         # color='gray', linestyle='--', alpha=0.5
                        # )

    
    # axs[0, 1].set_title(f"Filtered Envelope With Extent ≈ {extentDB:.2f} dB")
    # axs[0, 1].set_ylabel("Envelope Amplitude")
    # axs[0, 1].set_xlabel("Time (s)")
    # axs[0, 1].legend(loc="lower right")
    # axs[0, 1].grid(True)
    
    # time_ds = np.arange(len(env_ds)) / target_fs
    wavelength = 1.0 / vibRate * samplerate
    window = int(np.floor(wavelength * 0.75))
    if window <= 0:
        window = 1

    maxPeaks = find_peaks(env_rms, distance=window)[0]
    prominences = peak_prominences(env_rms, maxPeaks)[0] / 2
    # meanAmp = env_rms.mean()
    meanAmp = env_rms.mean()
    extentEstimate = prominences[1:-1].mean()
    # extentDB = 20 * np.log10(extentEstimate / meanAmp)
    extentDB = 94 + 20 * np.log10(extentEstimate / refRMS)
    env_SPL = 94 + 20 * np.log10(env_rms / refRMS)
    # promDB = 94 + 20 * np.log10(prominences/ refRMS)

    # 2. Filtered amplitude envelope (zoomed in view with peaks)
    # axs[1, 1].plot(time_rms, env_rms, label="Rolling SPL", linewidth=1.2)
    # axs[1, 1].plot(time_rms[maxPeaks], env_rms[maxPeaks], 'rx', label="Peaks")
    # axs[1, 1].plot(time_rms, env_SPL, label="Rolling SPL", linewidth=1.2)
    # axs[1, 1].plot(time_rms[maxPeaks], env_SPL[maxPeaks], 'rx', label="Peaks")

    
    # for i in range(1, len(maxPeaks)-1):
    for i in range(0, len(maxPeaks)):
        peak = maxPeaks[i]
        prom = prominences[i]
        # prom = promDB[i]
        # axs[1, 1].vlines(x=time_rms[peak],
                         # ymin=env_SPL[peak] - prom,
                         # ymax=env_SPL[peak],
                         # color='gray', linestyle='--', alpha=0.5
                        # )

    
    # axs[1, 1].set_title(f"Middle 2s Rolling SPL With Extent ≈ {extentDB:.2f} dB")
    # axs[1, 1].set_ylabel("SPL")
    # axs[1, 1].set_xlabel("Time (s)")
    # axs[1, 1].legend(loc="lower right")
    # axs[1, 1].grid(True)
    
    # === Define parameters *before* using them ===
    fs = samplerate
    frame_len = int(1 / (2 * vibRate) * fs)  # e.g. 20 ms analysis window
    hop_len = int(0.01 * fs)                    # e.g. 10 ms hop

    # === 1️⃣ Find peaks and valleys in the rolling RMS ===
    # env_rms = env_audio
    peaks, _ = find_peaks(env_rms, distance=window)
    valleys, _ = find_peaks(-env_rms, distance=window)

    # Sort to be safe
    peaks = np.sort(peaks)
    valleys = np.sort(valleys)

    # === 2️⃣ Pair each peak with the nearest preceding valley ===
    pairs = []
    for p in peaks:
        v_candidates = valleys[valleys < p]
        if len(v_candidates) > 0:
            v = v_candidates[-1]
            pairs.append((p, v))

    # === 3️⃣ Compute the local dB differences ===
    db_diffs = []
    for p, v in pairs:
        A_peak = env_rms[p]
        A_valley = env_rms[v]
        if A_peak > 0 and A_valley > 0:  # avoid log(0)
            db_diffs.append(20 * np.log10(A_peak / A_valley))

    # === 4️⃣ Average the extent across all cycles ===
    vibAmpExtent_dB = np.mean(db_diffs) if len(db_diffs) > 0 else np.nan

    # === 5️⃣ Compute time axes (in seconds) ===
    t_env_audio = (np.arange(len(env_rms)) * hop_len + frame_len / 2) / fs
    t_peaks = (peaks * hop_len + frame_len / 2) / fs
    t_valleys = (valleys * hop_len + frame_len / 2) / fs

    # === 6️⃣ Plot in existing figure panel [1,1] ===
    ax = axs[1, 1]

    # ax.plot(t_env_audio, env_rms, label='Envelope (RMS)')
    # ax.plot(t_peaks, env_rms[peaks], 'ro', label='Peaks')
    # ax.plot(t_valleys, env_rms[valleys], 'bo', label='Valleys')
    t_env_rms = np.arange(len(env_rms))/samplerate
    ax.plot(t_env_rms, env_rms, label='Envelope (RMS)')
    ax.plot(t_env_rms[peaks], env_rms[peaks], 'ro', label='Peaks')
    ax.plot(t_env_rms[valleys], env_rms[valleys], 'bo', label='Valleys')

    ax.legend(loc='lower left')
    ax.set_title(f'Local Vibrato Extent = {vibAmpExtent_dB:.2f} dB')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (a.u.)')


    
    # --- Row 2 ---
    # --- Panel [2,0]: Autocorrelation with vibrato-rate annotations ---
    # pd.plotting.autocorrelation_plot(pitch_contour2, ax=axs[2, 0])
    f_peak, acf_peak, ci = plot_autocorr_with_peak_on_ax(pitch_contour2, f_s_contour2, axs[2,0])

    # Reference lines at 3 Hz and 10 Hz
    frame10Hz = math.floor(1 / 10 * f_s_contour2)
    frame3Hz = math.floor(1 / 3 * f_s_contour2)

    # axs[2, 0].axvline(frame10Hz, color="r")
    # axs[2, 0].text(frame10Hz + 1, -1, "10 Hz", rotation=90)

    # axs[2, 0].axvline(frame3Hz, color="r")
    # axs[2, 0].text(frame3Hz + 1, -1, "3 Hz", rotation=90)

    # Vibrato rate marker (vibFreq = vibRate)
    maxLag = round(f_s_contour2 / vibRate)
    # axs[2, 0].plot(maxLag, 0.85, "x", color="k")
    # axs[2, 0].text(maxLag, 0.9, f"{vibRate:.1f} Hz", ha="center")

    # axs[2, 0].set_title("Autocorrelation and Rate Calculation")
    # axs[2, 0].set_ylabel("Autocorrelation")





    # compute autocorrelation values (nlags = enough to cover 2 Hz)
    # max_lag = int(target_fs / 2)   # lag for 2 Hz
    # acf_vals = acf(filtered_env, nlags=max_lag, fft=True)

    # lags = np.arange(len(acf_vals))
    # freqs = target_fs / lags  # convert lag -> frequency, watch out for lag=0
    # pd.plotting.autocorrelation_plot(filtered_env, ax=axs[2, 1])
    # Skip lag=0 to avoid div by zero
    # lags = lags[1:]
    # freqs = freqs[1:]
    # acf_vals = acf_vals[1:]

    # Mask frequencies between 2 and 20 Hz
    # mask = (freqs >= 2) & (freqs <= 20)

    # axs[2, 1].plot(freqs[mask], acf_vals[mask])
    # axs[2, 1].set_xlabel("Frequency (Hz)")
    # axs[2, 1].set_ylabel("Autocorrelation")
    # axs[2, 1].set_title("Autocorrelation (2–20 Hz)")

    # for f in [3, 10]:
        # axs[2, 1].axvline(f, color="r", linestyle="--")
        # axs[2, 1].text(f + 0.2, 0.8, f"{f} Hz", rotation=90)
    # f_peak, acf_peak, ci = plot_autocorr_with_peak_on_ax(env, samplerate, axs[1,1])
    f_peak, acf_peak, ci = plot_autocorr_with_peak_on_ax(env_rms, samplerate, axs[2,1])




    # Add panel letters
    panel_labels = {
        (0, 0): "A",
        (1, 0): "B",
        (2, 0): "C",
        (0, 1): "D",
        (1, 1): "E",
        (2, 1): "F",
    }

    for (row, col), label in panel_labels.items():
        axs[row, col].text(
            -0.1, 1.05, label, transform=axs[row, col].transAxes,
            fontsize=18, fontweight="bold", va="bottom", ha="right"
        )
    # plt.tight_layout()
    plt.show()

###Full calculation
import numpy as np
import pandas as pd
import librosa
import pyloudnorm as pyln
from scipy.signal import butter, filtfilt, hilbert, find_peaks, peak_prominences

# === CONSTANTS ===
P_REF = 20e-6
TARGET_LUFS = -23.0
N_HARMONICS = 20
BANDWIDTH = 40  # Hz for harmonic filters



# === HELPER FUNCTIONS ===
def bandpass_filter(x, fs, f0, bandwidth):
    nyq = fs / 2.0
    low = max(1.0, f0 - bandwidth/2) / nyq
    high = min(nyq - 1, f0 + bandwidth/2) / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, x)

def compute_rms_envelope(x, fs, rate_hz=6, overlap=0.5):
    win = (1/8)/rate_hz
    window_size = int(win * fs)
    step = int(window_size * (1 - overlap))
    pad = window_size // 2
    rms_env = np.zeros(len(x))
    for i in range(0, len(x) - window_size, step):
        window = x[i:i+window_size]
        rms = np.sqrt(np.mean(window**2))
        center = i + pad
        rms_env[center:center+step] = rms
    return rms_env

def compute_vibrato_extent(envelope, fs, vibRate):
    """Estimate vibrato extent in linear (Pa) or SPL dB difference."""
    wavelength = int(fs / vibRate)
    peaks, _ = find_peaks(envelope, distance=wavelength//2)
    valleys, _ = find_peaks(-envelope, distance=wavelength//2)
    peaks, valleys = np.sort(peaks), np.sort(valleys)
    pairs = []
    for p in peaks:
        v = valleys[valleys < p]
        if len(v): pairs.append((p, v[-1]))
    db_diffs = []
    for p, v in pairs:
        A_p, A_v = envelope[p], envelope[v]
        if A_p > 0 and A_v > 0:
            db_diffs.append(20 * np.log10(A_p / A_v))
    return np.mean(db_diffs) if len(db_diffs) else np.nan

# === MAIN ANALYSIS ===
def analyze_vibrato(signal_pa, fs, f0, vibRate_f0=6, vibExtent_f0=100, file_id="unknown", max_freq=8000):
    results = []

    # --- RMS vibrato extent (unfiltered)
    env_rms = compute_rms_envelope(signal_pa, fs, vibRate_f0)
    vib_extent_rms_db = compute_vibrato_extent(env_rms, fs, vibRate_f0)
    mean_spl_rms = 20 * np.log10(np.mean(np.abs(signal_pa)) / P_REF + 1e-12)

    results.append({
        "file_id": file_id,
        "harmonic": 0,
        "f0_hz": f0,
        "extent_pa": np.nan,
        "extent_spl": vib_extent_rms_db,
        "mean_spl": mean_spl_rms,
        "metric": "RMS Vibrato Extent",
        "type": "original"
    })

    f0 = float(f0)  # ensure scalar

    max_harmonic = int(np.floor(max_freq / f0))
    harmonics = np.arange(1, max_harmonic+1)

    # --- Instantaneous amplitude for harmonics
    for h in harmonics:
        f_h = h * f0
        vib_cents = vibExtent_f0  # expected vibrato extent
        bandwidth_h = 2 * f_h * (2**(vib_cents/1200) - 1) * 1.3
        filtered = bandpass_filter(signal_pa, fs, f_h, bandwidth_h)

        env_pa = np.abs(hilbert(filtered))
        env_spl = 20 * np.log10(env_pa / P_REF + 1e-12)
        extent_pa = compute_vibrato_extent(env_pa, fs, vibRate_f0)
        extent_spl = compute_vibrato_extent(env_spl, fs, vibRate_f0)
        mean_spl = np.mean(env_spl)  # mean SPL of this harmonic

        results.append({
            "file_id": file_id,
            "harmonic": h,
            "f0_hz": f_h,
            "extent_pa": extent_pa,
            "extent_spl": extent_spl,
            "mean_spl": mean_spl,
            "metric": "Instantaneous Amplitude",
            "type": "original"
        })

    return pd.DataFrame(results)


# === LOAD SIGNAL + NORMALIZE ===
def process_file(signal_pa, fs, f0, file_id="unknown", vibRate=6, vibExtent=100):
    if np.isnan(vibRate):
        vibRate = 5.5
    meter = pyln.Meter(fs)
    loudness = meter.integrated_loudness(signal_pa)
    signal_norm = pyln.normalize.loudness(signal_pa, loudness, TARGET_LUFS)

    df_orig = analyze_vibrato(signal_pa, fs, f0, vibRate, vibExtent, file_id)
    df_norm = analyze_vibrato(signal_norm, fs, f0, vibRate, vibExtent, file_id)
    df_norm["type"] = "normalized"
    return pd.concat([df_orig, df_norm], ignore_index=True)

P_REF = 20e-6  # Pa

def pcm_to_float(signal):
    """Convert integer PCM to float in [-1,1] (or pass-through for floats)."""
    if signal.dtype.kind == 'i':
        max_val = np.iinfo(signal.dtype).max
        return signal.astype(np.float64) / (max_val + 1)
    elif signal.dtype.kind == 'f':
        return signal.astype(np.float64)
    else:
        raise ValueError("Unsupported dtype")

def to_mono(x):
    return x.mean(axis=1) if x.ndim > 1 else x

def bandpass(signal, fs, center, bw=100, order=4):
    """Simple Butterworth bandpass around center +/- bw/2 (Hz)."""
    nyq = fs / 2.
    low = max((center - bw/2) / nyq, 1e-6)
    high = min((center + bw/2) / nyq, 0.999999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def rms(signal):
    return np.sqrt(np.mean(signal**2))

def compute_calibration_factor(wav_path, fs_expected=None, tone_freq=1000.0,
                               segment=None, bw=100.0, auto_select=True,
                               plot=False):
    """
    Compute sensitivity S = digital_units_per_Pa using a calibration tone of known 94 dB SPL (=> 1 Pa RMS).
    Args:
      - wav_path: path to calibration WAV (should contain the 1 kHz @ 94 dB tone)
      - fs_expected: optional expected sampling rate (will error if mismatch)
      - tone_freq: calibration tone frequency in Hz (default 1000)
      - segment: tuple (start_s, end_s) in seconds to use for RMS. If None and auto_select True, function will pick the highest-energy contiguous window.
      - bw: bandpass width around tone_freq (Hz) to isolate tone
      - auto_select: if True and segment is None, automatically pick window with max RMS
      - plot: show helpful plots
    Returns:
      dict with keys: 'S' (digital RMS per Pa), 'rms_digital', 'rms_pa' (should be ~1), 'estimated_SPL_dB', 'fs'
    """
    fs, sig = wavfile.read(wav_path)
    if fs_expected is not None and fs != fs_expected:
        raise ValueError(f"Samplerate mismatch: file has {fs}, expected {fs_expected}")
    sig = to_mono(sig)
    x = pcm_to_float(sig)
    x = x - np.mean(x)  # remove DC

    # bandpass to isolate the tone (helps if there is noise)
    x_bp = bandpass(x, fs, center=tone_freq, bw=bw, order=4)

    # choose segment
    total_len = len(x_bp) / fs
    if segment is not None:
        start_s, end_s = segment
        i0 = max(0, int(start_s * fs))
        i1 = min(len(x_bp), int(end_s * fs))
    elif auto_select:
        # slide a window (e.g., 0.5 s) and pick the window with maximum RMS
        win_s = min(0.5, total_len/4)
        win = int(win_s * fs)
        if win < 1:
            win = len(x_bp)
        # compute moving-rms efficiently
        sq = x_bp**2
        cumsum = np.concatenate(([0.0], np.cumsum(sq)))
        rms_vals = np.sqrt((cumsum[win:] - cumsum[:-win]) / win)
        best_idx = np.argmax(rms_vals)
        i0 = best_idx
        i1 = best_idx + win
    else:
        # default use entire file
        i0, i1 = 0, len(x_bp)

    seg = x_bp[i0:i1]
    rms_dig = rms(seg)

    # Known: 94 dB SPL => 1.0 Pa RMS
    p_calib = 1.0  # Pa
    S = rms_dig / p_calib   # digital RMS units per Pa

    # Convert the measured RMS into Pa and compute SPL to verify
    rms_pa = rms_dig / S   # should be ~1.0
    spl_est = 20.0 * np.log10(rms_pa / P_REF)

    if plot:
        t = np.arange(len(x))/fs
        plt.figure(figsize=(10,6))
        plt.subplot(3,1,1)
        plt.plot(t, x, label='raw (float)')
        plt.axvspan(i0/fs, i1/fs, color='orange', alpha=0.25, label='selected segment')
        plt.title('Calibration wav (float)')
        plt.legend()
        plt.subplot(3,1,2)
        plt.plot(t, x_bp, label=f'bandpassed {tone_freq:.0f}Hz')
        plt.axvspan(i0/fs, i1/fs, color='orange', alpha=0.25)
        plt.title('Bandpassed around tone')
        plt.subplot(3,1,3)
        seg_t = np.arange(i0, i1)/fs
        seg_pa = seg / S
        plt.plot(seg_t, seg_pa)
        plt.title(f'Selected segment in Pascals (RMS ≈ {rms(seg_pa):.3f} Pa)')
        plt.ylabel('Pressure [Pa]')
        plt.xlabel('Time [s]')
        plt.tight_layout()
        plt.show()

    return {
        'S': S,
        'rms_digital': rms_dig,
        'rms_pa': rms_pa,
        'estimated_SPL_dB': spl_est,
        'fs': fs,
        'segment_samples': (i0, i1)
    }

def apply_calibration_to_signal(x, S):
    """
    Convert PCM array (int or float) to Pascals using sensitivity S (digital RMS units per Pa).
    Uses same pcm_to_float conversion as above.
    Returns p(t) in Pa.
    """
    # x = to_mono(signal_pcm) if signal_pcm.ndim > 1 else signal_pcm
    x_float = pcm_to_float(x)
    x_float = x_float - np.mean(x_float)
    p = x_float / S
    return p

def instant_spl_from_pa(p):
    eps = 1e-12
    spl = 20.0 * np.log10(np.abs(p) + eps) - 20.0 * np.log10(P_REF)
    return spl

# 1) Compute calibration factor from your calibration file:





###Pseudocode
#Load Wooding/Nix Database
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
vibRatings = pd.read_csv('C:/Users/Reuben/Documents/Code/Promotionsvorhaben/Sandbox/RandiWoodingDatabase.csv')
#Convert (peak-to-peak) cents to mean-to-peak
vibRatings.loc[:,'EXTENT (CENTS)'] = vibRatings['EXTENT (CENTS)']/2

# Step 3: Calculate subgroup agreement
subgroups = ['STUDENT', 'PRO/SEMI-PRO', 'CHORAL DIRECTOR', 'VOICE INSTRUCTOR', 'SPEECH PATH']
experts = ['PRO/SEMI-PRO', 'CHORAL DIRECTOR', 'VOICE INSTRUCTOR', 'SPEECH PATH']

df = vibRatings.copy()

df['EXPERT COUNT YES'] = (
    df['PRO/SEMI-PRO COUNT YES'] +
    df['VOICE INSTRUCTOR COUNT YES'] +
    df['CHORAL DIRECTOR COUNT YES'] +
    df['SPEECH PATH COUNT YES']
)

df['EXPERT COUNT NO'] = (
    df['PRO/SEMI-PRO COUNT NO'] +
    df['VOICE INSTRUCTOR COUNT NO'] +
    df['CHORAL DIRECTOR COUNT NO'] +
    df['SPEECH PATH COUNT NO']
)

df['EXPERT TOTAL'] = (
    df['PRO/SEMI-PRO TOTAL'] +
    df['VOICE INSTRUCTOR TOTAL'] +
    df['CHORAL DIRECTOR TOTAL'] +
    df['SPEECH PATH TOTAL']
)
df['non-vibrato_EXPERT'] = df['EXPERT COUNT NO']/df['EXPERT TOTAL']

    
### And the logistic regression version?
def add_expert_vibrato_label(df, yes_col='EXPERT COUNT YES', no_col='EXPERT COUNT NO', new_col='EXPERT VIBRATO LABEL'):
    """
    Adds a binary vibrato label to the DataFrame based on expert counts.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        yes_col (str): Column name for expert YES counts.
        no_col (str): Column name for expert NO counts.
        new_col (str): Name of the new output column.
    
    Returns:
        pd.DataFrame: Updated DataFrame with new binary label.
    """
    df[new_col] = (df[yes_col] > df[no_col]).astype(int)
    return df

df = add_expert_vibrato_label(df)

# Your data
X = df[['RATE (HZ)', 'EXTENT (CENTS)']].values# vibrato rate and extent (shape [40, 2])
y = df['EXPERT VIBRATO LABEL'].values# binary yes/no labels (shape [40])
weights = df['EXPERT TOTAL'].values# sample weights (shape [40])

# Classifiers to try
models = {
    # 'Logistic Regression': LogisticRegression(),
    # 'SVM': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(max_depth=2),
    # 'Random Forest': RandomForestClassifier(),
    # 'k-NN': KNeighborsClassifier()
}

# Custom cross-validation with sample weights
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    fold_accuracies = []
    
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        w_train = weights[train_idx]
        
        if name == 'k-NN':
            # k-NN doesn't support sample weights during fitting
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, sample_weight=w_train)
        
        y_pred = model.predict(X_test)
        fold_accuracies.append(accuracy_score(y_test, y_pred))
    
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    print(f"{name}: Mean Accuracy = {mean_acc:.3f} (+/- {std_acc:.3f})")

# vibratoDF = pd.read_csv("C:\Users\Reuben\Documents\Code\Promotionsvorhaben\Sandbox\VibratoUntersuchung.csv")
#with open('classVib_Anfang_Retro.pkl', 'rb') as f:
# with open('classVib_Anfang_Retro.pkl', 'rb') as f:
    # anamDB = pickle.load(f)  
# anamDB = pd.read_pickle("C:\Users\Reuben\Documents\Code\Promotionsvorhaben\Sandbox\classVib_Anfang_Retro.pkl")
# df = pd.merge(vibratoDF, anamDB[['audioID', 'stimme.lage.beginn', 'geschlecht']], left_on='id', right_on='audioID')
df = pd.DataFrame({})
#Randfälle Untersuchung
#df = df[df['duration'] > 28]
#df = df[df['id'] == 31]
#for i in [121,372]:#df.index:
#Sort by date, then by id.
# df = pd.read_pickle('vib20250903.pkl')
indexArray = df.index#[0:1]

candFiles = ['0178&2013_12_03&test2.wav',
'0178&2016_04_12&Test 2.wav',
'0000&2014_01_28&test2.wav',
'0080&2010_11_25&Test 2.wav',
'0016&2011_06_07&test2.wav']

path = os.getcwd()
wav_files = glob.glob(os.path.join(path, "*.wav*"))#"*.xlsx"))
mancaFiles = []
avezzoFiles = []
dreiklangFiles = []
vokalFiles = []
comeFiles = []
dBFiles = []
for i in wav_files:
    if i[-5] == '1':
        vokalFiles.append(i)
    if i[-5] == '2':
        dreiklangFiles.append(i)
    if i[-5] == '5':
        mancaFiles.append(i)
    if i[-5] == '6':
        avezzoFiles.append(i)
    if i[-5] == '7':
        comeFiles.append(i)
    if i[-5] == 'B':
        dBFiles.append(i)
        
singingSamples =  dreiklangFiles #dBFiles +   + #vokalFiles #+  + comeFiles vokalFiles +  avezzoFiles + 


# indexArray = df.sort_values('Vibrato-Umfang (F_0)',ascending=False)[['Vibrato-Umfang (F_0)','id','date']].index
for i in singingSamples:#[indexArray == 222]:  range(len(candFiles)):
    wavFilename = i#"0081&2017_11_08&test2.wav"#"0006&2014_07_10&Test 2.wav"#"0081&2017_11_08&test2.wav"#i#df.loc[i].loc['newFilename']#df.sample()['newFilename'].iloc[0]# candFiles[i]
    #Corrupted Files
    if wavFilename in ['C:\\Users\\Reuben\\Documents\\Code\\Promotionsvorhaben\\Sandbox\\0011&2004_03_09&TEST2.wav',
                       'C:\\Users\\Reuben\\Documents\\Code\\Promotionsvorhaben\\Sandbox\\0038&2004_03_16&TEST2.wav',
                       'C:\\Users\\Reuben\\Documents\\Code\\Promotionsvorhaben\\Sandbox\\0055&2005_02_01&test2.wav',
                       'C:\\Users\\Reuben\\Documents\\Code\\Promotionsvorhaben\\Sandbox\\0101&2004_03_09&TEST2.wav',
                       'C:\\Users\\Reuben\\Documents\\Code\\Promotionsvorhaben\\Sandbox\\0105&2004_03_09&TEST2.wav',
                       'C:\\Users\\Reuben\\Documents\\Code\\Promotionsvorhaben\\Sandbox\\0110&2004_03_16&TEST2.wav',
                       'C:\\Users\\Reuben\\Documents\\Code\\Promotionsvorhaben\\Sandbox\\0172&2009_11_17&test2.wav'### THIS ONE ISN'T CORRUPTED
                       ]:
        continue
    geschlecht = np.nan#df.loc[i].loc['geschlecht']
    samplerate, data = read(wavFilename)
    duration = len(data)/samplerate #seconds
    trialNum = wavFilename.split('\\')[-1][-5]
    idNum = int(wavFilename.split('\\')[-1].split('&')[0])
    date = wavFilename.split('\\')[-1].split('&')[1]
    
    # if trialNum == 'B':
        # ref_RMS = np.sqrt(np.mean(data**2))
        # resultDict = {'id':idNum, 
              # 'date':date,
              # 'trialNum':trialNum,
              # 'duration':duration,
              # 'ref_RMS':ref_RMS
              # }
        # df = pd.concat([df, pd.DataFrame.from_records([resultDict])])
        # continue
    # else:
        ##We want to check and see if there is a dB value to calculate:
        ## Check for matching B row
        # mask = (
            # (df["id"] == idNum) &
            # (df["date"] == date) &
            # (df["trialNum"] == "B")
        # )
        # if mask.any():
            # refRMS = df.loc[mask, "ref_RMS"].iloc[0]
            # if refRMS == np.nan:
                # continue
        # else:
            # continue
            # refRMS = np.nan
    refRMS = np.nan
    #Select Middle Trial
    samplerate, middleTrial = selectMiddleTrial(wavFilename)
    #samplerate, finalTrial = selectFinalTrial(wavFilename)
    #Isolate middle 50% of highest pitch
    #samplerate, highestPitch, maxFreq = isolateHighestPitch50(samplerate, middleTrial)

    samplerate, highestPitch, maxFreq, meanFreq = isolateHighestPitch50MF(samplerate, middleTrial, gender=geschlecht)
    #Let's record the duration of our highestPitch sample
    sampleDuration50 = highestPitch.size/samplerate
    # df.loc[i, 'meanFreq'] = meanFreq
    # df.loc[i, 'sampleDuration50'] = sampleDuration50
    #vibrato_Frequency, vibratoPercentage, vibratoStd, amplitudeCents, amplitudeCentsStd, vibFreqTotal, vibAmpTotal, vibAmpStdTotal = vibratoCalcMF(highestPitch, samplerate, gender='weibl.', windowSecs=1)
    #vibrato_Frequency, vibratoStd, amplitudeCents, amplitudeCentsStd, vibratoPercentage = vibratoCalcTraining(highestPitch, samplerate, X_0, classifier, gender='weibl.', windowSecs=1)
    #tremorRate = tremorRateCalc(highestPitch, samplerate)
    # vibRate_f0, vibExtent_f0, vibRate_amp, vibExtent_amp = vibTremorDecision(highestPitch, samplerate, model)
    # vibRate_f0, vibExtent_f0, vibRate_amp, vibExtent_amp, vibExtent_SPL, vibExtent_dB, vibPercent = apply_vibTremorDecision_rolling(highestPitch, samplerate, model, refRMS, window_duration=1)
    
    result = apply_vibTremorDecision_rolling_harmonics(highestPitch, samplerate, model, refRMS, meanFreq,
                                              window_duration=1.0, step_duration=0.01,
                                              max_freq=8000)
    vibRate_f0, vibExtent_f0, vibRate_amp, vibExtent_amp, vibExtent_SPL, vibExtent_dB, vibPercent, vibExtentPa_roll, vibExtentSPL_roll, harmonicSPL_mean  = result
    dummyFilename = r"C:\Users\Reuben\Documents\Code\Promotionsvorhaben\Sandbox\00575&2024_05_17&94dB.wav"
    dBfilename = wavFilename[:-9] + "94dB.wav"
    try:
        cal = compute_calibration_factor(dBfilename, fs_expected=samplerate, tone_freq=1000.0)
        calibration = True
    except:
        cal = compute_calibration_factor(dummyFilename, fs_expected=48000, tone_freq=1000.0)
        calibration = False
        
    refRMS = cal['rms_digital']
    p = apply_calibration_to_signal(highestPitch, cal['S'])  # p is now in Pa

    df_vibrato = process_file(p, samplerate, meanFreq, idNum, vibRate=vibRate_f0, vibExtent=vibExtent_f0)
    # print(df_vibrato)
    origMask = ((df_vibrato['type'] =='original') & (df_vibrato['metric'] == 'RMS Vibrato Extent'))
    normMask0 = ((df_vibrato['type'] =='normalized') & (df_vibrato['metric'] == 'RMS Vibrato Extent'))
    harmMask = ((df_vibrato['type'] =='original') & (df_vibrato['metric'] == 'Instantaneous Amplitude'))
    normMask1 = ((df_vibrato['type'] =='normalized') & (df_vibrato['metric'] == 'Instantaneous Amplitude'))
    
    resultDict = {'id':idNum, 
              'date':date,
              'trialNum':trialNum,
              'duration':duration,
              'sampleDuration50':sampleDuration50,
              'meanFreq':meanFreq,
              'ref_RMS':refRMS,
              'vibRate_f0':vibRate_f0,
              'vibExtent_f0':vibExtent_f0,
              'vibRate_amp':vibRate_amp,
              'vibExtent_amp':vibExtent_amp,
              'vibExtent_SPL':vibExtent_SPL,
              'vibExtent_dB':vibExtent_dB,
              'vibPercent':vibPercent,
              'vibExtentPa_roll':vibExtentPa_roll,
              'vibExtentSPL_roll':vibExtentSPL_roll,
              'harmonicSPL_mean':harmonicSPL_mean,
              'vibExtent2_SPL':df_vibrato.loc[origMask, 'extent_spl'].iloc[0],
              'vibExtent2_SPL':df_vibrato.loc[normMask0, 'extent_spl'].iloc[0],
              'vibExtent_harm':df_vibrato.loc[harmMask,'extent_spl'],
              'vibExtent_hNorm':df_vibrato.loc[normMask1,'extent_spl'],
              'calibration':calibration
              }
    
    #print('TremorRate: ' + str(tremorRate))
    #plt.close('all')
    # df.loc[i, 'vibRate'] = vibrato_Frequency
    # df.loc[i, 'vibPercent'] = vibratoPercentage
    # df.loc[i, 'vibRateStd'] = vibratoStd
    # df.loc[i, 'vibExtent'] = amplitudeCents
    # df.loc[i, 'vibExtentStd'] = amplitudeCentsStd
    
    # visualizeResults(wavFilename, middleTrial, highestPitch, samplerate, gender=geschlecht)
    # if wavFilename == 'D:\\Post-Covid Audio\\00591&2024_11_18&test2.wav':
    # visualizeResultsFull(wavFilename, refRMS, vibRate=vibRate_f0)
    # plot_hilbert_envelope(highestPitch, samplerate, meanFreq, bandwidth=20)
    # prompt = input("Press Enter to continue, q to quit...")
    # if prompt == 'q':
       # break
    df = pd.concat([df, pd.DataFrame.from_records([resultDict])])
    plt.close('all')
    # vibratoFreq3 = vibratoCalc3(highestPitch, samplerate)
    # print('id: ' + str(df.loc[i].loc['id']) + 
         # ', Vibrato Rate: ' + str(round(vibRate_f0, 2))+ 
         # ' Hz')#, Vibrato Percentage: ' + str(round(vibratoPercentage, 2)) + 
    #      ', Vibrato Std: ' + str(round(vibratoStd,2)) +
    #      ', Vibrato Ampitude: ' + str(round(amplitudeCents,2)))
df.to_pickle('vib20251110cal.pkl')

plt.close('all')
#Plot scatterplots:
df_f = df.mask(df.isna(), np.nan)
maskPercent = df_f['Vibrato-Percent'] == 1
df_f = df_f[maskPercent]
# df_f= df_f[df_f['Vibrato-Umfang (F_0)'] < 100]
sns.scatterplot(x='Vibrato-Rate (F_0)', y='Vibrato-Umfang (F_0)', data=df_f)
plt.title('Vibrato-Rate vs. Vibrato-Umfang (F_0)')
plt.savefig('RateUmfangF0.png')
plt.close('all')
sns.scatterplot(x='Vibrato-Rate (Amp)', y='Vibrato-Umfang (Amp)', data=df_f)
plt.title('Vibrato-Rate vs. Vibrato-Umfang (Amp)')
plt.savefig('RateUmfangAmp.png')
plt.close('all')
sns.scatterplot(x='Vibrato-Rate (F_0)', y='Vibrato-Rate (Amp)', data=df_f)
plt.title('Vibrato-Rate (F_0) vs. Vibrato-Rate (Amp)')
plt.savefig('RateF0RateAmp.png')
plt.close('all')
sns.scatterplot(x='Vibrato-Umfang (F_0)', y='Vibrato-Umfang (Amp)', data=df_f)
plt.title('Vibrato-Umfang (F_0) vs. Vibrato-Umfang (Amp)')
plt.savefig('ExtentF0ExtentAmp.png')

sns.scatterplot(x='vibExtent_amp', y='vibExtent_SPL', data=df)
plt.title('Vibrato-Umfang (Amp) vs. Vibrato-Umfang (SPL)')
plt.xlabel('Vibrato-Umfang from Signal RMS (dB)')
plt.ylabel('Vibrato-Umfang from 94 dB Reference Tone (dB)')
plt.show()
plt.savefig('UmfangSPLScatter.png')
plt.close('all')

df['vibExtent_SPL'].hist()
plt.xlabel('Vibrato-Umfang (SPL)')
plt.ylabel('Count')
plt.title('Vibrato-Umfang (SPL) Post-Covid')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

# Clean dataframe
df_f = df.mask(df.isna(), np.nan)

maskSlow = df['Vibrato-Rate (F_0)'] < 5
maskFast = df['Vibrato-Rate (F_0)'] > 7
df_f['VibSpeed'] = 'Normal'
df_f.loc[maskSlow, 'VibSpeed'] = 'Slow'
df_f.loc[maskFast, 'VibSpeed'] = 'Fast'

# 1. Vibrato-Rate (F0) vs. Vibrato-Umfang (F0), color by Extent (Amp)
sns.scatterplot(
    x='Vibrato-Rate (F_0)',
    y='Vibrato-Umfang (F_0)',
    hue='Vibrato-Umfang (Amp)',  # color encodes 3rd variable
    size='Vibrato-Percent',
    palette='viridis',
    data=df_f
)
plt.title('Vibrato-Rate vs. Vibrato-Umfang (F_0) \nHue: Umfang (Amp)')
# plt.colorbar()  # add colorbar
plt.savefig('RateUmfangF0_hueAmp.png')
plt.close('all')

# 2. Vibrato-Umfang (F0) vs. Vibrato-Umfang (Amp), color by Rate (F0)
sns.scatterplot(
    x='Vibrato-Umfang (F_0)',
    y='Vibrato-Umfang (Amp)',
    hue='VibSpeed',
    palette='magma',
    data=df_f
)
plt.title('Vibrato-Umfang (F0) vs. Vibrato-Umfang (Amp) \nHue: Rate (F0)')
# plt.colorbar()
plt.savefig('ExtentF0ExtentAmp_hueRateF0.png')
plt.close('all')

df.tremorRate.hist()
df.to_csv('VibStudy.csv')
#df.to_csv('VibratoCalcTraining.csv')

import statsmodels.api as sm
mask = df[['Vibrato-Rate (F_0)','Vibrato-Rate (Amp)']].notna().any(axis=1)
X = df.loc[mask,'Vibrato-Rate (F_0)'].values
y = df.loc[mask,'Vibrato-Rate (Amp)'].values
model = sm.OLS(y, X).fit()
print(model.params)  # intercept, slope

# df = pd.merge(df, anamDB[['audioID', 'stimme.lage.beginn', 'geschlecht']], left_on='id', right_on='audioID')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mask = df[['Vibrato-Rate (F_0)','Vibrato-Umfang (F_0)','Vibrato-Umfang (Amp)']].notna().any(axis=1)
# --- Example data (replace with your real arrays) ---
# --- 3D scatter plot ---
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(df['Vibrato-Rate (F_0)'].values, df['Vibrato-Umfang (F_0)'].values, df['Vibrato-Umfang (Amp)'].values, 
                c=df['Vibrato-Rate (F_0)'].values, cmap='viridis', s=50, alpha=0.8)

ax.set_xlabel("Vibrato Rate (Hz)")
ax.set_ylabel("Vibrato Extent (Cents)")
ax.set_zlabel("Vibrato Extent (dB)")
ax.set_title("3D Visualization of Vibrato Features")

fig.colorbar(sc, ax=ax, label="Vibrato Rate (Hz)")
plt.show()

moi = ['Vibrato-Rate (F_0)','Vibrato-Umfang (F_0)','Vibrato-Umfang (Amp)']
maskPercent = df['Vibrato-Percent'] == 1
df.loc[maskPercent, moi].describe()


# #Randfälle Untersuchung
# #dfLow = df[df['vibratoPercentage'] < 0.1]
# #for i in [121,372]:#df.index:
# #investigate = df[df['id'].isin(idList)][['id', 'date', 'vibratoFreq4']].sort_values(by=['id','date'])
# #investigate = df[df['id'] == 69].sort_values(by=['id','date'])
# #Slow vibrato Untersuchung
# vibrato3Hz = pd.read_csv('VibratoMFNarrow1SecWindow.csv')#'VibratoMF1SecWindow.csv')
# with open('classVib_Anfang_Retro.pkl', 'rb') as f:
    # anamDB = pickle.load(f)
    
# #df = pd.merge(vibrato3Hz, anamDB[['id', 'stimme.lage.beginn', 'vibrato.stabilitaet', 'geschlecht']], on='id')
# #df = vibrato3Hz
# df = df1
#df = pd.merge(df, anamDB[['id', 'stimme.lage.beginn', 'vibrato.stabilitaet', 'geschlecht']], on='id')
#df = pd.read_csv('VibratoMFAmpFinal.csv')
df = pd.read_csv('VibratoCalcTraining.csv')
#df = df.sort_values('vibFreqAmp', ascending=False)
df = df.sort_values('vibRate', ascending=False)
#df = df.sort_values('vibratoFreqMF')#, ascending=False)
###PERFORM DATETIME TRANSFORM!!! Need function
#df = df1.copy()
#investigate = df1[df1['id'].isin(list(slow_final['id']))].sort_values(by=['id','date'])
#investigate = df[df['id'].isin(list(fast_final['id']))].sort_values(by=['id','date'])
investigate = df#[df['vibFreqTotal'].isna()]#[df['geschlecht'].isna()]#[df['id'].isin(list(df_0['audioID']))].sort_values(by=['id','date'])
#investigate = df[((df['geschlecht'] == 'weibl.') & (df['meanFreq'] < 400))]
#indexArray = investigate.index
#indexArray = [561, 601, 489, 560, 91, 289, 403, 430, 488, 431] #default sort
#indexArray = [602, 598, 607, 304, 604, 597, 606, 589, 241, 619] #weibl.
indexArray = d.index#[188, 160, 52, 53, 54, 55, 227, 226, 257, 363]
for i in indexArray:#[indexArray == 222]:
    wavFilename = d.loc[i].loc['newFilename']
    geschlecht = d.loc[i].loc['geschlecht']
    samplerate, data = read(wavFilename)
    #Select Middle Trial
    samplerate, middleTrial = selectMiddleTrial(wavFilename)
    #Isolate middle 50% of highest pitch
    #samplerate, highestPitch, maxFreq = isolateHighestPitch90(samplerate, middleTrial)
    samplerate, highestPitch, maxFreq, meanFreq = isolateHighestPitch50MF(samplerate, middleTrial, geschlecht)
    #Piano key frequencies:
    #g5: 784
    #g4: 392
    #c4: 261
    #c3: 131
    #if maxFreq > 1000:
    #    samplerate, highestPitch, maxFreq = isolateHighestPitch2(samplerate, middleTrial)
    #vibrato_Frequency, vibratoPercentage, vibratoStd, amplitudeCents, amplitudeCentsStd, x, y, z = vibratoCalcMF(highestPitch, samplerate, gender='weibl.', windowSecs=1)
    #vibrato_Frequency, vibratoStd, amplitudeCents, amplitudeCentsStd, vibratoPercentage = vibratoCalcTraining(highestPitch, samplerate, X_0, classifier, gender='weibl.', windowSecs=1)

    visualizeResults(wavFilename, middleTrial, highestPitch, samplerate, geschlecht)
    print('i: ', str(i), ', Gender: ',str(geschlecht),', meanFreq: ', str(meanFreq))
    print(df.loc[i, 'newFilename'])
    #print('id: ' + str(df.loc[i,'id']) + 
    #      ' date: ' + str(df.loc[i,'date']) +
    #      ', Vibrato8: ' + str(round(df.loc[i,'vibRate'], 2))+ 
    #      ' Hz, Vibrato Percentage: ' + str(round(df.loc[i,'vibPercent'], 2)) + 
    #      ', Vibrato Std: ' + str(round(df.loc[i,'vibRateStd'],2)) +
    #      ', Vibrato Amp: ' + str(round(df.loc[i,'vibExtent'],2)),
    #      ', Vibrato Amp Std: ' + str(round(df.loc[i,'vibExtentStd'],2)))

    prompt = input("Press Enter to continue, q to quit...")
    if prompt == 'q':
       break
    plt.close('all')


def visualizeResultsPaper(wavFilename, middleTrial, isolatedHighestPitch, samplerate, begin50, end50, gender=np.nan, ):
    plt.close('all')
    samplerate, data = read(wavFilename)
    #Visualize results:ipy
    fig, ax = plt.subplots(4)
    #fig.tight_layout(pad=100.0)
    #fig.SubplotParams(bottom=0.1)
    plt.rcParams['font.size'] = '14'
    #ax[0].plot(np.arange(len(data))/samplerate,data)
    #ax[0].set_title('Original Audio Waveform')
    #ax[0].axes.xaxis.set_visible(False)
    #ax[0].axes.yaxis.set_visible(False)
    ax[0].plot(np.arange(len(middleTrial))/samplerate,middleTrial)
    ax[0].set_title('Audio File')
    #ax[0].axes.xaxis.set_visible(False)
    ax[0].axes.yaxis.set_visible(False)
    sound = Sound(middleTrial, samplerate)
    #Create a praat pitch object,
    if gender == 'männl.':
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c3-g4
    elif gender == 'weibl.':
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c4-g5
    else:
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c4-g5 
    #This provides the frequencies of the sample.
    pitch_contour = pitch.selected_array['frequency']
    pitchContLength = pitch.selected_array['frequency'].size
    wavLength = len(middleTrial)
    f_s_contour = pitchContLength/wavLength*samplerate
    ax[1].plot(np.arange(len(pitch_contour))/f_s_contour,pitch_contour)
    ax[1].set_title('Pitch Contour With Middle 50% of Sustained Tone Highlighted')
    ax[1].set_ylabel('Freq (Hz)', fontsize=14)
    #ax[1].set_xlabel('Time (s)', fontsize=14, labelpad=0)
    ax[1].plot(np.arange(begin50,end50)/f_s_contour,pitch_contour[begin50:end50], color='r')
    ax[1].text(0.13, 220, 'A3')
    ax[1].text(0.82, 277, 'C#4')
    ax[1].text(2, 329, 'E4')
    #ax[2].axes.xaxis.set_visible(False)
    sound2 = Sound(isolatedHighestPitch, samplerate)
    #Create a praat pitch object,
    if gender == 'männl.':
        pitch2 = call(sound2, "To Pitch", 0.0, 100, 390) #c3-g4
    elif gender == 'weibl.':
        pitch2 = call(sound2, "To Pitch", 0.0, 261, 784) #c4-g5
    else:
        pitch2 = call(sound2, "To Pitch", 0.0, 60, 784) #c4-g5
    #This provides the frequencies of the sample.
    pitch_contour2 = pitch2.selected_array['frequency']
    pitchContLength2 = pitch2.selected_array['frequency'].size
    wavLength2 = len(isolatedHighestPitch)
    f_s_contour2 = pitchContLength2/wavLength2*samplerate
    vibFreq = df.loc[i,'vibratoFreqMF']
    wavLengthWindow = 0.75*1/vibFreq*f_s_contour2
    meanFreq = pitch_contour2.mean()
    maxPeaks = find_peaks(pitch_contour2, distance=wavLengthWindow)[0]
    prominences = scipy.signal.peak_prominences(pitch_contour2, maxPeaks)[0]/2
    contour_heights = pitch_contour2[maxPeaks] - prominences
    #plt.close()
    #plt.plot(pitch_contour)
    #plt.ylabel('Frequency (Hz)')
    #plt.title('Amplitude Calculation')
    
    ax[2].plot(np.arange(begin50,(begin50+len(pitch_contour2)))/f_s_contour2,pitch_contour2)
    ax[2].plot((begin50+maxPeaks)/f_s_contour2, pitch_contour2[maxPeaks], "x")
    ax[2].vlines(x=((begin50+maxPeaks)/f_s_contour2), ymin=contour_heights, ymax=pitch_contour2[maxPeaks],color='r')
    ax[2].set_title('Middle 50% with Amplitude Calculation')
    ax[2].set_ylabel('Freq (Hz)', fontsize=14)
    #ax[3].axes.xaxis.set_visible(False)
    pd.plotting.autocorrelation_plot(pitch_contour2, ax=ax[3])
    frame10Hz = math.floor(1/10*f_s_contour)
    frame3Hz = math.floor(1/3*f_s_contour)
    plt.axvline(frame10Hz, color='r')
    plt.text((frame10Hz+1),0.5,'10 Hz',rotation=90)
    plt.axvline(frame3Hz, color='r')
    plt.text((frame3Hz+1),0.5,'3 Hz',rotation=90)
    ax[3].set_title('Autocorrelation and Frequency Calculation')
    maxLag = round(f_s_contour/vibFreq)
    ax[3].plot(24, 0.8817, "x")
    plt.text(24,0.925,'5.6 Hz',horizontalalignment='center')
    #ax[2].set_ylabel('Freq (Hz)', fontsize=14)
    #lag = np.abs(acorr)[frame12Hz:frame3Hz].argmax() + 1 + frame12Hz
    #maxLag = covariance[frame10Hz:frame4Hz].argmax() + frame10Hz
    #maxLag = acorr[frame10Hz:frame3Hz].argmax() + frame10Hz
    plt.show()

#visualizeResultsPaper(wavFilename, middleTrial, highestPitch, samplerate, 283, 506, geschlecht)

def isolateHighestPitch50Paper(samplerate, selectedMiddleTrial, gender=np.nan):
    #Can we get the pitch contour?
    sound = Sound(selectedMiddleTrial, samplerate)
    #Create a praat pitch object,
    #Probably need upper frequency bound 2x potential sung frequency
    #Piano key frequencies:
    #g5: 784
    #g4: 392
    #c4: 261
    #c3: 131
    if gender == 'männl.':
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c3-g4
    elif gender == 'weibl.':
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c4-g5
    else:
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c4-g5
    #This provides the frequencies of the sample.
    pitch_contour = pitch.selected_array['frequency']
    #What is the new samplingrate?
    f_s_Audio = sound.sampling_frequency
    wavLength = sound.values.size
    pitchContLength = pitch.selected_array['frequency'].size
    f_s_contour = pitchContLength/wavLength*f_s_Audio
    #So we have an interval of a minor third between the highest note and the middle note. 
    #Yeah!
    ###Ok, now we want to find the corresponding interval in the pitch_contour array    
        #That are within a minor third of the maximum pitch. 
    #This is a little sensitive to pitch artifacts.
    #maxFreq = max(pitch_contour)
    #maxIndex = argmax(pitch_contour)
    #Let's just grab the middle value of the selection and hope.
    maxIndex = round(len(pitch_contour)/2)
    maxFreq = pitch_contour[maxIndex]
    #Minor 3rd ratio is 6:5
    thresholdFreq = maxFreq*5/6
    beginInterval = np.where(pitch_contour[:maxIndex] < thresholdFreq)[0][-1]
    if np.where(pitch_contour[maxIndex:] < thresholdFreq)[0].size != 0:
        endInterval = maxIndex + np.where(pitch_contour[maxIndex:] < thresholdFreq)[0][0]
    else:
        endInterval = len(pitch_contour)  
    #Let's take the middle fifty percent of this interval.
    #If you save the audio file here, you could use it for all data analysis.
    #close('all')
    begin50 = beginInterval + round((endInterval - beginInterval)*.25)
    print(str(begin50))
    end50 =  beginInterval + round((endInterval - beginInterval)*.75)
    print(str(end50))
    visualCheckSelection(pitch_contour, begin50, end50)
    #prompt = input("Press Enter to continue...")
    #beginAudioInterval = startMiddleAttempt + round(begin50*f_s_Audio/f_s_contour)
    beginAudioInterval = round(begin50*f_s_Audio/f_s_contour) #+ startMiddleAttempt
    
    #endAudioInterval = startMiddleAttempt + round(end50*f_s_Audio/f_s_contour)
    endAudioInterval = round(end50*f_s_Audio/f_s_contour) #+ startMiddleAttempt 
        
    middleFiftyPercentHighestPitch = selectedMiddleTrial[beginAudioInterval:endAudioInterval]
    #Let's get the mean pitch of this interval
    meanFreq = pitch_contour[begin50:end50].mean()
    return samplerate, middleFiftyPercentHighestPitch, maxFreq, meanFreq

i = 599
wavFilename = df.loc[i].loc['newFilename']
geschlecht = df.loc[i].loc['geschlecht']
samplerate, data = read(wavFilename)
#Select Middle Trial
samplerate, middleTrial = selectMiddleTrial(wavFilename)
#Isolate middle 50% of highest pitch
#samplerate, highestPitch, maxFreq = isolateHighestPitch50(samplerate, middleTrial)
samplerate, highestPitch, maxFreq, meanFreq = isolateHighestPitch50Paper(samplerate, middleTrial, gender=geschlecht)
#Let's record the duration of our highestPitch sample
sampleDuration50 = highestPitch.size/samplerate
df.loc[i, 'meanFreq'] = meanFreq
df.loc[i, 'sampleDuration50'] = sampleDuration50
#vibrato_Frequency, vibratoPercentage, vibratoStd, amplitudeCents, amplitudeCentsStd = vibratoCalcMF(highestPitch, samplerate, gender='weibl.', windowSecs=1)

#plt.close('all')
#df.loc[i, 'vibratoFreqMF'] = vibrato_Frequency
#df.loc[i, 'vibratoPercentageMF'] = vibratoPercentage
#df.loc[i, 'vibratoStdMF'] = vibratoStd
#df.loc[i, 'vibFreqAmp'] = amplitudeCents
#df.loc[i, 'vibFreqAmpStd'] = amplitudeCentsStd
sound = Sound(middleTrial, samplerate)
pitch = call(sound, "To Pitch", 0.0, 261, 784) #c4-g5 
pitch_contour = pitch.selected_array['frequency']
#What is the new samplingrate?
f_s_Audio = sound.sampling_frequency
wavLength = sound.values.size
pitchContLength = pitch.selected_array['frequency'].size
f_s_contour = pitchContLength/wavLength*f_s_Audio

visualizeResultsPaper(wavFilename, middleTrial, highestPitch, samplerate, 283, 506, geschlecht)


# #visualCheckSelection(pitch_contour, 283, 506)
# def visualCheckSelection(sample0, beginSample, endSample):
    # plt.close('all')
    # plt.subplots(1)
    # plot(sample0, color='b')
    # plot(np.arange(beginSample,endSample),sample0[beginSample:endSample], color='r')
    

plt.close('all')
begin50 = 283
end50 = 506
samplerate, data = read(wavFilename)
#Visualize results:ipy
fig, ax = plt.subplots(4)
#fig.tight_layout(pad=100.0)
#fig.SubplotParams(bottom=0.1)
plt.rcParams['font.size'] = '14'
#ax[0].plot(np.arange(len(data))/samplerate,data)
#ax[0].set_title('Original Audio Waveform')
#ax[0].axes.xaxis.set_visible(False)
#ax[0].axes.yaxis.set_visible(False)
ax[0].plot(np.arange(len(middleTrial))/samplerate,middleTrial/middleTrial.max())
ax[0].set_title('Audio File')
ax[0].set_xlabel('Time (s)', fontsize=20)#, labelpad=0)
ax[0].set_ylabel('Amplitude', fontsize=20)#, labelpad=0)

#ax[0].axes.xaxis.set_visible(False)
#ax[0].axes.yaxis.set_visible(False)
sound = Sound(middleTrial, samplerate)
#Create a praat pitch object,
if geschlecht == 'männl.':
    pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c3-g4
elif geschlecht == 'weibl.':
    pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c4-g5
else:
    pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c4-g5 
#This provides the frequencies of the sample.
pitch_contour = pitch.selected_array['frequency']
pitchContLength = pitch.selected_array['frequency'].size
wavLength = len(middleTrial)
f_s_contour = pitchContLength/wavLength*samplerate
ax[1].plot(np.arange(len(pitch_contour))/f_s_contour,pitch_contour)
ax[1].set_title('Pitch Contour With Middle 50% of Sustained Tone Highlighted')
ax[1].set_ylabel('Freq (Hz)', fontsize=20)
ax[1].set_xlabel('Time (s)', fontsize=20)#, labelpad=0)
ax[1].plot(np.arange(begin50,end50)/f_s_contour,pitch_contour[begin50:end50], color='r')
ax[1].text(0.13, 220, 'A3')
ax[1].text(0.82, 277, 'C#4')
ax[1].text(2, 329, 'E4')
#ax[2].axes.xaxis.set_visible(False)
sound2 = Sound(highestPitch, samplerate)
#Create a praat pitch object,
if geschlecht == 'männl.':
    pitch2 = call(sound2, "To Pitch", 0.0, 100, 390) #c3-g4
elif geschlecht == 'weibl.':
    pitch2 = call(sound2, "To Pitch", 0.0, 261, 784) #c4-g5
else:
    pitch2 = call(sound2, "To Pitch", 0.0, 60, 784) #c4-g5
#This provides the frequencies of the sample.
pitch_contour2 = pitch2.selected_array['frequency']
pitchContLength2 = pitch2.selected_array['frequency'].size
wavLength2 = len(highestPitch)
f_s_contour2 = pitchContLength2/wavLength2*samplerate
vibFreq = df.loc[i,'vibratoFreqMF']
wavLengthWindow = 0.75*1/vibFreq*f_s_contour2
meanFreq = pitch_contour2.mean()
maxPeaks = find_peaks(pitch_contour2, distance=wavLengthWindow)[0][1:-1]
prominences = scipy.signal.peak_prominences(pitch_contour2, maxPeaks)[0]/2
#contour_heights = pitch_contour2[maxPeaks] - prominences
contour_heights = pitch_contour2[maxPeaks] - prominences
#plt.close()
#plt.plot(pitch_contour)
#plt.ylabel('Frequency (Hz)')
#plt.title('Amplitude Calculation')

ax[2].plot(np.arange(begin50,(begin50+len(pitch_contour2)))/f_s_contour2,pitch_contour2)
ax[2].plot((begin50+maxPeaks)/f_s_contour2, pitch_contour2[maxPeaks], "x")
ax[2].vlines(x=((begin50+maxPeaks)/f_s_contour2), ymin=contour_heights, ymax=pitch_contour2[maxPeaks],color='r')
ax[2].set_title('Middle 50% with Extent Calculation')
ax[2].set_ylabel('Freq (Hz)', fontsize=20)
ax[2].set_xlabel('Time (s)', fontsize=20)#, labelpad=0)

pd.plotting.autocorrelation_plot(pitch_contour2, ax=ax[3])
ax[3].set_xlabel('Lag (frames)')
frame10Hz = math.floor(1/10*f_s_contour)
frame3Hz = math.floor(1/3*f_s_contour)
ax[3].axvline(frame10Hz, color='r')
ax[3].text((frame10Hz+1),0.5,'10 Hz',rotation=90)#, fontsize=24)
ax[3].axvline(frame3Hz, color='r')
ax[3].text((frame3Hz+1),0.5,'3 Hz',rotation=90)#, fontsize=24)
ax[3].set_title('Autocorrelation and Rate Calculation')
maxLag = round(f_s_contour/vibFreq)
ax[3].plot(24, 0.8817, "x")
ax[3].text(24,0.925,'5.6 Hz',horizontalalignment='center')#, fontsize=24)

fig.tight_layout()
#ax[3].axes.xaxis.set_visible(False)

#ax[2].set_ylabel('Freq (Hz)', fontsize=14)
#lag = np.abs(acorr)[frame12Hz:frame3Hz].argmax() + 1 + frame12Hz
#maxLag = covariance[frame10Hz:frame4Hz].argmax() + frame10Hz
#maxLag = acorr[frame10Hz:frame3Hz].argmax() + frame10Hz
#fig.tight_layout(h_pad=5.0)
#plt.show()
fig.subplots_adjust(hspace=0.4)

for i in range(len(ax)):
    
    for item in ([ax[i].title, ax[i].xaxis.label, ax[i].yaxis.label] +
                 ax[i].get_xticklabels() + ax[i].get_yticklabels()):
        item.set_fontsize(24)



plt.close('all')
begin50 = 283
end50 = 506
samplerate, data = read(wavFilename)
#Visualize results:ipy
fig, ax = plt.subplots(4)
#fig.tight_layout(pad=100.0)
#fig.SubplotParams(bottom=0.1)
plt.rcParams['font.size'] = '14'
#ax[0].plot(np.arange(len(data))/samplerate,data)
#ax[0].set_title('Original Audio Waveform')
#ax[0].axes.xaxis.set_visible(False)
#ax[0].axes.yaxis.set_visible(False)
ax[0].plot(np.arange(len(middleTrial))/samplerate,middleTrial)
ax[0].set_title('Audio File')
ax[0].set_xlabel('Time (s)', fontsize=14)#, labelpad=0)
ax[0].set_ylabel('Amplitude', fontsize=14)#, labelpad=0)

#ax[0].axes.xaxis.set_visible(False)
#ax[0].axes.yaxis.set_visible(False)
sound = Sound(middleTrial, samplerate)
#Create a praat pitch object,
if geschlecht == 'männl.':
    pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c3-g4
elif geschlecht == 'weibl.':
    pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c4-g5
else:
    pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c4-g5 
#This provides the frequencies of the sample.
pitch_contour = pitch.selected_array['frequency']
pitchContLength = pitch.selected_array['frequency'].size
wavLength = len(middleTrial)
f_s_contour = pitchContLength/wavLength*samplerate
ax[1].plot(np.arange(len(pitch_contour))/f_s_contour,pitch_contour)
ax[1].set_title('Pitch Contour With Middle 50% of Sustained Tone Highlighted')
ax[1].set_ylabel('Freq (Hz)', fontsize=14)
ax[1].set_xlabel('Time (s)', fontsize=14)#, labelpad=0)
ax[1].plot(np.arange(begin50,end50)/f_s_contour,pitch_contour[begin50:end50], color='r')
ax[1].text(0.13, 220, 'A3')
ax[1].text(0.82, 277, 'C#4')
ax[1].text(2, 329, 'E4')
#ax[2].axes.xaxis.set_visible(False)
sound2 = Sound(highestPitch, samplerate)
#Create a praat pitch object,
if geschlecht == 'männl.':
    pitch2 = call(sound2, "To Pitch", 0.0, 100, 390) #c3-g4
elif geschlecht == 'weibl.':
    pitch2 = call(sound2, "To Pitch", 0.0, 261, 784) #c4-g5
else:
    pitch2 = call(sound2, "To Pitch", 0.0, 60, 784) #c4-g5
#This provides the frequencies of the sample.
pitch_contour2 = pitch2.selected_array['frequency']
pitchContLength2 = pitch2.selected_array['frequency'].size
wavLength2 = len(highestPitch)
f_s_contour2 = pitchContLength2/wavLength2*samplerate
vibFreq = df.loc[i,'vibratoFreqMF']
wavLengthWindow = 0.75*1/vibFreq*f_s_contour2
meanFreq = pitch_contour2.mean()
maxPeaks = find_peaks(pitch_contour2, distance=wavLengthWindow)[0][1:-1]
prominences = scipy.signal.peak_prominences(pitch_contour2, maxPeaks)[0]/2
#contour_heights = pitch_contour2[maxPeaks] - prominences
contour_heights = pitch_contour2[maxPeaks] - prominences
#plt.close()
#plt.plot(pitch_contour)
#plt.ylabel('Frequency (Hz)')
#plt.title('Amplitude Calculation')

pd.plotting.autocorrelation_plot(pitch_contour2, ax=ax[2])
ax[2].set_xlabel('Lag (frames)')
frame10Hz = math.floor(1/10*f_s_contour)
frame3Hz = math.floor(1/3*f_s_contour)
ax[2].axvline(frame10Hz, color='r')
ax[2].text((frame10Hz+1),0.5,'10 Hz',rotation=90)
ax[2].axvline(frame3Hz, color='r')
ax[2].text((frame3Hz+1),0.5,'3 Hz',rotation=90)
ax[2].set_title('Autocorrelation and Frequency Calculation')
maxLag = round(f_s_contour/vibFreq)
ax[2].plot(24, 0.8817, "x")
ax[2].text(24,0.925,'5.6 Hz',horizontalalignment='center')

ax[3].plot(np.arange(begin50,(begin50+len(pitch_contour2)))/f_s_contour2,pitch_contour2)
ax[3].plot((begin50+maxPeaks)/f_s_contour2, pitch_contour2[maxPeaks], "x")
ax[3].vlines(x=((begin50+maxPeaks)/f_s_contour2), ymin=contour_heights, ymax=pitch_contour2[maxPeaks],color='r')
ax[3].set_title('Middle 50% with Extent Calculation')
ax[3].set_ylabel('Freq (Hz)', fontsize=14)
ax[3].set_xlabel('Time (s)', fontsize=14)#, labelpad=0)
#ax[3].axes.xaxis.set_visible(False)

#ax[2].set_ylabel('Freq (Hz)', fontsize=14)
#lag = np.abs(acorr)[frame12Hz:frame3Hz].argmax() + 1 + frame12Hz
#maxLag = covariance[frame10Hz:frame4Hz].argmax() + frame10Hz
#maxLag = acorr[frame10Hz:frame3Hz].argmax() + frame10Hz
#fig.tight_layout(h_pad=5.0)
#plt.show()
fig.subplots_adjust(hspace=0.4)

for i in range(len(ax)):
    
    for item in ([ax[i].title, ax[i].xaxis.label, ax[i].yaxis.label] +
                 ax[i].get_xticklabels() + ax[i].get_yticklabels()):
        item.set_fontsize(24)

from scipy.stats import zscore

# Select numeric columns
# Calculate z-scores for numeric columns
# df_z = df.copy()
df = pd.read_pickle('vib20250904.pkl')
mask = df[moi].notna().any(axis=1)
df_z = df.loc[mask,moi].apply(zscore)

# 'filename' remains intact

df_z = pd.concat([df.loc[mask,'newFilename'], df_z], axis=1)
for i in moi:
    df_z[i] = df_z[i].apply(lambda x: np.abs(x))
df_z['zSum'] = df_z[moi].sum(axis=1)
print(df_z.sort_values('zSum')['newFilename'])

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import find_peaks, spectrogram
import parselmouth
from parselmouth.praat import call
import scipy.signal
import math

# --- your isolate function (unchanged) ---
def isolateHighestPitch50Paper(samplerate, selectedMiddleTrial, gender=np.nan):
    sound = Sound(selectedMiddleTrial, samplerate)
    if gender == 'männl.':
        pitch = call(sound, "To Pitch", 0.0, 60, 1000)
    elif gender == 'weibl.':
        pitch = call(sound, "To Pitch", 0.0, 60, 1000)
    else:
        pitch = call(sound, "To Pitch", 0.0, 60, 1000)

    pitch_contour = pitch.selected_array['frequency']
    f_s_Audio = sound.sampling_frequency
    wavLength = sound.values.size
    pitchContLength = pitch_contour.size
    f_s_contour = pitchContLength / wavLength * f_s_Audio

    maxIndex = round(len(pitch_contour) / 2)
    maxFreq = pitch_contour[maxIndex]
    thresholdFreq = maxFreq * 5 / 6
    beginInterval = np.where(pitch_contour[:maxIndex] < thresholdFreq)[0][-1]

    if np.where(pitch_contour[maxIndex:] < thresholdFreq)[0].size != 0:
        endInterval = maxIndex + np.where(pitch_contour[maxIndex:] < thresholdFreq)[0][0]
    else:
        endInterval = len(pitch_contour)

    begin50 = beginInterval + round((endInterval - beginInterval) * .25)
    end50 = beginInterval + round((endInterval - beginInterval) * .75)

    beginAudioInterval = round(begin50 * f_s_Audio / f_s_contour)
    endAudioInterval = round(end50 * f_s_Audio / f_s_contour)
    middleFiftyPercentHighestPitch = selectedMiddleTrial[beginAudioInterval:endAudioInterval]

    meanFreq = pitch_contour[begin50:end50].mean()
    return samplerate, middleFiftyPercentHighestPitch, maxFreq, meanFreq, begin50, end50, pitch_contour, f_s_contour


import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import find_peaks, spectrogram, peak_prominences
import parselmouth
from parselmouth.praat import call
import math


visualizeResultsFull(wavFilename, vibRate=vibRate_f0)
# ref_wav = '0178&2013_12_03&test2.wav'
ref_wav = '0000&2014_01_28&test2.wav'

visualizeResultsFull(ref_wav,vibRate=10, bandwidth=8)

from scipy.signal import hilbert

# 1. Original audio
samplerate, data = read(wavFilename)
wavData = data / np.max(np.abs(data))
time_raw = np.arange(len(wavData)) / samplerate

# 2. Hilbert envelope (same length as audio)
analytic = hilbert(wavData)
env = np.abs(analytic)

# 3. Scale envelope to match waveform max amplitude
env_scaled = env / np.max(env) * np.max(wavData)

fig, axs = plt.subplots(3, 2, figsize=(16, 12))
# 4. Plot
axs[1,1].plot(time_raw, wavData, color="steelblue", linewidth=0.8, label="Raw Signal")
axs[1,1].plot(time_raw, env_scaled, color="darkorange", linewidth=1.2, label="Amplitude Envelope")
axs[1,1].set_title("Raw Signal with Envelope Overlay")
axs[1,1].set_ylabel("Amplitude")
axs[1,1].legend()
axs[1,1].grid(True)
plt.show()


###Mad Ramblings
df_full = df[df['vibPercent'] == 1]
cols = ['vibRate_f0', 'vibExtent_f0', 'vibRate_amp', 'vibExtent_dB']
zCols = ['ZvibRate_f0', 'ZvibExtent_f0', 'ZvibRate_amp', 'ZvibExtent_dB']
df_full[zCols] = (df_full[cols] - df_full[cols].mean()) / df_full[cols].std()
df_full['zSum'] = df_full[zCols].sum(axis=1)
df_full['xSum'] = df_full['ZvibExtent_f0'] - df_full['ZvibExtent_dB']

print(df_full.sort_values('xSum',ascending=False).head()[['id','date']])
      # id       date
# 417    6 2014-07-10
# 413    6 2014-07-10
# 537  236 2013-07-09
# 538  236 2014-07-10
# 175   65 2006-05-16


print(df_full.sort_values('xSum',ascending=True).head()[['id','date']])
      # id       date
# 711   54 2015-12-08
# 767   81 2017-11-08
# 723  143 2019-04-30
# 189    9 2006-05-16
# 194    9 2006-05-16

mask = df_full['geschlecht'] == 'männl.'
print(df_full[mask].sort_values('xSum',ascending=False).head()[['id','date']])
      # id       date
# 417    6 2014-07-10
# 413    6 2014-07-10
# 537  236 2013-07-09
# 538  236 2014-07-10
# 175   65 2006-05-16


print(df_full[mask].sort_values('xSum',ascending=True).head()[['id','date']])
      # id       date
# 711   54 2015-12-08
# 767   81 2017-11-08
# 723  143 2019-04-30
# 189    9 2006-05-16
# 194    9 2006-05-16

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

P_REF = 20e-6  # reference pressure in Pascals

def bandpass_filter(signal, fs, center_freq, bandwidth=20, order=4):
    """Apply a Butterworth bandpass filter centered at center_freq ± bandwidth/2."""
    nyquist = fs / 2.0
    low = max((center_freq - bandwidth/2) / nyquist, 1e-6)
    high = min((center_freq + bandwidth/2) / nyquist, 0.999999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def plot_hilbert_envelope(signal_pa, fs, f0, bandwidth=20):
    """
    Compute and plot Hilbert amplitude envelopes for the fundamental and first 3 harmonics.
    The signal should already be calibrated in Pascals.
    """
    harmonics = np.arange(1,6)
    t = np.arange(len(signal_pa)) / fs
    eps = 1e-12

    results = {}

    fig, axs = plt.subplots(len(harmonics), 1, figsize=(12, 10))
    fig.suptitle("Hilbert Amplitude Envelopes (in Pascals and dB SPL)", fontsize=14, y=0.98)

    for i, h in enumerate(harmonics):
        # Bandpass filter around harmonic
        filtered = bandpass_filter(signal_pa, fs, h*f0, bandwidth)
        envelope_pa = np.abs(hilbert(filtered))
        envelope_db = 20 * np.log10(envelope_pa + eps) - 20 * np.log10(P_REF)

        results[h] = {"filtered": filtered, "envelope_pa": envelope_pa, "envelope_db": envelope_db}

        # Plot
        ax1 = axs[i]
        color1, color2 = "tab:blue", "tab:red"
        ax1.plot(t, envelope_pa, color=color1, label="Envelope [Pa]")
        ax1.set_ylabel("Amplitude [Pa]", color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_title(f"Harmonic {h}: {h*f0:.1f} Hz", loc="left")

        ax2 = ax1.twinx()
        ax2.plot(t, envelope_db, color=color2, alpha=0.7, label="Envelope [dB SPL]")
        ax2.set_ylabel("Amplitude [dB SPL]", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

    axs[-1].set_xlabel("Time [s]")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    return results


# Example usage:
# plot_hilbert_envelope("example.wav", fs=44100, f0=220, bandwidth=30)

### Computing Pa and SPL
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

P_REF = 20e-6  # Pa

def pcm_to_float(signal):
    """Convert integer PCM to float in [-1,1] (or pass-through for floats)."""
    if signal.dtype.kind == 'i':
        max_val = np.iinfo(signal.dtype).max
        return signal.astype(np.float64) / (max_val + 1)
    elif signal.dtype.kind == 'f':
        return signal.astype(np.float64)
    else:
        raise ValueError("Unsupported dtype")

def to_mono(x):
    return x.mean(axis=1) if x.ndim > 1 else x

def bandpass(signal, fs, center, bw=100, order=4):
    """Simple Butterworth bandpass around center +/- bw/2 (Hz)."""
    nyq = fs / 2.
    low = max((center - bw/2) / nyq, 1e-6)
    high = min((center + bw/2) / nyq, 0.999999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def rms(signal):
    return np.sqrt(np.mean(signal**2))

def compute_calibration_factor(wav_path, fs_expected=None, tone_freq=1000.0,
                               segment=None, bw=100.0, auto_select=True,
                               plot=True):
    """
    Compute sensitivity S = digital_units_per_Pa using a calibration tone of known 94 dB SPL (=> 1 Pa RMS).
    Args:
      - wav_path: path to calibration WAV (should contain the 1 kHz @ 94 dB tone)
      - fs_expected: optional expected sampling rate (will error if mismatch)
      - tone_freq: calibration tone frequency in Hz (default 1000)
      - segment: tuple (start_s, end_s) in seconds to use for RMS. If None and auto_select True, function will pick the highest-energy contiguous window.
      - bw: bandpass width around tone_freq (Hz) to isolate tone
      - auto_select: if True and segment is None, automatically pick window with max RMS
      - plot: show helpful plots
    Returns:
      dict with keys: 'S' (digital RMS per Pa), 'rms_digital', 'rms_pa' (should be ~1), 'estimated_SPL_dB', 'fs'
    """
    fs, sig = wavfile.read(wav_path)
    if fs_expected is not None and fs != fs_expected:
        raise ValueError(f"Samplerate mismatch: file has {fs}, expected {fs_expected}")
    sig = to_mono(sig)
    x = pcm_to_float(sig)
    x = x - np.mean(x)  # remove DC

    # bandpass to isolate the tone (helps if there is noise)
    x_bp = bandpass(x, fs, center=tone_freq, bw=bw, order=4)

    # choose segment
    total_len = len(x_bp) / fs
    if segment is not None:
        start_s, end_s = segment
        i0 = max(0, int(start_s * fs))
        i1 = min(len(x_bp), int(end_s * fs))
    elif auto_select:
        # slide a window (e.g., 0.5 s) and pick the window with maximum RMS
        win_s = min(0.5, total_len/4)
        win = int(win_s * fs)
        if win < 1:
            win = len(x_bp)
        # compute moving-rms efficiently
        sq = x_bp**2
        cumsum = np.concatenate(([0.0], np.cumsum(sq)))
        rms_vals = np.sqrt((cumsum[win:] - cumsum[:-win]) / win)
        best_idx = np.argmax(rms_vals)
        i0 = best_idx
        i1 = best_idx + win
    else:
        # default use entire file
        i0, i1 = 0, len(x_bp)

    seg = x_bp[i0:i1]
    rms_dig = rms(seg)

    # Known: 94 dB SPL => 1.0 Pa RMS
    p_calib = 1.0  # Pa
    S = rms_dig / p_calib   # digital RMS units per Pa

    # Convert the measured RMS into Pa and compute SPL to verify
    rms_pa = rms_dig / S   # should be ~1.0
    spl_est = 20.0 * np.log10(rms_pa / P_REF)

    if plot:
        t = np.arange(len(x))/fs
        plt.figure(figsize=(10,6))
        plt.subplot(3,1,1)
        plt.plot(t, x, label='raw (float)')
        plt.axvspan(i0/fs, i1/fs, color='orange', alpha=0.25, label='selected segment')
        plt.title('Calibration wav (float)')
        plt.legend()
        plt.subplot(3,1,2)
        plt.plot(t, x_bp, label=f'bandpassed {tone_freq:.0f}Hz')
        plt.axvspan(i0/fs, i1/fs, color='orange', alpha=0.25)
        plt.title('Bandpassed around tone')
        plt.subplot(3,1,3)
        seg_t = np.arange(i0, i1)/fs
        seg_pa = seg / S
        plt.plot(seg_t, seg_pa)
        plt.title(f'Selected segment in Pascals (RMS ≈ {rms(seg_pa):.3f} Pa)')
        plt.ylabel('Pressure [Pa]')
        plt.xlabel('Time [s]')
        plt.tight_layout()
        plt.show()

    return {
        'S': S,
        'rms_digital': rms_dig,
        'rms_pa': rms_pa,
        'estimated_SPL_dB': spl_est,
        'fs': fs,
        'segment_samples': (i0, i1)
    }

def apply_calibration_to_signal(x, S):
    """
    Convert PCM array (int or float) to Pascals using sensitivity S (digital RMS units per Pa).
    Uses same pcm_to_float conversion as above.
    Returns p(t) in Pa.
    """
    # x = to_mono(signal_pcm) if signal_pcm.ndim > 1 else signal_pcm
    x_float = pcm_to_float(x)
    x_float = x_float - np.mean(x_float)
    p = x_float / S
    return p

def instant_spl_from_pa(p):
    eps = 1e-12
    spl = 20.0 * np.log10(np.abs(p) + eps) - 20.0 * np.log10(P_REF)
    return spl

# 1) Compute calibration factor from your calibration file:

wavFilename = "00575&2024_12_09&test2.wav"
dBfilename = wavFilename[:-9] + "94dB.wav"
samplerate, middleTrial = selectMiddleTrial(wavFilename)
samplerate, highestPitch, maxFreq, meanFreq = isolateHighestPitch50MF(samplerate, middleTrial, gender=geschlecht)
cal = compute_calibration_factor(dBfilename, fs_expected=samplerate, tone_freq=1000.0)
refRMS = cal['rms_digital']
vibRate_f0, vibExtent_f0, vibRate_amp, vibExtent_amp, vibExtent_SPL, vibExtent_dB, vibPercent = apply_vibTremorDecision_rolling(highestPitch, samplerate, model, refRMS, window_duration=1)


print("S (digital RMS units per Pa) =", cal['S'])
print("Estimated SPL of calibration file (dB) =", cal['estimated_SPL_dB'])  # should be ~94 dB

# 2) Convert another recording to Pascals:
from scipy.io import wavfile
# fs2, rec = wavfile.read("my_recording.wav")
p = apply_calibration_to_signal(highestPitch, cal['S'])  # p is now in Pa

# 3) Compute overall SPL (RMS) of the recording
rms_rec_pa = np.sqrt(np.mean(p**2))
spl_rec = 20*np.log10(rms_rec_pa / P_REF)
print("Recording RMS SPL (dB) =", spl_rec)


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# --- assume you've already done your calibration ---
# S = calibration factor (digital RMS units per Pa)
# e.g. from `calibrate_and_plot_in_pa()` earlier:
# cal = calibrate_and_plot_in_pa("calibration_1kHz_94dB.wav")
# S = cal["S"]

def pcm_to_float(signal):
    if signal.dtype.kind == 'i':
        return signal.astype(np.float64) / (np.iinfo(signal.dtype).max + 1)
    elif signal.dtype.kind == 'f':
        return signal.astype(np.float64)
    else:
        raise ValueError("Unsupported dtype")

def to_mono(x):
    return x.mean(axis=1) if x.ndim > 1 else x

def plot_calibrated_signal_in_pa(sig, fs, S, fs_expected=None):
    """
    Converts a recording to Pascals using the calibration factor S and plots it.
    """
    # fs, sig = wavfile.read(wav_path)
    # if fs_expected is not None and fs != fs_expected:
        # raise ValueError(f"Samplerate mismatch: file has {fs}, expected {fs_expected}")

    # Convert PCM → float → Pascals
    x = pcm_to_float(to_mono(sig))
    x = x - np.mean(x)  # remove DC
    p = x / S           # now in Pascals

    # Time axis
    t = np.arange(len(p)) / fs

    # Plot
    plt.figure(figsize=(12,5))
    plt.plot(t, p, color='steelblue')
    plt.title("Calibrated signal (in Pascals)")
    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [Pa]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"RMS pressure: {np.sqrt(np.mean(p**2)):.3f} Pa")
    return {'fs': fs, 'p': p}

# Example usage:
# plot_calibrated_signal_in_pa("my_recording.wav", S=cal["S"], fs_expected=44100)

# Example use:
# cal = calibrate_and_plot_in_pa("calibration_1kHz_94dB.wav")
S = cal["S"]
p = plot_calibrated_signal_in_pa(highestPitch, samplerate, S, fs_expected=samplerate)

import pandas as pd
import numpy as np
from scipy.signal import hilbert

def compute_vibrato_extent(envelope, vibAmpGuess):
    wavelength = 1.0 / vibAmpGuess * samplerate
    window = int(np.floor(wavelength * 0.75))
    if window <= 0:
        window = 1

    maxPeaks = find_peaks(envelope, distance=window)[0]
    prominences = peak_prominences(envelope, maxPeaks)[0] / 2
    meanAmp = envelope.mean()
    extentEstimate = np.nanmedian(prominences[1:-1])
    # extentEstimate = prominences.mean()
    return extentEstimate

def analyze_vibrato(signal_pa, fs, f0, bandwidth=20, file_id="unknown",n_h=20, vibRate=5.5):
    """Compute vibrato extent for fundamental + 3 harmonics and store in a DataFrame."""
    harmonics = np.arange(1,n_h)
    results = []

    for h in harmonics:
        # Filter + Hilbert
        filtered = bandpass_filter(signal_pa, fs, h*f0, bandwidth)
        env_pa = np.abs(hilbert(filtered))
        env_spl = 20 * np.log10(env_pa / P_REF + 1e-12)

        # Compute extent
        extent_pa = compute_vibrato_extent(env_pa, vibRate)

        
        # --- Vibrato extent in SPL (peak-to-trough / 2) ---
        extent_spl = compute_vibrato_extent(env_spl, vibRate)

        # --- Store results ---
        results.append({
            "file_id": file_id,
            "harmonic": h,
            "freq": f0*h,
            "vibrato_extent_pa": extent_pa,
            "vibrato_extent_spl": extent_spl,
        })


    df = pd.DataFrame(results)
    return df

# p = your calibrated signal in Pascals
# fs = samplerate
# f0 = mean fundamental (Hz)

df_vibrato = analyze_vibrato(p, samplerate, meanFreq, bandwidth=30, file_id="singer_01_noteA3", vibRate=vibRate_f0)
print(df_vibrato)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# Constants

fs = 44100  # Sampling rate (Hz)
duration = 1.0  # seconds
t = np.arange(0, duration, 1/fs)
p_ref = 20e-6  # Reference pressure (Pa)

# Synthetic sine parameters

f0 = 440  # Fundamental frequency (Hz)
A = 0.1   # Peak amplitude in Pa (~80 dB SPL)

# Simple sine wave

sine = A * np.sin(2 * np.pi * f0 * t)

# Vibrato sine wave (modulate frequency)

f_vib = 5  # Hz vibrato rate
depth = 10  # Hz vibrato extent
vib_sine = A * np.sin(2*np.pi*(f0 + depth*np.sin(2*np.pi*f_vib*t)) * t)

# Compute Hilbert amplitude envelopes

env_sine = np.abs(hilbert(sine))
env_vib = np.abs(hilbert(vib_sine))

# Convert envelopes to dB SPL

env_sine_spl = 20 * np.log10(env_sine / p_ref + 1e-12)
env_vib_spl = 20 * np.log10(env_vib / p_ref + 1e-12)

# Plot results

plt.figure(figsize=(12,6))

plt.subplot(2,1,1)
plt.plot(t, env_sine_spl)
plt.title("Hilbert Envelope - Pure Sine (dB SPL)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [dB SPL]")

plt.subplot(2,1,2)
plt.plot(t, env_vib_spl)
plt.title("Hilbert Envelope - Vibrato Sine (dB SPL)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [dB SPL]")

plt.tight_layout()
plt.show()

###Test plot
test = plot_calibrated_signal_in_pa(highestPitch, samplerate, S, fs_expected=samplerate)
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

# --- Helper: bandpass filter ---
def bandpass_filter(x, fs, f0, bandwidth=20):
    nyq = fs / 2.0
    low = max(1.0, f0 - bandwidth/2) / nyq
    high = min(nyq - 1, f0 + bandwidth/2) / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, x)

# --- Reference SPL conversion ---
P_REF = 20e-6  # Pa reference for 0 dB SPL

# === USER SIGNAL ===
# Replace this with your calibrated signal in Pascals:
# e.g. test['p'] and test['fs'] are your measured data
fs = test['fs']
signal_pa = test['p']

# --- Time vector based on actual signal length ---
sig_len = len(signal_pa)
t = np.arange(sig_len) / fs

# --- Fundamental frequency (mean fundamental) ---
f0 = meanFreq  # Hz

# --- Harmonics to analyze ---
harmonics = np.arange(1, 7)  # first six harmonics
bandwidth = 40  # Hz per harmonic filter

# ===============================================
# FIGURE 1: Hilbert amplitude envelopes in dB SPL
# ===============================================
plt.figure(figsize=(12, 10))

for i, h in enumerate(harmonics, 1):
    f_h = h * f0
    # Filter the harmonic
    filtered = bandpass_filter(signal_pa, fs, f_h, bandwidth)
    
    # Hilbert amplitude envelope (in Pa)
    env_pa = np.abs(hilbert(filtered))
    
    # Convert to SPL
    env_spl = 20 * np.log10(env_pa / P_REF + 1e-12)
    
    # Plot SPL envelope
    ax = plt.subplot(6, 1, i)
    ax.plot(t, env_spl, lw=1.2, color='tab:red')
    ax.set_ylabel("SPL [dB]")
    # ax.set_xlim(0, min(0.2, t[-1]))  # zoom into first 200 ms
    ax.set_title(f"Harmonic {h}: {f_h:.1f} Hz")
    if i < 6:
        ax.set_xticklabels([])

plt.xlabel("Time [s]")
plt.tight_layout()
plt.show()

# ===============================================
# FIGURE 2: Hilbert amplitude envelopes in Pascals
# ===============================================
plt.figure(figsize=(12, 10))

for i, h in enumerate(harmonics, 1):
    f_h = h * f0
    # Filter the harmonic
    filtered = bandpass_filter(signal_pa, fs, f_h, bandwidth)
    
    # Hilbert amplitude envelope (in Pa)
    env_pa = np.abs(hilbert(filtered))
    
    # Plot PA envelope
    ax = plt.subplot(6, 1, i)
    ax.plot(t, env_pa, lw=1.2, color='tab:blue')
    ax.set_ylabel("Pressure [Pa]")
    # ax.set_xlim(0, min(0.2, t[-1]))
    ax.set_title(f"Harmonic {h}: {f_h:.1f} Hz")
    if i < 6:
        ax.set_xticklabels([])

plt.xlabel("Time [s]")
plt.tight_layout()
plt.show()


###Loudness Normalization
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pyloudnorm as pyln
from scipy.signal import butter, filtfilt, hilbert

# === PARAMETERS ===
P_REF = 20e-6   # reference for SPL (Pa)
harmonics = np.arange(1, 7)
bandwidth = 40  # Hz for bandpass per harmonic
target_loudness = -23.0  # LUFS for perceptual normalization

# === SIGNAL ===
signal_pa = np.array(test['p'])
fs = test['fs']
T = len(signal_pa) / fs
t = np.linspace(0, T, len(signal_pa))
f0 = meanFreq

# --- Bandpass filter helper
def bandpass_filter(x, fs, f0, bandwidth=20):
    nyq = fs / 2.0
    low = max(1.0, f0 - bandwidth/2) / nyq
    high = min(nyq - 1, f0 + bandwidth/2) / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, x)

# --- Perceptual loudness normalization (ITU-R BS.1770)
meter = pyln.Meter(fs)
loudness = meter.integrated_loudness(signal_pa)
signal_norm = pyln.normalize.loudness(signal_pa, loudness, target_loudness)

# === PROCESS EACH HARMONIC ===
fig, axes = plt.subplots(6, 2, figsize=(12, 14))
plt.subplots_adjust(hspace=0.5)

for i, h in enumerate(harmonics):
    f_h = h * f0

    # --- Filter harmonic
    filtered_orig = bandpass_filter(signal_pa, fs, f_h, bandwidth)
    filtered_norm = bandpass_filter(signal_norm, fs, f_h, bandwidth)

    # --- Hilbert envelopes
    env_orig = np.abs(hilbert(filtered_orig))
    env_norm = np.abs(hilbert(filtered_norm))

    # --- Convert to SPL
    env_orig_spl = 20 * np.log10(env_orig / P_REF + 1e-12)
    env_norm_spl = 20 * np.log10(env_norm / P_REF + 1e-12)

    # --- Left panel: unnormalized
    ax1 = axes[i, 0]
    ax1.plot(t, env_orig_spl, color='tab:blue', lw=1)
    # ax1.set_xlim(0, 0.2)
    ax1.set_ylim(20, 120)
    ax1.set_ylabel("SPL [dB]")
    ax1.set_title(f"Harmonic {h}: {f_h:.1f} Hz (Original)")

    # --- Right panel: normalized
    ax2 = axes[i, 1]
    ax2.plot(t, env_norm_spl, color='tab:red', lw=1)
    # ax2.set_xlim(0, 0.2)
    ax2.set_ylim(20, 120)
    ax2.set_ylabel("SPL [dB]")
    ax2.set_title(f"Harmonic {h}: {f_h:.1f} Hz (Normalized)")

for ax in axes[-1, :]:
    ax.set_xlabel("Time [s]")

plt.tight_layout()
plt.show()

print(f"Original loudness: {loudness:.2f} LUFS → Normalized to {target_loudness:.2f} LUFS")

###Full calculation
import numpy as np
import pandas as pd
import librosa
import pyloudnorm as pyln
from scipy.signal import butter, filtfilt, hilbert, find_peaks, peak_prominences

# === CONSTANTS ===
P_REF = 20e-6
TARGET_LUFS = -23.0
N_HARMONICS = 20
BANDWIDTH = 40  # Hz for harmonic filters



# === HELPER FUNCTIONS ===
def bandpass_filter(x, fs, f0, bandwidth):
    nyq = fs / 2.0
    low = max(1.0, f0 - bandwidth/2) / nyq
    high = min(nyq - 1, f0 + bandwidth/2) / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, x)

def compute_rms_envelope(x, fs, rate_hz=6, overlap=0.5):
    win = (1/8)/rate_hz
    window_size = int(win * fs)
    step = int(window_size * (1 - overlap))
    pad = window_size // 2
    rms_env = np.zeros(len(x))
    for i in range(0, len(x) - window_size, step):
        window = x[i:i+window_size]
        rms = np.sqrt(np.mean(window**2))
        center = i + pad
        rms_env[center:center+step] = rms
    return rms_env

def compute_vibrato_extent(envelope, fs, vibRate):
    """Estimate vibrato extent in linear (Pa) or SPL dB difference."""
    wavelength = int(fs / vibRate)
    peaks, _ = find_peaks(envelope, distance=wavelength//2)
    valleys, _ = find_peaks(-envelope, distance=wavelength//2)
    peaks, valleys = np.sort(peaks), np.sort(valleys)
    pairs = []
    for p in peaks:
        v = valleys[valleys < p]
        if len(v): pairs.append((p, v[-1]))
    db_diffs = []
    for p, v in pairs:
        A_p, A_v = envelope[p], envelope[v]
        if A_p > 0 and A_v > 0:
            db_diffs.append(20 * np.log10(A_p / A_v))
    return np.mean(db_diffs) if len(db_diffs) else np.nan

# === MAIN ANALYSIS ===
def analyze_vibrato(signal_pa, fs, f0, vibRate_f0=6, vibExtent_f0=100, file_id="unknown"):
    results = []

    # --- RMS vibrato extent (unfiltered)
    env_rms = compute_rms_envelope(signal_pa, fs, vibRate_f0)
    vib_extent_rms_db = compute_vibrato_extent(env_rms, fs, vibRate_f0)
    results.append({
        "file_id": file_id,
        "harmonic": 0,
        "f0_hz": f0,
        "extent_pa": np.nan,
        "extent_spl": vib_extent_rms_db,
        "metric": "RMS Vibrato Extent",
        "type": "original"
    })

    # --- Instantaneous amplitude for harmonics
    for h in range(1, N_HARMONICS+1):
        f_h = h * f0
        vib_cents = vibExtent_f0  # expected vibrato extent
        bandwidth_h = 2 * (h * f0) * (2**(vib_cents/1200) - 1) * 1.3
        filtered = bandpass_filter(signal_pa, fs, h * f0, bandwidth_h)

        env_pa = np.abs(hilbert(filtered))
        env_spl = 20 * np.log10(env_pa / P_REF + 1e-12)
        extent_pa = compute_vibrato_extent(env_pa, fs, vibRate_f0)
        extent_spl = compute_vibrato_extent(env_spl, fs, vibRate_f0)
        results.append({
            "file_id": file_id,
            "harmonic": h,
            "f0_hz": f_h,
            "extent_pa": extent_pa,
            "extent_spl": extent_spl,
            "metric": "Instantaneous Amplitude",
            "type": "original"
        })
    return pd.DataFrame(results)

# === LOAD SIGNAL + NORMALIZE ===
def process_file(signal_pa, fs, f0, file_id="unknown", vibRate=5.5, vibExtent=100):
    if vibRate == np.nan:
        vibRate = 5.5
    meter = pyln.Meter(fs)
    loudness = meter.integrated_loudness(signal_pa)
    signal_norm = pyln.normalize.loudness(signal_pa, loudness, TARGET_LUFS)

    df_orig = analyze_vibrato(signal_pa, fs, f0, vibRate, ,vibExtent, file_id)
    df_norm = analyze_vibrato(signal_norm, fs, f0, vibRate, ,vibExtent, file_id)
    df_norm["type"] = "normalized"
    return pd.concat([df_orig, df_norm], ignore_index=True)

# === EXAMPLE USAGE ===
# signal_pa = test['p']
# fs = test['fs']
# f0 = meanFreq
# df_vibrato = process_file(signal_pa, fs, f0, "singer_01_noteA3", vibRate=6)
# print(df_vibrato)

###Ok, we have our multidimensional DF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# --- Make sure each row has matching harmonic numbers ---
df = df.copy()

df['harmonic_nums'] = df['vibExtent_harm'].apply(lambda x: list(range(1, len(x)+1)))
df['harmonic_nums_norm'] = df['vibExtent_hNorm'].apply(lambda x: list(range(1, len(x)+1)))

# === EXPAND ORIGINAL HARMONIC DATA ===
df_exp = df.explode(['vibExtent_harm', 'harmonic_nums'], ignore_index=True)
df_exp.rename(columns={'harmonic_nums': 'harmonic'}, inplace=True)
df_exp['vibExtent_harm'] = pd.to_numeric(df_exp['vibExtent_harm'], errors='coerce')

# === EXPAND NORMALIZED HARMONIC DATA ===
df_norm = df.explode(['vibExtent_hNorm', 'harmonic_nums_norm'], ignore_index=True)
df_norm.rename(columns={'harmonic_nums_norm': 'harmonic'}, inplace=True)
df_norm['vibExtent_hNorm'] = pd.to_numeric(df_norm['vibExtent_hNorm'], errors='coerce')

# === GROUP AND DESCRIBE ===
stats_orig = df_exp.groupby('harmonic')['vibExtent_harm'].describe()
stats_norm = df_norm.groupby('harmonic')['vibExtent_hNorm'].describe()

# Compute standard errors
stats_orig['sem'] = df_exp.groupby('harmonic')['vibExtent_harm'].sem()
stats_norm['sem'] = df_norm.groupby('harmonic')['vibExtent_hNorm'].sem()

# === COMPARATIVE PLOT ===
plt.figure(figsize=(10, 5))
plt.errorbar(
    stats_orig.index, stats_orig['mean'],
    yerr=stats_orig['sem'],
    fmt='-o', lw=2, capsize=3, label='Original'
)
plt.errorbar(
    stats_norm.index, stats_norm['mean'],
    yerr=stats_norm['sem'],
    fmt='--s', lw=2, capsize=3, label='Loudness-normalized'
)
plt.title("Vibrato Extent by Harmonic — Original vs Loudness-Normalized")
plt.xlabel("Harmonic Number")
plt.ylabel("Mean Vibrato Extent (units of vibExtent_harm)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# === OPTIONAL: Boxplot comparison ===
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

df_exp.boxplot(column='vibExtent_harm', by='harmonic', ax=axes[0], grid=False, color='b')
axes[0].set_title("Original Signal")
axes[0].set_ylabel("Vibrato Extent")

df_norm.boxplot(column='vibExtent_hNorm', by='harmonic', ax=axes[1], grid=False, color='r')
axes[1].set_title("Loudness-Normalized Signal")
axes[1].set_xlabel("Harmonic Number")
axes[1].set_ylabel("Vibrato Extent")

plt.suptitle("Distribution of Vibrato Extent Across Harmonics")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

### Frequency-Region Visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Gather harmonic frequencies
df_exp = df.copy().explode('vibExtent_hNorm', ignore_index=True)
df_exp['harmonic'] = np.tile(np.arange(1, 21), len(df))
df_exp['vibExtent_hNorm'] = pd.to_numeric(df_exp['vibExtent_hNorm'], errors='coerce')
df_exp['f_h'] = df_exp['harmonic'] * df_exp['meanFreq']

# --- Make adaptive bins based on quantiles of f_h
n_bins = 10  # adjust as needed
freq_bins = np.quantile(df_exp['f_h'], np.linspace(0, 1, n_bins + 1))
freq_labels = [f"{int(freq_bins[i])}-{int(freq_bins[i+1])} Hz" for i in range(n_bins)]
df_exp['freq_bin'] = pd.cut(df_exp['f_h'], bins=freq_bins, labels=freq_labels, include_lowest=True)

# --- Group stats
stats = df_exp.groupby('freq_bin')['vibExtent_hNorm'].describe()
stats['sem'] = df_exp.groupby('freq_bin')['vibExtent_hNorm'].sem()

# --- Plot (mean ± sem)
plt.figure(figsize=(10, 5))
plt.errorbar(stats.index, stats['mean'], yerr=stats['sem'], fmt='-o', lw=2, capsize=4)
plt.title("Vibrato Extent by Frequency Region (Loudness-Normalized)")
plt.xlabel("Frequency Bin (Hz)")
plt.ylabel("Vibrato Extent (SPL)")
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Optional: inspect bin fill levels
print(df_exp['freq_bin'].value_counts().sort_index())

### Spectrogram with Vibrato Extent (limited to 8 kHz)
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

# --- Parameters ---
signal_pa = p['p']        # calibrated signal in Pascals
f0 = meanFreq             # sustained fundamental frequency (Hz)
fs = samplerate
MAX_FREQ = 8000           # limit analysis to 8 kHz
P_REF = 20e-6             # Pa reference for 0 dB SPL

# --- Helper: robust bandpass filter ---
def bandpass_filter(x, fs, f0, bandwidth):
    x = np.atleast_1d(np.squeeze(x))
    if x.ndim != 1 or len(x) == 0:
        return np.zeros_like(x)
    nyq = fs / 2
    low = (f0 - bandwidth / 2) / nyq
    high = (f0 + bandwidth / 2) / nyq
    low = max(low, 1e-6)
    high = min(high, 0.999)
    if low >= high:
        return np.zeros_like(x)
    from scipy.signal import butter, filtfilt, lfilter
    b, a = butter(4, [low, high], btype='band')
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        try:
            return lfilter(b, a, x)
        except ValueError:
            return np.zeros_like(x)
    try:
        return filtfilt(b, a, x)
    except ValueError:
        return np.zeros_like(x)

# --- Storage ---
freqs, mean_spl, extent_spl = [], [], []

# --- Analyze harmonics ---
h = 1
while True:
    f_h = h * f0
    if f_h > MAX_FREQ:
        break

    # Vibrato bandwidth: ±1 semitone around harmonic
    vib_cents = 120
    bandwidth_hz = 2 * f_h * (2**(vib_cents / 1200) - 1)

    filtered = bandpass_filter(signal_pa, fs, f_h, bandwidth_hz)
    if np.all(filtered == 0):
        h += 1
        continue

    # Envelope & SPL
    env = np.abs(hilbert(filtered))
    env_spl = 20 * np.log10(env / P_REF + 1e-12)

    mean_level = np.mean(env_spl)
    vib_extent = np.std(env_spl) * 2  # approximate peak-to-peak modulation depth

    freqs.append(f_h)
    mean_spl.append(mean_level)
    extent_spl.append(vib_extent)
    h += 1

# --- Convert to arrays ---
freqs = np.array(freqs)
mean_spl = np.array(mean_spl)
extent_spl = np.array(extent_spl)

# --- Plot Vibrato Spectrum ---
plt.figure(figsize=(10, 6))
plt.errorbar(freqs, mean_spl, yerr=extent_spl / 2, fmt='-o', capsize=4,
             color='tab:red', alpha=0.85, label='Harmonic Vibrato Spectrum')
plt.title("Harmonic Spectrum with Vibrato Extent (≤ 8 kHz)")
plt.xlabel("Harmonic Frequency (Hz)")
plt.ylabel("Level [dB SPL]")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

