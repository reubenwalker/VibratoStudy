#PSEUDOCODE
#Load DataFrame
#Run through folder for all audio files
#Calculate:
    #If more than one trial, use middle trial:
        #Beginning Frame
        #End Frame
    #Mean Frequency
    #Identify first L/R frames within a minor third of this frequency
        #Beginning Frame
        #End Frame
    #Middle 50% of sustained pitch
        #Beginning Frame
        #End Frame
    #Calculate measures over rolling 1 s window
        #Mean vibrato frequency
            #Std vibrato frequency
        #Mean vibrato extent
            #Std vibrato extent
        #Vibrato percentage


#Necessary Libraries:
from parselmouth.praat import call
from parselmouth import Sound
import scipy
from scipy.signal import hilbert, butter, sosfilt, find_peaks
from scipy.io.wavfile import read
import statsmodels.api as sm
from statsmodels.tsa.stattools import acovf, acf
import matplotlib
import pandas as pd
#from scipy.signal import savgol_filter
import pickle

#Necessary functions:
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
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
    minIndex = peaks[0][argmin(distance_from_midpoint)]
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


###Method:
#Select and read .wav file
wavFilename = 'sample.wav'
#We'll approximate the gender as male 'männl.' as the pitch was E4
geschlecht = 'männl.'
#from scipy.io.wavfile import read
samplerate, data = read(wavFilename)
#If more than one trial, use middle trial:
    #Beginning Frame
    #End Frame
#Select Middle Trial
samplerate, middleTrial = selectMiddleTrial(wavFilename)
#The pitch contour of the exercise was calculated using the PRAAT Parselmouth library in python and
    #the middle fifty percent of the highest sustained tone was isolated.
#Mean Frequency
#Identify first L/R frames within a minor third of this frequency
    #NOTE: For the synthetic sample file,
        #there is a pitch artifact that makes this calculation not function correctly.
        #Thankfully, the sample selected for the calculation will still serve the illustration.
    #Beginning Frame
    #End Frame
#Isolate middle 50% of highest pitch
def isolateHighestPitch50MF(samplerate, selectedMiddleTrial, gender=np.nan):
    #Calculate pitch contour with PRAAT Parselmouth
    sound = Sound(selectedMiddleTrial, samplerate)
    #Create a praat pitch object,
    #Piano key frequencies:
    #g5: 784
    #g4: 392
    #c4: 261
    #c3: 131
    #Performed pitches are D4/E4 for tenor/bass, D5/E5 for alto/soprano
    ###Octave doesn't matter for isolating the middle pitch.
    #Just look between 60-1000 Hz
    if gender == 'männl.':
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c3-g4
    elif gender == 'weibl.':
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c4-g5
    else:
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c4-g5
    #pitch_contour provides the frequencies of the sample.
    pitch_contour = pitch.selected_array['frequency']
    #Calculate new samplingrate
    f_s_Audio = sound.sampling_frequency
    wavLength = sound.values.size
    pitchContLength = pitch.selected_array['frequency'].size
    f_s_contour = pitchContLength/wavLength*f_s_Audio
    #So we have an interval of a minor third between the highest note and the middle note.
    ###Ok, now we want to find the corresponding interval in the pitch_contour array
        #that are within a minor third of the maximum pitch.
    #This is a little sensitive to pitch artifacts.
    #maxFreq = max(pitch_contour)
    #maxIndex = argmax(pitch_contour)
    #Let's just grab the middle value of the selection look to either side.
    maxIndex = round(len(pitch_contour)/2)
    maxFreq = pitch_contour[maxIndex]
    #Minor 3rd ratio is 6:5
    thresholdFreq = maxFreq*5/6
    #Find earliest value to the left of maxIndex within this threshold
    beginInterval = np.where(pitch_contour[:maxIndex] < thresholdFreq)[0][-1]
    #Find latest value to the right of maxIndex within this threshold
    if np.where(pitch_contour[maxIndex:] < thresholdFreq)[0].size != 0:
        endInterval = maxIndex + np.where(pitch_contour[maxIndex:] < thresholdFreq)[0][0]
    else:
        endInterval = len(pitch_contour)
    #Let's take the middle fifty percent of this interval.
    begin50 = beginInterval + round((endInterval - beginInterval)*.25)
    #print(str(begin50))
    end50 =  beginInterval + round((endInterval - beginInterval)*.75)
    #visualCheckSelection(pitch_contour, begin50, end50)
    #prompt = input("Press Enter to continue...")
    beginAudioInterval = round(begin50*f_s_Audio/f_s_contour) #+ startMiddleAttempt
    endAudioInterval = round(end50*f_s_Audio/f_s_contour) #+ startMiddleAttempt
    middleFiftyPercentHighestPitch = selectedMiddleTrial[beginAudioInterval:endAudioInterval]
    #Let's get the mean pitch of this interval
    meanFreq = pitch_contour[begin50:end50].mean()
    return samplerate, middleFiftyPercentHighestPitch, maxFreq, meanFreq
samplerate, highestPitch, maxFreq, meanFreq = isolateHighestPitch50MF(samplerate, middleTrial, gender=geschlecht)
#Record the duration of our highestPitch sample
sampleDuration50 = highestPitch.size/samplerate
#df.loc[i, 'meanFreq'] = meanFreq
#df.loc[i, 'sampleDuration50'] = sampleDuration50
#The mean and standard deviation of vibrato frequency were calculated by
    #performing an autocorrelation over windows with a one second duration.
###Massive Calculation
    #subsequent vibratoCalc needs autocorrVib and vibAmpRoll functions
def autocorrVib3Hz(pitch_contour, f_s_contour):
    #Store pitch contour
    x = pitch_contour
    #Take length of pitch contour
    n = len(x)
    #Calculate autocorrelation:
        #The correlation of the signal with a time-delayed version of itself.
    acorr = sm.tsa.acf(x, nlags = n-1)
    #Calculate confidence interval:
        #95% Confidence interval is +- 1.96/sqrt(n)
    highCI = 1.96/sqrt(n)
    lowCI = -highCI
    #Desired vibrato region of interest is 3 Hz - 10 Hz
        #=> Desired period is 1/10 s - 1/4 s
        #=> Desired lag times in frames are:
            #{1/10*f_s_contour:1/4*f_s_contour}
    frame10Hz = math.floor(1/10*f_s_contour)
    frame3Hz = math.floor(1/3*f_s_contour)
    #The lag value between these two frames with a maximum value is our suspected vibrato frequency
    maxLag = acorr[frame10Hz:frame3Hz].argmax() + frame10Hz
#If the extrema of the autocorrelation function exceeded the 95\% confidence intervals
#between the equivalent lags for 3 Hz and 10 Hz,
#the vibrato frequency was recorded for the maximum lag and
#the mean vibrato extent was recorded for the window.
    if (acorr[:maxLag].min() < lowCI) & (acorr[maxLag] > highCI):
        vibratoFreq = 1/maxLag*f_s_contour
#If the extrema did not exceed the confidence intervals,
#the vibrato frequency and extent were recorded as null values.
    else:
        vibratoFreq = np.nan
    #pd.plotting.autocorrelation_plot(pitch_contour, ax=ax[1])
    #prompt = input("Press Enter to continue...")
    #plt.close()
    return vibratoFreq

def vibAmpRoll(pitch_contour, f_s_contour, rollingVib, windowFactor=0):
    #Take vibrato frequency calculated from autocorrelation
    vibFreq = rollingVib[pitch_contour.index.max()]
    #If no frequency was logged, store the vibrato amplitude as null value.
    if math.isnan(vibFreq):
        ampCents = np.nan
        return ampCents
    #Calculate wavelength of interest to look for peaks.
    wavelength = 1.0/vibFreq*f_s_contour # in frames
    #This is left over code. Not sure what error I was trying to avoid.
    try:
        window = math.floor(wavelength*windowFactor)
    except ValueError:
        window = math.floor(1.0/5.5*f_s_contour*0.75)
    if window == 0:
        window = 1
#The vibrato extent was calculated by taking the mean of
#all pitch contour prominences within 75\% of the wavelength extracted from the autocorrelation function.
    maxPeaks = find_peaks(pitch_contour, distance=window)[0]
    #Calculate the peak amplitude in Hz by dividing the peak-to-peak prominence by two.
    prominences = scipy.signal.peak_prominences(pitch_contour, maxPeaks)[0]/2
    #Since the first and final waves are often clipped,
        #Do not calculate extent with the first and final peaks.
    ampEstimate = prominences[1:-1].mean()
    meanFreq = pitch_contour.mean()
    #print('Mean freq: ' + str(meanFreq))
    ampStd = prominences[1:-2].std()
#This peak amplitude was converted to cents with reference to the mean frequency.
    ampCents = 1200*log(1 + ampEstimate/meanFreq)/log(2)
    #vibStd = 1200*log(1 + ampStd/meanFreq)/log(2)
    #if vibFreq == np.nan:
    #    ampCents = np.nan
    return ampCents#, vibStd # amplitude in cents

def vibratoCalcMF(stableWavArray, samplerate, gender=np.nan, windowSecs=1):
#1 s window, 3Hz autoCorr
    #Convert sustained audio sample to a PRAAT object
    sound = Sound(stableWavArray, samplerate)
    #Creates PRAAT sound file from .wav array, default 44100 Hz sampling frequency?
    #We actually just need the single pitches for this calculation, between C-E
        #Let's look at the following pitch ranges for the different voice types:
    if gender == 'männl.':
        pitch = call(sound, "To Pitch", 0.0, 100, 390) #c3-g4
    elif gender == 'weibl.':
        pitch = call(sound, "To Pitch", 0.0, 261, 784) #c4-g5
    else:
        pitch = call(sound, "To Pitch", 0.0, 60, 784) #c4-g5
    pitch_contour = pitch.selected_array['frequency']
    plt.close('all')
    plt.plot(pitch_contour)
    prompt = input('Press enter to continue')
    #Calculate the contour's sample rate from the differing array sizes
    f_s_contour = pitch.selected_array['frequency'].size/sound.values.size*sound.sampling_frequency
    #Convert the window of interest to frames for the pitch contour
    window = math.ceil(windowSecs*f_s_contour) # 1 s * sampling frequency of pitch contour
    #If the window is larger than the contour, take the length of the window.
    if window > pitch_contour.shape[0]:
        window = pitch_contour.shape[0]-1
    #Convert the pitch contour to a pandas series
    pandasContour = pd.Series(pitch_contour)
    #Over the pandas series, calculate the vibrato frequency for each one second window.
    rollingVib = pandasContour.rolling(window).apply(lambda x: autocorrVib3Hz(x, f_s_contour))
    #Over the pandas series, calculate the vibrato extent for each one second window.
    rollingAmp = pandasContour.rolling(window).apply(lambda x: vibAmpRoll(x, f_s_contour,rollingVib, windowFactor=0.75))
    #Descriptive statistics for vibrato frequency and extent
    vibrato_Frequency = rollingVib.mean()
    vibratoStd = rollingVib.std()
    vibratoAmplitude = rollingAmp.mean()
    vibAmpStd = rollingAmp.std()

#Finally, a vibrato percentage was calculated from all portions of the entire sample that had recorded non-null values.
    vibratoPercentage = len(rollingVib[window:][rollingVib[window:].notna()])/len(rollingVib[window:])

    return vibrato_Frequency, vibratoPercentage, vibratoStd, vibratoAmplitude, vibAmpStd

vibrato_Frequency, vibratoPercentage, vibratoStd, amplitudeCents, amplitudeCentsStd = vibratoCalcMF(highestPitch, samplerate, gender=geschlecht, windowSecs=1)

def visualizeResults(wavFilename, middleTrial, isolatedHighestPitch, samplerate, gender=np.nan):
    samplerate, data = read(wavFilename)
    #Visualize results:
    fig, ax = plt.subplots(4)
    plt.rcParams['font.size'] = '14'
    ax[0].plot(np.arange(len(data))/samplerate,data)
    ax[0].set_title('Original Audio Waveform, (produced by Madde for illustration)')
    ax[0].axes.xaxis.set_visible(False)
    ax[0].axes.yaxis.set_visible(False)
    ax[1].plot(np.arange(len(middleTrial))/samplerate,middleTrial)
    ax[1].set_title('Middle Trial, if more than one repetition present')
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
    #ax[3].axes.xaxis.set_visible(False)


visualizeResults(wavFilename, middleTrial, highestPitch, samplerate, gender=geschlecht)
#vibratoFreq3 = vibratoCalc3(highestPitch, samplerate)
print('filename: ' + str(wavFilename) +
      ', Vibrato Frequency: ' + str(round(vibrato_Frequency, 2))+
      ' Hz, Vibrato Percentage: ' + str(round(vibratoPercentage, 2)) +
      ', Vibrato Std: ' + str(round(vibratoStd,2)) +
      ', Vibrato Extent: ' + str(round(amplitudeCents,2)),
      ', Vibrato Std: ' + str(round(amplitudeCentsStd,2)))