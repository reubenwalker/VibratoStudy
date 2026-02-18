import pickle
from datetime import datetime
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
###How to find the ID number of the audio files:

    #Two objects:
        #IDNumbers.csv
#IDNumbers and names from audio files
IDNames = pd.read_csv('IDNumbers.csv')
        #studDB
#IDNames.columns: id, Name
#Might need to add years 
#Deidentified ID Numbers with time stamps
deIDMeta = pd.read_csv('DeidentifizierteMetadatenTimestamp.csv')
from datetime import datetime
deIDMeta['date'] = deIDMeta.date.apply(lambda x: datetime.strptime(x,  "%Y-%m-%d"))
#deIDMeta['date'].min()
    #2002
#Let's grab someone from 2002
#studDB.loc[0, 'Nachname']
#Following provides us with the excel ID numbers, the names, and intake year
with open('studDB_IDs.pkl', 'rb') as f:
    studDB_IDs = pd.read_pickle(f)
#For the audio DB we just want years with an audio recording
audioDB = studDB_IDs[studDB_IDs['Jahr'].astype(int) >= 2002]
#Make a df with id, name, and each date of recording
IDNamesDate = pd.merge(IDNames[['id', 'name']], deIDMeta[['date', 'id']], on='id')
IDNamesDate['year'] = IDNamesDate['date'].apply(lambda x: x.year)
#nachname = audioDB['Nachname'].iloc[0]
#vorname = audioDB['Vorname'].iloc[0]
#Need to convert audioDB's years to int
IDNamesDate = IDNamesDate[['id', 'name', 'year']].groupby('id').min().reset_index()
#IDNamesDate[IDNamesDate['name'].str[:12] == 'Böhme, Julia']
    #There are two Julia Böhme

#With the exception of the single duplicate, we'll search for both Vorname and Nachname
def findIDNum(nachname, vorname, jahr, IDNames):
    idSer = IDNames[((IDNames['name'].str.contains(nachname)) & 
                             (IDNames['name'].str.contains(vorname)) #&
                             #(IDNamesDate['year'] == jahr)
                             #(abs(IDNamesDate['year'] - jahr)<= 2)
                             ) ]['id']
    #If there is more than one response, we'll look for a recording within a year of intake
    if idSer.size > 1:
        print(idSer)
        idSer = IDNames[((IDNames['name'].str.contains(nachname)) & 
                             (IDNames['name'].str.contains(vorname)) &
                             #(IDNamesDate['year'] == jahr)
                             (abs(IDNames['year'] - jahr) <= 1)
                             ) ]['id']
    if idSer.size < 1: 
        idSer = np.nan
    try:
        idNum = int(idSer)
        
    except ValueError:
        idNum = np.nan
    return idNum
    return idNum
    
audioDB['audioID'] = audioDB.apply(lambda x: findIDNum(x.Nachname, x.Vorname, int(x.Jahr), IDNamesDate), axis=1)
audioDB['audioID'].count()
#293 Entries
#214 ID'd Anamnesen
#69 Anamnese since 2002 without audio recordings
#29 Audio files (243-214) without Anamnesen??? Marie said this shouldn't be possible
###Find missing Anamnesen:
#Rejoin audioDB to IDNames
missingNames = pd.merge(IDNames, audioDB['audioID'], how='left', left_on='id', right_on='audioID')
missingNames = missingNames[missingNames['audioID'].isna()]
###Just do it manually:
#missingNames.to_csv('missingAudioAnamnesen.csv')
foundNames = pd.read_csv('missingAudioAnamnesen.csv', dtype='str')
foundNames = foundNames.rename(columns={'audioID':'anamnesenID'})
mask = audioDB['audioID'].isna()
#audioDB.loc[mask, 'audioID'] = audioDB.loc[mask, 'Nummer'].apply(lambda x: str(foundNames[foundNames['anamnesenID'] == x]))
#audioDB[mask]['audioID'] = audioDB.loc[mask, 'Nummer'].apply(lambda x: str(foundNames[foundNames['anamnesenID'] == x]))
foundAudio = pd.merge(audioDB[audioDB['audioID'].isna()], foundNames[['id', 'anamnesenID']], how='inner', left_on='Nummer', right_on= 'anamnesenID')
foundAudio = foundAudio[['Jahr', 'Nummer', 'Nachname', 'Vorname', 'id']].rename(columns={'id':'audioID'})
###Fill in the IDs from the found files
def fillFoundIDs(anamnesenID, foundAudio):
    try:
        audioID = int(foundAudio[foundAudio['Nummer'] == anamnesenID]['audioID'])
    except TypeError:
        audioID = np.nan
    return audioID
audioDB.loc[mask,'audioID'] = audioDB.loc[mask, 'Nummer'].apply(lambda x: fillFoundIDs(x, foundAudio))


#Let's exclude jazzPop
with open('jazzPopDB.pkl', 'rb') as f:
    jazzPopDB = pd.read_pickle(f)
noJazz = pd.merge(audioDB, jazzPopDB[jazzPopDB['jazzPop'] == 0], on='Nummer')
#Removed 27 students, 29 total in jazzPopDBFvib
#I think we need to mke all non-NA values in audioID integers.
#First remove NA values
noJazz = noJazz[noJazz['audioID'].notna()]
#change type to int
noJazz['audioID'] = noJazz['audioID'].astype('int')
#vibDB = pd.read_csv('Vibrato3Hz.csv')
#vibDB = pd.read_csv('VibratoMFAmpFinal.csv')#'VibratoMF1SecWindow.csv')
#vibDB = pd.read_csv('TimbreSF.csv')
# with open('LTASMed.pkl', 'rb') as f:

# with open('vokalAusgleich796.pkl', 'rb') as f:
    # vibDB1 = pd.read_pickle(f)


# with open('avezzo784.pkl', 'rb') as f:
    # vibDB2 = pd.read_pickle(f)

# with open('dreiklang794.pkl', 'rb') as f:
    # vibDB3 = pd.read_pickle(f)
    
# vibDB = pd.concat([vibDB1, vibDB2, vibDB3])

# with open('PEVOC_f.pkl', 'rb') as f:

# First dSPL run
# d = pd.read_pickle('Just_dSPL.pkl')
# with open('normedRun.pkl', 'rb') as f:
    # vibDB = pd.read_pickle(f)
# vibDB = vibDB.drop('dSPL', axis=1)
# vibDB = pd.merge(vibDB, d[['id','date','dSPL']], on=['id','date'])

# vibDB0 = pd.read_pickle('melSpecOriginal20250723.pkl')
vibDB = pd.read_pickle('vib20251111Sand.pkl')#[['id','date','testNum','meanFreq', 'sampleDuration50', 'Vibrato-Rate (F_0)', 'Vibrato-Umfang (F_0)', 'Vibrato-Rate (Amp)', 'Vibrato-Umfang (Amp)','Vibrato-Percent']]
# vibDB = pd.concat([vibDB0,vibDB1])
# vibDB = pd.read_pickle('PEVOC_f.pkl')

# vibDB = vibDB.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.2'])
vibDBTest = vibDB#.drop_duplicates()
vibDBTest['id'] = vibDBTest['id'].astype('int64')
classVib = pd.merge(noJazz[['Jahr', 'Nummer', 'audioID']], 
                    #vibDB[['id', 'date', 'meanFreq',
                    vibDBTest,#[['id', 'date', 'meanFreq',
                        #'vibratoFreqMF', 'vibratoPercentageMF', 'vibratoStdMF', 
                        #   'vibFreqAmp', 'vibFreqAmpStd']],
                    left_on='audioID', right_on='id')
#Let's rename the vibrato calculations:
vibDict = {'vibratoFreqMF':'vibFreq',
           'vibratoPercentageMF':'vibPerc',
           'vibratoStdMF':'vibStd'}
#classVib = classVib.rename(columns=vibDict)
#classVib = classVib.drop(column='id')
#classVib = classVib.drop_duplicates()


#Now let's grab the control variables from the categorical DB
with open('nullDB.pkl', 'rb') as f:
    nullDB = pd.read_pickle(f)
# nullDB = pd.read_csv('nullDB.csv',index_col=False)
#nullDB = nullDB.drop(columns='Unnamed: 0')
#classVib['Nummer'] = classVib['Nummer'].astype('int64')
nullDB.loc[nullDB['vpnummer'] == '00415','geschlecht'] = 'weibl.'
df = pd.merge(classVib, nullDB, left_on='Nummer', right_on='vpnummer')

###Sidebar: Let's group the singers into E5, D5, E4 and D4
# df.loc[df['meanFreq'] > 623.3, 'meanPitch'] = 'E5'
# df.loc[((df['meanFreq'] < 623.3) & (df['meanFreq'] > 440)), 'meanPitch'] = 'D5'
# df.loc[((df['meanFreq'] < 440) & (df['meanFreq'] > 311.64)), 'meanPitch'] = 'E4'
# df.loc[df['meanFreq'] < 311.64, 'meanPitch'] = 'D4'

# with open('SNRdraft.pkl', 'rb') as f:
    # snr = pd.read_pickle(f)

# snr['id'] = snr['id'].astype(int)
# df = pd.merge(df, snr[['id', 'trialNum','date','SNR']], on=['id', 'trialNum','date'])


df['date'] = df['date'].apply(lambda x: x.replace('_', '-'))

###Ok, let's find the first recording.
#Code taken from metaData exploration
from datetime import datetime
from datetime import timedelta
#datetime.strptime(df.date[0], "%Y-%m-%d")
#df.date.apply(lambda x: datetime.strptime(x,  "%Y-%m-%d"))
#Didn't work. Some aren't in that format.
df['dateLen'] = df.date.apply(lambda x: len(x))
#Still have a couple of holdovers. 
    #One 2012-6-6 and one DATE - Kopie
#Convert 2012-6-6, remove Kopie
mask = df['dateLen'] == 8
df.loc[mask, 'date'] = '2012-06-06'
###remove Kopie
df['dateLen'] = df.date.apply(lambda x: len(x))
df = df[df['dateLen'] == 10].copy()
#Transition date strings to date format
#df1.date.apply(lambda x: datetime.strptime(x,  "%Y-%m-%d"))
    #Didn't work.
#Find inconsistently formatted dates
df['dateLen2'] = df.date.apply(lambda x: len(x.split('-')[0]))
#Only one entry with %d-%m-%Y
df[df['dateLen2'] == 2]
mask = df['dateLen2'] == 2
df.loc[mask, 'date'] = '2011-11-23'
df['date'] = df.date.apply(lambda x: datetime.strptime(x,  "%Y-%m-%d"))
df = df.drop(columns=['dateLen', 'dateLen2'])

df['minDate'] = df['id'].apply(lambda x: df[['id', 'date']].loc[df['id'] == x].groupby('id').min().iloc[0])
#Calculate difference:
df['dateDiff'] = df['date'] - df['minDate']

df['beginDate'] = df.studienbeginn.apply(lambda x: datetime(int(x),9,25))
def negativeDate(beginDate, minDate):
    if (beginDate - minDate) < timedelta(0):
        return beginDate
    else:
        return minDate

df['beginDate'] = df[['beginDate', 'minDate']].apply(lambda x: negativeDate(x.beginDate, x.minDate), axis=1)
df['beginDiff'] = df['date'] - df['beginDate']
df['beginDiff'] = df['beginDiff'].apply(lambda x: x.days)

###Let's restrict this to singers with extant recordings between 3-5 years after the begin of studies
df['yearFloor'] = df['beginDiff'].apply(lambda x: np.floor(x/365))
df['yearCeiling'] = df['beginDiff'].apply(lambda x: np.ceil(x/365))
df['Year'] = df['date'].apply(lambda x: x.year)
#Fix 0 yearCeiling
maskCeil = df['yearCeiling'] == 0
df.loc[maskCeil, 'yearCeiling'] = 1
# yearFloor
# 0.0    220
# 1.0    166
# 2.0    150
# 3.0    120
# 4.0     73
# 5.0     34
# 6.0     16
# 7.0      7
# 8.0      1


df2 = df[df['trialNum'] == '2'].copy()
df2['Stimmfach'] = 'Sop/Mezzo/Alt'
df2.loc[df2['meanFreq'] < 450, 'Stimmfach'] = 'Ten/Bar/Bass'
# df1 = df[df['trialNum'] == '1'].copy()
# df1['Stimmfach'] = 'Sop/Mezzo/Alt'
# df1.loc[df1['pitchMed'] < 325, 'Stimmfach'] = 'Ten/Bar/Bass'


#Let's remove duplicate recordings:
df2 = df2.groupby(['id','date','trialNum']).first().reset_index()
# df1 = df1.groupby(['id','date','trialNum']).first().reset_index()

df2['yearDiff'] = df2['beginDiff'].apply(lambda x: x/365)
mask = df2['yearDiff'] < 4
maskM = ((df2['yearDiff'] < 4) & (df2['geschlecht'] == 'männl.'))
maskF = ((df2['yearDiff'] < 4) & (df2['geschlecht'] == 'weibl.'))

df2.loc[mask,['id','yearCeiling', 'yearDiff', 'sampleDuration50', 'Stimmfach', 'meanFreq', 'zulassung.hno',
     'Vibrato-Rate (F_0)','Vibrato-Umfang (F_0)', 'Vibrato-Rate (Amp)', 'Vibrato-Umfang (Amp)', 'Vibrato-Percent']].to_csv('vib20250905.csv')
    
### Do vibrato 
#Test 
dfTest = df2[mask]
dfTest_clean = dfTest.dropna(subset=['vibExtent_dB', 'yearDiff', 'id'])

import statsmodels.formula.api as smf


model = smf.mixedlm("vibExtent_dB ~ yearDiff", dfTest_clean,
                    groups=dfTest_clean["id"], re_formula="~yearDiff")
result = model.fit()
print(result.summary())
             # Mixed Linear Model Regression Results
# ===============================================================
# Model:               MixedLM  Dependent Variable:  vibExtent_dB
# No. Observations:    543      Method:              REML
# No. Groups:          159      Scale:               0.1903
# Min. group size:     1        Log-Likelihood:      -512.4779
# Max. group size:     10       Converged:           Yes
# Mean group size:     3.4
# ---------------------------------------------------------------
                     # Coef.  Std.Err.   z    P>|z| [0.025 0.975]
# ---------------------------------------------------------------
# Intercept             1.028    0.062 16.506 0.000  0.906  1.150
# yearDiff              0.056    0.024  2.313 0.021  0.009  0.103
# Group Var             0.410    0.199
# Group x yearDiff Cov -0.080    0.058
# yearDiff Var          0.044    0.027
# ===============================================================

#Test 
dfTest_clean = dfTest.dropna(subset=['vibExtent_f0', 'yearDiff', 'id'])

import statsmodels.formula.api as smf

model = smf.mixedlm("vibExtent_f0 ~ yearDiff", dfTest_clean,
                    groups=dfTest_clean["id"], re_formula="~yearDiff")
result = model.fit()
print(result.summary())
             # Mixed Linear Model Regression Results
# ================================================================
# Model:              MixedLM   Dependent Variable:   vibExtent_f0
# No. Observations:   543       Method:               REML
# No. Groups:         159       Scale:                38.1386
# Min. group size:    1         Log-Likelihood:       -2053.5626
# Max. group size:    10        Converged:            Yes
# Mean group size:    3.4
# ----------------------------------------------------------------
                      # Coef.  Std.Err.   z    P>|z| [0.025 0.975]
# ----------------------------------------------------------------
# Intercept             24.899    1.228 20.283 0.000 22.493 27.305
# yearDiff               2.380    0.404  5.890 0.000  1.588  3.172
# Group Var            195.238    5.374
# Group x yearDiff Cov  -8.625    1.038
# yearDiff Var          14.144    0.496
# ================================================================
x = dfTest_clean['vibExtent_f0']
y = dfTest_clean['vibExtent_dB']

r = np.corrcoef(x, y)[0, 1]
print("Pearson r:", r)

x = df2['vibRate_f0']
y = df2['vibRate_amp']

r = np.corrcoef(x, y)[0, 1]
print("Pearson r:", r)

### Anfang des Studiums
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Prepare data (one row per 'id')
# df_grouped = df.groupby('id').first().reset_index()
df_grouped = df[df['yearCeiling'] == 1].groupby('id').first().reset_index()

# Define features
features = ['vibRate_f0', 'vibExtent_f0', 'vibExtent_dB', 'vibPercent']

# Combine "ohne" and "labil" into one category
df_grouped['stability_group'] = df_grouped['vibrato.stabilitaet'].replace({
    'ohne': 'ohne/labil',
    'labil': 'ohne/labil',
    'stabil': 'stabil'
})

# Define order and color palette (Okabe–Ito)
order = ['ohne/labil', 'stabil']
palette = ['#E69F00', '#0072B2']  # orange and blue (colorblind safe)

# Ensure categorical ordering
df_grouped['stability_group'] = pd.Categorical(
    df_grouped['stability_group'],
    categories=order,
    ordered=True
)

# Apply clean Seaborn style
sns.set(style="whitegrid", context="talk")

# Plot violin + stripplot for each feature
for feature in features:
    plt.figure(figsize=(6, 4))
    
    # Violin plot
    sns.violinplot(
        data=df_grouped,
        x='stability_group',
        y=feature,
        order=order,
        palette=palette,
        inner='box',
        cut=0,
        linewidth=1.2
    )
    
    # Overlay stripplot
    if feature != 'vibPercent':
        sns.stripplot(
            data=df_grouped,
            x='stability_group',
            y=feature,
            order=order,
            color='black',
            size=3,
            jitter=0.15,
            alpha=0.6
        )
    else:
        sns.stripplot(
            data=df_grouped,
            x='stability_group',
            y=feature,
            order=order,
            color='black',
            size=3,
            jitter=0.3,
            alpha=0.6
        )
    
    plt.title(f'{feature}: ohne/labil vs stabil', fontsize=13)
    plt.xlabel('vibrato.stabilitaet group')
    plt.ylabel(feature)
    plt.tight_layout()
    # plt.show()
    plt.savefig(feature + '.png')


###Randfälle

### Test interaction between extents and rates
import statsmodels.formula.api as smf

# Ensure your data is clean
mask = ((df['vibPercent'] == 1) & (df['vibExtent_f0'] < 100))
df_model = df[mask]  # one row per subject

# Fit linear model with interaction
model = smf.ols('vibRate_f0 ~ vibExtent_f0 * vibExtent_dB', data=df_model).fit()

# Show results
print(model.summary())

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Prepare data (one row per 'id')
df_grouped = df_model

# Set Seaborn style
sns.set(style="whitegrid", context="talk")

# Scatter plot with regression line
plt.figure(figsize=(6, 4))
sns.regplot(
    data=df_grouped,
    x='vibExtent_f0',
    y='vibRate_f0',
    scatter_kws={'s': 40, 'alpha': 0.7, 'color': 'black'},
    line_kws={'color': '#0072B2', 'linewidth': 2}
)

plt.xlabel('Vibrato Extent (f0)')
plt.ylabel('Vibrato Rate (f0)')
plt.title('Linear Regression: vibRate_f0 ~ vibExtent_f0')
plt.tight_layout()
plt.savefig('vibRateUmfangNoOutlier.png')

mask = ((df['vibPercent'] == 1))# & (df['vibExtent_f0'] < 100))
df_model = df[mask]  # one row per subject

# Fit linear model with interaction
model = smf.ols('vibRate_f0 ~ beginDiff*vibExtent_f0', data=df_model).fit()

import statsmodels.formula.api as smf

# Fit the model
model = smf.ols('vibRate_f0 ~ beginDiff * (vibExtent_f0 + vibExtent_dB)', data=df).fit()

# Summarize results
print(model.summary())

# Fit the model
model = smf.ols('vibRate_f0 ~ beginDiff + vibExtent_f0 + vibExtent_dB', data=df).fit()

# Summarize results
print(model.summary())

# Fit the model
model = smf.ols('vibRate_f0 ~ vibExtent_f0 + vibExtent_dB', data=df).fit()

# Summarize results
print(model.summary())

# Fit the model
model = smf.ols('vibRate_f0 ~ vibExtent_f0*vibExtent_dB', data=df).fit()

# Summarize results
print(model.summary())



###
dfTest = df2[mask]
dfTest_clean = dfTest.dropna(subset=['vibExtent2_SPL', 'yearDiff', 'id'])

import statsmodels.formula.api as smf


model = smf.mixedlm("vibExtent2_SPL ~ yearDiff", dfTest_clean,
                    groups=dfTest_clean["id"], re_formula="~yearDiff")
result = model.fit()
print(result.summary())
             # Mixed Linear Model Regression Results
# ===============================================================
# Model:             MixedLM  Dependent Variable:  vibExtent2_SPL
# No. Observations:  653      Method:              REML
# No. Groups:        230      Scale:               0.5647
# Min. group size:   1        Log-Likelihood:      -979.0429
# Max. group size:   5        Converged:           Yes
# Mean group size:   2.8
# ---------------------------------------------------------------
                     # Coef.  Std.Err.   z    P>|z| [0.025 0.975]
# ---------------------------------------------------------------
# Intercept             1.808    0.087 20.769 0.000  1.638  1.979
# yearDiff              0.098    0.032  3.099 0.002  0.036  0.160
# Group Var             1.175    0.310
# Group x yearDiff Cov -0.114    0.077
# yearDiff Var          0.062    0.035
# ===============================================================


# Fit the model
model = smf.ols('vibRate_f0 ~ vibExtent_f0*vibExtent2_SPL', data=df_model).fit()

# Summarize results
print(model.summary())

# Fit the model
model = smf.ols('vibRate_f0 ~ vibExtent_f0 + vibExtent2_SPL', data=df_model).fit()

# Summarize results
print(model.summary())
                            # OLS Regression Results
# ==============================================================================
# Dep. Variable:             vibRate_f0   R-squared:                       0.037
# Model:                            OLS   Adj. R-squared:                  0.033
# Method:                 Least Squares   F-statistic:                     8.342
# Date:                Wed, 12 Nov 2025   Prob (F-statistic):           0.000279
# Time:                        07:28:41   Log-Likelihood:                -398.13
# No. Observations:                 437   AIC:                             802.3
# Df Residuals:                     434   BIC:                             814.5
# Df Model:                           2
# Covariance Type:            nonrobust
# ==================================================================================
                     # coef    std err          t      P>|t|      [0.025      0.975]
# ----------------------------------------------------------------------------------
# Intercept          5.5378      0.079     70.529      0.000       5.383       5.692
# vibExtent_f0      -0.0064      0.002     -3.551      0.000      -0.010      -0.003
# vibExtent2_SPL     0.0870      0.026      3.388      0.001       0.037       0.137
# ==============================================================================
# Omnibus:                       38.660   Durbin-Watson:                   1.017
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):               67.987
# Skew:                           0.558   Prob(JB):                     1.73e-15
# Kurtosis:                       4.578   Cond. No.                         122.
# ==============================================================================


### vibExtent f0 v SF
df0818 = df2[((df2['Jahr'].astype(int) <= 2018) & (df2['Jahr'].astype(int) >= 2008))]

df_expanded = (
    df0818.explode(['vibExtentSPL_roll', 'harmonicSPL_mean'])
    .reset_index(drop=True)
)

# Add harmonic number and frequency
df_expanded['harmonic'] = df_expanded.groupby('audioID').cumcount() + 1
df_expanded['harmonic_freq'] = df_expanded['harmonic'] * df_expanded['meanFreq']

df_f0 = df_expanded.query("harmonic == 1").copy()
df_f0['vibExtentSPL_roll_f0'] = df_f0['vibExtentSPL_roll']

df_2500_3200 = (
    df_expanded
    .query("meanFreq < 400 and 2500 <= harmonic_freq <= 3200")
    .groupby(['id', 'date'], as_index=False)
    ['vibExtentSPL_roll'].mean()
    .rename(columns={'vibExtentSPL_roll': 'vibExtentSPL_roll_2500_3200'})
)

# --- Non-treble singers (<400 Hz): average vib extent between 2500–3200 Hz
df_2500_3200 = (
    df_expanded
    .query("meanFreq < 400 and 2500 <= harmonic_freq <= 3200")
    .groupby(['id', 'date'], as_index=False)
    .agg({
        'vibExtentSPL_roll': 'mean',
        'meanFreq': 'first',
        'yearDiff': 'first'
    })
    .rename(columns={'vibExtentSPL_roll': 'vibExtentSPL_roll_2500_3200'})
)

# --- Treble singers (>400 Hz): find harmonic closest to 3200 Hz
df_treble = df_expanded.query("meanFreq > 400").copy()
df_treble['diff'] = np.abs(df_treble['harmonic_freq'] - 3200)

df_3200 = (
    df_treble
    .loc[df_treble.groupby(['id', 'date'])['diff'].idxmin()]
    .copy()
    .rename(columns={'vibExtentSPL_roll': 'vibExtentSPL_roll_3200'})
)

# Keep only relevant columns
df_3200 = df_3200[['id', 'date', 'meanFreq', 'yearDiff', 'vibExtentSPL_roll_3200']]


# Assuming `Jahr` is the recording year and you can reference first measurement per singer
# df['yearDiff'] = df['Jahr'] - df.groupby('id')['Jahr'].transform('min')

import statsmodels.formula.api as smf

# 1️⃣ Fundamental
df_f0['vibExtentSPL_roll_f0'] = pd.to_numeric(df_f0['vibExtentSPL_roll_f0'], errors='coerce')

model_f0 = smf.mixedlm(
    "vibExtentSPL_roll_f0 ~ yearDiff + meanFreq",
    df_f0, groups=df_f0["id"]
).fit()

# 2️⃣ Non-treble region
df_2500_3200['vibExtentSPL_roll_2500_3200'] = pd.to_numeric(df_2500_3200['vibExtentSPL_roll_2500_3200'], errors='coerce')

model_mid = smf.mixedlm(
    "vibExtentSPL_roll_2500_3200 ~ yearDiff",
    df_2500_3200, groups=df_2500_3200["id"]
).fit()

# 3️⃣ Treble region
df_3200['vibExtentSPL_roll_3200'] = pd.to_numeric(df_3200['vibExtentSPL_roll_3200'], errors='coerce')

model_high = smf.mixedlm(
    "vibExtentSPL_roll_3200 ~ yearDiff",
    df_3200, groups=df_3200["id"]
).fit()

print(model_f0.summary())
print(model_mid.summary())
print(model_high.summary())

sns.lmplot(
    data=df_f0, x='yearDiff', y='vibExtentSPL_roll_f0',
    hue='id', ci=None
)

df2.loc[mask,['id','yearCeiling', 'yearDiff', 'sampleDuration50', 'Stimmfach', 'meanFreq', 'zulassung.hno',
     'Vibrato-Rate (F_0)','Vibrato-Umfang (F_0)', 'Vibrato-Rate (Amp)', 'Vibrato-Umfang (Amp)', 'Vibrato-Percent']].to_csv('vib20250918.csv')
    
### Do vibrato extents interact?
mask1 = ((df2['zulassung.hno'].notna()) & (df2['zulassung.hno'] != 'Pädagogik') & (df2['yearDiff'] < 1))
hueOrder = ['sicher, Solo', 'fragl., Solo', 'sicher, Chor', 'fragl., Chor']
# 2. Vibrato-Umfang (F0) vs. Vibrato-Umfang (Amp), color by Rate (F0)
plt.close('all')
sns.scatterplot(
    x='Vibrato-Umfang (F_0)',
    y='Vibrato-Umfang (Amp)',
    hue='zulassung.hno',
    hue_order=hueOrder,
    # palette='magma',
    data=df2[mask1]
)
plt.title('Vibrato-Umfang (F0) vs. Vibrato-Umfang (Amp) \nHue: Rate (F0)')
plt.xlabel('Vibrato-Umfang (Hz)')
plt.ylabel('Vibrato-Umfang (Cents)')
# plt.colorbar()
plt.show()
# plt.savefig('ExtentF0ExtentAmp_hueRateF0.png')
plt.close('all')

###Let's combine the non-frequent options:
def soloChor(text):
    if type(text) != str:
        return np.nan
    if 'Chor' in text:
        return 'Chor'
    elif 'Solo' in text:
        return 'Solo'

import pandas as pd

# df assumed to already exist
df = df2f.copy()

# Ensure categorical variables are coded correctly
df["id"] = df["id"].astype("category")
df["SoloChor"] = df["SoloChor"].astype("category")

# Optional but recommended: center time
df["yearDiff_c"] = df["yearDiff"] - df["yearDiff"].min()