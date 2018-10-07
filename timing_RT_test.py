from __future__ import division;
import mne;
import scipy.io; # for matlab read
import matplotlib.pyplot as plt;
import numpy as np; #, h5py;
import pandas as pd;
from itertools import groupby #, izip; # for runlenghencode like function
import argparse
import warnings



fn='MEG_11346_20141230_tc.mat'

### BEHAVIORAL
# task file
mat   = scipy.io.loadmat(fn,struct_as_record=True);
subj  = mat.get('subject');
order = subj['order'][0][0];

#trial = np.ndarray((order.size,12))
trial = [];
for t in order:
    trial.append( [i[0] if isinstance(i[0],basestring) else  i[0][0] for i in t[0][0]]  ) ## collect trial info into 504 by 12 array. 504 trials, 12 data fields


# convert trial info into pnadas dataframe
df_trial = pd.DataFrame(trial)
df_trial.columns = [ 'function','run','trial','block','null','starttime','mag','scoreinc','freq','ev','RT','emotion'] # here are the 12 data fields

# two identical df with trial info...?
columns=['function','run','trial','block','NA','start','mag','inc','freq','ev','rsptime','emotion']
df = pd.DataFrame(trial,columns=columns) # it seems to me the response time unit is in ms?


### MEG 
# meg file
# testing 10637 run num 2 that has RT = 0 from Michael's testing
#raw = mne.io.Raw(args.fiffile) # 2015008 -- newer mne call
fif_path ='/home/kahwang/11346_Clock_run1_raw.fif'
raw =  mne.io.read_raw_fif(fif_path)# ,preload=True) # preload to enable editing
runnum = 1 
## get only the trials that are in this run
df_trial = df_trial[df_trial['block']==runnum] #run 2, 63 trials 
# originalTrialNums must be exctacted from list, if just =, will hold current values when we want to change 
originalTrialNums = [ x for x in df_trial['trial'] ]  #vector of original trian N, not sure what this is for.


# re number the trial to start at 1 (fif doesn't know there were any trials before)
starttrial=min(df_trial['trial'])
df_trial['trial'] = df_trial['trial'] - starttrial + 1  #this is to reorder the trialN in df. Before it doesn't start with 1

ttl = raw.ch_names.index('STI101');  ## ttl channel from stim presentation computer # in most cases, should be ch310
pdio  = raw.ch_names.index('MISC007');  ## photodio chn 309
bpsh= raw.ch_names.index('STI102'); ## button press chanel 311
data,times = raw[ [pdio,ttl,bpsh], :]  ## extrat data from chnnel 309-311


#fancy function from Will to find timepointso signal deflection in both photodiode and ttl channels
# digitized position, length, startidx, stopidx -- start and stop are inclusuve a a a b b b b -> a,3,0,2; b,2,3,6 
def rledig(hist):
  rle = np.array( [ (i,len(list(j)) ) for i,j in groupby(hist) ] );
  idx = rle[:,1].cumsum();
  # matrix of:        histval, length,          start,stop
  rleidx = np.vstack((rle.T, (idx - rle[:,1] ).T, idx.T - 1)).T;
  rleidx= rleidx.tolist() # for faster del operation
  
  # remove jumps (len==1) and 
  #  merge identical histval sections it might have separated
  i=1;
  while i  < len(rleidx):
      j=i+1;
      while j<len(rleidx) and (rleidx[j][0] == rleidx[i][0] or rleidx[j][1] < 2):
          # update end position
          rleidx[i][3] = rleidx[j][3]
          # reset length
          rleidx[i][1] = rleidx[i][3] - rleidx[i][2] +1
          # remove this bogus entry          
          del(rleidx[j])
          #np.delete(rleidx,j,0)
          # j+=1; # b/c of delete next is current
      i+=1; 
      
  return np.array(rleidx)


pdioToLabel = {
 1: 'ISI', # black
 2: 'score',    # gray [204 204 204]
 3: 'face',     # white
}
ttlToLabel = {
 1: 'score', # 0 (??)
 5: 'score', # 135-235
 2: 'ISI',   # 10
 3: 'ITI',   # 15
 4: 'face',  # 24-130
 6: 'done',  #255
}

# stop looking at photodiode after end of trial
# data[1,:]==255
IdxEnd=np.where(data[1,:]==255);
trialIdxEnd=IdxEnd[0][1] if np.size(IdxEnd)>0 and IdxEnd[0][1]>200000 else np.size(data[1,:])

# PHOTODIODE 
# 1. face  | white
# 2. ISI   | black
# 3. score | [204 204 204]
# 4. ITI   | black

### look at histogram of photodio data, there should be "3" boms cprres[pdomg tp face. score, and ISI/ITI
pdio_inds = np.digitize(data[0,1:trialIdxEnd],np.histogram(data[0,:],bins=3)[1])
# remove anything that went too high (why did this happend?)
pdio_inds[pdio_inds>3] = 3

### do the same thing for ttl, group them into groups
# PARALLEL PORT -- TTL
# 1. face  | 25-130
# 2. ISI   | 10 
# 3. score | 135 - 235 ( + 4 if correct) 
# 3. score | 135 - 235 (face+107)
# 4. ITI   | 15
# - done 255
#                                 |junk/score? |10=ISI| 15=ITI | 25-130=face | 135-239=score |
ttl_inds  = np.digitize(data[1,:],[0,           5,     12,      20,          132,           250 ])
# turn junk triggers into score
ttl_inds = [ 5 if x==1 else x for x in ttl_inds ];
# "6" is end-of-run code
ttl_inds = [ 3 if x==6 else x for x in ttl_inds ];


# get when there is a change
ttl_rleidx = rledig(ttl_inds)
pdio_rleidx = rledig(pdio_inds)


# truncate photiodiode events (remove get ready)
# remove anythere where the photodiode starts before the first tigger ends
# -- pdio changes after first trigger sent
pdio_rleidx = np.array([ p for p in pdio_rleidx  if  p[2] - ttl_rleidx[0,3]  > 0 ])

# trial numbers for pdio_rleidx, trial starts with face (histogram value = 3 for photodiode)
trial = np.array([ p[0] == 3 for p in pdio_rleidx ]).cumsum()
pdio_rleidx = np.c_[pdio_rleidx, trial]

#turn pdio data into dataframe
# as one step, get bad data conversion
#pdio_df = pd.DataFrame(np.c_[pdio_rleidx,np.array([ pdioToLabel[x] for x in  pdio_rleidx[:,0] ])])
pdio_df = pd.DataFrame(pdio_rleidx)
pdio_df['event'] = np.array([ pdioToLabel[x] for x in  pdio_rleidx[:,0] ])
pdio_df.columns = ['pd.histval','pd.len','pd.start','pd.stop','trial','event']

# build array of (event + trialnum*10). were this overlaps, there is an ITI.
# this has to be done in this ugly way because for whatever reason the ITI and ISI had the same pdio intensity "1"
itiIdxs=np.r_[pdio_rleidx[:,0]+pdio_rleidx[:,4]*10, 0, 0] - np.r_[0,0,pdio_rleidx[:,0]+pdio_rleidx[:,4]*10] == 0;
pdio_df['event'][np.where(itiIdxs)[0]] = 'ITI'


# retype those pesky strings, not sure why this is needed.
tofloat = [ x for x in pdio_df.columns if x != 'event' ]
pdio_df[tofloat] = pdio_df[tofloat].astype(float)

# only the last trial should not have 4 types (face,ISE,score,ITI)
if len( [t for t,g  in groupby(pdio_df['trial']) if len(list(g))!=4 ] ) > 1:
    Exception('do not have pdio face,ISI,score,ITI for all expected trials (not last)') 


## now do the same thing for ttl trigger channels, put into dataframe
# trial number for trigger, starts at face (value of 4)
# count up trials based on number of starts
trial = np.array([ t[0] == 4 for t in ttl_rleidx ]).cumsum()
ttl_rleidx = np.c_[ttl_rleidx, trial]
ttl_df = pd.DataFrame(ttl_rleidx)
ttl_df['event'] = [ ttlToLabel[x] for x in  ttl_rleidx[:,0] ]
ttl_df.columns = ['tt.histval','tt.len','tt.start','tt.stop','trial','event']
# retype those pesky strings, for all columns that are not event
tofloat = [ x for x in ttl_df.columns if x != 'event' ]
ttl_df[tofloat] = ttl_df[tofloat].astype(float)


## check if number of trials in ttl and pdio are the same
if pdio_df['trial'].tail(1).values != ttl_df['trial'].tail(1).values:
    warnings.warn('photodiode (n.t=' +
                     str(pdio_df['trial'].tail(1).item())+
                    ') and triggers (n.t='+
                    str(ttl_df['trial'].tail(1).item())+
                    ') do not align -- using ttl instead of pdio')
    
    ## pretend  ttl channel is the pdio channel
    #pdio_df_real = pdio_df    
    pdio_df = ttl_df.copy()
    pdio_df.columns = ['pd.histval','pd.len','pd.start','pd.stop','trial','event']
    ## plot triggers
    plt.plot(data[0,:]);
    for v in np.histogram(data[0,:],bins=3)[1]:
        plt.axhline(y=v,color='r')
    plt.savefig(fn + '_badPDIO.png')
else:
    plt.plot(data[0,:]);
    for v in np.histogram(data[0,:],bins=3)[1]:
        plt.axhline(y=v,color='r')
    plt.savefig(fn + '_goodPDIO.png')
    plt.close()
    plt.plot(data[2,:]);plt.plot(ttl_inds[:]);plt.plot(pdio_inds[:]);
    plt.savefig(fn+'_allTrigs.pdf')
  

# check if number of trials are the same in ttl and matlab log
if ttl_df['trial'].tail(1).values != df_trial['trial'].tail(1).values:
    raise Exception('trials from trigger do not match matlab file!')    


#original df is in df and df_trial

new_df = pd.merge(pdio_df,ttl_df)


for e in set(pdio_df['event']):
    ename=e + '.start'
    df_trial[ename] = 0;
    for t in set(df_trial['trial']):
        startidx=pdio_df.loc[ (pdio_df['trial']==t) & (pdio_df['event']==e)]['pd.start']
        if len(startidx)!=1:
            warnings.warn(e+" on trial "+str(t)+" has "+str(len(startidx)) + " occurances",Warning)
            startidx=startidx.head(1)
        
        df_trial[ename][(df_trial['trial']==t)] = np.array(startidx) # NaN if not cast to array first (why?)



runTypeToVolt = {
  #face: 25 - 130 
  'CEV.fear.face'   : 29,
  'CEVR.fear.face'  : 38,
  'DEV.fear.face'   : 47,
  'IEV.fear.face'   : 56,
  'CEV.happy.face'  : 65,
  'CEVR.happy.face' : 74,
  'DEV.happy.face'  : 83,
  'IEV.happy.face'  : 92,
  'CEV.scram.face'  : 101,
  'CEVR.scram.face' : 110,
  'DEV.scram.face'  : 119,
  'IEV.scram.face'  : 128,
  #score: 135 - 235 == face+107
  'CEV.fear.score'  : 136,
  'CEVR.fear.score' : 145,
  'DEV.fear.score'  : 154,
  'IEV.fear.score'  : 163,
  'CEV.happy.score' : 172,
  'CEVR.happy.score': 181,
  'DEV.happy.score' : 190,
  'IEV.happy.score' : 199,
  'CEV.scram.score' : 208,
  'CEVR.scram.score': 217,
  'DEV.scram.score' : 226,
  'IEV.scram.score' : 235,
}

# sorting events too see if order was wrong
x=df_trial[['face.start','ISI.start','score.start','ITI.start']].stack()
if( not all( x[:-1].values < x[1:].values) ):
    Exception('indexes are not ordered correct (face<ISI<score<ITI!')    

### write out eve file
# tile up the triggers per trial, write them out for each trial
funcEmo =  df_trial['function'].head(1) + '.' +  df_trial['emotion'].head(1)
triggers = [ runTypeToVolt[ funcEmo.values[0] + '.face' ], 15,  runTypeToVolt[ funcEmo.values[0] + '.score' ], 10 ] 

#           index, time,      "volt":     was                                                 is
eve = np.c_[ x,    x/1000,    np.tile([ triggers[a] for a in [3, 0, 1, 2] ], int(len(x)/4)),    np.tile(triggers, int(len(x)/4)) ]
# add first row of zeros for MNE
eve = np.vstack(([0,0,0,0],eve))  
# make actual first transtion value from zero  
eve[1,2] = 0
# save 
np.savetxt(fn+'eve',eve,fmt='%d %.03f %d %d')



### write out csv file

df_trial['imagefile']='NA'
df_trial['ITIideal']=0
df_trial['trial'] = originalTrialNums
df_trial['totalscore']=df_trial['scoreinc'].cumsum()
# use the button push STI102 trigger to get a more accurate RT
# -- dependant on face.start beign accurate

### now, get RT in df based on button press channel info, compare it to the adjusted face onset (from phodio) to recalculate RT
# use the button push STI102 trigger to get a more accurate RT
# This is where things could go wrong:
#   1. for some subjects, the response button channel (STI102) is flat, this would cause all the RT be set to zero and "pidshidx" to be set to face/clock onset
#   2. when subject did not press a button, the RT will be set to the difference between face onset and ISI onset, which will be around 4000ms
#   3. for correctedly recorded trials, the RT is calculated by counting the time from the last timestamp of clock onset to the first timstamp of ISI onset. Depending on how 
#      button press were recorded by the stimulus computer, this could make RT saved out to the CSV file consistenly faster than the ones in the matlab logs 
df_trial['pushidx'] = [ np.argmax(data[2,s:e]>=1.5) + s   for s,e in np.array(df_trial[['face.start','ISI.start']]) ]
df_trial['RT.push'] = [ np.argmax(data[2,s:e]>=1.5)  for s,e in np.array(df_trial[['face.start','ISI.start']]) ]


df_final = df_trial[ ['block','trial', 'function', 'emotion', 'mag','freq','scoreinc','ev','RT.push',
                      'face.start','ISI.start','score.start','ITI.start','ITIideal','imagefile','pushidx'] ]
df_final.columns=['run','trial','rewFunc','emotion','magnitude','probability','score','ev','rt',
                  'clock_onset','isi_onset','feedback_onset','iti_onset','iti_ideal','image','pushidx']         
df_final.to_csv(fn+'.csv',index=False)





