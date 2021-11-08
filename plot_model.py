#### script to plot model outputs from Alex and Michael
import numpy as np
import pandas as pd
import mne as mne
import pyreadr
import matplotlib.pyplot as plt
plt.ion()

#mkae paths global
datapath = '/data/backed_up/kahwang/Clock/'
save_path='/data/backed_up/kahwang/Clock/'


def create_param_tfr(sdf, fdf, term, threshold = True):
    ''' function to create mne objet for ploting from R data frame
    two inputs, sdf: the dtaframe
    term, the variable for plotting
    You also have the option to threshold the data, if True, then only data with fdr_p<0.05  will be saved
    '''

    # creat TFR epoch object for plotting. Use the "info" in this file for measurement info
    template_TFR = mne.time_frequency.read_tfrs(datapath + 'Group/group_feedback_power-tfr.h5')[0]
    # the data array in this template is 306 ch by 20 freq by 464 time

    # create custom tfr data array
    try:
        tdf=sdf.loc[sdf.term==term]
    except:
        tdf = df

    time = np.sort(tdf.Time.unique()) #what is the diff between "Time" and "t" in the dataframe?
    freq = np.sort(tdf.Freq.unique())
    new_data = np.zeros((306, len(freq), len(time)))

    if threshold:
        fdf=fdf.loc[fdf.term==term]

    # now plut in real stats into the dataframe
    for index, row in tdf.iterrows():
        t = row.Time
        f = row.Freq
        try:
            ch = row.level
            ch = 'MEG'+ '{:0>4}'.format(ch)
        except:
            ch = row.Sensor
            ch = 'MEG'+ '{:0>4}'.format(ch)
        #print(ch)
        ch_idx = mne.pick_channels(template_TFR.ch_names, [ch])
        try:
            if threshold & (fdf.loc[(fdf.Time==t) & (fdf.Freq ==f)]['p_fdr'].values<0.05):
                new_data[ch_idx, np.where(freq==f)[0], np.where(time==t)[0]] = row.estimate
            elif threshold:
                new_data[ch_idx, np.where(freq==f)[0], np.where(time==t)[0]] = 0
        except:
            new_data[ch_idx, np.where(freq==f)[0], np.where(time==t)[0]] = row.estimate

    new_tfr = mne.time_frequency.AverageTFR(template_TFR.info, new_data, time, freq, 1)

    return new_tfr


def extract_sensor_random_effect(rdata, alignment):
    ''' take r data frame, extract sensor level random effect, remeber to specity the alignment 'rt' or 'clock' '''
    df = rdata[None]

    # save a different df that contains the fix effect so we can get the p values
    fdf = df.loc[df.effect=='fixed']
    fdf = fdf.loc[fdf.alignment==alignment]
    fdf.Freq = fdf.Freq.astype('float')

    # cut it down to only include sensor data
    df = df.loc[df.group=='Sensor']
    #select random effects
    df = df.loc[df.effect=='ran_coefs']
    #which alignment?
    df = df.loc[df.alignment==alignment]
    df.Freq = df.Freq.astype('float')

    return df, fdf


def create_tfr(df):
    ''' function to create mne objet for ploting from R data frame from Michael
    '''
    df['Freq'] = df.Freq.str.lstrip('f_')
    df['Freq'] = df.Freq.astype('float')

    # creat TFR epoch object for plotting. Use the "info" in this file for measurement info
    template_TFR = mne.time_frequency.read_tfrs(datapath + 'Group/group_feedback_power-tfr.h5')[0]
    # the data array in this template is 306 ch by 20 freq by 464 time

    # create custom tfr data array
    time = np.sort(df.Time.unique()) #what is the diff between "Time" and "t" in the dataframe?
    freq = np.sort(df.Freq.unique())
    new_data = np.zeros((306, len(freq), len(time)))
    # now plut in real stats into the dataframe
    for index, row in df.iterrows():
        t = row.Time
        f = row.Freq
        try:
            ch = row.level
            ch = 'MEG'+ '{:0>4}'.format(ch)
        except:
            ch = row.Sensor
            ch = 'MEG'+ '{:0>4}'.format(ch)
        #print(ch)
        ch_idx = mne.pick_channels(template_TFR.ch_names, [ch])
        new_data[ch_idx, np.where(freq==f)[0], np.where(time==t)[0]] = row.Estimate

    new_tfr = mne.time_frequency.AverageTFR(template_TFR.info, new_data, time, freq, 1)

    return new_tfr


def create_fixed_effect_tfr(inputdf, reward = 'Omission', regressor = 'RT_t', effect = 'zhigh'):
    ''' function to create mne objet for ploting from fixed effect data frame from Alex
    '''
    #df['Freq'] = df.Freq.str.lstrip('f_')
    inputdf['Freq'] = inputdf.Freq.astype('float')
    df = inputdf.loc[(inputdf['reward_t']==reward) & (inputdf['regressor']==regressor)]
    # creat TFR epoch object for plotting. Use the "info" in this file for measurement info
    template_TFR = mne.time_frequency.read_tfrs(datapath + 'Group/group_feedback_power-tfr.h5')[0]
    # the data array in this template is 306 ch by 20 freq by 464 time

    # create custom tfr data array
    time = np.sort(df.Time.unique()) #what is the diff between "Time" and "t" in the dataframe?
    freq = np.sort(df.Freq.unique())
    new_data = np.zeros((306, len(freq), len(time)))
    # now plut in real stats into the dataframe
    for index, row in df.iterrows():
        t = row.Time
        f = row.Freq
        try:
            ch = row.level
            ch = 'MEG'+ '{:0>4}'.format(ch)
        except:
            ch = row.Sensor
            ch = 'MEG'+ '{:0>4}'.format(ch)
        #print(ch)
        ch_idx = mne.pick_channels(template_TFR.ch_names, [ch])
        new_data[ch_idx, np.where(freq==f)[0], np.where(time==t)[0]] = row[effect]

    new_tfr = mne.time_frequency.AverageTFR(template_TFR.info, new_data, time, freq, 1)

    return new_tfr



####################
#### Read data! MASSIVE data
####################
# sign PE
# spe_data = pyreadr.read_r(datapath + 'meg_ddf_wholebrain_signed_pe.rds')
# spe_rt_df, spe_rt_fdf, = extract_sensor_random_effect(spe_data, 'rt')
# spe_t_rt_tfr = create_param_tfr(spe_rt_df, spe_rt_fdf, 'pe_max_sc')
# spe_t_rt_utfr = create_param_tfr(spe_rt_df, spe_rt_fdf, 'pe_max_sc', threshold = False)
# spe_t_rt_tfr.plot_topo(yscale='log', picks='grad')
# spe_t_rt_utfr.plot_topo(yscale='log', picks='grad')
#
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(0, 1, 0.05)
# for n, time in enumerate(times):
#     spe_t_rt_tfr.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=4, fmax=8, ch_type ='grad', title = ('4 to 8 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(0, 1, 0.05)
# for n, time in enumerate(times):
#     spe_t_rt_tfr.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=12, fmax=17, ch_type ='grad', title = ('12 to 17 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()


## absolute PE, read and plot
pe_data = pyreadr.read_r(datapath + 'meg_ddf_wholebrain_abs_pe.rds')
pe_rt_df, pe_rt_fdf, = extract_sensor_random_effect(pe_data, 'rt')
pe_t_rt_tfr = create_param_tfr(pe_rt_df, pe_rt_fdf, 'scale(abs_pe)')
pe_t_rt_utfr = create_param_tfr(pe_rt_df, pe_rt_fdf, 'scale(abs_pe)', threshold = False)
pe_t_rt_tfr.plot_topo(yscale='log', picks='grad')
pe_t_rt_utfr.plot_topo(yscale='log', picks='grad')
pe_t_rt_utfr.plot_topomap(baseline=None, tmin = 0.7, tmax = 1, fmin=8, fmax=20, ch_type ='grad', cmap = 'Blues_r', size = 3, colorbar = True)
pe_t_rt_utfr.plot_topomap(baseline=None, tmin = 0.7, tmax = 1, fmin=8, fmax=20, vmax=0, ch_type ='grad', cmap = 'Blues_r', size = 3, contours=0 , colorbar = True)
pe_t_rt_utfr.plot_topomap(baseline=None, tmin = 0.5, tmax = 0.6, fmin=4, fmax=8, vmin=0, ch_type ='grad', cmap = 'Reds', size = 3, contours=0 , colorbar = True)

# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(0.5, 1, 0.05)
# for n, time in enumerate(times):
#     pe_t_rt_tfr.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=8, fmax=20, ch_type ='grad', title = ('8 to 20 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(-0.2, 0.5, 0.05)
# for n, time in enumerate(times):
#     pe_t_rt_utfr.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=4, fmax=8, ch_type ='grad', title = ('4 to 8 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()

reward_t_rt_tfr = create_param_tfr(pe_rt_df, pe_rt_fdf, 'reward_t')
reward_t_rt_tfr.plot_topomap(baseline=None, tmin = 0.5, tmax = 0.7, fmin=4, fmax=8, vmax= -0.3, vmin= -0.31, ch_type ='grad', cmap = 'Blues_r', size = 3, colorbar = True)
reward_t_rt_tfr.plot_topo(yscale='log', picks='grad',cmap='Blues_r')

entropy_change_rdata = pyreadr.read_r(datapath + 'entropy_change/meg_ddf_wholebrain_entropy_change.rds') #whole brain data
entropy_change_rt_df, entropy_change_rt_fdf = extract_sensor_random_effect(entropy_change_rdata, 'rt')
#entropy_change_clock_df, entropy_change_clock_fdf = extract_sensor_random_effect(entropy_change_rdata, 'clock')
entropy_change_t_rt_tfr = create_param_tfr(entropy_change_rt_df, entropy_change_rt_fdf, 'entropy_change_t')
entropy_change_t_rt_tfr.plot_topo(yscale='log', picks='grad')
entropy_change_t_rt_tfr.plot_topomap(baseline=None, tmin = 0.8, tmax = 0.85, fmin=6, fmax=16, vmax= 0, ch_type ='grad', cmap = 'Blues_r', contours=0, size = 3, colorbar = True)
entropy_change_t_rt_tfr.plot_topomap(baseline=None, tmin = 1, tmax = 1.05, fmin=3, fmax=6, vmax= 0, ch_type ='grad', cmap = 'Blues_r', contours=0, size = 3, colorbar = True)

entropy_rdata = pyreadr.read_r(datapath + 'entropy/meg_ddf_wholebrain_entropy.rds') #whole brain data
entropy_rt_df, entropy_rt_fdf, = extract_sensor_random_effect(entropy_rdata, 'rt')
# entropy_clock_df, entropy_clock_fdf, = extract_sensor_random_effect(entropy_rdata, 'clock')
v_entropy_wi_rt_tfr = create_param_tfr(entropy_rt_df, entropy_rt_fdf, 'v_entropy_wi', threshold = False)
v_entropy_wi_rt_tfr.plot_topo(yscale='log', picks='grad')


# kld_rdata = pyreadr.read_r(datapath + 'kld/meg_ddf_wholebrain_kld.rds') #whole brain data
# kld_rt_df, kld_rt_fdf = extract_sensor_random_effect(kld_rdata, 'rt')
#kld_clock_df, kld_clock_fdf = extract_sensor_random_effect(kld_rdata, 'clock')

#reward_rdata = pyreadr.read_r(datapath + 'meg_ddf_wholebrain_reward.rds') #whole brain data
#reward_rt_df, reward_rt_fdf = extract_sensor_random_effect(reward_rdata, 'rt')
#eward_clock_df, reward_clock_fdf = extract_sensor_random_effect(reward_rdata, 'clock')

## entropy change pos and neg
# entropy_change_neg_rdata = pyreadr.read_r(datapath + 'meg_ddf_wholebrain_entropy_change_neg.rds') #whole brain data
# entropy_change_neg_rt_df, entropy_change_neg_rt_fdf = extract_sensor_random_effect(entropy_change_neg_rdata, 'rt')
# #entropy_change_neg_clock_df, entropy_change_neg_clock_fdf = extract_sensor_random_effect(entropy_change_neg_rdata, 'clock')
#
# entropy_change_pos_rdata = pyreadr.read_r(datapath + 'meg_ddf_wholebrain_entropy_change_pos.rds') #whole brain data
# entropy_change_pos_rt_df, entropy_change_pos_rt_fdf = extract_sensor_random_effect(entropy_change_pos_rdata, 'rt')
#entropy_change_pos_clock_df, entropy_change_pos_clock_fdf = extract_sensor_random_effect(entropy_change_pos_rdata, 'clock')

# turn dataframe into mne object for plotting
# v_entropy_wi_rt_tfr = create_param_tfr(entropy_rt_df, entropy_rt_fdf, 'v_entropy_wi')
# entropy_change_t_rt_tfr = create_param_tfr(entropy_change_rt_df, entropy_change_rt_fdf, 'entropy_change_t')
# kld_v_entropy_wi_rt_tfr = create_param_tfr(kld_rt_df, kld_rt_fdf, 'v_entropy_wi')
# v_entropy_wi_clock_tfr = create_param_tfr(entropy_clock_df, entropy_clock_fdf, 'v_entropy_wi')
# entropy_change_t_clock_tfr = create_param_tfr(entropy_change_clock_df, entropy_change_clock_fdf, 'entropy_change_t')
# kld_v_entropy_wi_clock_tfr = create_param_tfr(kld_clock_df, kld_clock_fdf, 'v_entropy_wi')
# entropy_change_neg_t_rt_tfr = create_param_tfr(entropy_change_neg_rt_df, entropy_change_neg_rt_fdf, 'entropy_change_neg_t')
# entropy_change_pos_t_rt_tfr = create_param_tfr(entropy_change_pos_rt_df, entropy_change_pos_rt_fdf, 'entropy_change_pos_t')
# entropy_change_neg_t_clock_tfr = create_param_tfr(entropy_change_neg_clock_df, entropy_change_neg_clock_fdf, 'entropy_change_neg_t')
# entropy_change_pos_t_clock_tfr = create_param_tfr(entropy_change_pos_clock_df, entropy_change_pos_clock_fdf, 'entropy_change_pos_t')
# reward_t_rt_tfr = create_param_tfr(reward_rt_df, reward_rt_fdf, 'reward_t')
# reward_t_clock_tfr = create_param_tfr(reward_clock_df, reward_clock_fdf, 'reward_t')
# pe_t_rt_tfr = create_param_tfr(pe_rt_df, pe_rt_fdf, 'scale(abs_pe)')
# reward_t_rt_tfr = create_param_tfr(pe_reward_t_df, pe_reward_t_fdf, 'reward_t')

# pe_t_rt_tfr.plot_topomap(baseline=None, tmin = 0.4, tmax = 0.6, fmin=4, fmax=8, ch_type ='grad')
# pe_t_rt_tfr.plot_topomap(baseline=None, tmin = 0.5, tmax = 0.9, fmin=8, fmax=9.9, ch_type ='grad')


#bayesian meta analsis from Michael
# int_contrast_by_sensor_df = pd.read_csv(datapath + 'int_contrast_by_sensor.csv')
# hilo_reward_by_sensor_df = pd.read_csv(datapath + 'hilo_reward_by_sensor.csv')
# hilo_omission_by_sensor_df = pd.read_csv(datapath + 'hilo_omission_by_sensor.csv')
#
# int_contrast_by_sensor_tfr= create_tfr(int_contrast_by_sensor_df)
# hilo_reward_by_sensor_tfr = create_tfr(hilo_reward_by_sensor_df)
# hilo_omission_by_sensor_tfr = create_tfr(hilo_omission_by_sensor_df)


# fixed effects RT prediction
#wholebrain_fixefrandom_slope = pyreadr.read_r(datapath+'meg_rdf_wholebrain_fixefrandom_slope.rds')
wholebrain_zstats_random_slope = pyreadr.read_r(datapath + 'meg_rdf_wholebrain_zstats_random_slope.rds')
zstats_df = wholebrain_zstats_random_slope[None]
Omission_RT_t_zdiff = create_fixed_effect_tfr(zstats_df, 'Omission', 'RT_t' ,'zdiff')
Reward_RT_t_zdiff = create_fixed_effect_tfr(zstats_df, 'Reward', 'RT_t' ,'zdiff')
Reward_RT_vmax_zdiff = create_fixed_effect_tfr(zstats_df, 'Reward', 'RT_Vmax_t' ,'zdiff')

Reward_RT_vmax_zdiff.plot_topo(yscale='log', picks='grad')
Omission_RT_t_zdiff.plot_topo(yscale='log', picks='grad')
Reward_RT_t_zdiff.plot_topo(yscale='log', picks='grad')

Reward_RT_vmax_zdiff.plot_topomap(baseline=None, tmin = 1, tmax = 1.05, fmin=10, fmax=18, vmin= 0, ch_type ='grad', cmap = 'Reds', contours=0, size = 3, colorbar = True)
Omission_RT_t_zdiff.plot_topomap(baseline=None, tmin = 0.5, tmax = 0.55, fmin=2, fmax=6, vmax= 0, ch_type ='grad', cmap = 'Blues_r', contours=0, size = 3, colorbar = True)

#
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(-1.5, 1.5, 0.2)
# for n, time in enumerate(times):
#     Omission_RT_t_zdiff.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=2.1, fmax=4, vmax=-2, cmap='Blues_r', ch_type ='grad', title = ('2 to 4 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(-1.5, 1.5, 0.2)
# for n, time in enumerate(times):
#     Omission_RT_t_zdiff.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=4, fmax=8, vmax=-2, cmap='Blues_r', ch_type ='grad', title = ('4 to 8 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(-1.5, 1.5, 0.2)
# for n, time in enumerate(times):
#     Omission_RT_t_zdiff.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=8, fmax=14, vmax=-2, cmap='Blues_r', ch_type ='grad', title = ('8 to 14 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(-1.5, 1.5, 0.2)
# for n, time in enumerate(times):
#     Omission_RT_t_zdiff.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=14, fmax=20, vmax=-2, cmap='Blues_r', ch_type ='grad', title = ('14 to 20 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
#
# Reward_RT_t_zdiff.plot_topo(yscale='log', picks='grad', vmin=1, cmap='Reds_r')
#
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(-1.5, 1.5, 0.2)
# for n, time in enumerate(times):
#     Reward_RT_t_zdiff.plot_topomap(baseline=None, tmin = time, tmax = time+0.2, fmin=2.1, fmax=4, vmin=1, cmap='Reds', ch_type ='grad', title = ('2 to 4 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(-1.5, 1.5, 0.2)
# for n, time in enumerate(times):
#     Reward_RT_t_zdiff.plot_topomap(baseline=None, tmin = time, tmax = time+0.2, fmin=4, fmax=8, vmin=1, cmap='Reds', ch_type ='grad', title = ('4 to 8 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(-1.5, 1.5, 0.2)
# for n, time in enumerate(times):
#     Reward_RT_t_zdiff.plot_topomap(baseline=None, tmin = time, tmax = time+0.2, fmin=8, fmax=14, vmin=1, cmap='Reds', ch_type ='grad', title = ('8 to 14 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
#
# Reward_RT_vmax_zdiff.plot_topo(yscale='log', picks='grad', vmin=2, cmap='Reds_r')
#
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(-1.5, 1.5, 0.2)
# for n, time in enumerate(times):
#     Reward_RT_vmax_zdiff.plot_topomap(baseline=None, tmin = time, tmax = time+0.2, fmin=2.1, fmax=4, vmin=2, cmap='Reds', ch_type ='grad', title = ('2 to 4 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(-1.5, 1.5, 0.2)
# for n, time in enumerate(times):
#     Reward_RT_vmax_zdiff.plot_topomap(baseline=None, tmin = time, tmax = time+0.2, fmin=4, fmax=8, vmin=2, cmap='Reds', ch_type ='grad', title = ('4 to 8 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(-1.5, 1.5, 0.2)
# for n, time in enumerate(times):
#     Reward_RT_vmax_zdiff.plot_topomap(baseline=None, tmin = time, tmax = time+0.2, fmin=8, fmax=14, vmin=2, cmap='Reds', ch_type ='grad', title = ('8 to 14 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()

####################
#### rt prediction
####################
# rt_prediction_randomslope_rdata = pyreadr.read_r(datapath + 'meg_rdf_wholebrain_zstats_random_slope.rds')
# df = rt_prediction_randomslope_rdata[None]
# # cut it down to only include sensor data
# df = df.loc[df.alignment=='RT']
# df = df.loc[df.reward_t=='Reward']
# #select random effects
# df = df.loc[df.regressor=='RT_t']
# #which alignment?
# df.Freq = df.Freq.astype('float')
# df['estimate'] = df['zdiff']
#
# df = rt_prediction_randomslope_rdata[None]
# df = df.loc[df.alignment=='RT']
# df = df.loc[df.reward_t=='Omission']
# #select random effects
# df = df.loc[df.regressor=='RT_t']
# #which alignment?
# df.Freq = df.Freq.astype('float')
# df['estimate'] = df['zdiff']
#
# df = rt_prediction_randomslope_rdata[None]
# df = df.loc[df.alignment=='RT']
# df = df.loc[df.reward_t=='Reward']
# #select random effects
# df = df.loc[df.regressor=='RT_Vmax_t']
# #which alignment?
# df.Freq = df.Freq.astype('float')
# df['estimate'] = df['zdiff']



####################
#### Plot!!
####################

# this function plots the sensor-wide TFR plot (the one you can click around with)
# v_entropy_wi_clock_tfr.plot_topo(yscale='log', picks='grad')
#
# entropy_change_t_rt_tfr.plot_topo(yscale='log', picks='grad')
# entropy_change_t_clock_tfr.plot_topo(yscale='log', picks='grad')
#
# kld_v_entropy_wi_rt_tfr.plot_topo(yscale='log', picks='grad')
# kld_v_entropy_wi_clock_tfr.plot_topo(yscale='log', picks='grad')
#
# entropy_change_neg_t_rt_tfr.plot_topo(yscale='log', picks='grad')
# entropy_change_neg_t_clock_tfr.plot_topo(yscale='log', picks='grad')
#
# reward_t_rt_tfr.plot_topo(yscale='log', picks='grad')
# reward_t_clock_tfr.plot_topo(yscale='log', picks='grad')
#
# pe_t_rt_tfr.plot_topo(yscale='log', picks='grad')
#
# tfr_omission_RT_t.plot_topo(yscale='log', picks='grad')
# tfr_reward_RT_t.plot_topo(yscale='log', picks='grad')
# tfr_reward_RTvmax_t.plot_topo(yscale='log', picks='grad')
#
# Reward_RT_vmax_zdiff.plot_topomap(baseline=None, tmin = 0.5, tmax = 0.6, fmin=10, fmax=14, ch_type ='grad')

# this plots the topographic map with specific time-frequency interval
# under the 'timefreqs' flag, you can specifiy a list of (time, frequency) montage that you would like to plot
#v_entropy_wi_tfr.plot_joint(baseline=None, yscale='log', timefreqs=[(-1.5, 10), (-1, 10), (-0.5, 10), (0, 10), (0.5, 10), (1, 10)], picks='grad')


########
# publication plots

## abs pe, theta signal, 5 to 8 hz, 0.2 to 0.4 s
#from matplotlib.pyplot import figure
#f = figure(figsize=(8, 6), dpi=300)
#fig, axis = plt.subplots(3, 3, squeeze = True, figsize=(3,4))
# f = pe_t_rt_tfr.plot_topomap(baseline=None, tmin = -0.3, tmax = -0.1, fmin=5, fmax=8, ch_type ='grad', cmap = 'plasma', vmin = -0.08, vmax = 0.08, show=False, size = 3, colorbar = False)
# f.savefig('figures/pe_theta_-0.3.tiff')
# f = pe_t_rt_tfr.plot_topomap(baseline=None, tmin = 0.5, tmax = 0.7, fmin=5, fmax=8, ch_type ='grad', cmap = 'plasma', vmin = -0.08, vmax = 0.08, show=False, size = 3, colorbar = False)
# f.savefig('figures/pe_theta_0.5.tiff')
# f = pe_t_rt_tfr.plot_topomap(baseline=None, tmin = 0.8, tmax = 0.95, fmin=8, fmax=10, ch_type ='grad', cmap = 'plasma', vmin = -0.08, vmax = 0.08, show=False, size = 3, colorbar = False)
# f.savefig('figures/pe_alpha_-0.8.tiff')
#plt.show()
#a.set_size_inches(4, 4, forward=True)

### reard signal, 2 to 4 hz, 4 to 8 hz, 0.2 to 0.8 s
# f = reward_t_rt_tfr.plot_topomap(baseline=None, tmin = 0.5, tmax = 0.7, fmin=4, fmax=8, ch_type ='grad', cmap = 'plasma', vmin = -0.46, vmax = -0.35, show=False, size = 3, colorbar = False)
# f.savefig('figures/reward_delta_0.5.tiff')


### entropy change, alpha, theta, low beta 0.8 to 1.5
# f= entropy_change_t_rt_tfr.plot_topomap(baseline=None, tmin = 0.8, tmax = 1.5, fmin=14, fmax=20, ch_type ='grad', cmap = 'plasma', vmin = -0.009, vmax = 0.0005, show=False, size = 3, colorbar = False)
# f.savefig('figures/entropy_change_lowbeta_0.8.tiff')
#
# f= entropy_change_t_rt_tfr.plot_topomap(baseline=None, tmin = 0.8, tmax = 1.5, fmin=8, fmax=14, ch_type ='grad', cmap = 'plasma', vmin = -0.02, vmax = 0.0005, show=False, size = 3, colorbar = False)
# f.savefig('figures/entropy_change_alpha_0.8.tiff')
#
# f= entropy_change_t_rt_tfr.plot_topomap(baseline=None, tmin = 0.8, tmax = 1.5, fmin=4, fmax=8, ch_type ='grad', cmap = 'plasma', vmin = -0.009, vmax = 0.0005, show=False, size = 3, colorbar = False)
# f.savefig('figures/entropy_change_theta_0.8.tiff')
#
# f= entropy_change_t_rt_tfr.plot_topomap(baseline=None, tmin = 0, tmax = 0.3, fmin=8, fmax=14, ch_type ='grad', cmap = 'plasma', vmin = -0.008, vmax = 0.08, show=False, size = 3, colorbar = True)
# f.savefig('figures/entropy_change_alpha_0.tiff')
# f= entropy_change_t_rt_tfr.plot_topomap(baseline=None, tmin = 0, tmax = 0.3, fmin=8, fmax=14, ch_type ='grad', cmap = 'plasma', vmin = -0.08, vmax = 0.0005, show=False, size = 3, colorbar = False)
# f.savefig('figures/entropy_change_alpha_0.tiff')
#


### v_max_wi??

########
# another way to generate time-frequency montage
# Here let us plot the montage of alpha
# you would have to play around the colorbar scale (vmin and vmax).
#
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(0.5, 0.9, 0.05)
# for n, time in enumerate(times):
#     pe_t_rt_tfr.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=5, fmax=8, ch_type ='grad', title = ('5 to 8 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(0.5, 1.3, 0.05)
# for n, time in enumerate(times):
#     pe_t_rt_tfr.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=8, fmax=12, ch_type ='grad', title = ('8 to 12 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
#
#
#
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(0, 1.5, 0.1)
# for n, time in enumerate(times):
#     reward_t_rt_tfr.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=4, fmax=9, ch_type ='grad', title = ('3 to 7 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
#
#
# ##### entropy_change_neg_t_rt_tfr, effect around -1.5 s (clock alignment)
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(-0.5, 1.5, 0.1)
# for n, time in enumerate(times):
#     entropy_change_pos_t_rt_tfr.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=8, fmax=14, ch_type ='grad',vmin = -0.4, vmax = 0.4, title = ('8 to 14 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(-0.5, 1.5, 0.1)
# for n, time in enumerate(times):
#     entropy_change_neg_t_rt_tfr.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=4, fmax=7, ch_type ='grad', title = ('3 to 7 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
#
#
# ##### v_entropy_wi, effect around -1.5 s (clock alignment)
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(-1.7, -1.45, 0.05)
# for n, time in enumerate(times):
#     v_entropy_wi_clock_tfr.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=8, fmax=14, ch_type ='grad', title = ('8 to 14 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
# ##### entropy_change_t_rt_tfr, alpha effect around 0 to 1sec
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(0.8, 1.2, 0.05)
# for n, time in enumerate(times):
#     entropy_change_t_rt_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=8, fmax=14, ch_type ='grad', title = ('8 to 14 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
# ##### entropy_change_t_rt_tfr, low beta effect around 0 to 1sec
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(0.1, 0.5, 0.05)
# for n, time in enumerate(times):
#     entropy_change_t_rt_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=12, fmax=18, vmin = -0.15, vmax = 0.15, ch_type ='grad', title = ('12 to 18 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
# ##### entropy_change_t_rt_tfr, alpha effect around 0 to 1sec
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(0.8, 1.2, 0.05)
# for n, time in enumerate(times):
#     entropy_change_t_rt_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=1, fmax=18, vmin = -0.15, vmax = 0.15, ch_type ='grad', title = ('8 to 14 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
# ##### entropy_change_t_rt_tfr, theta effect around 1sec
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(0.5, 1.5, 0.1)
# for n, time in enumerate(times):
#     entropy_change_t_rt_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=3, fmax=5, vmin = -0.055, vmax = 0.055, ch_type ='grad', title = ('3 to 5 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
#
# ##### entropy_change_t_rt_tfr, detla effect around 1sec
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(1, 1.5, 0.05)
# for n, time in enumerate(times):
#     entropy_change_t_rt_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=2.1, fmax=4, vmin = -0.08, vmax = 0.08, ch_type ='grad', title = ('3 to 5 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
#
#
#
#
# ##### entropy_change_t_clock_tfr, alpha effect
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(-1.7, -1.3, 0.025)
# for n, time in enumerate(times):
#     entropy_change_t_clock_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=8, fmax=14, vmin = -0.15, vmax = 0.15, ch_type ='grad', title = ('8 to 14 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
#
# ##### kld, alpha effect
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(-1.7, -1.3, 0.025)
# for n, time in enumerate(times):
#     entropy_change_t_clock_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=3, fmax=8, vmin = -0.15, vmax = 0.15, ch_type ='grad', title = ('3 to 8 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
#
#
# ##### v_entropy_wi, effect around -1.5 s (clock alignment)
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(-1.5, 1.8, 0.3)
# for n, time in enumerate(times):
#     kld_v_entropy_wi_rt_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=8, fmax=14, ch_type ='grad', vmin = -0.06, vmax = 0.06, title = ('8 to 14 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
# ##### v_entropy_wi, effect around -1.5 s (clock alignment)
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(1.4, 2, 0.1)
# for n, time in enumerate(times):
#     kld_v_entropy_wi_rt_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=8, fmax=14, ch_type ='grad', title = ('8 to 14 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
# ##### v_entropy_wi, effect around -1.5 s (clock alignment)
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(1.4, 2, 0.1)
# for n, time in enumerate(times):
#     kld_v_entropy_wi_rt_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=13, fmax=20, ch_type ='grad', title = ('13 to 20 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
# ##### v_entropy_wi, effect around -1.5 s (clock alignment)
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(1.4, 2, 0.1)
# for n, time in enumerate(times):
#     kld_v_entropy_wi_rt_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=13, fmax=20, ch_type ='grad', title = ('13 to 20 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#
# ##### v_entropy_wi, effect around -1.5 s (clock alignment)
# fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
# times = np.arange(-1.7, 0, 0.1)
# for n, time in enumerate(times):
#     kld_v_entropy_wi_clock_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=13, fmax=20, ch_type ='grad', title = ('13 to 20 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
# plt.show()
#

########################################################################
######### graveyard

#entropy_change_tfr = create_param_tfr(sdf, 'entropy_change_t')
#entropy_change_tfr.plot_topo(yscale='log', picks='grad')

#entropy_change_tfr = create_param_tfr(sdf, 'entropy_change_t')
# a few different types of plots
#entropy_change_tfr.plot_topo(yscale='log', picks='grad')
#entropy_change_tfr.plot_joint(mode='mean', yscale='log', timefreqs=[(-0.1, 20), (0.25, 12), (0.6, 5), (0.9, 10), (1, 5)], picks='grad')


#entropy_wi_tfr = create_param_tfr(sdf, 'v_entropy_wi')
#entropy_change_tfr.plot_joint(mode='mean', timefreqs=[(-1, 10), (1, 20)], picks='grad')
#entropy_change_tfr.plot_topomap(ch_type='grad', tmin=-1, tmax=-0.8, fmin=15, fmax=40, title='Beta at time -1 to -.8')
#
#
# reward_tfr = create_param_tfr(sdf, 'reward_t')
# reward_tfr.plot_joint(mode='mean', timefreqs=[(-1, 10),(0.6, 24), (1, 20)], picks='grad')
# reward_tfr.plot_topomap(ch_type='grad', tmin=0.4, tmax=0.8, fmin=15, fmax=30, title='Beta at time 0.4 to 0.8')
#
# entropy_wi_tfr = create_param_tfr(sdf, 'v_entropy_wi')
#
# fdf = df.loc[pd.isna(df.group)]
# fdf = fdf.loc[fdf.alignment=='rt']
# fdf.Freq = fdf.Freq.astype('float')
#
# def create_param_fixed_tfr(sdf, term):
#     # creat TFR epoch object for plotting. Use the "info" in this file for measurement info
#     template_TFR = mne.time_frequency.read_tfrs(datapath + 'Group/group_feedback_power-tfr.h5')[0]
#     # the data array in this template is 306 ch by 20 freq by 464 time
#
#     # create custom tfr data array
#     tdf=sdf.loc[sdf.term==term]
#     time = tdf.Time.unique() #what is the diff between "Time" and "t" in the dataframe?
#     freq = tdf.Freq.unique()
#     new_data = np.zeros((306, len(freq), len(time)))
#
#     # now plut in real stats into the dataframe
#     for index, row in tdf.iterrows():
#         t = row.Time
#         f = row.Freq
#         #ch = 'MEG'+row.level
#         #ch_idx = mne.pick_channels(template_TFR.ch_names, [ch])
#         new_data[:, np.where(freq==f)[0], np.where(time==t)[0]] = row.estimate
#     new_tfr = mne.time_frequency.AverageTFR(template_TFR.info, new_data, time, freq, 1)
#
#     return new_tfr
#
#
#
# entropy_wi_fixed_tfr = create_param_fixed_tfr(fdf, 'v_entropy_wi')
#
#

# end
