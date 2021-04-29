import mne
import pandas as pd
import numpy as np
from Clock import raw_to_epoch
from Clock import get_dropped_trials_list
import os.path as op
import nibabel as nib
import nilearn
import glob
from nibabel.processing import resample_from_to
from nilearn.image import index_img
from nilearn.input_data import NiftiLabelsMasker

subjects_dir = '/data/backed_up/kahwang/Clock/'
subjects = np.loadtxt('/home/kahwang/bin/Clock/subjects', dtype=int)
channels_list = np.load('/home/kahwang/bin/Clock/channel_list.npy')

# use subject 10997 as an example, but you can do for subject in subjects: to loop through each subject
for subject in [11313]:
    print(subject)

    #create dataframe for each subject
    subject_df = pd.DataFrame()

    # read inverse and source space
    filter_fn = subjects_dir + '%s/MEG/%s-vol-beamformer-lcmv.h5' %(subject, subject)
    filters = mne.beamformer.read_beamformer(filter_fn)
    src_fn = subjects_dir +'%s/MEG/%s-vol-src.fif' %(subject, subject)
    vol_src = mne.read_source_spaces(src_fn)

    # read epochs
    Event_types = ['feedback']
    fb_ep = raw_to_epoch(subject, Event_types)
    drops = get_dropped_trials_list(fb_ep) #0 based

    # hack 11313!
    if subject == 11313:
        drops = np.array([63,125,187])

    # project
    stcs = mne.beamformer.apply_lcmv_epochs(fb_ep['feedback'], filters, max_ori_out='signed')
    # temporal info for this source object tmin : -4000.0 (ms), tmax : 1000.0 (ms), tstep : 4.0 (ms)

    # read in Michael's ROIs
    roipath = subjects_dir + 'dan_subject_1mm/%s*dan_1mm.nii.gz' %subject # get rid of the data prefix
    roifile = glob.glob(roipath)
    roifile = nib.load(roifile[0])

    # source image in 5mm voxle, so need to resample
    # interpolate ROI template to 5mm (source resolution)
    template_img = index_img(stcs[0].as_volume(vol_src),0)  # turn into nii object for nilearn and nibabel manipulations
    roi_5mm = resample_from_to(roifile, template_img, mode='nearest')

    #rois of interest from Michael, create a new roi mask file with just these ROIs
    rois_list = [137,34,147,43,35,139,138,36,141,38,145,41,142,33,144,40,140,37,143,39,146,42,135,136,31,32,25,128]
    mask_data = np.zeros((roi_5mm.shape))
    for r in rois_list:
        mask_data[roi_5mm.get_fdata()==r] = r
    roi_mask = nilearn.image.new_img_like(roi_5mm, mask_data)
    # create masker with this roi image
    masker = NiftiLabelsMasker(labels_img=roi_mask, standardize=False) #roi masker

    # loop through trials, extract ts for each trial
    # log the trial index, so that it will "skip" over the dropped trials
    trials = np.arange(0,504)
    if drops.size > 0:
        trials = np.delete(trials, drops, axis=0)

    # create df for each trial
    for i, trial in enumerate(trials):
        source_epoch = stcs[i].as_volume(vol_src) # turn source data into a nii object
        ts = masker.fit_transform(source_epoch) # 1251 time points by 25 ROIs, ROIs will be ranked by its integer values
        trial_df = pd.DataFrame(columns = masker.labels_, data = ts)
        trial_df['time'] = stcs[i].times
        trial_df['trial_num'] = trial+1  # adjust 0 base indexing

        # concate each trial df to the subject df.
        # the end result is a subject_df that has the timeseries for each trial and each ROI
        subject_df = pd.concat([subject_df, trial_df])
        output_path = subjects_dir + 'csv_data/%s_source_ts.csv' %subject
        # the output will be about 300mb per subjet
        subject_df.to_csv(output_path)















# end of file