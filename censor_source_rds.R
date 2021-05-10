# apply censoring to RDS data
library(pacman)
p_load(checkmate, data.table, R6, doParallel)
source("/proj/mnhallqlab/users/michael/fmri.pipeline/R/rle_dt.R") # sparse storage class

epoch <- Sys.getenv("epoch")
if (epoch == "") {
    epoch <- "clock"
}

basedir <- "/proj/mnhallqlab/projects/Clock_MEG/fif_data/csv_data"
inputdir <- file.path(basedir, paste0(epoch, "_rds"))
csvdata_dir <- "/proj/mnhallqlab/projects/clock_analysis/meg/data/csv_timings"
behav_files <- list.files(path = csvdata_dir, pattern = ".*alltimes\\.csv", full.names = TRUE)
allbehav <- rbindlist(lapply(behav_files, fread))
source_files <- list.files(inputdir, pattern = ".*_source\\.rds", full.names = TRUE)

# We should dump MEG data for super-short trials, right? There are many where RTs < 100
# allbehav[rt < 100] #155 trials with excessively short RTs

# Also dump trials where RT > 4000 and there is no score (time out events)
# r$> summary(allbehav$rt > 4000 & allbehav$score==0)
#    Mode   FALSE    TRUE    NA's
# logical   35453     268      63

for (ff in source_files) {
    xx <- readRDS(ff)
    xx <- xx$get()

    # load behavioral data for this subject
    behav_data <- allbehav[id == xx$Subject[1]]
    setnames(behav_data, c("id", "trial"), c("Subject", "Trial")) # to align with MEG

    bad_trials <- behav_data$Trial[is.na(behav_data$rt) | behav_data$rt < 100 |
        (behav_data$rt > 4000 & behav_data$score == 0)]
    xx <- xx[!Trial %in% bad_trials, ] # drop bad trials from neural data

    if (xx$Event[1] == "RT") {
        # for RT aligned data, censor onset of next trial (as noted above, this never happens in a 2s interval)
        # and censor pre-clock period (offline) -- so that we are always measuring online choice

        behav_data[, rt_censor_late := max(iti_offset - isi_onset, 2), by = Trial]
        behav_data[, rt_censor_early := max(-2, rt / -1000), by = Trial]

        # Conclusion: for RT aligned data, the +2 interval never intersects the next trial onset because:
        #   .3 ISI + .85 feedback + 1.0-1.5s ITI --> min is 2.15
        # r$> summary(behav_data$rt_censor_late)
        #  Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
        # 2.150   2.250   2.400   2.400   2.550   2.651

        # It is possible to have the -2 interval encompass an offline period, however,
        # since RTs can be less than 2s
        # r$> summary(behav_data$rt_censor_early)
        #   Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
        # -2.000  -2.000  -1.185  -1.295  -0.730  -0.005

        tomerge <- behav_data[, c("Trial", "rt_censor_early", "rt_censor_late")]
        xx <- merge(xx, tomerge, by = "Trial")
        xx <- xx[Time >= rt_censor_early & Time <= rt_censor_late]
        xx[, rt_censor_early := NULL]
        xx[, rt_censor_late := NULL]
    } else if (xx$Event[1] == "clock") {
        # censor anything before the previous ITI
        behav_data[, prior_iti := dplyr::lag(iti_ideal, 1, order_by = Trial, default = 0)]
        tomerge <- behav_data[, c("Trial", "prior_iti", "rt")]
        #behav_data[, trial_offset_cum := trial_offset + clock_onset]
        xx <- merge(xx, tomerge, by = "Trial")
        xx <- xx[Time >= -1*prior_iti & Time <= rt/1000]
        xx[, rt := NULL]
        xx[, prior_iti := NULL]
    }

    saveRDS(xx, sub("_source.rds", "_source_censor.rds", ff, fixed = TRUE))
}