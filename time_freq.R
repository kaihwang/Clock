library(biwavelet)
library(dplyr)
library(gplots)
library(data.table)
library(ggplot2)
library(foreach)
library(iterators)
library(doParallel)
# setwd("~/Data_Analysis/clock_analysis/meg/code")
setwd("/proj/mnhallqlab/projects/Clock_MEG/code")
source("tf_functions.R")

# external settings
epoch <- Sys.getenv("epoch")
if (epoch == "") {
  epoch <- "RT"
}

#support a single source RDS input for subject-level DAN sources
sourcefile <- Sys.getenv("sourcefile")
if (sourcefile != "") {
  use_source <- TRUE
  stopifnot(file.exists(sourcefile))
  sensors <- sourcefile
} else {
  use_source <- FALSE
  sensors <- Sys.getenv("sensor")
  if (sensors == "") {
    sensors <- c("0612", "0613", "0542", "0543", "1022", "1823", "1822", "2222", "2223")
  }
}

tmin <- Sys.getenv("tmin")
if (tmin == "") {
  tmin <- -Inf
} else {
  tmin <- as.numeric(tmin)
}

tmax <- Sys.getenv("tmax")
if (tmax == "") {
  tmax <- Inf
} else {
  tmax <- as.numeric(tmax)
}

# paths
if (isTRUE(use_source)) {
  datapath <- paste0("/proj/mnhallqlab/projects/Clock_MEG/dan_source_rds/", epoch, "_time")
  outputpath <- sub("_time", "_timefreq", datapath, fixed = TRUE)
} else {
  datapath <- paste0("/proj/mnhallqlab/projects/Clock_MEG/tfr_rds/", epoch, "/original")
  outputpath <- sub("/original", "/time_freq", datapath, fixed = TRUE)
}

if (!dir.exists(outputpath)) {
  dir.create(outputpath)
}

for (ss in sensors) {
  if (isTRUE(use_source)) {
    infile <- ss
    outfile <- file.path(outputpath, sub("_source.rds", "_source_timefreq.rds", basename(ss), fixed = TRUE))
  } else {
    infile <- file.path(datapath, paste0("MEG", ss, ".rds"))   
    outfile <- file.path(outputpath, paste0("MEG", ss, "_tf.rds"))
  }

  if (file.exists(outfile)) next # skip existing files
  result <- timefreq_calc(infile,
    filetype = ifelse(isTRUE(use_source), "source", "sensor"),
    ncpus = 1, tmin = tmin, tmax = tmax
  )

  saveRDS(result, file = outfile)
}