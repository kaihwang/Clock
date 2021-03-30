library(biwavelet)
library(dplyr)
library(gplots)
library(data.table)
library(ggplot2)
library(foreach)
library(iterators)
library(doParallel)
#setwd("~/Data_Analysis/clock_analysis/meg/code")
setwd("/proj/mnhallqlab/projects/Clock_MEG/code")

compute_wavelet <- function(ts, time, dt=.004, dj=1/4, maxfreq=NULL, minfreq=NULL) {
  max.scale <- ifelse(is.null(maxfreq), NULL, 1/minfreq)
  s0 <- ifelse(is.null(maxfreq), 2*dt, 1/maxfreq)
  
  res <- wt(cbind(time, ts), dt=dt, s0=s0, max.scale=max.scale, dj=dj, do.sig=FALSE, pad=TRUE)
  odf <- data.table(time=time, t(res$power))
  setnames(odf, c("Time", paste0("f_", sprintf("%06.3f", 1/res$scale))))
  odf <- data.table::melt(odf, id.vars="Time", variable.name="Freq", value.name="Pow")
  return(odf)
}


subsample_dt <- function(dt, keys=key(dt), dfac=1L, method="subsamp") {
  checkmate::assert_data_table(dt)
  downsamp <- function(col, dfac=1L) { col[seq(1, length(col), dfac)] }
  
  if (method=="subsamp") {
    dt[, lapply(.SD, downsamp, dfac=dfac), by=keys]
  } else if (method=="mean") {
    dt[, chunk := rep(1:ceiling(.N/dfac), each=dfac, length.out=.N), by=keys]
    
    dt <- dt[, lapply(.SD, mean), by=c(keys, "chunk")] #compute mean of every k samples
    dt[, chunk := NULL]
  } else if (method=="time_mean") {
    #dt[, chunk:=ggplot2::cut_interval(Time, n = NULL, length = NULL, ...)]
  }
  
  return(dt)
}


timefreq_sensor <- function(ff, downsamp=12, ncpus=4) {
  df <- readRDS(ff)
  df <- df$get()
  df[, Channel:=NULL]
  df[, Event:=NULL] #all feedback for now
  setkeyv(df, c("Subject", "Run", "Trial"))
  setorderv(df, c("Subject", "Run", "Trial", "Time")) #make sure we sort properly before subsampling
  
  #wavelet settings: .004s bins, 25
  dt <- .004
  freq <- 1/dt
  minfreq <- 2
  maxfreq <- 40
  
  keys <- key(df)

  if (ncpus > 1) {
    cl <- makeCluster(ncpus)
    registerDoParallel(cl)
    on.exit(try(stopCluster(cl)))
  } else {
    registerDoSEQ()
  }
  
  #ram blows up if we go across all subjects
  #thus, split by subject and do the time-frequency + downsampling per subject
  splitdt <- split(df, df$Subject)
  rm(df) #garbage collect unsplit data

  ff <- foreach(thisdf=iter(splitdt), .noexport="splitdt", .packages=c("biwavelet", "data.table"), .export=c("subsample_dt", "compute_wavelet")) %dopar% {
    
    timefreq_dt <- thisdf[, .(lapply(.SD, compute_wavelet, ts=Signal, time=Time, minfreq=minfreq, maxfreq=maxfreq)), by=keys, .SDcols="Signal"]
    #timefreq_dt <- timefreq_dt[, .(V1[[1]]), by=.(Subject, Run, Trial)] #only needed if we return odf as a list in function (bad idea)
    
    #unnest by selecting out data.table from V1 -- V1 is a 'list-column'
    timefreq_dt <- timefreq_dt[, V1[[1]], by=keys]
    setkeyv(timefreq_dt, c("Subject", "Run", "Trial", "Freq"))
    setorderv(timefreq_dt, c("Subject", "Run", "Trial", "Freq", "Time")) #make sure we sort properly before subsampling
    
    ss <- subsample_dt(timefreq_dt, dfac=downsamp, method="mean")
    return(ss)
  }
  
  return(rbindlist(ff))
  
}

sensors <- c("0612", "0613", "0542", "0543","1022", "1823", "1822", "2222", "2223")

for (ss in sensors) {
  result <- timefreq_sensor(paste0("../r_channel_combined/MEG", ss, ".rds"))
  saveRDS(result, file=paste0("../time_frequency/MEG", ss, "_tf.rds"))
}

# dd <- expand.grid(unique(df$Subject), unique(df$Run), unique(df$Trial))

#example <- compute_wavelet(t1$Signal, t1$Time, minfreq=minfreq, maxfreq=maxfreq)

# ss1 <- timefreq_dt %>% filter(Subject==10637 & Run==1 & Trial==1)
# 
# 
# 
# t1 <- df %>% filter(Subject==11329 & Run==1 & Trial==1) %>% mutate(Rownum=1:n())
# #diff(t1$Time)
# 
# #test <- wt(t1 %>% select(Rownum, Signal), pad=TRUE, do.sig=FALSE, dt = .004) #, J1 = 20)
# test <- wt(t1 %>% select(Time, Signal), pad=TRUE, do.sig=FALSE, dt = .004, s0=1/maxfreq, max.scale=1/minfreq, dj=1/12)
# 1/test$scale
# 
# powmelt <- reshape2::melt(test$power)
# 
# 
# 
# 
# example <- compute_wavelet(t1$Signal, t1$Time, minfreq=minfreq, maxfreq=maxfreq)
# 
# 
# timefreq_dt <- df
# 
# 
# ggplot(example, aes(x=time, y=freq, fill=pow))+ geom_tile()
# 
# #####
# 
# plot(test, plot.cb=TRUE)
# 
# heatmap.2(test$power,dendrogram='none', Rowv=FALSE, Colv=FALSE,trace='none')
# 
# t1 <- cbind(1:100, rnorm(100))
# 
# ## Continuous wavelet transform
# wt.t1 <- wt(t1)
# 
# par(oma = c(0, 0, 0, 1), mar = c(5, 4, 4, 5) + 0.1)
# plot(wt.t1, plot.cb = TRUE, plot.phase = FALSE)
# 
# 
# # library(WaveletComp)
# # my.wt <- analyze.wavelet(my.data, "x",
# #                              loess.span = 0,
# #                              dt = 1/24, dj = 1/20,
# #                              lowerPeriod = 1/4,
# #                              make.pval = TRUE, n.sim = 10,
# #                              date.format = "%F %T", date.tz = "")
