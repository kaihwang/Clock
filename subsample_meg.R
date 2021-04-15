library(data.table)

epoch <- Sys.getenv("epoch")
if (epoch=="") { epoch <- "feedback" }


subsample_dt <- function(dt, keys=key(dt), dfac=1L, method="subsamp") {
  checkmate::assert_data_table(dt)
  downsamp <- function(col, dfac=1L) { col[seq(1, length(col), dfac)] }

  if (method=="subsamp") {
    dt[, lapply(.SD, downsamp, dfac=dfac), by=keys]
  } else if (method=="mean") {
    dt[, chunk := rep(1:ceiling(.N/dfac), each=dfac, length.out=.N), by=keys]

    dt <- dt[, lapply(.SD, mean), by=c(keys, "chunk")] #compute mean of every k samples
    dt[, chunk := NULL]
  }

  return(dt)
}


library(foreach)
library(doParallel)
cl <- makeForkCluster(8)
registerDoParallel(cl)

datapath <- paste0("/proj/mnhallqlab/projects/Clock_MEG/tfr_rds/", epoch, "/original")
outputpath <- sub("/original", "/downsamp_20Hz_mean", datapath, fixed=TRUE)
if (!dir.exists(outputpath)) { dir.create(outputpath) }

megfiles <- list.files(path=datapath, pattern="MEG[0-9]+\\.rds", full.names=TRUE)

extant_files <- file.path(outputpath, sub(".rds", "_20Hz.rds", basename(megfiles), fixed=TRUE))
megfiles <- megfiles[!file.exists(extant_files)]

if (length(megfiles) == 0L) { quit(status=0, save="no") } #end gracefully

#for (mm in megfiles) {
ff <- foreach(mm = iter(megfiles), .packages=c("data.table")) %dopar% {
  xx <- readRDS(mm)
  dt <- xx$get() #rehydrate for visualization
  dt[, Channel:=NULL]
  dt[, Event:=NULL] #all feedback for now
  setkeyv(dt, c("Subject", "Run", "Trial"))
  setorderv(dt, c("Subject", "Run", "Trial", "Time")) #make sure we sort properly before subsampling
  rm(xx)

  #downsample by 12x -- 20.83 Hz
  dt_down <- subsample_dt(dt, dfac=12, method="mean")

  saveRDS(dt_down, file=file.path(datapath, "downsamp_20Hz_mean", sub(".rds", "_20Hz.rds", basename(mm), fixed=TRUE)))
  return(NULL)
}

stopCluster(cl)
