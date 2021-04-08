library(pacman)
p_load(checkmate, data.table, R6, doParallel)
source("/proj/mnhallqlab/users/michael/fmri.pipeline/R/rle_dt.R") #sparse storage class

channels <- list.dirs("../csv_data_update", recursive=FALSE)
chnames <- sub(".*/ch_(MEG\\d{4})", "\\1", channels, perl=TRUE)
cl <- makeForkCluster(8)
registerDoParallel(cl)

extant_files <- sub(".rds", "", list.files("../r_channel_combined", pattern=".*\\.rds", recursive=FALSE), fixed=TRUE)
channels <- channels[!chnames %in% extant_files] #don't re-run old files

ff <- foreach(cc=channels, .inorder=FALSE, .packages="data.table") %dopar% {
  cat("Working on channel_dir:", cc, "\n")
  ch_files <- list.files(path=cc, pattern="\\.csv", recursive=TRUE, full.names=TRUE)
  ch_data <- sapply(ch_files, function(ff) {
    dt <- data.table::fread(ff)
    dt[, V1 := NULL] #dummy row number
    return(dt)
  }, simplify=FALSE, USE.NAMES=TRUE)

  big_dt <- rbindlist(ch_data) #, idcol="file_name")
  chnum <- big_dt$Channel[1]
  rm(ch_data)
  sparse_dt <- rle_dt$new(big_dt, keys=c("Channel", "Event", "Subject", "Run", "Trial", "Time"))
  rm(big_dt)
  saveRDS(sparse_dt, file=paste0("../r_channel_combined/", chnum, ".rds"))
  rm(sparse_dt) #encourage R to garbage clean
  return(NULL)
}

stopCluster(cl)
