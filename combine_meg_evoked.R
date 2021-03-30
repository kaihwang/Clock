library(pacman)
p_load(checkmate, data.table, R6)
source("/proj/mnhallqlab/users/michael/fmri.pipeline/R/rle_dt.R") #sparse storage class

channels <- list.dirs("../csv_data", recursive=FALSE)

for (cc in channels) {
  cat("Working on channel_dir:", cc, "\n")
  ch_files <- list.files(path=cc, pattern="\\.csv", recursive=TRUE, full.names=TRUE)
  ch_data <- sapply(ch_files, function(ff) {
    dt <- data.table::fread(ff)
    dt[, V1 := NULL] #dummy row number
    return(dt)
  }, simplify=FALSE, USE.NAMES=TRUE)

  big_dt <- rbindlist(ch_data) #, idcol="file_name")
  sparse_dt <- rle_dt$new(big_dt, keys=c("Channel", "Event", "Subject", "Run", "Trial", "Time"))
  saveRDS(sparse_dt, file=paste0("../r_channel_combined/", big_dt$Channel[1], ".rds"))
  rm(ch_data, big_dt, sparse_dt) #encourage R to garbage clean
}
