library(pacman)
p_load(checkmate, data.table, R6, doParallel)
source("/proj/mnhallqlab/users/michael/fmri.pipeline/R/rle_dt.R") #sparse storage class

epoch <- Sys.getenv("epoch")
if (epoch == "") {
    epoch <- "RT"
}

basedir <- "/proj/mnhallqlab/projects/Clock_MEG/fif_data/csv_data"

inputdir <- file.path(basedir, epoch)
outputdir <- file.path(basedir, paste0(epoch, "_rds"))

stopifnot(dir.exists(inputdir))
if (!dir.exists(outputdir)) { dir.create(outputdir, recursive=TRUE) }

source_files <- list.files(pattern=".*source_ts.csv", inputdir, recursive=FALSE, full.names=TRUE)
subids <- sub(".*/(\\d{5})_.*", "\\1", source_files, perl = TRUE)
cl <- makeForkCluster(8)
registerDoParallel(cl)

extant_files <- sub(".rds", "", list.files(outputdir, pattern=".*\\.rds", recursive=FALSE), fixed=TRUE)
source_files <- source_files[!subids %in% extant_files] # don't re-run old files

ff <- foreach(cc=seq_along(source_files), .inorder = FALSE, .packages = "data.table") %dopar% {
  cat("Working on subject:", cc, "\n")
  dat <- data.table::fread(source_files[cc])
  dat[, V1 := NULL]
  setnames(dat, sub("([0-9]+)\\.\\d+", "\\1", names(dat)))
  setnames(dat, c("time", "trial_num"), c("Time", "Trial"))
  dat <- data.table::melt(dat,
    id.vars = c("Time", "Trial"),
    variable.name = "roinum", value.name = "source_est"
  )

  dat[, Subject := subids[cc]]
  dat[, Event := epoch]

  sparse_dt <- rle_dt$new(dat, keys = c("Subject", "Event", "roinum", "Trial", "Time"))
  rm(dat)
  saveRDS(sparse_dt, file = file.path(outputdir, paste0(subids[cc], "_source.rds")))
  rm(sparse_dt) #encourage R to garbage clean
  return(NULL)
}

stopCluster(cl)
