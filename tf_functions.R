# TF functions

compute_wavelet <- function(ts, time, delta_t = .004, dj = 1 / 4, maxfreq = NULL, minfreq = NULL, pad_tails = 1) {
    require(biwavelet)
    require(data.table)
    max.scale <- ifelse(is.null(maxfreq), NULL, 1 / minfreq)
    s0 <- ifelse(is.null(maxfreq), 2 * delta_t, 1 / maxfreq)

    if (pad_tails > 0) {
        npad <- ceiling(pad_tails / delta_t)
        time_before <- seq(min(time) - npad * delta_t, min(time) - delta_t, by = delta_t)
        signal_before <- rev(ts[1:npad])
        time_after <- seq(max(time) + delta_t, max(time) + npad * delta_t, by = delta_t)
        signal_after <- rev(ts[(length(ts) - npad + 1):length(ts)])
        time <- c(time_before, time, time_after)
        ts <- c(signal_before, ts, signal_after)
    }

    res <- wt(cbind(time, ts), dt = delta_t, s0 = s0, max.scale = max.scale, dj = dj, do.sig = FALSE, pad = TRUE)
    odf <- data.table(time = time, t(res$power))
    if (pad_tails > 0) {
        odf <- odf[(npad + 1):(nrow(odf) - npad), ]
    } # remove padding
    setnames(odf, c("Time", paste0("f_", sprintf("%06.3f", 1 / res$scale))))
    odf <- data.table::melt(odf, id.vars = "Time", variable.name = "Freq", value.name = "Pow")

    return(odf)
}

subsample_dt <- function(dt, keys = key(dt), dfac = 1L, method = "subsamp") {
    checkmate::assert_data_table(dt)
    downsamp <- function(col, dfac = 1L) {
        col[seq(1, length(col), dfac)]
    }

    if (method == "subsamp") {
        dt[, lapply(.SD, downsamp, dfac = dfac), by = keys]
    } else if (method == "mean") {
        dt[, chunk := rep(1:ceiling(.N / dfac), each = dfac, length.out = .N), by = keys]

        dt <- dt[, lapply(.SD, mean), by = c(keys, "chunk")] # compute mean of every k samples
        dt[, chunk := NULL]
    } else if (method == "time_mean") {
        # dt[, chunk:=ggplot2::cut_interval(Time, n = NULL, length = NULL, ...)]
    }

    return(dt)
}


timefreq_calc <- function(ff, filetype="sensor", downsamp = 12, ncpus = 4, tmin = -Inf, tmax = Inf) {
    require(data.table)
    df <- readRDS(ff)
    df <- df$get()

    df[, Event := NULL] # use directory structure to indicate event (cut down on object size)
    df <- df[Time > tmin & Time < tmax] # filter out times, if requested

    if (filetype=="sensor") {
        # df[, Channel:=NULL]
        df[, Channel := sub("MEG", "", Channel, fixed = TRUE)] # keep channel for Alex scripts
        df[, Signal := Signal * 1e10] # scale up to reasonable level
        setkeyv(df, c("Subject", "Channel", "Run", "Trial"))
        setorderv(df, c("Subject", "Channel", "Run", "Trial", "Time")) # make sure we sort properly before subsampling
    } else if (filetype=="source") {
        setnames(df, "source_est", "Signal") # use Signal for consistency
        df[, Signal := Signal * 1e2] # scale up to reasonable level
        df[, Run := ceiling(Trial/63)] # for consistency with sensor data
        setkeyv(df, c("Subject", "roinum", "Run", "Trial"))
        setorderv(df, c("Subject", "roinum", "Run", "Trial", "Time")) # make sure we sort properly before subsampling
    }


    # wavelet settings: .004s bins, 25
    delta_t <- .004
    freq <- 1 / delta_t
    minfreq <- 2 # Hz
    maxfreq <- 80 # Hz
    db_transform <- TRUE

    keys <- key(df)

    if (ncpus > 1) {
        cl <- makeCluster(ncpus)
        registerDoParallel(cl)
        on.exit(try(stopCluster(cl)))
    } else {
        registerDoSEQ()
    }

    # ram blows up if we go across all subjects
    # thus, split by subject and do the time-frequency + downsampling per subject
    if (filetype == "sensor") {
        splitdt <- split(df, df$Subject)
    } else {
        splitdt <- split(df, df$roinum) #source
    }
    rm(df) # garbage collect unsplit data

    ff <- foreach(
        thisdf = iter(splitdt), .noexport = "splitdt",
        .packages = c("biwavelet", "data.table"),
        .export = c("subsample_dt", "compute_wavelet")
    ) %dopar% {

        timefreq_dt <- thisdf[, .(lapply(.SD, function(dt) {
            compute_wavelet(
                ts = Signal, time = Time, minfreq = minfreq,
                maxfreq = maxfreq, delta_t = delta_t, pad_tails = 1
            )
        })), by = keys, .SDcols = "Signal"]

        # only needed if we return odf as a list in function (bad idea)
        # timefreq_dt <- timefreq_dt[, .(V1[[1]]), by=.(Subject, Run, Trial)]

        # unnest by selecting out data.table from V1 -- V1 is a 'list-column'
        timefreq_dt <- timefreq_dt[, V1[[1]], by = keys]

        if (isTRUE(db_transform)) {
            timefreq_dt[, Pow := 10 * log10(Pow + 1e-25)]
        } # don't save original scaling, too (storage issue)

        setkeyv(timefreq_dt, c(keys, "Freq"))
        setorderv(timefreq_dt, c(keys, "Freq", "Time")) # make sure we sort properly before subsampling

        ss <- subsample_dt(timefreq_dt, dfac = downsamp, method = "mean")
        return(ss)
    }

    return(rbindlist(ff))
}