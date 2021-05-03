library(tidyverse)
library(devtools)
library(ggforce)

path = "/proj/mnhallqlab/projects/Clock_MEG/r_channel_combined/downsamp_20Hz_mean/"
setwd(dir="/proj/mnhallqlab/projects/Clock_MEG/r_channel_combined/downsamp_20Hz_mean/")
file.names <- dir(path)

for(i in 1:length(file.names)){
  data <- tryCatch(readRDS(file.names[i]), error=function(e) {
    message("Couldn't read :", file.names[i]); return(NULL)
  })

  if (is.null(data)) { next }

  summary =
    data %>%
    group_by(Subject, Run, Time) %>%
    summarize(response=mean(Signal, na.rm=TRUE), resp_sd=sd(Signal, na.rm=TRUE))
  
  pdf(file.names[i], "_pdf.pdf")
  
  for(j in 1:length(unique(data$Subject))){
    print(ggplot(data = summary) +
            geom_point(aes(x=Time, y=response)) +
            geom_ribbon(aes(ymin=response-resp_sd, ymax=response+resp_sd, x=Time), color="blue", alpha=0.5) +
            geom_line(aes(x=Time, y=response)) + geom_vline(xintercept = 0, color = "red") +
            facet_wrap_paginate(~Subject + Run, nrow=4, ncol=2, scales="free_y", page=j))
  }
  
  dev.off()
  
}
