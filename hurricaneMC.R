# Importing data
rm(list = ls())
library(tidyverse)

setwd("/Users/jasonluo/Documents/Hurricane_proj")

h1 <- read.csv("finalPreprocData/final_dataset1.csv")
h2 <- read.csv("finalPreprocData/final_dataset2.csv")
h3 <- read.csv("finalPreprocData/final_dataset3.csv")
h4 <- read.csv("finalPreprocData/final_dataset4.csv")
h5 <- read.csv("finalPreprocData/final_dataset5.csv")
h6 <- read.csv("finalPreprocData/final_dataset6.csv")
h7 <- read.csv("finalPreprocData/final_dataset7.csv")


set.seed(580)
timesteps <- 50 # 50 timesteps into the future
simulations <- 10
windspeeds <- h3$windspeed


sim_data <- c()
for (i in 1:simulations) {
  sim_list <- c()
  
  for (j in 1:timesteps) {
    simulated_obs <- sample(size = 1, x = windspeeds, replace = TRUE)
    sim_list <- c(sim_list, simulated_obs)
    windspeeds <- c(windspeeds, simulated_obs)
    #MEAN <- sum(windspeeds) / length(windspeeds) + 1
    #STD <- sum(windspeeds - MEAN)^2 / length(windspeeds) - 1
  }

  sim_data <- rbind(sim_data, sim_list)
  
}

sim_data <- as.data.frame(sim_data)
colnames(sim_data) <- c((nrow(h1)+1):(nrow(h1)  + timesteps))
sim_data$sim_num <- as.factor(c(1:nrow(sim_data)))
sim_data <- pivot_longer(sim_data, cols = -sim_num,
                         names_to = "time",
                         values_to = "windspeed")

# Original time series
ggplot() + 
  geom_line(data = h1, aes(x = c(1:nrow(h1)), y = windspeed)) 

# Multiple simulated time series 50 time steps into the future
ggplot() + 
  geom_line(data = sim_data, aes(x = time, 
                                 y = windspeed, 
                                 group = sim_num, 
                                 color = sim_num,
                                 alpha = 0.4)) + 
  theme(legend.position = "none")







