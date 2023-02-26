library(tidyr)
library(ggplot2)
library(dplyr)
library(ggthemes)
library(openxlsx)
library(scales)
library(httpgd)

# setup data
A <- read.xlsx('processed_data hand-corrected.xlsx', sheet='processed_data') %>%
  select(A_gender, A_meet_again_bool, A_marks_out_of_10_float, date) %>%
  transmute(gender = A_gender, meet_again = A_meet_again_bool, marks = A_marks_out_of_10_float, date=date)
B <- read.xlsx('processed_data hand-corrected.xlsx', sheet='processed_data') %>%
  select(B_gender, B_meet_again_bool, B_marks_out_of_10_float, date) %>%
  transmute(gender = B_gender, meet_again = B_meet_again_bool, marks = B_marks_out_of_10_float, date=date)
data <- rbind(A, B)

data %>%
  filter(marks < 10) %>%
  ggplot(aes(marks)) +
  geom_histogram(aes(fill=gender), bins=10) +
  scale_x_continuous(breaks=seq(1, 10, 1))

y_data <- data %>%
  filter(meet_again=='y')
hist(y_data$marks, prob=TRUE)
n_data <- data %>%
  filter(meet_again=='n')
hist(n_data$marks, prob=TRUE)

hist(data$marks, prob=TRUE)

pairs <- A %>%
    select(marks, meet_again, date) %>%
    rename(A_marks = marks, A_meet_again = meet_again) %>%
    left_join(B %>%
                select(marks, meet_again, date) %>%
                rename(B_marks = marks, B_meet_again = meet_again),
              by='date') %>%
    select(-date)

pairs %>%
  ggplot(aes(A_marks, B_marks)) +
    geom_point() +
    geom_smooth(method='lm')
cor.test(pairs$A_marks, pairs$B_marks, method='spearman')