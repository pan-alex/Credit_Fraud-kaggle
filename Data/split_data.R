# To reduce the size of the individual data file, I'm adding this pre-processing
# step to split the data file into two.

library(tidyverse)

credit <- read_csv('creditcard.csv')

n <- dim(credit)[1]
split <- as.integer(n / 2)

credit_1 <- credit[1:split, ]
credit_2 <- credit[(split+1):n, ]

write_csv(credit_1, 'creditcard_1.csv')
write_csv(credit_2, 'creditcard_2.csv')
