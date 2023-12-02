setwd("/home/camilo/arena/momics/data/generated")

library(dplyr)
library(readr)
library(DataExplorer)

dna <- read.csv("DNA", header=FALSE)
rna <- read.csv("Expression", header=FALSE)
meth <- read.csv("Methylation", header=FALSE)
mut <- read.csv("Mutation", header=FALSE)

colnames(dna) <- paste0("dna",colnames(dna))
colnames(rna) <- paste0("rna",colnames(rna))
colnames(meth) <- paste0("meth",colnames(meth))
colnames(mut) <- paste0("mut",colnames(mut))


data_list <- list(dna, rna, meth, mut)
plot_str(data_list)

final_data <- cbind(dna, rna, meth, mut)
introduce(final_data)
plot_intro(final_data)
plot_missing(final_data)

plot_histogram(dna[,1:10])
plot_histogram(rna[,1:10])
plot_histogram(meth[,1:10])
#plot_histogram(mut[,1:10])


qq_data <- final_data[, c("dnaV1", "rnaV1", "methV1", "mutV1")]
plot_qq(qq_data, sampled_rows = 1000L)

plot_correlation(na.omit(qq_data), maxcat = 5L)

plot_boxplot(qq_data, by = "dnaV1")



# read csv file
# df <- 
#   read_delim(
#     file = "/data/cardio_train.csv",
#     col_types = "iifidiiffffff",
#     delim=";")


# pre-processing
df <- 
  # remove the id
  select(df, -id) %>%
  # age: days -> years
  mutate(age = round(age / 365))


# observe first rows
head(df)


library(gtools)
?rdirichlet
dirich
rdirichlet(rgamma(5, 2, 1), rep(1,5)/5)

plot(sort(rgamma(100, 2, 1)))

      