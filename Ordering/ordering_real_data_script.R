# set working directorty if necessary
source('multi_tree_bic.R')
source('multi_sample_pl.R')

# Read in CNA data
raw.ccf.data <- read.csv('./159pat_feat_cnas_public.csv', stringsAsFactors = FALSE, header = TRUE)

# preprocess the UK data  - data set specific
ccf.data <- raw.ccf.data[,c(2:26,28)]       # select columns to be used in ordering
rownames(ccf.data) <- raw.ccf.data$Sample

classes <- multi_tree_bic(ccf.data)       # calculate the BIC scores
multi_sample_pl(ccf.data, classes)    # calculate the Plackett-Luce values for each ordering profile



