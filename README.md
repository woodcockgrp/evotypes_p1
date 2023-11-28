# evotypes_p1

Code used in the manuscript "Genomic evolution shapes prostate cancer disease type"

## Installation

No istallation is required.  Code consists of Matlab (Feature extraction) and R (Ordering) scripts that can be run from the command line in the respective pacakges.  

## Usage

Feature extraction:

Start with the extract_features.m file - this performs feature extraction with rbm_feature_learning.m and saves the input/feature map in the variable amalgamated_weights.

Once the weight matrix is established, this is used as an input into rbm_feature_scores.m, which itself outputs the feaature representation of the data (feature_representation) and the feature scores (class_weights)


Ordering: 

Start with the ordering_real_data_script.R.  This inputs and processes the CCF data, calcualtes the BIC scores with the multi_tree_bic function (output to bic_mat.txt), and then calculates the worth values for each ordering profile in multi_sample_pl function (output to pl_ordering_X.txt, where X is the ordering number).


## Contributions

Code written by D J Woodcock.  R Tesloiuanu contributed to ideas in the Ordering package.  Project supervised by D C Wedge.

https://doi.org/10.5281/zenodo.10214666
