This repository contains code and datasets for paper in accepted in AIES 2024, "Hidden or Inferred: Fair Learning-To-Rank With Unknown Demographics"

## Datasets ##
The datasets used in the paper are available in the `Datasets` directory. The datasets are described in the paper.

## Code ##

Run all code in fair_rank.py
Example steps to run the experiments in the paper are as follows: NOTE: Current settings file will run experiment for LAW dataset

## Step 1 
Rename settings file if needed
Manually rename `settings-<Dataset>.json` to `settings.json` where `<Dataset>` is the name of the dataset for which you are
running experiments. The current settings file is for the (W)NBA dataset.

## Step 2 (optional)
Clean the Dataset (You can skip this step if you are using the same datasets already in the repository)

    Clean()

## Step 3 (optional)
Split the Dataset (You can skip this step if you are using the same datasets already in the repository)

    Split()

## Step 4 (optional)
Infer demographic information using the test split for case studies. (You can skip this step if you are using the same datasets already in the repository)
Refer to license agreements for the APIs used. API keys are required.

Add keys to settings.json. You may use multiple APIs for the BTN API(comma separated)
    
    BehindTheName()
    NameSor()
    GenderAPI()

## Step 5 (important)
Train the model using the train split. (You can skip this step if you are using the same datasets already in the repository)
The model is trained using the inferred demographic information
1. Train fairness unaware model with inferred demographic information
Set 'gamma' to 0.0 in 'settings.json'. Number of iterations per dataset is given below.
   (W)NBA: 1000
   Boston Marathon: 1500
   COMPAS: 1500
   LAW: 3000

    Train()

2. Train fairness aware model (You can skip this step if you are using the same datasets already in the repository)

Set 'gamma' > 1 in 'settings.json'. See paper for details on setting parameters.
In our experiments, we use the following gamma values for the datasets:
   (W)NBA: 477500, number of iterations: 1000
   Boston Marathon: 1000000, number of iterations: 100
   COMPAS: 497000, number of iterations: 1500
   LAW: 72000, number of iterations: 3000

    Train()

3. Train fairness unaware model without inferred demographic information (You can skip this step if you are using the same datasets already in the repository)
    
    TrainBlind()

## Step 6 (Can be included in the full experiment)
Simulate errors in inferred demographic information for controlled studies for 5 seeds

    for flip_choice in flip_choices:
            for seed in seeds:
                if flip_choice != "CaseStudies":
                    VariantSplit(flip_choice, seed)


## Step 7
Run full experiment (after train) for each simulation option (flip_choice) as described in paper.
    
    full_experiment() 


This runs the following each flip_choice option as described in the paper:

    # Calculate Test Data metrics before ranking (testing)
    CalculateInitialMetrics(flip_choice)
    
    # Rank Ground Truth
    RankGroundTruth(flip_choice)
    
    # Rank with hidden demographic information
    RankColorblind(flip_choice)
    
    # Rank with inferred demographic information
    RankInferred(flip_choice)
    
    # Re-rank with DetConstSort
    DetConstSortHidden(flip_choice)
    DetConstSortNotHidden(flip_choice)
    DetConstSortBlind(flip_choice)


## Final Steps 
(after running full experiment for all "flip choice" options)
Calculate metrics for the full experiment

Calculate metrics for the full experiment

    CalculateResultsMetrics()
    CollateNDCGandSkews()
    Make_Metric_Csvs()


after running the experiments for all the datasets

    PlotGraphs()
    ParetoPlots()


## optional 
These plot the loss graphs for the training stages

    PlotLoss()
    PlotLossExposure()
    PlotListLoss()


All ranked files and graphs have been added to the repository. Training models were however not included in the repository due to size constraints.

More details on the settings file are below:
## Gender Data Define ##
This is used
so that the values on the left side that may appear in the data sets of the experiment
can be replaced with either a 1 or a 0. This is done for the LTR model that we are using
that only accepts 1’s and 0’s for the protected attribute column. A value of 1 indicates
that a group is protected and a value of 0 indicates that a group is non-protected. In
the WNBA/NBA experiment, females are the protected group and males are the non-
protected group.

## Read File Settings ##

This portion of
the settings consists of the path which is the path to the file that you want to run the
experiment on. This file should be placed within the FairRank package. Then you must
specify whether it is a gender experiment or a race experiment by setting one of them True
and the other to False. The SCORE COL is is the column name of the scoring feature or
what the LTR model is going to learn. 

## Data Split ##
This portion of the settings describes how the testing and training will be split. The TRAIN_PCT
accepts any values between 0.0 where 0% of the original data will be split to the training
data and 1.0 where 100% of the original data will be split to the training data. A value of 0.8
means that 80% of the original data will be split to the training data and 20%
will be split to the testing data.

## Inference Methods ##
This portion of the settings is used by the inference algorithms in the experiment.
INFER_COL is the column name in the original data set that the inference algorithms can
use to predict the necessary demographic information. In the case of the WNBA/NBA
experiment the column name is ”PlayerName”. The three different inference algorithms,
Behind The Name (BTN), NameSor (NMSOR), and GenderAPI (GAPI) have the same
two essential pieces. The API_KEY value is your own individual API_KEY that can be
used to make the inference requests to the website. In the case of Behind the Name
you need at least two API_KEYS for the code to be functional. The URL, unlike the
API_KEYS, should not be touched

## DELTR Options ##
This portion of the settings configured the DELTR options. Gamma can be any value greater
than 0.0 where 0.0 is training a fairness-unaware LTR model and any value higher is a
fairness-aware LTR model. The ”num iterations” and ”standardize” are both values that
DELTR requires. SCORE_COLUMN is the column name in the original data set that the
LTR model is trying to learn. Finally NORMALIZE_SCORE_COLUMN is a value that is either
True or False indicating whether or not you want all the values in the scoring column to
be normalized to a value between 0 and 1

