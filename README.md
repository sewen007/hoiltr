
The FairRank Python package is used to run Fair ranking machine learning experiments. The main in-processing algorithm used is DELTR and the post-processing algorithm included is DetConsSort. These can be used together to compare rankings produced by either or both of these methods. The goal of these ranking algorithms are to produce a fair ranking, balancing both utility and fairness dependant on a protected attribute (i.e. gender, income, race, etc).


The FairRank Python package is used to run Fair ranking machine learning experiments. The main in-processing algorithm used is DELTR and the post-processing algorithm included is DetConsSort. These can be used together to compare rankings produced by either or both of these methods. The goal of these ranking algorithms are to produce a fair ranking, balancing both utility and fairness dependant on a protected attribute (i.e. gender, income, race, etc).

## Running the Experiment Example ##

```
from FairRank import *

# Clean the Dataset
Clean()

# Split the Dataset
Split()

# Infer demographic information using the test split
BehindTheName()
NameSor()
GenderAPI()

# Train the model using the train split
Train()

# Rank Ground Truth Datasets, Rank Inferred Datasets
RankGroundTruth()
RankInferred()

# DetConstSort Ranking
DetConstSort()

# Calculating the Metrics
CalculateResultsMetrics()
```

## settings.json Example ##

```
{
  "GENDER_DATA_DEFINE": {
    "F" : "1",
    "f" : "1",
    "FEMALE" : "1",
    "Female" : "1",
    "female" : "1",
    "mostly_female" : "1",
    "M" : "0",
    "m" : "0",
    "MALE" : "0",
    "Male" : "0",
    "male": "0",
    "mostly_male" : "0",
    "Default" : "0"
  },
  "READ_FILE_SETTINGS": {
    "PATH": <String: filepath to dataset used in the experiment>,
    "GENDER_EXPERIMENT": <String: True/False>,
    "RACE_EXPERIMENT" : <String: True/False>,
    "DEMO_COL" : <String: name of demographic column in the dataset>,
    "ADDITIONAL_COLUMNS" : <Array: string values of columns to be included>
  },
  "DATA_SPLIT" : {
    "TRAIN_PCT" : <Double: percent to split the full dataset into train/test ex. 0.8 results in 80%  used for testing>
  },
  "INFERENCE_METHODS" : {
    "INFER_COL" : <String: colomn used for inference>,
    "BTN" : {
      "API_KEY" : [] <StringArray: API key(s) from Behind the Name>,
      "URL" : <String: URL to the inference method>
    },
    "NAPI" : {
      "API_KEY" : <String: API key from Name API>,
      "URL" : <String: URL to the inference method>
    },
    "NMSOR" : {
      "API_KEY" : <String: API key from NameSor>,
      "URL" : <String: URL to the inference method>
    },
    "GAPI" : {
      "API_KEY" : <String: API key from GenderAPI>,
      "URL" : <String: URL to the inference method>
    }
  },
  "DELTR_OPTIONS" : {
    "gamma" : <Double: gamma value to indicate whether to train DELTR on fairness or utility>,
    "num_iterations" : <Int: number of iterations when training a DELTR model>,
    "standardize" : <Boolean: whether to standardize the dataset when training the DELTR model>,
    "SCORE_COLUMN" : <String: column in the dataset that will be used as the score for training DELTR>,
    "NORMALIZE_SCORE_COLUMN" : <Boolean: whether to standardize the score column before training DELTR>
  }
}
```

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






