This repository contains code and datasets for paper in submission to AEIS 2024, "Hidden or Inferred: Fair Learning-To-Rank With Unknown Demographics"


## Datasets
The datasets used in the paper are available in the `Datasets` directory. The datasets are described in the paper.

## Code

Example steps to run the experiments in the paper are as follows:

# Rename settings file if needed
Manually rename `settings-<Dataset>.json` to `settings.json` where `<Dataset>` is the name of the dataset for which you are
running experiments.

# Clean the Dataset
Clean()

# Split the Dataset (into train and test
Split()

# Infer demographic information using the test split for case studies
BehindTheName()
NameSor()
GenderAPI()

# Simulate errors in inferred demographic information for controlled studies for 5 seeds
VariantSplit()

# Train the model using the train split. 
# The model is trained using the inferred demographic information
1. Train fairness unaware model with inferred demographic information
Set 'gamma' to 0.0 in 'settings.json'. Number of iterations per dataset is given below.
   (W)NBA: 1000
   Boston Marathon: 1500
   COMPAS: 1500
   LAW: 3000

Train()

2. Train fairness aware model
Set 'gamma' > 1 in 'settings.json'. See paper for details on setting parameters.
In our experiments, we use the following gamma values for the datasets:
   (W)NBA: 477500, number of iterations: 1000
   Boston Marathon: 1000000, number of iterations: 100
   COMPAS: 497000, number of iterations: 1500
   LAW: 72000, number of iterations: 3000

Train()

3. Train fairness unaware model without inferred demographic information
TrainBlind()

# Run full experiment (after train) for each simulation option (flip_choice) as described in paper.

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


# Calculate metrics for the full experiment
