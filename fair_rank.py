from HOIRank import *

start = time.time()

flip_choices = settings["DELTR_OPTIONS"]["flip_choices"]

seeds = settings["DELTR_OPTIONS"]["seeds"]


# Clean the Dataset
# Clean()

# Split the Dataset (into train and test
# Split()

# Infer demographic information using the test split for case studies
# BehindTheName()
# NameSor()
# GenderAPI()

# Simulate errors in inferred demographic information for controlled studies for 5 seeds
# for flip_choice in flip_choices:
#     for seed in seeds:
#         if flip_choice != "CaseStudies":
#             VariantSplit(flip_choice, seed)

# Train the models using the train split
# Train()
# TrainBlind()


def full_experiment(flip_choice):
    for seed in seeds:
        print(seed)
        print("VariantSplit()")
        if flip_choice != "CaseStudies":
            VariantSplit(flip_choice, seed)
        RankInferred("both")

    CalculateInitialMetrics(flip_choice)

    # Rank Ground Truth Datasets, Rank Inferred Datasets. It is important to use the order set
    print("RankGroundTruth()")
    RankGroundTruth(flip_choice)  # good

    # Rank Colorblind (Hidden)
    print("RankColorblind()")
    RankColorblind(flip_choice)  # good

    print("RankInferred()")
    RankInferred(flip_choice)  #good

    # DetConstSort Ranking
    print("DetConstSortHidden()")
    DetConstSortHidden(flip_choice) # good
    print("DetConstSortNotHidden()")
    DetConstSortNotHidden(flip_choice) # good
    print("DetConstSortBlind()")
    DetConstSortBlind(flip_choice) #  good

    # Calculating the Metrics
    print("CalculateResultsMetrics()")
    CalculateResultsMetrics(seed, flip_choice)


DetConstSortHidden("both")
DetConstSortNotHidden("both")
DetConstSortBlind("both")




# full_experiment(flip_choice)

# CollateNDCGandSkews()
# Make_Metric_Csvs()
#
# # after doing the experiments for all the datasets
# PlotGraphs()
# ParetoPlots()

# --------------------------------------
# optional
# --------------------------------------
# PlotLoss()
# PlotLossExposure()
# PlotListLoss()


# ----------------------------------------------------------------------------------------------------
########################################################################################################
########################################################################################################

# PlotGraphs()


# Clean the Dataset
# Clean()

# Split the Dataset
# Split()
# RankColorblind(flip_choice)
# DetConstSortHidden(flip_choice)
# DetConstSortNotHidden(flip_choice)
# CalculateInitialMetrics()
#
# # Infer demographic information using the test split
# BehindTheName()
# NameSor()
# GenderAPI()
#
# # Train the model using the train split
# #Train()
#
# # Rank Ground Truth Datasets, Rank Inferred Datasets. It is important to use the order set
# RankGroundTruth()
# RankColorblind()
# VariantSplit()
# RankInferred()
# #
# # # DetConstSort Ranking
# DetConstSortHidden()

# DetConstSortNotHidden()
#
# # Calculating the Metrics
# CalculateResultsMetrics()


# PlotLoss()
# PlotLossExposure()
# PlotListLoss()

# CombineResults()

# Make_Metric_Csvs()

# CalculateInitialMetrics("both")
# full_experiment("CaseStudies")
# for flip_choice in flip_choices:
#     print(flip_choice)
#     # CalculateInitialMetrics(flip_choice)
#     full_experiment(flip_choice)
# CollateNDCGandSkews()
# PlotGraphs(flip_choice)

# CollateNDCGandSkews()
# for flip_choice in flip_choices:
#     print(flip_choice)
#     full_experiment(flip_choice)
# CollateNDCGandSkews()
# Make_Metric_Csvs()
# PlotGraphs()

# full_experiment("both")
# CombineResults()

# Generate multiple results for each seed

# for seed in seeds:
# VariantSplit(seed)
# RankInferred()
# #
# # # DetConstSort Ranking
# DetConstSortHidden()
# DetConstSortNotHidden()
#
# # Calculating the Metrics
# CalculateResultsMetrics(seed, 'CaseStudies')
# Specify the directory where you want to start the renaming process
# directory_to_search = 'FairRank/Results'
#
# # Call the function to rename folders with '.csv' extension
# rename_folders_with_csv(directory_to_search)

end = time.time()

print("time taken = ", end - start)
