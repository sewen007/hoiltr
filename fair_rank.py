from HOIRank import *

start = time.time()

flip_choices = settings["DELTR_OPTIONS"]["flip_choices"]

seeds = settings["DELTR_OPTIONS"]["seeds"]

# Define the source directory containing the files
source_directory = './HOIRank/Datasets/NBAWNBA/Ranked/both'


# Clean the Dataset
# Clean()

# Split the Dataset (into train and test)
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

    # CalculateInitialMetrics(flip_choice)
    #
    # # Rank Ground Truth Datasets, Rank Inferred Datasets. It is important to use the order set
    # print("RankGroundTruth()")
    # RankGroundTruth(flip_choice)
    #
    # # Rank Colorblind (Hidden)
    # print("RankColorblind()")
    # RankColorblind(flip_choice)
    #
    # print("RankInferred()")
    # RankInferred(flip_choice)
    #
    # # DetConstSort Ranking
    # print("DetConstSortHidden()")
    # DetConstSortHidden(flip_choice)
    # print("DetConstSortNotHidden()")
    # DetConstSortNotHidden(flip_choice)
    # # print("DetConstSortBlind()")
    # DetConstSortBlind(flip_choice)
    #
    # # Calculating the Metrics
    # print("CalculateResultsMetrics()")


# for flip_choice in flip_choices:
#     for seed in seeds:
#         CalculateResultsMetrics(flip_choice, seed)

# CollateNDCGandSkews()
# Make_Metric_Csvs()


# # after doing the experiments for all the datasets
# PlotGraphs()
# ParetoPlots()

# --------------------------------------
# optional
# --------------------------------------
# PlotLoss()
# PlotLossExposure()
# PlotListLoss()


end = time.time()

print("time taken = ", end - start)
