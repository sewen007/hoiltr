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


def full_experiment(flip_choice, seed):
    # run experiments for all inference choices except case studies
    if flip_choice != "CaseStudies":
        # Rank Ground Truth Datasets, Rank Inferred Datasets. It is important to use the order set here
        VariantSplit(flip_choice, seed)
        RankGroundTruth(flip_choice, seed)
        RankColorblind(flip_choice, seed)
        RankInferred(flip_choice, seed)
        # DetConstSort Ranking
        print("DetConstSortHidden()")
        DetConstSortHidden(flip_choice, seed)
        print("DetConstSortNotHidden()")
        DetConstSortNotHidden(flip_choice, seed)
        print("DetConstSortBlind()")
        DetConstSortBlind(flip_choice, seed)
        CalculateResultsMetrics(flip_choice, seed)


# for flip_choice in flip_choices:
#     print(flip_choice)
#     for seed in seeds:
#         print(seed)
#         full_experiment(flip_choice, seed)


# # #
#
# if "CaseStudies" in flip_choices:
#     RankGroundTruth("CaseStudies")
#     RankColorblind("CaseStudies")
#     RankInferred("CaseStudies")
#     DetConstSortHidden("CaseStudies")
#     DetConstSortNotHidden("CaseStudies")
#     DetConstSortBlind("CaseStudies")
#     CalculateResultsMetrics("CaseStudies")
# # #
# #
# CollateNDCGandSkews()
# Make_Metric_Csvs()

# # after doing the experiments for all the datasets
#
PlotGraphs()
# # ParetoPlots()


# --------------------------------------
# optional after training models
# --------------------------------------
# PlotLoss()
# PlotLossExposure()
# PlotListLoss()


end = time.time()

print("time taken = ", end - start)
