from scipy.stats import rankdata
import numpy as np
import pandas as pd

data_shd = {
        "Dataset": ["Sprinkler", "Cancer", "Asia", "Sachs", "Child", "Insurance", "Alarm", "Barley", "Hepar2", "Win95PTS", "Andes"],
        "Standard PC": [2, 0, 3, 17, 12, 18, 4, 9, 9, 12, 10],
        "Typed PC (generic Prompt)": [2, 0, 3, 17, 8, 18, 9, 42, 7, 39, 97],
        "Tag PC - Tag Majority (generic Prompt)": [2, 0, 1, 17, 4, 0, 6, 9, 8, 7, 9],
        "Tag PC - Tag Weighted (generic Prompt)": [2, 0, 8, 17, 4, 4, 2, 9, 10, 13, 6],
        "Tag PC - Tag Majority (domain Prompt)": [None, None, 6, 7, None, 2, None, 12, None, 12, 9],
        "Tag PC - Tag Weighted (domain Prompt)": [None, None, 2, 13, None, 3, None, 8, None, 12, 9]
}

data_sid = {
    "Dataset": ["Sprinkler", "Cancer", "Asia", "Sachs", "Child", "Insurance", "Alarm", "Barley"],
    "Standard PC": [9, 0, 18, 62, 228, 275, 46, 237],
    "Typed PC (generic Prompt)": [9, 0, 18, 62, 104, 233, 125, 972],
    "Tag PC - Tag Majority (generic Prompt)": [5, 0, 5, 62, 44, 0, 67, 237],
    "Tag PC - Tag Weighted (generic Prompt)": [5, 0, 28, 62, 24, 62, 27, 237],
    "Tag PC - Tag Majority (domain Prompt)": [None, None, 18, 24, None, 42, None, 205],
    "Tag PC - Tag Weighted (domain Prompt)": [None, None, 7, 39, None, 62, None, 89]
}


# convert to DataFrames for easier processing
df_shd = pd.DataFrame(data_shd).set_index("Dataset")
df_sid = pd.DataFrame(data_sid).set_index("Dataset")

print(df_shd)

ranked_shd = df_shd.T.apply(lambda row: rankdata(row, method='average', nan_policy="omit"), axis=0).T
ranked_sid = df_sid.T.apply(lambda row: rankdata(row, method='average', nan_policy="omit"), axis=0).T

print(ranked_shd)
print(ranked_sid)

# calculate average percentage rank
average_rank_shd = ranked_shd.mean()
average_rank_sid = ranked_sid.mean()

# output results
print("Average Percentage Ranks for SHD (using 'average' method):")
print(average_rank_shd)

print("\nAverage Percentage Ranks for SID (using 'average' method):")
print(average_rank_sid)

