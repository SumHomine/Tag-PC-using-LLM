from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
import numpy as np
from pgmpy.readwrite import BIFWriter
import bnlearn as bn
import pandas as pd
import matplotlib.pyplot as plt
import os

#For Tu Server
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


def get_data_from_csv_int(path_to_file):
    # Read CSV file and extract node names from first line
    df = pd.read_csv(path_to_file, header=None)
    node_names = df.iloc[0].values
    # Remove first row and get data as nd.array
    data = df.iloc[1:].values

    data = data.astype(int)  # Convert all columns to int as failsafe

    print(node_names)
    print(data)

    return data, node_names

# get datat that is stored in strings in csv file
def get_data_from_csv_string(path_to_folder, bnfdata = "insurance"):
    # Read CSV file and extract node names from first line
    path_to_file = os.path.join(path_to_folder, (bnfdata + "_data.csv"))
    df = pd.read_csv(path_to_file, header=None)
    node_names = df.iloc[0].values

    # Remove first row
    df = df.iloc[1:].copy()
    df.columns = node_names  # Ensure columns have the correct names
    # use onehot encoding to get string data as numbers 
    df_hot, df_num = bn.df2onehot(df = df) 
    data = df_num.values # get data as nd.array
    data = data.astype(int) # Convert all columns to int as failsafe

    print(node_names)
    print(data)

    return data, node_names


# alternativly get data via bif file from bnlearn
def get_data_from_bnf(bnfdata="asia"):
    data = bn.import_example(bnfdata) 

    # Extract values and node names
    node_names = data.columns.tolist()
    data = data.values

    print(node_names)
    print(data)

    return data, node_names

#get data from csv:
#forest:
path_forest = "/home/ml-stud19/Tag-PC-using-LLM/generated_forestdata.csv"
# data, node_names = get_data_from_csv_int(path_to_file=path_forest)
#bnlearn:
path_bnlearn = "/home/ml-stud19/Tag-PC-using-LLM/additionalData" 
bnfdata = "asia"  # change depending on example
# data, node_names = get_data_from_csv_string(path_to_folder=path_bnlearn,bnfdata=bnfdata)


# or bnf example
bnfdata = "asia" # use here the example you wnt to use
data, node_names = get_data_from_bnf(bnfdata)

# save_bnf_true_graph(bnfdata) # if necessary and not on website https://www.bnlearn.com/bnrepository/ get the true graph


alpha = 0.05 #significance level - default 0.05
indep_test = "fisherz" #default is fisherz, gibt fisherz, chisq, gsq, mv_fisherz, kci (killed PC!), fisherz seems to wield similar results as mv_fisherz, chisq seems to wield smiliar results as gsq, chisq seems to work the best
uc_rule = 0 #how unshielded colliders are oriented - default 0 
uc_priority = 2 #default 2 (prio existing colliders), 0 (overwrite) interessant aber oft zyklisch, 3 (prioritize stronger colliders) auch interessant
mvpc = False #use missing value PC - default is false
verbose = False #shows more infos - default is false
cg = pc(data, alpha=alpha, indep_test=indep_test, uc_rule=uc_rule, uc_priority=uc_priority, mvpc=mvpc, verbose=verbose)

# save graph
fname = "dag_" + indep_test + str(alpha) + ".png"
dir = "Tag-PC-using-LLM/normal-PC/figures" #change depending on data
pyd = GraphUtils.to_pydot(cg.G, labels=node_names)
pyd.write_png(os.path.join(dir,fname))


# for saving true graphs from bnf
def save_bnf_true_graph(bnfdata="asia"):
   # Import DAG
    model = bn.import_DAG(bnfdata)
    dir_default = "/home/ml-stud19/Tag-PC-using-LLM/normal-PC/figures"
    dir = os.path.join(dir_default, f"{bnfdata}_true_graph.png")
    # Plot DAG
    fig, ax = plt.subplots(figsize=(15, 10))  # Create figure and axes
    bn.plot(model, params_static={'figsize': (15, 10)}, verbose=0)   
    # Save plot as PNG file
    plt.savefig(dir, format='png')
    plt.close(fig)  # Close figure to free up memory

