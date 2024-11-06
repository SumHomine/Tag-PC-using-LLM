import numpy as np
import bnlearn as bn
import pandas as pd
import os
from tpc import tpc
from tpc_utils import get_typelist_from_text
from visualization import create_graph_viz

#For Tu Server
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# get dataset that is stored in int in csv file
def get_data_from_csv_int(path_to_file):
    # Read CSV file and extract node names from first line
    df = pd.read_csv(path_to_file, header=None)
    node_names = df.iloc[0].values
    # Remove first row and get data as nd.array
    data = df.iloc[1:].values
    data = data.astype(int)  # Convert all columns to int for forest

    #TODO better way to get types
    if (path_to_file == "/home/ml-stud19/tagged-pc-using-LLM/generated_forestdata.csv"):
        #TODO types besser übergeben
        # MAKE SURE THAT NODES ARE IN THE CORRECT ORDER
        types = """
            A : Producer
            R : 1st Consumer
            S : 1st Consumer
            H : 2nd Consumer
            B : 2nd Consumer
            W : 3rd Consumer
            F : 3rd Consumer
            """

    print(data)


    return data, node_names, types

# get dataset that is stored in strings in csv file
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

    #TODO better way to get types
    match bnfdata:
        case "insurance":
            # like constintou et. al: we assigned types to variables by randomly partitioning the topological ordering of the DAGs into groups
            # Types from Typing Assumption Paper
            types = """
            GoodStudent : First
            Age : First
            SocioEcon : First
            RiskAversion : First
            VehicleYear : Second
            ThisCarDam : Fourth
            RuggedAuto : Third
            Accident : Fourth
            MakeModel : Second
            DrivQuality : Fourth
            Mileage : First
            Antilock : Third
            DrivingSkill : Third
            SeniorTrain : Second
            ThisCarCost : Fifth
            Theft : Fourth
            CarValue : Third
            HomeBase : Sixth
            AntiTheft : Second
            PropCost : Fifth
            OtherCarCost : Fourth
            OtherCar : First
            MedCost : Fourth
            Cushioning : Fourth
            Airbag : Third
            ILiCost : Fourth
            DrivHist : Fourth
            """

            # alternatively Types assigned by Human Expert (me)
        case "asia":
            # like constintou et. al: we assigned types to variables by randomly partitioning the topological ordering of the DAGs into groups
            types = """
            asia : first
            tub : first
            smoke : second
            lung : second
            bronc : third
            either : third
            xray : fourth
            dysp : fourth
            """

    print(data)

    return data, node_names, types


# alternativly get data from bnlearn example
def get_data_from_bnf(bnfdata="asia"):
    data = bn.import_example(bnfdata)
    # get corrosponding types
    match bnfdata:
        case "asia":
            # like constintou et. al: we assigned types to variables by randomly partitioning the topological ordering of the DAGs into groups
            types = """
            asia : first
            tub : first
            smoke : second
            lung : second
            bronc : third
            either : third
            xray : fourth
            dysp : fourth
            """

        case "sprinkler":
            types_Weather = """
            Cloudy : Weather
            Sprinkler : Object
            Rain : Weather
            Wet_Grass : Object    
            """
            types_intuitiv = """
            Cloudy : Weather
            Sprinkler : Object
            Rain : Weather
            Wet_Grass : Plant_Con    
            """
   
            types = types_Weather
   
    # Extract values and node names
    node_names = data.columns.tolist()
    data = data.values

    return data, node_names, types

#get data from csv:
#forest:
path_forest = "/home/ml-stud19/tagged-pc-using-LLM/generated_forestdata.csv"
bnfdata = "forest" #for naming faile
# data, node_names, types = get_data_from_csv_int(path_to_file=path_forest)
#bnlearn:
path_bnlearn = "/home/ml-stud19/tagged-pc-using-LLM/additionalData" 
bnfdata = "insurance"  # change depending on example
data, node_names, types = get_data_from_csv_string(path_to_folder=path_bnlearn,bnfdata=bnfdata)

# or bnf example -> not deterministic
# bnfdata = "asia" # use here the example you wnt to use
# data, node_names, types = get_data_from_bnf(bnfdata)



alpha = 0.05
indep_test = "fisherz"
majority_rule = False
dag, stat_tests, typelist = tpc(data=data, types=types, node_names=node_names, alpha=alpha, indep_test=indep_test, majority_rule=majority_rule) #kci gives good results but takes alot of cumputing power

#TODO comment out
print(dag)
print(stat_tests)

dir = "/home/ml-stud19/tagged-pc-using-LLM/typed-PC"
fname = "tdag_" + bnfdata + "_" + indep_test + str(alpha) + ("majority" if majority_rule else "naive") + "0"
create_graph_viz(dag=dag, var_names=node_names, types=typelist, save_to_dir=dir, fname=fname)


