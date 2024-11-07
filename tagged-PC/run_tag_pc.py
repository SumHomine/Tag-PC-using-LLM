import numpy as np
import bnlearn as bn
import pandas as pd
import os
from tag_pc import tag_pc
from tag_pc_utils import get_typelist_from_text, get_taglist_of_int_from_text
from run_llm_tag_engineering import run_llm, run_llm_default  
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
    if (path_to_file == "Tag-PC-using-LLM/generated_forestdata.csv"):
        #TODO types besser übergeben
        # MAKE SURE THAT NODES ARE IN THE CORRECT ORDER
        tags = """
            A : Producer
            R : 1st Consumer
            S : 1st Consumer
            H : 2nd Consumer
            B : 2nd Consumer
            W : 3rd Consumer
            F : 3rd Consumer
            """

    print(data)


    return data, node_names, tags

# get dataset that is stored in strings in csv file #XXX NOT RECOMMENDED 
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
            tags = """
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
            tags = """
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

    return data, node_names, tags


# alternativly get data from bnlearn example
def get_data_from_bnf(bnfdata="asia"):
    data = bn.import_example(bnfdata)
    # get corrosponding types

    match bnfdata:
        case "asia":
            # like constintou et. al: we assigned types to variables by randomly partitioning the topological ordering of the DAGs into groups
            tags = """
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
            tags = """
            Cloudy : Weather, Weather, NotWatering
            Sprinkler : Watering, NotWeather, Watering
            Rain : Weather, Weather, Watering
            Wet_Grass : Plant_Con, NotWeather, NotWatering   
            """
   
    # Extract values and node names
    node_names = data.columns.tolist()
    data = data.values

    return data, node_names, tags

#get data from csv:
#forest:
path_forest = "Tag-PC-using-LLM/generated_forestdata.csv"
# data, node_names, tags = get_data_from_csv_int(path_to_file=path_forest)
#bnlearn:
path_bnlearn = "Tag-PC-using-LLM/additionalData" 
# bnfdata = "insurance"  # change depending on example
# data, node_names, tags = get_data_from_csv_string(path_to_folder=path_bnlearn,bnfdata=bnfdata)

# or bnf example XXX not deterministic for asia or bigger data due to underterminism in Algo when presented with incorrect skeleton
bnfdata = "asia" # use here the example you wnt to use
data, node_names, tags = get_data_from_bnf(bnfdata)



llm_generated_tags = False #when False, check assigned tag in line 160 #TODO check line number before publishing

if llm_generated_tags: # use run_llm for already prompt engineered data or write your own prompt with run_llm_default
    tags, node_names = run_llm(bnfdata, deterministic=True)
#    prompt = "Lorem ipsum dolor sit amet" 
#    tags, node_names = run_llm_default(prompt, node_names)

    


alpha = 0.05
indep_test = "fisherz"
majority_rule_typed = True
equal_majority_rule_tagged = True
dag, stat_tests, tag_list = tag_pc(data=data, tags=tags, node_names=node_names, alpha=alpha, indep_test=indep_test, equal_majority_rule_tagged=equal_majority_rule_tagged, majority_rule_typed=majority_rule_typed) #kci gives good results but takes alot of cumputing power

print(node_names)

dir = "Tag-PC-using-LLM/tagged-PC"
fname = "tdag_" + bnfdata + "_" + indep_test + str(alpha) + ("AI_Tag_" if llm_generated_tags else "") + ("majoritytag_" if equal_majority_rule_tagged else "weightedtag_") + ("majoritytype" if majority_rule_typed else "naivetype") + "_1"
create_graph_viz(dag=dag, var_names=node_names, types=tag_list[0], save_to_dir=dir, fname=fname) #print using first tag
if llm_generated_tags: print(f"Ai generated Tags:\n{tags}")  # for debuging TODO ggf. remove for publishen