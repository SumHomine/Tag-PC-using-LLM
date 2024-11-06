import numpy as np
import bnlearn as bn
import pandas as pd
import os
from tpc import tpc_from_true_skeleton
from tpc_utils import get_typelist_from_text
from visualization import create_graph_viz, create_graph_viz_colorless
from llm_interface_typed import run_llm_generic_prompt_typed


#For Tu Server
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


tag_insurance = """
    SocioEcon : First
    GoodStudent : First
    Age : First
    RiskAversion : First
    VehicleYear : Second
    Accident : Fourth
    ThisCarDam : Fourth
    RuggedAuto : Third
    MakeModel : Second
    Antilock : Third
    Mileage : First
    DrivQuality : Fourth
    DrivingSkill : Third
    SeniorTrain : Second
    ThisCarCost : Fifth
    CarValue : Third
    Theft : Fourth
    AntiTheft : Second
    HomeBase : Sixth
    OtherCarCost : Fourth
    PropCost : Fifth
    OtherCar : First
    MedCost : Fourth
    Cushioning : Fourth
    Airbag : Third
    ILiCost : Fourth
    DrivHist : Fourth
    """

#not type consistent
asia_type2 = """ 
    asia : nosick
    tub : sick
    smoke : nosick
    lung : sick
    bronc : sick
    either : sick
    xray : nosick
    dysp : sick
    """

sprinkler_types_Weather = """
    Cloudy : Weather
    Sprinkler : NotWeather
    Rain : Weather
    Wet_Grass : NotWeather  
    """

sprinkler_types_Watervapor = """
    Cloudy : Uses Watervapor
    Sprinkler : Uses Watervapor
    Rain : NotUses Watervapor
    Wet_Grass : NotUses Watervapor   
    """

sprinkler_types_watering = """
    Cloudy : NotWatering
    Sprinkler : Watering
    Rain : Watering
    Wet_Grass : NotWatering   
    """

sprinkler_types_intuitiv = """
    Cloudy : Weather
    Sprinkler : Object
    Rain : Weather
    Wet_Grass : Plant_Con    
    """

sprinkler_types_true_graph = """
    Cloudy : clouds
    Sprinkler : Watering
    Rain : Watering
    Wet_Grass : NotWatering   
    """

sprinkler_types_true_graph_2 = """
    Cloudy : Before Water
    Sprinkler : Notclouds
    Rain : Notclouds
    Wet_Grass : After Water   
    """

sprinkler_ai = """
    Cloudy : Weather
    Sprinkler : Action
    Rain : Weather
    Wet_Grass : State
    """

asia_ai = """
    asia : Disease
    tub : Disease
    smoke : Risk Factor
    lung : Symptom
    bronc : Disease
    either : Exposure
    xray : Test/Procedure
    dysp : Symptom
    """

insurance_ai = """
    SocioEcon : Demographic
    GoodStudent : Demographic
    Age : Demographic
    RiskAversion : Risk
    VehicleYear : Vehicle
    Accident : Risk
    ThisCarDam : Risk
    RuggedAuto : Vehicle
    MakeModel : Vehicle
    Antilock : Vehicle
    Mileage : Vehicle
    DrivQuality : Driving
    DrivingSkill : Driving
    SeniorTrain : Driving
    ThisCarCost : Financial
    CarValue : Financial
    Theft : Risk
    AntiTheft : Risk
    HomeBase : Demographic
    OtherCarCost : Financial
    PropCost : Financial
    OtherCar : Vehicle
    MedCost : Financial
    Cushioning : Vehicle
    Airbag : Vehicle
    ILiCost : Financial
    DrivHist : Driving
    """

# node names for AI generation
node_names_sprinkler = ["Cloudy", "Sprinkler", "Rain", "Wet_Grass"]
node_names_asia = ["asia", "tub", "smoke", "lung", "bronc", "either", "xray",  "dysp"]
node_names_insurance = ['SocioEcon', 'GoodStudent', 'Age', 'RiskAversion', 'VehicleYear', 'Accident', 'ThisCarDam', 'RuggedAuto', 'MakeModel', 'Antilock', 'Mileage', 'DrivQuality', 'DrivingSkill', 'SeniorTrain', 'ThisCarCost', 'CarValue', 'Theft', 'AntiTheft', 'HomeBase', 'OtherCarCost', 'PropCost', 'OtherCar', 'MedCost', 'Cushioning', 'Airbag', 'ILiCost', 'DrivHist']
node_names_andes = ['DISPLACEM0', 'RApp1', 'SNode_3', 'GIVEN_1', 'RApp2', 'SNode_8', 'SNode_16', 'SNode_20', 'NEED1', 'SNode_21', 'GRAV2', 'SNode_24', 'VALUE3', 'SNode_15', 'SNode_25', 'SLIDING4', 'SNode_11', 'SNode_26', 'CONSTANT5', 'SNode_47', 'VELOCITY7', 'KNOWN6', 'RApp3', 'KNOWN8', 'RApp4', 'SNode_27', 'GOAL_2', 'GOAL_48', 'COMPO16', 'TRY12', 'TRY11', 'SNode_5', 'GOAL_49', 'SNode_6', 'GOAL_50', 'CHOOSE19', 'SNode_17', 'SNode_51', 'SYSTEM18', 'SNode_52', 'KINEMATI17', 'GOAL_53', 'IDENTIFY10', 'SNode_28', 'IDENTIFY9', 'TRY13', 'TRY14', 'TRY15', 'SNode_29', 'VAR20', 'SNode_31', 'SNode_10', 'SNode_33', 'GIVEN21', 'SNode_34', 'GOAL_56', 'APPLY32', 'GOAL_57', 'CHOOSE35', 'SNode_7', 'SNode_59', 'MAXIMIZE34', 'SNode_60', 'AXIS33', 'GOAL_61', 'WRITE31', 'GOAL_62', 'WRITE30', 'GOAL_63', 'RESOLVE37', 'SNode_64', 'NEED36', 'SNode_9', 'SNode_41', 'SNode_42', 'SNode_43', 'IDENTIFY39', 'GOAL_66', 'RESOLVE38', 'SNode_67', 'SNode_54', 'IDENTIFY41', 'GOAL_69', 'RESOLVE40', 'SNode_70', 'SNode_55', 'IDENTIFY43', 'GOAL_72', 'RESOLVE42', 'SNode_73', 'SNode_74', 'KINE29', 'SNode_4', 'SNode_75', 'VECTOR44', 'GOAL_79', 'EQUATION28', 'VECTOR27', 'RApp5', 'GOAL_80', 'RApp6', 'GOAL_81', 'TRY25', 'TRY24', 'GOAL_83', 'GOAL_84', 'CHOOSE47', 'SNode_86', 'SYSTEM46', 'SNode_156', 'NEWTONS45', 'GOAL_98', 'DEFINE23', 'SNode_37', 'IDENTIFY22', 'TRY26', 'SNode_38', 'SNode_40', 'SNode_44', 'SNode_46', 'SNode_65', 'NULL48', 'SNode_68', 'SNode_71', 'GOAL_87', 'FIND49', 'SNode_88', 'NORMAL50', 'NORMAL52', 'INCLINE51', 'SNode_91', 'SNode_12', 'SNode_13', 'STRAT_90', 'HORIZ53', 'BUGGY54', 'SNode_92', 'SNode_93', 'IDENTIFY55', 'SNode_94', 'WEIGHT56', 'SNode_95', 'WEIGHT57', 'SNode_97', 'GOAL_99', 'FIND58', 'SNode_100', 'IDENTIFY59', 'SNode_102', 'FORCE60', 'GOAL_103', 'APPLY61', 'GOAL_104', 'CHOOSE62', 'SNode_106', 'SNode_152', 'GOAL_107', 'WRITE63', 'GOAL_108', 'WRITE64', 'GOAL_109', 'GOAL_110', 'GOAL65', 'GOAL_111', 'GOAL66', 'NEED67', 'RApp7', 'RApp8', 'SNode_112', 'GOAL_113', 'GOAL68', 'GOAL_114', 'SNode_115', 'SNode_116', 'VECTOR69', 'SNode_117', 'SNode_118', 'VECTOR70', 'SNode_119', 'EQUAL71', 'SNode_120', 'GOAL_121', 'GOAL72', 'SNode_122', 'SNode_123', 'VECTOR73', 'SNode_124', 'NEWTONS74', 'SNode_125', 'SUM75', 'GOAL_126', 'GOAL_127', 'RApp9', 'RApp10', 'SNode_128', 'GOAL_129', 'GOAL_130', 'SNode_131', 'SNode_132', 'SNode_133', 'SNode_134', 'SNode_135', 'SNode_154', 'SNode_136', 'SNode_137', 'GOAL_142', 'GOAL_143', 'GOAL_146', 'RApp11', 'RApp12', 'RApp13', 'GOAL_147', 'GOAL_149', 'TRY76', 'GOAL_150', 'APPLY77', 'SNode_151', 'GRAV78', 'GOAL_153', 'SNode_155', 'SNode_14', 'SNode_18', 'SNode_19']

# Change Here
dataname = "sprinkler" # asia, insurance .....
majority_rule = True
llm_generated_types = False

if llm_generated_types:
    types, node_names = run_llm_generic_prompt_typed(node_names=node_names_andes, determinstic=True) #XXX you need top update node_names depending on dataset
else:
    types = sprinkler_types_watering #see tags above 

dag, stat_tests, node_names, typelist = tpc_from_true_skeleton(dataname=dataname, types=types, majority_rule=majority_rule)


dir = "/home/ml-stud19/tagged-pc-using-LLM/typed-PC"
fname = "tdag_" + dataname + "_" + "true_skeleton_"  + ("AI_Type_New_" if llm_generated_types else "") +  ("majority" if majority_rule else "naive") + ""
create_graph_viz(dag=dag, var_names=node_names, types=typelist, save_to_dir=dir, fname=fname) #print using first tag
# create_graph_viz_colorless(dag=dag, var_names=node_names, types=typelist, save_to_dir=dir, fname="colorless_" + fname) #print using colorless edges
