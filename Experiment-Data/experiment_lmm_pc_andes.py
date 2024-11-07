import os

import bnlearn as bn
import networkx as nx

from pc_from_true_skeleton import standard_pc_deconstructed

from visualization_experiment import create_graph_viz, create_graph_viz_colorless 
from llm_interface_typed import run_llm_generic_prompt_typed

from tpc_utils import get_typelist_from_text
from tpc import tpc_from_true_skeleton

from tag_pc_utils import calculate_shd_sid, get_undirect_graph
from run_llm_tag_engineering import run_llm, run_llm_generic_prompt
from tag_pc import tag_pc_from_true_skeleton


#XXX BEFORE USING EDIT def calculate_shd_sid(dag, true_dag): in tag_pc_utils edit out SID and return:R     
    # return shd, "N/A"
def experiment_llm_pc(dataname):
    """
    : param dataname: use data from .bif file in additionaldata or one of the following from bnlearn: "asia", 'sprinkler', 'alarm', 'andes', 'sachs'
    
    step 1: get node_names and load skeleton from bif file or bnlearn, draw skeleton + true dag
    step 1.5: generate Types using LLM, save Types
    step 2: run standard PC using true skeleton, draw dag and save SHD,SID
    step 3: run Typed PC, draw dag and save SHD, SID
    step 4.1: generate Tags using LLM generic Prompt, save Tags
    step 4.2: run Tag PC - tag majority, draw dag and save SHD, SID
    step 4.3: run Tag PC - tag weighted, draw dag and save SHD, SID
    if domain prompt exists
    step 5.1: generate Tags using LLM domain Prompt, save Tags
    step 5.2: run Tag PC - tag majority, draw dag and save SHD, SID
    step 5.3: run Tag PC - tag weighted, draw dag and save SHD, SID
    """



    #folder for experiment
    experiment_dir = os.path.join("Tag-PC-using-LLM/Experiment-Data/Experiment-Graphs-and-Tags", dataname)
    os.makedirs(experiment_dir, exist_ok=True)
    # file with SHD and SID for all experiments
    resultsfile = open(os.path.join(experiment_dir, (dataname + "_shd_sid.txt")), "w")
    resultsfile.write("Algo,SHD,SID,(Amount Tags)\n")


    #-----------------step 1: get node names--------------------------
    print("------Step 1: Load Bnf-------")
    match dataname:
        # since we already get the true skeleton with the correct parameters for causallearn we do just that
        case "forest":
            node_names = ["A","R","S","H","B","W","F"]
            print("Not implemented yet")
            # TODO was an path / (data geben)
        case _:
            match dataname:
                        #bnlearn directly supports those, we do not need a bif file
                case "asia" | 'sprinkler' | 'sachs':
                    path = dataname
                #we search for a .bif file in Tag-PC-using-LLM/additionalData 
                case _:
                    path = os.path.join("Tag-PC-using-LLM/additionalData", (dataname + ".bif"))
                    if not (os.path.isfile(path)):
                        raise FileNotFoundError(f"There is no true graph for {dataname}. Check your spelling or create a .bif file in {path}. (If you are lucky there might be one at https://www.bnlearn.com/bnrepository/).") 
            
            # get Dag from bnf data
            model = bn.import_DAG(path)
            print("imported Model") #TODO DELETE
            adjacency_mat_true_graph = model['adjmat']
            adjacency_mat_true_graph = adjacency_mat_true_graph.astype(int) #get int representation
            node_names = adjacency_mat_true_graph.columns.tolist()
            adjacency_mat_true_graph = adjacency_mat_true_graph.values #delete headers
            skeleton_adjacency_mat = get_undirect_graph(adjacency_mat_true_graph)
            stat_tests = skeleton_adjacency_mat
            print(node_names)
            # draw skeleton and true graph
            create_graph_viz_colorless(dag=skeleton_adjacency_mat, var_names=node_names, save_to_dir=experiment_dir, fname=(dataname + "_skeleton"))
            create_graph_viz_colorless(dag=adjacency_mat_true_graph, var_names=node_names, save_to_dir=experiment_dir, fname=(dataname + "_true_dag"))
            print("CALC SHD") #TODO DELETE
            # save skeleton SHD, SID, amount types
            shd, sid = calculate_shd_sid(dag=skeleton_adjacency_mat, true_dag=adjacency_mat_true_graph)
            resultsfile.write("Skeleton," + str(shd) + "," + str(sid) + ",(0)\n")


    #---------------------- step 1.5: get types LLM ---------------
    print("------Step 1.5: Generating Types-------")
    types, type_list, node_names = run_llm_generic_prompt_typed(node_names=node_names, determinstic=True) #XXX if LLM produces bs types, change here
    #save types
    f = open(os.path.join(experiment_dir, (dataname + "_types_generic_llm.txt")), "w")
    f.write(types)
    f.close()
    # typelist = get_typelist_from_text(types)


    #-------------------- step 2: run PC -------------------------
    print("------Step 2: Running PC-------")
    type_andes = """
    DISPLACEM0 : State  
    RApp1 : Action  
    SNode_3 : State  
    GIVEN_1 : Property  
    RApp2 : Action  
    SNode_8 : State  
    SNode_16 : State  
    SNode_20 : State  
    NEED1 : Goal  
    SNode_21 : State  
    GRAV2 : Property  
    SNode_24 : State  
    VALUE3 : Property  
    SNode_15 : State  
    SNode_25 : State  
    SLIDING4 : Action  
    SNode_11 : State  
    SNode_26 : State  
    CONSTANT5 : Property  
    SNode_47 : State  
    VELOCITY7 : Property  
    KNOWN6 : Property  
    RApp3 : Action  
    KNOWN8 : Property  
    RApp4 : Action  
    SNode_27 : State  
    GOAL_2 : Goal  
    GOAL_48 : Goal  
    COMPO16 : Property  
    TRY12 : Action  
    TRY11 : Action  
    SNode_5 : State  
    GOAL_49 : Goal  
    SNode_6 : State  
    GOAL_50 : Goal  
    CHOOSE19 : Action  
    SNode_17 : State  
    SNode_51 : State  
    SYSTEM18 : Property  
    SNode_52 : State  
    KINEMATI17 : Property  
    GOAL_53 : Goal  
    IDENTIFY10 : Action  
    SNode_28 : State  
    IDENTIFY9 : Action  
    TRY13 : Action  
    TRY14 : Action  
    TRY15 : Action  
    SNode_29 : State  
    VAR20 : Property  
    SNode_31 : State  
    SNode_10 : State  
    SNode_33 : State  
    GIVEN21 : Property  
    SNode_34 : State  
    GOAL_56 : Goal  
    APPLY32 : Action  
    GOAL_57 : Goal  
    CHOOSE35 : Action  
    SNode_7 : State  
    SNode_59 : State  
    MAXIMIZE34 : Action  
    SNode_60 : State  
    AXIS33 : Property  
    GOAL_61 : Goal  
    WRITE31 : Action  
    GOAL_62 : Goal  
    WRITE30 : Action  
    GOAL_63 : Goal  
    RESOLVE37 : Action  
    SNode_64 : State  
    NEED36 : Goal  
    SNode_9 : State  
    SNode_41 : State  
    SNode_42 : State  
    SNode_43 : State  
    IDENTIFY39 : Action  
    GOAL_66 : Goal  
    RESOLVE38 : Action  
    SNode_67 : State  
    SNode_54 : State  
    IDENTIFY41 : Action  
    GOAL_69 : Goal  
    RESOLVE40 : Action  
    SNode_70 : State  
    SNode_55 : State  
    IDENTIFY43 : Action  
    GOAL_72 : Goal  
    RESOLVE42 : Action  
    SNode_73 : State  
    SNode_74 : State  
    KINE29 : Property  
    SNode_4 : State  
    SNode_75 : State  
    VECTOR44 : Property  
    GOAL_79 : Goal  
    EQUATION28 : Property  
    VECTOR27 : Property  
    RApp5 : Action  
    GOAL_80 : Goal  
    RApp6 : Action  
    GOAL_81 : Goal  
    TRY25 : Action  
    TRY24 : Action  
    GOAL_83 : Goal  
    GOAL_84 : Goal  
    CHOOSE47 : Action  
    SNode_86 : State  
    SYSTEM46 : Property  
    SNode_156 : State  
    NEWTONS45 : Property  
    GOAL_98 : Goal  
    DEFINE23 : Action  
    SNode_37 : State  
    IDENTIFY22 : Action  
    TRY26 : Action  
    SNode_38 : State  
    SNode_40 : State  
    SNode_44 : State  
    SNode_46 : State  
    SNode_65 : State  
    NULL48 : Property  
    SNode_68 : State  
    SNode_71 : State  
    GOAL_87 : Goal  
    FIND49 : Action  
    SNode_88 : State  
    NORMAL50 : Property  
    NORMAL52 : Property  
    INCLINE51 : Property  
    SNode_91 : State  
    SNode_12 : State  
    SNode_13 : State  
    STRAT_90 : Property  
    HORIZ53 : Property  
    BUGGY54 : Property  
    SNode_92 : State  
    SNode_93 : State  
    IDENTIFY55 : Action  
    SNode_94 : State  
    WEIGHT56 : Property  
    SNode_95 : State  
    WEIGHT57 : Property  
    SNode_97 : State  
    GOAL_99 : Goal  
    FIND58 : Action  
    SNode_100 : State  
    IDENTIFY59 : Action  
    SNode_102 : State  
    FORCE60 : Property  
    GOAL_103 : Goal  
    APPLY61 : Action  
    GOAL_104 : Goal  
    CHOOSE62 : Action  
    SNode_106 : State  
    SNode_152 : State  
    GOAL_107 : Goal  
    WRITE63 : Action  
    GOAL_108 : Goal  
    WRITE64 : Action  
    GOAL_109 : Goal  
    GOAL_110 : Goal  
    GOAL_65 : Goal  
    GOAL_111 : Goal  
    GOAL_112 : Goal  
    NEED67 : Goal  
    RApp7 : Action  
    RApp8 : Action  
    SNode_112 : State  
    GOAL_113 : Goal  
    GOAL_114 : Goal  
    SNode_115 : State  
    SNode_116 : State  
    VECTOR69 : Property  
    SNode_117 : State  
    SNode_118 : State  
    VECTOR70 : Property  
    SNode_119 : State  
    EQUAL71 : Property  
    SNode_120 : State  
    GOAL_121 : Goal  
    GOAL_122 : Goal  
    SNode_123 : State  
    SNode_124 : State  
    VECTOR73 : Property  
    SNode_125 : State  
    NEWTONS74 : Property  
    SNode_126 : State  
    SUM75 : Property  
    GOAL_126 : Goal  
    GOAL_127 : Goal  
    RApp9 : Action  
    RApp10 : Action  
    SNode_128 : State  
    GOAL_129 : Goal  
    GOAL_130 : Goal  
    SNode_131 : State  
    SNode_132 : State  
    SNode_133 : State  
    SNode_134 : State  
    SNode_135 : State  
    SNode_154 : State  
    SNode_136 : State  
    SNode_137 : State  
    GOAL_142 : Goal  
    GOAL_143 : Goal  
    GOAL_146 : Goal  
    RApp11 : Action  
    RApp12 : Action  
    RApp13 : Action  
    GOAL_147 : Goal  
    GOAL_149 : Goal  
    TRY76 : Action  
    GOAL_150 : Goal  
    APPLY77 : Action  
    SNode_151 : State  
    GRAV78 : Property  
    GOAL_153 : Goal  
    SNode_155 : State  
    SNode_14 : State  
    SNode_18 : State  
    SNode_19 : State  
    """
    pc_adjacency_matrix = standard_pc_deconstructed(dataname=dataname, tags=type_andes)
    #save PC
    create_graph_viz_colorless(dag=pc_adjacency_matrix, var_names=node_names, save_to_dir=experiment_dir, fname=(dataname + "_Standard_PC_")) #print using first tag    

    # save PC SHD, SID, amount types
    shd, sid = calculate_shd_sid(dag=pc_adjacency_matrix, true_dag=adjacency_mat_true_graph)
    resultsfile.write("PC," + str(shd) + "," + str(sid) + ",(0)\n")


    #-------------------- step 3: run typed PC -------------------
    print("------Step 3: Running Typed-PC-------")
    majority_rule = True
    dag_typed, stat_tests, node_names, typelist = tpc_from_true_skeleton(dataname=dataname, types=type_andes, majority_rule=majority_rule)
    #save typed dag
    fname = dataname + "_Typed_PC_LLM_Generic_" +  ("majority" if majority_rule else "naive")
    create_graph_viz(dag=dag_typed, var_names=node_names, types=typelist, save_to_dir=experiment_dir, fname=fname) #print using first tag

    #save Typed PC SHD, SID, amount types
    shd, sid = calculate_shd_sid(dag=dag_typed, true_dag=adjacency_mat_true_graph)
    resultsfile.write("Typed-PC," + str(shd) + "," + str(sid) + ",(" + "5" + ")\n")


    # ----- step 4.1: generate Tags ------------
    print("------Step 4.1: Generating Tags-------")
    tags, tag_list, node_names = run_llm_generic_prompt(node_names=node_names, determinstic=True)
    #save tags
    f = open(os.path.join(experiment_dir, (dataname + "_tags_generic_llm.txt")), "w")
    f.write(tags)
    f.close()


    # ----- step 4.2: run Tag PC - tag majority -----------
    print("------Step 4.2 Running Tag PC - Tag-Majority-------")
    equal_majority_rule_tagged = True #true means majority tag, false is weighted tag
    majority_rule_typed = True #majority rule of typing algo, true is normaly the better choice

    dag_tagged_majority, stat_tests, node_names, taglist = tag_pc_from_true_skeleton(dataname=dataname, tags=tags, equal_majority_rule_tagged=equal_majority_rule_tagged, majority_rule_typed=majority_rule_typed) #TODO Add data for Forest


    fname = dataname + "_Tagged_PC_LLM_Generic_" +  "Tag_Majority_" + ("majoritytype" if majority_rule_typed else "naivetype")
    create_graph_viz(dag=dag_tagged_majority, var_names=node_names, types=taglist[0], save_to_dir=experiment_dir, fname=fname) #print using first tag

    # save Typed PC SHD, SID, amount types
    shd, sid = calculate_shd_sid(dag=dag_tagged_majority, true_dag=adjacency_mat_true_graph)
    resultsfile.write("Tagged-PC_Tag-Majority," + str(shd) + "," + str(sid) + ",(" + str(len(tag_list)) + ")\n")
    

    # ----- step 4.3: run Tag PC - tag weighted ----------
    print("------Step 4.3 Running Tag PC - Tag-Weighted-------")
    equal_majority_rule_tagged = False #true means majority tag, false is weighted tag
    majority_rule_typed = True #majority rule of typing algo, true is normaly the better choice

    dag_tagged_weighted, stat_tests, node_names, taglist = tag_pc_from_true_skeleton(dataname=dataname, tags=tags, equal_majority_rule_tagged=equal_majority_rule_tagged, majority_rule_typed=majority_rule_typed) #TODO Add data for Forest


    fname = dataname + "_Tagged_PC_LLM_Generic_" +  "Tag_Weighted_" + ("majoritytype" if majority_rule_typed else "naivetype")
    create_graph_viz(dag=dag_tagged_weighted, var_names=node_names, types=taglist[0], save_to_dir=experiment_dir, fname=fname) #print using first tag

    # save Typed PC SHD, SID, amount types
    shd, sid = calculate_shd_sid(dag=dag_tagged_weighted, true_dag=adjacency_mat_true_graph)
    resultsfile.write("Tagged-PC_Tag-Weighted," + str(shd) + "," + str(sid) + ",(" + str(len(tag_list)) + ")\n")
    

    # if domain prompt exists for andes so no try catch:
    # ---------------------- step 5.1: generate Tags using LLM domain Prompt ---------------
    print("------Step 5.1: Generating Domain Tags-------")
    tags_domain, tag_list, node_names = run_llm(data=dataname, deterministic=True)
    #save tags
    f = open(os.path.join(experiment_dir, (dataname + "_tags_domain_llm.txt")), "w")
    f.write(tags_domain)
    f.close()


    # ---------------------- step 5.2: run Tag PC - tag majority ----------------------
    print("------Step 5.2 Running Tag PC - Tag-Majority-Domain-------")
    equal_majority_rule_tagged = True #true means majority tag, false is weighted tag
    majority_rule_typed = True #majority rule of typing algo, true is normaly the better choice

    dag_tagged_majority_domain, stat_tests, node_names, taglist = tag_pc_from_true_skeleton(dataname=dataname, tags=tags_domain, equal_majority_rule_tagged=equal_majority_rule_tagged, majority_rule_typed=majority_rule_typed) #TODO Add data for Forest


    fname = dataname + "_Tagged_PC_LLM_Domain_" +  "Tag_Majority_" + ("majoritytype" if majority_rule_typed else "naivetype")
    create_graph_viz(dag=dag_tagged_majority_domain, var_names=node_names, types=taglist[0], save_to_dir=experiment_dir, fname=fname) #print using first tag

    # save Typed PC SHD, SID, amount types
    shd, sid = calculate_shd_sid(dag=dag_tagged_majority_domain, true_dag=adjacency_mat_true_graph)
    resultsfile.write("Tagged-PC_Tag-Majority_Domain," + str(shd) + "," + str(sid) + ",(" + str(len(tag_list)) + ")\n")

    # ---------------------- step 5.3: run Tag PC - tag weighted -----------
    print("------Step 5.3 Running Tag PC - Tag-Weighted-Domain-------")
    equal_majority_rule_tagged = False #true means majority tag, false is weighted tag
    majority_rule_typed = True #majority rule of typing algo, true is normaly the better choice

    dag_tagged_weighted_domain, stat_tests, node_names, taglist = tag_pc_from_true_skeleton(dataname=dataname, tags=tags_domain, equal_majority_rule_tagged=equal_majority_rule_tagged, majority_rule_typed=majority_rule_typed) #TODO Add data for Forest

    fname = dataname + "_Tagged_PC_LLM_Domain_" +  "Tag_Weighted_" + ("majoritytype" if majority_rule_typed else "naivetype")
    create_graph_viz(dag=dag_tagged_weighted_domain, var_names=node_names, types=taglist[0], save_to_dir=experiment_dir, fname=fname) #print using first tag

    # save Typed PC SHD, SID, amount types
    shd, sid = calculate_shd_sid(dag=dag_tagged_weighted_domain, true_dag=adjacency_mat_true_graph)
    resultsfile.write("Tagged-PC_Tag-Weighted_Domain," + str(shd) + "," + str(sid) + ",(" + str(len(tag_list)) + ")\n")

    resultsfile.close()