import os

import bnlearn as bn
import networkx as nx

from pc_from_true_skeleton import standard_pc_deconstructed

from visualization_experiment import create_graph_viz, create_graph_viz_colorless 
from llm_interface_typed import run_llm_generic_prompt_typed

from tpc_utils import get_typelist_from_text
from tpc import tpc_from_true_skeleton

from tag_pc_utils import calculate_shd, calculate_shd_sid, get_undirect_graph
from run_llm_tag_engineering import run_llm, run_llm_generic_prompt
from tag_pc import tag_pc_from_true_skeleton

def experiment_llm_pc(dataname, no_sid = False):
    """
    :param dataname: use data from .bif file in additionaldata or one of the following from bnlearn: "asia", 'sprinkler', 'alarm', 'andes', 'sachs'
    :param no_sid: bool, set to true, if dataset is too large for calculating SID
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
    experiment_dir = os.path.join("tagged-pc-using-LLM/Experiment-Data/Experiment-Graphs-and-Tags", dataname)
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
                case "asia" | 'sprinkler' :
                    path = dataname
                #we search for a .bif file in tagged-pc-using-LLM/additionalData 
                case _:
                    path = os.path.join("tagged-pc-using-LLM/additionalData", (dataname + ".bif"))
                    if not (os.path.isfile(path)):
                        raise FileNotFoundError(f"There is no true graph for {dataname}. Check your spelling or create a .bif file in {path}. (If you are lucky there might be one at https://www.bnlearn.com/bnrepository/).") 
            
            # get Dag from bnf data
            model = bn.import_DAG(path)
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
            # save skeleton SHD, SID, amount types
            if (no_sid):
                shd = calculate_shd(dag=skeleton_adjacency_mat, true_dag=adjacency_mat_true_graph)
                resultsfile.write("Skeleton," + str(shd) + "," + "N/A" + ",(0)\n")
            else:
                shd, sid = calculate_shd_sid(dag=skeleton_adjacency_mat, true_dag=adjacency_mat_true_graph)
                resultsfile.write("Skeleton," + str(shd) + "," + str(sid) + ",(0)\n")


    #---------------------- step 1.5: get types LLM ---------------
    print("------Step 1.5: Generating Types-------")
    types, type_list, node_names = run_llm_generic_prompt_typed(node_names=node_names, determinstic=True) #XXX if LLM produces bs types, change here
    #save types
    f = open(os.path.join(experiment_dir, (dataname + "_types_generic_llm.txt")), "w")
    f.write(types)
    f.close()

    #-------------------- step 2: run PC -------------------------
    print("------Step 2: Running PC-------")
    pc_adjacency_matrix = standard_pc_deconstructed(dataname=dataname, tags=types)
    #save PC
    create_graph_viz_colorless(dag=pc_adjacency_matrix, var_names=node_names, save_to_dir=experiment_dir, fname=(dataname + "_Standard_PC_")) #print using first tag    

    # save PC SHD, SID, amount types
    if (no_sid):
        shd = calculate_shd(dag=pc_adjacency_matrix, true_dag=adjacency_mat_true_graph)
        resultsfile.write("PC," + str(shd) + "," + "N/A" + ",(0)\n")
    else:
        shd, sid = calculate_shd_sid(dag=pc_adjacency_matrix, true_dag=adjacency_mat_true_graph)
        resultsfile.write("PC," + str(shd) + "," + str(sid) + ",(0)\n")


    #-------------------- step 3: run typed PC -------------------
    print("------Step 3: Running Typed-PC-------")
    majority_rule = True
    dag_typed, stat_tests, node_names, typelist = tpc_from_true_skeleton(dataname=dataname, types=types, majority_rule=majority_rule)
    #save typed dag
    fname = dataname + "_Typed_PC_LLM_Generic_" +  ("majority" if majority_rule else "naive")
    create_graph_viz(dag=dag_typed, var_names=node_names, types=typelist, save_to_dir=experiment_dir, fname=fname) #print using first tag

    # save Typed PC SHD, SID, amount types
    if (no_sid):
        shd = calculate_shd(dag=dag_typed, true_dag=adjacency_mat_true_graph)
        resultsfile.write("Typed-PC," + str(shd) + "," + "N/A" + ",(" + str(len(type_list)) + ")\n")
    else:
        shd, sid = calculate_shd_sid(dag=dag_typed, true_dag=adjacency_mat_true_graph)
        resultsfile.write("Typed-PC," + str(shd) + "," + str(sid) + ",(" + str(len(type_list)) + ")\n")


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
    if (no_sid):
        shd = calculate_shd(dag=dag_tagged_majority, true_dag=adjacency_mat_true_graph)
        resultsfile.write("Tagged-PC_Tag-Majority," + str(shd) + "," + "N/A" + ",(" + str(len(tag_list)) + ")\n")
    else:
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
    if (no_sid):
        shd = calculate_shd(dag=dag_tagged_weighted, true_dag=adjacency_mat_true_graph)
        resultsfile.write("Tagged-PC_Tag-Weighted," + str(shd) + "," + "N/A" + ",(" + str(len(tag_list)) + ")\n")
    else:
        shd, sid = calculate_shd_sid(dag=dag_tagged_weighted, true_dag=adjacency_mat_true_graph)
        resultsfile.write("Tagged-PC_Tag-Weighted," + str(shd) + "," + str(sid) + ",(" + str(len(tag_list)) + ")\n")
    

    # if domain prompt exists
    try:
    # ---------------------- step 5.1: generate Tags using LLM domain Prompt ---------------
        print("------Step 5.1: Generating Domain Tags-------")
        tags_domain, tag_list, node_names = run_llm(data=dataname, deterministic=True)
            #save tags
        f = open(os.path.join(experiment_dir, (dataname + "_tags_domain_llm.txt")), "w")
        f.write(tags_domain)
        f.close()
    except ValueError: # No Domain Prompt -> Terminate
        print(f"No Domain Prompt for {dataname}. Terminating Here")
        resultsfile.close
        return

    # ---------------------- step 5.2: run Tag PC - tag majority ----------------------
    print("------Step 5.2 Running Tag PC - Tag-Majority-Domain-------")
    equal_majority_rule_tagged = True #true means majority tag, false is weighted tag
    majority_rule_typed = True #majority rule of typing algo, true is normaly the better choice

    dag_tagged_majority_domain, stat_tests, node_names, taglist = tag_pc_from_true_skeleton(dataname=dataname, tags=tags_domain, equal_majority_rule_tagged=equal_majority_rule_tagged, majority_rule_typed=majority_rule_typed) #TODO Add data for Forest


    fname = dataname + "_Tagged_PC_LLM_Domain_" +  "Tag_Majority_" + ("majoritytype" if majority_rule_typed else "naivetype")
    create_graph_viz(dag=dag_tagged_majority_domain, var_names=node_names, types=taglist[0], save_to_dir=experiment_dir, fname=fname) #print using first tag

    # save Typed PC SHD, SID, amount types
    if (no_sid):
        shd = calculate_shd(dag=dag_tagged_majority_domain, true_dag=adjacency_mat_true_graph)
        resultsfile.write("Tagged-PC_Tag-Majority_Domain," + str(shd) + "," + "N/A" + ",(" + str(len(tag_list)) + ")\n")
    else:
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
    if (no_sid):
        shd = calculate_shd(dag=dag_tagged_weighted_domain, true_dag=adjacency_mat_true_graph)
        resultsfile.write("Tagged-PC_Tag-Weighted_Domain," + str(shd) + "," + "N/A" + ",(" + str(len(tag_list)) + ")\n")
    else:
        shd, sid = calculate_shd_sid(dag=dag_tagged_weighted_domain, true_dag=adjacency_mat_true_graph)
        resultsfile.write("Tagged-PC_Tag-Weighted_Domain," + str(shd) + "," + str(sid) + ",(" + str(len(tag_list)) + ")\n")

    resultsfile.close()