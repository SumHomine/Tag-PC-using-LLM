import logging
import networkx as nx
import numpy as np

from itertools import combinations, permutations
from typing import Tuple
from random import shuffle

from causallearn.utils.PCUtils import SkeletonDiscovery
from causallearn.utils.cit import *

import bnlearn as bn

from tag_pc_utils import _orient_typeless_with_priomat_cycle_check_adjmat, _orient_typeless_with_priomat_cycle_check_nx, assign_deleted_edges_in_priomat_evidence, exists_entry_in_forkevidencelist, fork_clashes_immorality, get_amount_difference_two_mats, get_taglist_of_int_from_text, get_undirect_graph, are_forks_clashing, orient_tagged_dag_according_majority_tag_matrix_using_prio_mat_cycle_check, set_tags_as_int, format_seperating_sets, _has_both_edges, _has_directed_edge, _orient_typeless_with_priomat, type_of_from_tag_all, get_majority_tag_matrix, get_majority_tag_matrix_using_priomat_type_majority, amount_of_matching_types_for_two_way_fork, get_priomat_from_skeleton, get_separating_sets_using_true_skeleton, typed_pc_from_true_skeleton
from visualization_experiment import plot_dag_state, plot_dag_state_only_visible
# from test_tag_pc import load_skeleton, save_skeleton 

def tag_pc (data, tags, node_names, alpha=0.05, indep_test="fisherz", equal_majority_rule_tagged=True, majority_rule_typed=True):
    """
    :param data: np.ndarray of datafile. instruction to get there are in run_tag
    :param tags: text of params and corresponding assigned tags with tags being seperated by comma in the form:
        Cloudy : Weather, Weather, NotWatering
        Sprinkler : Watering, NotWeather, Watering
        Rain : Weather, Weather, Watering
        Wet_Grass : Plant_Con, NotWeather, NotWatering  
    :param node_names: list of String of the column names of the dataset in the correct order, used to correct order of tags when mistakenly in wrong order and for debugging
    :param alpha: significance level for skeleton discovery
    :param indep_test: str, type of the used independence test. default: fisherz, 
                other options are: "mv_fisherz", "mc_fisherz", "kci", "chisq", "gsq", "d_separation"
    :param equal_majority_rule_tagged: bool, if true, uses the majority of tags to orient forks (with each type being equally important) =majority alg. if false priotize tags that change less edges =tag weight alg
    :param majority_rule_typed: bool, majority rule from typed PC Algo: if true use majority rule to orient forks, if false use naive rule to orient forks
    :return: adjacency matrix of the DAG of type np.ndarray 
    :return: taglist transformed as list of list of Int where every list is a typelist

    step 1: infer skeleton
    step 2: orient v-structures & two type forks + type consistency
    step 3: t-propagation
    
    step 2 and 3 are seperated in tpc_tag_weighted and tpc_tag_majority

    recommended usage regarding tag importance:
    if all tags have the same importance use tag majority (with type majority)
    if you want to prioritize tags with small changes use Tag Weighted
    """
    #step 1
    taglist = get_taglist_of_int_from_text(tags=tags, node_names_ordered=node_names) #get Tags as List of List of Int
    skeleton, separating_sets, stat_tests = create_skeleton_using_causallearn(data, taglist, alpha, indep_test) # get skeleton

    print("skeleton: \n", nx.adjacency_matrix(skeleton).todense()) 
    # plot_dag_state_only_visible(dag=nx.adjacency_matrix(skeleton).todense(), var_names=node_names, types=taglist[0], step_number=0, experiment_step="skeleton") #XXX prints intermediate dag states, comment out if necessary
#    save_skeleton(skeleton=skeleton, separating_sets=separating_sets, stat_tests=stat_tests)     # -> use for debugging when you want to save a new skeleton
#    skeleton, separating_sets, stat_tests = load_skeleton()                                 # -> use for debugging when you dont want to wait to create skeleton -> also comment out create skeleton/seperating_sets

    if (equal_majority_rule_tagged):
        dag, stat_tests = tpc_tag_majority(taglist, skeleton, separating_sets, stat_tests, majority_rule_typed, node_names=node_names)
    else:
        dag, stat_tests = tpc_tag_weighted(taglist, skeleton, separating_sets, stat_tests, majority_rule_typed, node_names=node_names)

    return dag, stat_tests, taglist

def tag_pc_from_true_skeleton (dataname : str, tags, equal_majority_rule_tagged=True, majority_rule_typed=True, data=None):
    """
    :param tags: text of params and corresponding assigned tags with tags being seperated by comma in the form:
        Cloudy : Weather, Weather, NotWatering
        Sprinkler : Watering, NotWeather, Watering
        Rain : Weather, Weather, Watering
        Wet_Grass : Plant_Con, NotWeather, NotWatering  
    :param equal_majority_rule_tagged: bool, if true, uses the majority of tags to orient forks (with each type being equally important) =majority alg. if false priotize tags that change less edges =tag weight alg
    :param majority_rule_typed: bool, majority rule from typed PC Algo: if true use majority rule to orient forks, if false use naive rule to orient forks
    :optional param data: string of data with nodes seperated with whitespace and entries by line break
    :return: adjacency matrix of the DAG of type np.ndarray
    :return: stat_tests of the DAG of type np.ndarray  
    :return: node names of type list<String> 
    :return: taglist transformed as list of list of Int where every list is a typelist

    step 1: infer skeleton
    step 2: orient v-structures & two type forks
    step 3: t-propagation
    
    step 2 and 3 are seperated in tpc_tag_weighted and tpc_tag_majority

    recommended usage regarding tag importance:
    if all tags have the same importance use tag majority (with type majority)
    if you want to prioritize tags with small changes use Tag Weighted 
    """
    #step 1 from true skelton
    skeleton, separating_sets, stat_tests, node_names, taglist = get_true_skeleton(dataname=dataname, tags=tags, data=data)

    print("skeleton: \n", nx.adjacency_matrix(skeleton).todense()) 
    plot_dag_state_only_visible(dag=nx.adjacency_matrix(skeleton).todense(), var_names=node_names, types=taglist[0], step_number=0, experiment_step="skeleton") #XXX prints intermediate dag states, comment out if necessary
#    save_skeleton(skeleton=skeleton, separating_sets=separating_sets, stat_tests=stat_tests)     # -> use for debugging when you want to save a new skeleton
#    skeleton, separating_sets, stat_tests = load_skeleton()                                 # -> use for debugging when you dont want to wait to create skeleton -> also comment out create skeleton/seperating_sets

    if equal_majority_rule_tagged:
        dag, stat_tests = tpc_tag_majority(taglist, skeleton, separating_sets, stat_tests, majority_rule_typed, node_names=node_names)
    else:
        dag, stat_tests = tpc_tag_weighted(taglist, skeleton, separating_sets, stat_tests, majority_rule_typed, node_names=node_names) 

    return dag, stat_tests, node_names, taglist

def tpc_tag_majority(taglist, skeleton, separating_sets, stat_tests, majority_rule_typed, node_names=[]):
    """
    :param taglist: list of list of int where every list is a typelist (can be obtained using get_taglist_of_int_from_text)
    :param skeleton: nx.graph of the datas' skeleton (meanin undirected edges between all statistically connected nodes)
    :param separating_sets: np.ndarray where each entry (i, j) is a list of separating sets for nodes i and j (obtained using causallearn - SkeletonDiscovery on data or get_separating_sets_using_true_skeleton for true skeleton).
    :param majority_rule_typed: bool, majority rule from typed PC Algo: if true use majority rule to orient forks, if false use naive rule to orient forks (true is recommended)
    :optional param node_names: list of string of the node names in the correct order (handed over by tpc or optained in get_true_skeleton) -> only needed for debugging can be empty
    :param + return: stat_tests of the DAG of type np.ndarray  (obtained in skeleton discovery and not changed here)
    :return: adjacency matrix of the DAG of type np.ndarray
    """  
    priomat = get_priomat_from_skeleton(nx.adjacency_matrix(skeleton).todense(), taglist)
    print("priomat \n", priomat) 

    #Step 2 - V Structues + Two Type Forks
    if (majority_rule_typed):
        dag, priomat = orient_forks_majority_tag_majority_top1(skeleton=skeleton, sep_sets=separating_sets, priomat=priomat, taglist=taglist, node_names=node_names)
    else:
        dag, priomat = orient_forks_majority_tag_naive_type(skeleton=skeleton, sep_sets=separating_sets, priomat=priomat, taglist=taglist, node_names=node_names)
    # plot_dag_state(dag=nx.adjacency_matrix(dag).todense(), var_names=node_names, types=taglist[0], step_number=1, experiment_step="orienting_forks_v_structures") # XXX comment out before publishing #comment in for intermediate dag states

    #Step 3 - T-Propagation without type consistency
    adjacency_mat, priomat = meek_majority_tag_without_typeconsistency(cpdag=nx.adjacency_matrix(dag).todense(), tags=taglist, priomat=priomat, node_names=node_names)
    # for typesnr in range(len(taglist)):
        # plot_dag_state(dag=adjacency_mat, var_names=node_names, types=taglist[typesnr], step_number=2, experiment_step="meek_no_consistency_tag" + str(typesnr)) # XXX comment out before publishing #comment in for intermediate dag states


    #Step 4 - T-Propagation with Tag_majority Consistency by using priomat
    adjacency_mat, priomat = typed_meek_majority_tag_with_consistency(cpdag=adjacency_mat, tags=taglist, priomat=priomat, majority_rule_typed=majority_rule_typed, node_names=node_names)
    # plot_dag_state(dag=adjacency_mat, var_names=node_names, types=taglist[0], step_number=3, experiment_step="meek_type_consistency") # XXX comment out before publishing #comment in for intermediate dag states


    print("priomat result: \n", priomat)
    print("adjacency_mat result: \n", adjacency_mat)
    print("stat_test result: \n", stat_tests)
    return adjacency_mat, stat_tests

def tpc_tag_weighted(taglist, skeleton, separating_sets, stat_tests, majority_rule_typed, node_names=[]):
    """
    :param taglist: list of list of int where every list is a typelist (can be obtained using get_taglist_of_int_from_text)
    :param skeleton: nx.graph of the datas' skeleton (meaning undirected edges between all statistically connected nodes)
    :param separating_sets: np.ndarray where each entry (i, j) is a list of separating sets for nodes i and j (obtained using causallearn - SkeletonDiscovery on data or get_separating_sets_using_true_skeleton for true skeleton).
    :param majority_rule_typed: bool, majority rule from typed PC Algo: if true use majority rule to orient forks, if false use naive rule to orient forks (true is recommended)
    :optional param node_names: list of string of the node names in the correct order (handed over by tpc or optained in get_true_skeleton) -> only needed for debugging can be empty
    :param + return: stat_tests of the DAG of type np.ndarray  (obtained in skeleton discovery and not changed here)
    :return: adjacency matrix of the DAG of type np.ndarray
    """  
    #step 1: make typed adjacency matrix for every type in tag
    adjacency_mats_list = get_adjacency_matrices_for_all_types(skeleton=skeleton, separating_sets=separating_sets, taglist=taglist, majority_rule_typed=majority_rule_typed)
    # for i in range(len(adjacency_mats_list)): #plot adjacency mat for all types: #not necessary -> comment out if needed
       # plot_dag_state(dag=adjacency_mats_list[i], var_names=node_names, types=taglist[i], step_number=1, experiment_step=f"type_{i}_adjacency_matrix") # XXX #prints intermediate dag states, comment out if necessary     

    #step 2: multiply entries in matrix by (number of total entries - changed entries of skeleton), so that the larger the number, the fewer changed entries 
    #TODO bessere(wissenschaftlichere) metric zum weighten finden - ggf. alpha einführen
    weight_mats_list = turn_adjacency_mats_to_weighted_prio_mats_depending_on_difference_to_skeleton(adjacency_mats_list=adjacency_mats_list, skeleton_mat=nx.adjacency_matrix(skeleton).todense())

    #step 3: sum weight matrices (so that you have a balance between many types and strong types), then make adjacency matrix for each stronger direction 
    adjacency_mat, priomat, weight_mat = get_adjacency_mat_from_tag_weight_mat_cycle_check(weight_mats_list=weight_mats_list, skeleton_mat=nx.adjacency_matrix(skeleton).todense(), taglist=taglist, node_names=node_names)
    # plot_dag_state(dag=adjacency_mat, var_names=node_names, types=taglist[0], step_number=3, experiment_step="weights_accumulated") # XXX #prints intermediate dag states, comment out if necessary
    print("weight matrix: \n", weight_mat)

    #step 4: meek without type consistency (same as majority alg)
    meek_majority_tag_without_typeconsistency(cpdag=adjacency_mat, tags=taglist, priomat=priomat, node_names=node_names)
    # plot_dag_state(dag=adjacency_mat, var_names=node_names, types=taglist[0], step_number=4, experiment_step="meek") # XXX #prints intermediate dag states, comment out if necessary

    print("priomat result: \n", priomat)
    print("adjacency_mat result: \n", adjacency_mat)
    print("stat_test result: \n", stat_tests)
    return adjacency_mat, stat_tests


# step 1 - infer skeleton (using the causallearn library):
def create_skeleton_using_causallearn(data, taglist, alpha, indep_test):
    """
    :param data: data (without node names) as panda dataframe
    :param taglist: list of list of int where every list is a typelist (can be obtained using get_taglist_of_int_from_text)
    :param alpha: float, desired significance level of independence tests (p_value) in (0,1), for skeleton discovery
    :param indep_test: string, the method used for independence test (as casuallearn ist used, the following are possible: ["fisherz", "mv_fisherz", "mc_fisherz", "kci", "chisq", "gsq"])
    :return skeleton: nx.graph of the datas' skeleton (meaning undirected edges between all statistically connected nodes)
    :return separating_sets: np.ndarray where each entry (i, j) is a list of separating sets for nodes i and j
    :return: stat_tests of the DAG of type np.ndarray (adjacency matrix of skeleton)
    """
    # get conditional independence test object and use it to create skeleton
    cit = CIT(data, method=indep_test)
    skeleton = SkeletonDiscovery.skeleton_discovery(data, alpha, indep_test=cit, stable=True,)
    # get seperating_sets from sekelton
    separating_sets = skeleton.sepset

    #get skeleton as nx.Graph
    skeleton.to_nx_skeleton()
    skeleton = skeleton.nx_skel

    node_ids = skeleton.number_of_nodes()

#    for (i, j) in list(combinations(range(node_ids), 2)):
#        print(f"seperating set of {i} and {j} is: {separating_sets[i][j]}")

    separating_sets = format_seperating_sets(separating_sets)
    
    print("seperating sets Formated \n") 
    for (i, j) in list(combinations(range(node_ids), 2)):
        print(f"seperating set of {i} and {j} is: {separating_sets[i][j]}") 

    stat_tests = nx.adjacency_matrix(skeleton).todense()

    skeleton = set_tags_as_int(skeleton, taglist)
    return skeleton, separating_sets, stat_tests

# step 1 - get correct skeleton from true graph (for assumption that skeleton is correct)
def get_true_skeleton(dataname : str, tags, data=None):
    """
    get the true skeleton (and trueish seperating sets), only works for forest example and the bnf dataset (you might have to download additional .bif files)
    
    :param dataname: string of the name of the desired data (i.e. asia, sprinkler, alarm, insurance, forest) -> https://www.bnlearn.com/bnrepository/ for more information
    :param tags: text of params and corresponding assigned tags with tags being seperated by comma in the form:
        Cloudy : Weather, Weather, NotWatering
        Sprinkler : Watering, NotWeather, Watering
        Rain : Weather, Weather, Watering
        Wet_Grass : Plant_Con, NotWeather, NotWatering  
    :optional param data: data (without node names) as panda dataframe (used wehere there is no bnlearn data (=forest))
    :return skeleton: nx.graph of the datas' skeleton (meaning undirected edges between all statistically connected nodes)
    :return separating_sets: np.ndarray where each entry (i, j) is a list of separating sets for nodes i and j
    :return stat_tests: of the DAG of type np.ndarray (adjacency matrix of skeleton)
    :return taglist: list of list of int where every list is a typelist (transformation of param tags)
    :return node_names: list of String of the column names of the dataset in the correct order
    """

    match dataname:
        # since we already get the true skeleton with the correct parameters for causallearn we do just that
        case "forest":
            if data is None:
                raise ValueError("since you are using non bnf data, you need give the function data ad np.ndarray")
            node_names = ["A","R","S","H","B","W","F"]
            taglist = get_taglist_of_int_from_text(tags, node_names)
            skeleton, separating_sets, stat_tests = create_skeleton_using_causallearn(data, taglist, alpha = 0.05, indep_test="fisherz")
            return skeleton, separating_sets, stat_tests, node_names, taglist
        #bnlearn directly supports those, we do not need a bif file
        case "asia" | 'sprinkler' :
            path = dataname
        #we search for a .bif file in Tag-PC-using-LLM/additionalData 
        case _:
            path = os.path.join("Tag-PC-using-LLM/additionalData", (dataname + ".bif"))
            if not (os.path.isfile(path)):
                raise FileNotFoundError(f"There is no true graph for {dataname}. Check your spelling or create a .bif file in {path}. (If you are lucky there might be one at https://www.bnlearn.com/bnrepository/).") 
        
    # get Dag from bnf data
    model = bn.import_DAG(path)
    adjacency_mat = model['adjmat']
    adjacency_mat = adjacency_mat.astype(int) #get int representation
    print("true graph from bnf: \n", adjacency_mat) #to debug
    node_names = adjacency_mat.columns.tolist()
    adjacency_mat = adjacency_mat.values #delete headers
    skeleton_adjacency_mat = get_undirect_graph(adjacency_mat)
    print("skeleton: \n", skeleton_adjacency_mat) #to debug
    stat_tests = skeleton_adjacency_mat

    #get skeleton as nx.Graph and set tags
    skeleton = nx.from_numpy_array(skeleton_adjacency_mat)
    taglist = get_taglist_of_int_from_text(tags=tags, node_names_ordered=node_names)
    skeleton = set_tags_as_int(skeleton, taglist)

    # Get separating sets
    separating_sets = get_separating_sets_using_true_skeleton(skeleton, adjacency_mat)

    node_ids = skeleton.number_of_nodes()
    for (i, j) in list(combinations(range(node_ids), 2)):
        logging.debug(f"seperating set of {i} {node_names[i]} and {j} {node_names[j]} is: {separating_sets[i][j]}")

    separating_sets = format_seperating_sets(separating_sets)

    print("seperating sets Formated \n") 
#    print("seperating sets: \n", separating_sets) 
    for (i, j) in list(combinations(range(node_ids), 2)):
        logging.debug(f"seperating set of {i} {node_names[i]} and {j} {node_names[j]} is: {separating_sets[i][j]}")

    return skeleton, separating_sets, stat_tests, node_names, taglist


# the rest of the code is taken from https://github.com/ServiceNow/typed-dag and strongly modified for tagged
# -------------------------------- algo-steps for Tag Majority -------------------------------------

#######
# This is the part where we orient all immoralities and two-type forks.
# The behavior used to orient t-edges depends on the chosen typed strategy (majority is recommended):
#   * Naive: orient as first encountered orientation
#   * Majority: orient using the most frequent orientation
#######

# step 2 - orient v-strucutes and two type forks

def orient_forks_majority_tag_naive_type(skeleton, sep_sets, priomat, taglist, node_names=[]):
    """
    Orient immoralities and two-type forks

    Type - Strategy: naive -- orient as first encountered
    Tag - Strategy: majority -- orient each edge taking all tags equaly into consideration
        
    :param skeleton: nx.graph of the datas' skeleton (meaning undirected edges between all statistically connected nodes)
    :param sep_sets: np.ndarray where each entry (i, j) is a list of separating sets for nodes i and j
    :param priomat: np.ndarray, priority matrix for type evidence.
    :param taglist: list of list of int where every list is a typelist 
    :optional param node_names: list of string of the node names in the correct order -> only needed for debugging, can be empty    
    :return dag: updated skeleton (nx.graph)
    :return priomat: uptated priomat (np.ndarray)
    """

    dag = skeleton.to_directed()
    node_ids = skeleton.nodes()

    if (len(node_names) < len(list(node_ids))): # make sure that code works if no node names are given -> if node names are empty just numerate nodes
        node_names = []
        for i in range(len(node_ids)):
            node_names.append(i)


    print("orient forks Tag Majority type naive")
    

    # Orient all immoralities and two-type forks
    # TODO: SERVICENOW DEBUG using shuffling to test hypothesis
    combos = list(combinations(node_ids, 2))
    shuffle(combos)
    # Iterate over all Edges
    for (i, j) in combos:
        adj_i = set(dag.successors(i))
        adj_j = set(dag.successors(j))

        # If j is a direct child of i
        if j in adj_i:
            continue

        # If i is a direct child of j
        if i in adj_j:
            continue

        # If i and j are directly connected, continue.
        if sep_sets[i][j] is None:
            continue

        common_k = adj_i & adj_j  # Common direct children of i and j
        for k in common_k:
            # Case: we have an immorality i -> k <- j
            if k not in sep_sets[i][j] and k in dag.successors(i) and k in dag.successors(j):
                # XXX: had to add the last two conditions in case k is no longer a child due to t-edge orientation
                print(
                    f"V: orient immorality {node_names[i]} (t{type_of_from_tag_all(dag, i)}) -> {node_names[k]} (t{type_of_from_tag_all(dag, k)}) <- {node_names[j]} (t{type_of_from_tag_all(dag, j)})"
                ) 
                # orient just v-structure ignore tags for now, but add strong entry in prio matrix (since v-strucutre comes from data it should not be overwritten easily later on)
                prio_weight = len(taglist)
                print("V-Structure_prio_weight: ", prio_weight)
                _orient_typeless_with_priomat(dag, i, k, priomat, prio_weight)
                _orient_typeless_with_priomat(dag, j, k, priomat, prio_weight)
                # print("priomat: \n", priomat)  
                

            # Case: we have an orientable two-type fork, i.e., it is not an immorality, so i <- k -> j 
            elif (
                ((_has_both_edges(dag, k, i)  and _has_both_edges(dag, k, j)) # we want to orient when both edges are unoriented
                or  (_has_both_edges(dag, k, i) and _has_directed_edge(dag, k, j)) # but also when already one edge points in the correct direction (going into k).
                or (_has_both_edges(dag, k, j) and _has_directed_edge(dag, k, i)) ) # For normal typed (or tag naive), type consistency would take care of that but here we would need meek first but since we are less confident in tag consistency here, we want to orient all possible two way forks before type consistency (changed for tagged naive)
                and ((prio_weight := amount_of_matching_types_for_two_way_fork(dag, i, j, k)) > 0) # check that there is a tag where typeof(i) == typeof(j) =! typeof(k)
            ):                
                print(
                    f"F: orient two-type fork {node_names[i]} (t{type_of_from_tag_all(dag, i)}) <- {node_names[k]} (t{type_of_from_tag_all(dag, k)}) -> {node_names[j]} (t{type_of_from_tag_all(dag, j)})"
                ) 
                # orient both edges if they dont already have a higher prio
                if (priomat[k][i] <= prio_weight and priomat[k][j] <= prio_weight):
                    _orient_typeless_with_priomat_cycle_check_nx(dag, k, i, priomat, prio_weight) 
                    _orient_typeless_with_priomat_cycle_check_nx(dag, k, j, priomat, prio_weight) # Since we do not use type consistency yet, we need to orient both i and j to k
                else: 
                    print("ERROR THIS SHOULD NOT HAPPEN")
        
        #search for additional forks that are already partily oriented bevor applying type consistency in t-propagation. This is necessary because we do not use type consistency yet (which would oriented this forks before). 
        pred_i = set(dag.predecessors(i))
        pred_j = set(dag.predecessors(j))
        common_pred_k = pred_i & pred_j  # Common direct predecessor of i and j (used for forks)
        for k in common_pred_k:
            # Case: we have an orientable two-type fork, since it is not an immorality, so i <- k -> j 
            if (
                ((_has_both_edges(dag, k, i)  and _has_both_edges(dag, k, j)) # we want to orient when both edges are unoriented
                or  (_has_both_edges(dag, k, i) and _has_directed_edge(dag, k, j)) # but also when already one edge points in the correct direction (going into k).
                or (_has_both_edges(dag, k, j) and _has_directed_edge(dag, k, i)) ) # For normal typed (or tag naive), type consistency would take care of that but here we would need meek first but since we are less confident in tag consistency here, we want to orient all possible two way forks before type consistency (changed for tagged naive)
                and ((prio_weight := amount_of_matching_types_for_two_way_fork(dag, i, j, k)) > 0) # check that there is a tag where typeof(i) == typeof(j) =! typeof(k)
            ):                
                print(
                    f"F: orient two-type fork to predecesor {node_names[i]} (t{type_of_from_tag_all(dag, i)}) <- {node_names[k]} (t{type_of_from_tag_all(dag, k)}) -> {node_names[j]} (t{type_of_from_tag_all(dag, j)})"
                ) 
                # orient both edges if they dont already have a higher prio
                if (priomat[k][i] <= prio_weight and priomat[k][j] <= prio_weight):
                    _orient_typeless_with_priomat_cycle_check_nx(dag, k, i, priomat, prio_weight) 
                    _orient_typeless_with_priomat_cycle_check_nx(dag, k, j, priomat, prio_weight) # Since we do not use type consistency yet, we need to orient both i and j to k
                else: 
                    print(f"Not enough Prio Evidence to orient {node_names[i]} (t{type_of_from_tag_all(dag, i)}) <- {node_names[k]} (t{type_of_from_tag_all(dag, k)}) -> {node_names[j]} (t{type_of_from_tag_all(dag, j)})")
                        

    print("orienting forks finished - current adj. matrix: \n", nx.adjacency_matrix(dag).todense()) 
    print("current priomat: \n", priomat) 
    return dag, priomat


def orient_forks_majority_tag_majority_top1(skeleton, sep_sets, priomat, taglist, node_names=[]): 
    """
    Orient immoralities and two-type forks -> adjusted naive typed since classic majority algo uses integraly type consistence

    Type - Strategy: majority -- orient using the most frequent orientation
    Particularity: Find two-type forks in tags and only orient for edge triples, where more tags support a specific direction than tags support the other direction.
    also do not use type consistency yet (in contrast to majority top in typing algo)
    Tag - Strategy: majority -- orient each edge taking all tags equaly into consideration
    
    :param skeleton: nx.graph of the datas' skeleton (meaning undirected edges between all statistically connected nodes)
    :param sep_sets: np.ndarray where each entry (i, j) is a list of separating sets for nodes i and j
    :param priomat: np.ndarray, priority matrix for type evidence.
    :param taglist: list of list of int where every list is a typelist 
    :optional param node_names: list of string of the node names in the correct order -> only needed for debugging, can be empty 
    :return dag: updated skeleton (nx.graph)
    :return priomat: uptated priomat (np.ndarray)
    """
    
    dag = skeleton.to_directed()
    node_ids = skeleton.nodes()
    
    if (len(node_names) < len(list(node_ids))): # make sure that code works if no node names are given -> if node names are empty just numerate nodes
        node_names = []
        for i in range(len(node_ids)):
            node_names.append(i)

    print("orient forks Tag Majority type majority \n")
    
    # Orient all immoralities and two-type forks
    # XXX: SERVICENOW DEBUG using shuffling to test hypothesis (very helpful actually thank you for including this <3)
    combos = list(combinations(node_ids, 2))
    shuffle(combos)
    # list for saving 2 way fork every tag: with the following entries [tag, i, k, j] with tag being the amound of tag with this tow way fork and the two way fork consisitng of the nodes: i <- k -> j
    two_way_evidence = []

    # Iterate over all Edges
    for (i, j) in combos:
        adj_i = set(dag.successors(i))
        adj_j = set(dag.successors(j))

        # If j is a direct child of i
        if j in adj_i:
            continue

        # If i is a direct child of j
        if i in adj_j:
            continue

        # If i and j are directly connected, continue.
        if sep_sets[i][j] is None:
            continue

        common_k = adj_i & adj_j  # Common direct children of i and j
        for k in common_k:
            # Case: we have an immorality i -> k <- j
            if k not in sep_sets[i][j] and k in dag.successors(i) and k in dag.successors(j):
                # XXX: had to add the last two conditions in case k is no longer a child due to new orientation
                print(
                    f"V: orient immorality {node_names[i]} (t{type_of_from_tag_all(dag, i)}) -> {node_names[k]} (t{type_of_from_tag_all(dag, k)}) <- {node_names[j]} (t{type_of_from_tag_all(dag, j)})"
                ) 
                # orient just v-structure ignore tags for now, but add strong entry in prio matrix (since v-strucutre comes from data it should not be overwritten easily later on)
                prio_weight = len(taglist)
                _orient_typeless_with_priomat(dag, i, k, priomat, prio_weight)
                _orient_typeless_with_priomat(dag, j, k, priomat, prio_weight)
                # print("priomat after orienten v-strucutre: \n", priomat)   #still comment out
                

            # Case: we have an orientable two-type fork, i.e., it is not an immorality, so i <- k -> j
            elif (
                ((_has_both_edges(dag, k, i)  and _has_both_edges(dag, k, j)) # we want to orient when both edges are unoriented
                or  (_has_both_edges(dag, k, i) and _has_directed_edge(dag, k, j)) # but also when already one edge points in the correct direction (going into k).
                or (_has_both_edges(dag, k, j) and _has_directed_edge(dag, k, i)) ) # For normal typed (or tag naive), type consistency would take care of that but here we would need meek first but since we are less confident in tag consistency here, we want to orient all possible two way forks before type consistency (changed for tagged naive)
                and ((prio_weight := amount_of_matching_types_for_two_way_fork(dag, i, j, k)) > 0) # check that there is a tag where typeof(i) == typeof(j) =! typeof(k)
            ):
                print(
                    f"F: found two-type fork {node_names[i]} (t{type_of_from_tag_all(dag, i)}) <- {node_names[k]} (t{type_of_from_tag_all(dag, k)}) -> {node_names[j]} (t{type_of_from_tag_all(dag, j)} saving for orienting)"
                ) 
                # safe two way fork just for now
                two_way_evidence.append([prio_weight, i, k, j])

       #search for additional forks that are already partily oriented bevor applying type consistency in t-propagation This is necessary because we do not use type consistency yet (which would oriented this forks before). 
        pred_i = set(dag.predecessors(i))
        pred_j = set(dag.predecessors(j))
        common_pred_k = pred_i & pred_j  # Common direct predecessor of i and j (used for forks)
        for k in common_pred_k:
            # Case: we have an orientable two-type fork, since it is not an immorality, so i <- k -> j 
            if (
                ((_has_both_edges(dag, k, i)  and _has_both_edges(dag, k, j)) # we want to orient when both edges are unoriented
                or  (_has_both_edges(dag, k, i) and _has_directed_edge(dag, k, j)) # but also when already one edge points in the correct direction (going into k).
                or (_has_both_edges(dag, k, j) and _has_directed_edge(dag, k, i)) ) # For normal typed (or tag naive), type consistency would take care of that but here we would need meek first but since we are less confident in tag consistency here, we want to orient all possible two way forks before type consistency (changed for tagged naive)
                and ((prio_weight := amount_of_matching_types_for_two_way_fork(dag, i, j, k)) > 0) # check that there is a tag where typeof(i) == typeof(j) =! typeof(k)
            ):                
                # save fork if we have not encountered it already
                if not (exists_entry_in_forkevidencelist(two_way_evidence, prio_weight, i, k, j)):
                    print(
                        f"F: found two-type fork to predecesor {node_names[i]} (t{type_of_from_tag_all(dag, i)}) <- {node_names[k]} (t{type_of_from_tag_all(dag, k)}) -> {node_names[j]} (t{type_of_from_tag_all(dag, j)} saving for orienting)"
                    ) 
                    two_way_evidence.append([prio_weight, i, k, j])
                else: 
                    # print(f"two way fork {node_names[i]} (t{type_of_from_tag_all(dag, i)}) <- {node_names[k]} (t{type_of_from_tag_all(dag, k)}) -> {node_names[j]} (t{type_of_from_tag_all(dag, j)}) found but already saved") #comment back in if necessary
                    continue

    # deleting all Forks that are impossible due to later oriented V-Structures
    print("Delete 2-Way-Fork that are now impossible due to later oriented Immoralites")
    two_way_evidence = [two_way_fork for two_way_fork in two_way_evidence if not fork_clashes_immorality(dag, two_way_fork)]

    # now orienting two way forks 
    print("\nall evidence collected, now orienting two way forks:")

    # iterate over all two way forks i <- k -> j to search for clashes
    for index, two_way_fork in enumerate(two_way_evidence):
        prio_weight, i, k, j = two_way_fork
        has_highest_prio_weight = True
        clash_found = False

        for other_index, other_fork in enumerate(two_way_evidence):
            if index != other_index:
                other_prio_weight, other_i, other_k, other_j = other_fork

                # if any other two_way_fork has an opposed edge -> we have a clash meaning different oriented forks and have to decide which one to orient 
                if are_forks_clashing(two_way_fork, other_fork):
                    clash_found = True
                    # fork with higher (or same) prio_weight found -> we will not orient current fork, but wait in loop for the higher weight fork to orient it then (if other fork has no conflict that is)
                    # if there are only clashing forks with the same priority do not orient any of them.
                    if other_prio_weight >= prio_weight:
                        has_highest_prio_weight = False
                        break             
        
        if clash_found:
            if has_highest_prio_weight:
            # still check prioweight for (redundant) safety (in _orienting it will also be checked if edge was directed in other way previously (=checked for class with immorality))
                if (priomat[k][i] <= prio_weight and priomat[k][j] <= prio_weight):
                    print(
                        f"Clash found -> orienting two-type fork {node_names[i]} (t{type_of_from_tag_all(dag, i)}) <- {node_names[k]} (t{type_of_from_tag_all(dag, k)}) -> {node_names[j]} (t{type_of_from_tag_all(dag, j)}. despite of clash with at least fork {node_names[other_i]} <- {node_names[other_k]} -> {node_names[other_j]} with prio: {other_prio_weight} < now orienting fork: {prio_weight}. )"
                    ) 
                    _orient_typeless_with_priomat_cycle_check_nx(dag, k, i, priomat, prio_weight)
                    _orient_typeless_with_priomat_cycle_check_nx(dag, k, j, priomat, prio_weight)
            else:
                print(f"Clash found -> No orientation of this fork: {node_names[i]} <- {node_names[k]} -> {node_names[j]} with prio: {prio_weight} due to a clashing fork {node_names[other_i]} <- {node_names[other_k]} -> {node_names[other_j]} with higher prio: {other_prio_weight}.")
        
        #k is not in i or j in any other two_way_fork, meaning we have no conflict and can orient fork safely
        else:
            # still check prioweight for (redundant) safety (in _orienting it will also be checked if edge was directed in other way previously (=checked for class with immorality))
            if (priomat[k][i] < prio_weight and priomat[k][j] < prio_weight):
                print(
                    f"orienting two-type fork {node_names[i]} (t{type_of_from_tag_all(dag, i)}) <- {node_names[k]} (t{type_of_from_tag_all(dag, k)}) -> {node_names[j]} (t{type_of_from_tag_all(dag, j)})"
                ) 
                _orient_typeless_with_priomat_cycle_check_nx(dag, k, i, priomat, prio_weight)
                _orient_typeless_with_priomat_cycle_check_nx(dag, k, j, priomat, prio_weight)
            # orient v-structures that are already partily oriented (and therefore have a higher priomat entry) (actually necessary)
            else:
                if (priomat[k][i] < prio_weight and _has_directed_edge(dag, k, j)): #case k-i unoriented, k->j oriented
                    print(
                        f"orienting part of two-type fork {node_names[i]} (t{type_of_from_tag_all(dag, i)}) <- {node_names[k]} (t{type_of_from_tag_all(dag, k)}) with already oriented: {node_names[k]} -> {node_names[j]} (t{type_of_from_tag_all(dag, j)} (with oriented prio {node_names[k]} -> {node_names[j]}: {priomat[k][j]} and new oriented lower prio {node_names[k]} -> {node_names[i]}: {prio_weight})."
                    ) 
                    _orient_typeless_with_priomat_cycle_check_nx(dag, k, i, priomat, prio_weight) #orient only k->i and only give it prioweight as k->j already has stronger evidence
                elif (priomat[k][j] < prio_weight and _has_directed_edge(dag, k, i)): #case k-j unoriented, k->i oriented
                    print(
                        f"orienting part of two-type fork {node_names[j]} (t{type_of_from_tag_all(dag, j)}) <- {node_names[k]} (t{type_of_from_tag_all(dag, k)}) with already oriented: {node_names[k]} -> {node_names[i]} (t{type_of_from_tag_all(dag, i)} (with oriented prio {node_names[k]} -> {node_names[i]}: {priomat[k][i]} and new oriented lower prio {node_names[k]} -> {node_names[i]}: {prio_weight})."
                    ) 
                    _orient_typeless_with_priomat_cycle_check_nx(dag, k, j, priomat, prio_weight) #orient only k->j and only give it prioweight as k->i already has stronger evidence
                else:
                    print(
                        f"not orienting two-type fork {node_names[i]} (t{type_of_from_tag_all(dag, i)}) <- {node_names[k]} (t{type_of_from_tag_all(dag, k)}) -> {node_names[j]} (t{type_of_from_tag_all(dag, j)} because edges are already oriented with better evidence)"
                    ) 


    print("orienting forks finished - current adj. matrix: \n", nx.adjacency_matrix(dag).todense()) 
    print("current priomat: \n", priomat) 
    return dag, priomat


#step 3 - t-propagation:

# taken and adjusted from https://github.com/ServiceNow/typed-dag/blob/main/typed_pc/tmec.py changed for tag-algo
def meek_majority_tag_without_typeconsistency(cpdag: np.ndarray, tags: list, priomat : np.matrix, node_names=[], iter_max: int = 100) -> Tuple[np.ndarray, np.matrix]:
    """
    Apply the Meek algorithm with the type consistency as described in Section 5 (from the CPDAG).

    :param cpdag: np.ndarray, adjacency matrix of the CPDAG
    :param tags: list of list of int
    :param priomat: np.ndarray, priority matrix for type evidence.
    :optional param node_names: list of string of the node names in the correct order -> only needed for debugging, can be empty 
    :optional param iter_max: The maximum number of iterations. If reached, an exception will be raised.
    :return G: updated cpdag (np.ndarray)
    :return priomat: uptated priomat (np.ndarray)
    """
    print("typed Meek without type consistency \n")
    # priomat checks are mostly redundant here, since a,b is undirected and should therefore have prioweight 0

    n_nodes = cpdag.shape[0]
    G = np.copy(cpdag)

    if (len(node_names) < n_nodes): # make sure that code works if no node names are given -> if node names are empty just numerate nodes
        node_names = []
        for i in range(n_nodes):
            node_names.append(i)

    # repeat until the graph is not changed by the algorithm
    # or too high number of iteration
    previous_G = np.copy(G)
    i = 0
    while True and i < iter_max: # break when dag does not change in iteration
        """
        Normaly we would use type consistency with Meeks rule here, but since we want to get as much information without type consistency for now, we just append meeks rule with a 2-way-fork rule

        """

        i += 1
        # Skip Type Consistency - it will have its turn in the next step

        # Apply Meek's rules (R1, R2, R3, R4) and the two-type fork rule (R5)
        for a, b, c in permutations(range(n_nodes), 3):
            # Orient any undirected edge a - b as a -> b if any of the following rules is satisfied:
            if G[a, b] != 1 or G[b, a] != 1:
                # Edge a - b is already oriented
                continue

            # R1: c -> a - b, b-/-c ==> a -> b
            if G[a, c] == 0 and G[c, a] == 1 and G[b, c] == 0 and G[c, b] == 0 and ((prioweight := min(priomat[a,c], priomat[c, a], priomat[b,c], priomat[c,b])) > max(priomat[b,a], priomat[a, b])): #check that prio evidence is stronger than the to orient edge
                print(f"meek R1: {node_names[c]} -> {node_names[a]} - {node_names[b]}: orient to  {node_names[a]} -> {node_names[b]}")  
                _orient_typeless_with_priomat_cycle_check_adjmat(G,a,b,priomat,prioweight) #delete edge b-a and update prio weight while checking for cycles
            # R2: a -> c -> b and a - b ==> a -> b
            elif G[a, c] == 1 and G[c, a] == 0 and G[b, c] == 0 and G[c, b] == 1 and ((prioweight := min(priomat[a,c], priomat[c, a], priomat[b,c], priomat[c,b])) > max(priomat[b,a], priomat[a, b])): #check that prio evidence is stronger than the to orient edge
                print(f"meek R2: {node_names[a]} -> {node_names[c]} -> {node_names[b]} and {node_names[a]} - {node_names[b]}: orient to {node_names[a]} -> {node_names[b]}")  
                _orient_typeless_with_priomat_cycle_check_adjmat(G,a,b,priomat,prioweight) #delete edge b-a and update prio weight while checking for cycles
            # XXX no R5 since all possible 2-Way Forks where already tested
            else:

                for d in range(n_nodes):
                    if d != a and d != b and d != c:
                        # R3: a - c -> b and a - d -> b and a - b and c -/- d ==> a -> b
                        if (
                            G[a, c] == 1
                            and G[c, a] == 1
                            and G[b, c] == 0
                            and G[c, b] == 1
                            and G[a, d] == 1
                            and G[d, a] == 1
                            and G[b, d] == 0
                            and G[d, b] == 1
                            and G[c, d] == 0
                            and G[d, c] == 0
                            and ((prioweight := min(priomat[b,c], priomat[c,b], priomat[b,d], priomat[d,b], priomat[c,d], priomat[d,c])) > max(priomat[b,a], priomat[a,b])) #check that prio evidence (of oriented edges) is stronger than the to orient edge 
                        ):
                            print(f"meek R3: {node_names[a]} - {node_names[c]} -> {node_names[b]} and {node_names[a]} - {node_names[d]} -> {node_names[b]} and {node_names[c]} -/- {node_names[d]} orient to {node_names[a]} -> {node_names[b]} ")  
                            _orient_typeless_with_priomat_cycle_check_adjmat(G,a,b,priomat,prioweight) #delete edge b-a and update prio weight while checking for cycles
                        # R4: a - d -> c -> b and a - - c ==> a -> b
                        elif (
                            G[a, d] == 1
                            and G[d, a] == 1
                            and G[c, d] == 0
                            and G[d, c] == 1
                            and G[b, c] == 0
                            and G[c, b] == 1
                            and (G[a, c] == 1 or G[c, a] == 1)
                            and ((prioweight := min(priomat[c,d], priomat[d,c], priomat[b,c], priomat[c,b])) > max(priomat[b,a], priomat[a,b])) #check that prio evidence (of oriented edges) is stronger than the to orient edge 
                        ):
                            print(f"meek R4: {node_names[a]} - {node_names[d]} -> {node_names[c]} -> {node_names[b]} and {node_names[a]} -> {node_names[c]} or {node_names[c]} -> {node_names[a]} orient to {node_names[a]} -> {node_names[b]} ")  
                            _orient_typeless_with_priomat_cycle_check_adjmat(G,a,b,priomat,prioweight) #delete edge b-a and update prio weight while checking for cycles

        if (previous_G == G).all():
            break
        if i >= iter_max:
            raise Exception(f"Typed Meek is stucked. More than {iter_max} iterations.")

        previous_G = np.copy(G)

    print("typed meek without type consistency finished - current adjacency matrix: \n", G) 
    print("current priomat: \n", priomat) 
    return G, priomat

# step 4 type-consistent t-propagation

# taken and adjusted from https://github.com/ServiceNow/typed-dag/blob/main/typed_pc/tmec.py
def typed_meek_majority_tag_with_consistency(cpdag: np.ndarray, tags: list, priomat : np.matrix, majority_rule_typed : bool = True, node_names=[], iter_max: int = 100) -> Tuple[np.ndarray, np.ndarray]: 
    """
    Apply the Meek algorithm with the type consistency as described in Section 5 (from the CPDAG).

    :param cpdag: np.ndarray, adjacency matrix of the CPDAG
    :param tags: list of list of int
    :param priomat: np.ndarray, priority matrix for type evidence.
    :optional param node_names: list of string of the node names in the correct order -> only needed for debugging, can be empty 
    :optional param iter_max: The maximum number of iterations. If reached, an exception will be raised.
    :return G: updated cpdag (np.ndarray)
    :return priomat: uptated priomat (np.ndarray)
    """
    print("\ntyped Meek with type consistency \n")
    # priomat checks are mostly redundant here, since a,b is undirected and should therefore have prioweight 0 

    n_nodes = cpdag.shape[0]

    G = np.copy(cpdag)

    if (len(node_names) < n_nodes): # make sure that code works if no node names are given -> if node names are empty just numerate nodes
        node_names = []
        for i in range(n_nodes):
            node_names.append(i)

    # repeat until the graph is not changed by the algorithm
    # or too high number of iteration
    previous_G = np.copy(G)
    i = 0
    while True and i < iter_max:
        """
        Each iteration is broken down into three stages:
        1) Get Matrix where each for each tag type consistency is applied (according to majority_rule)
        2) Orient edges where tag consitency evidence (from 1) is stronger than current evidence
        3) Apply the Meek rules (only one per iteration) to orient the remaining edges.

        Note: Edges are oriented one-by-one in step 3, but these orientations will be propagated to the whole
            t-edge once we return to step (1).

        """
        i += 1
        # Step 1: get adjacency matrix where each entry is the amount of tags (with each tag being one typelist) for which this edge will be oriented in this direction for the current Graph
        # depending of type majority rule, we either get this adhjacency matrix via first encountered type consistency in each type (=naive) or via the majority of t-edges (considering priomat) (=majority)
        majoritymat =  get_majority_tag_matrix_using_priomat_type_majority(G, taglist=tags, priomat=priomat) if majority_rule_typed else get_majority_tag_matrix(G, taglist=tags) 
        print(f"majority mat on t-propagation step: {i}: \n", majoritymat) 
        print(f"priomat before type consistency: \n", priomat) 
        print(f"adjacency mat before type consistency: \n", G) 

        # Step 2: Orient all Edges where there is a higher type consisitency evidence for the current Graph (entry in majoritymat) than already priority in priomat
        G, priomat = orient_tagged_dag_according_majority_tag_matrix_using_prio_mat_cycle_check(G, tags, majoritymat, priomat, node_names=node_names)

        # Apply Meek's rules (R1, R2, R3, R4) and the two-type fork rule (R5)
        for a, b, c in permutations(range(n_nodes), 3):
            # Orient any undirected edge a - b as a -> b if any of the following rules is satisfied:
            if G[a, b] != 1 or G[b, a] != 1:
                # Edge a - b is already oriented
                continue

            # R1: c -> a - b ==> a -> b
            if G[a, c] == 0 and G[c, a] == 1 and G[b, c] == 0 and G[c, b] == 0 and ((prioweight := min(priomat[a,c], priomat[c, a], priomat[b,c], priomat[c,b])) > max(priomat[b,a], priomat[a, b])): #check that prio evidence is stronger than the to orient edge
                print(f"meek R1: {node_names[c]} -> {node_names[a]} - {node_names[b]}: orient to  {node_names[a]} -> {node_names[b]}")  
                _orient_typeless_with_priomat_cycle_check_adjmat(G,a,b,priomat,prioweight) #delete edge b-a and update prio weight while checking for cycles
            # R2: a -> c -> b and a - b ==> a -> b
            elif G[a, c] == 1 and G[c, a] == 0 and G[b, c] == 0 and G[c, b] == 1 and ((prioweight := min(priomat[a,c], priomat[c, a], priomat[b,c], priomat[c,b])) > max(priomat[b,a], priomat[a, b])): #check that prio evidence is stronger than the to orient edge
                print(f"meek R2: {node_names[a]} -> {node_names[c]} -> {node_names[b]} and {node_names[a]} - {node_names[b]}: orient to {node_names[a]} -> {node_names[b]}")  
                _orient_typeless_with_priomat_cycle_check_adjmat(G,a,b,priomat,prioweight) #delete edge b-a and update prio weight while checking for cycles
            # No R5 because why should I? R 5: b - a - c and a-/-c and t(c) = t(b) ==> a -> b and a -> c (two-type fork)
            else:
                for d in range(n_nodes):
                    if d != a and d != b and d != c:
                        # R3: a - c -> b and a - d -> b, c -/- d ==> a -> b
                        if (
                            G[a, c] == 1
                            and G[c, a] == 1
                            and G[b, c] == 0
                            and G[c, b] == 1
                            and G[a, d] == 1
                            and G[d, a] == 1
                            and G[b, d] == 0
                            and G[d, b] == 1
                            and G[c, d] == 0
                            and G[d, c] == 0
                            and ((prioweight := min(priomat[b,c], priomat[c,b], priomat[b,d], priomat[d,b], priomat[c,d], priomat[d,c])) > max(priomat[b,a], priomat[a,b])) #check that prio evidence (of oriented edges) is stronger than the to orient edge 
                        ):
                            print(f"meek R3: {node_names[a]} - {node_names[c]} -> {node_names[b]} and {node_names[a]} - {node_names[d]} -> {node_names[b]} and {node_names[c]} -/- {node_names[d]} orient to {node_names[a]} -> {node_names[b]} ")  
                            _orient_typeless_with_priomat_cycle_check_adjmat(G,a,b,priomat,prioweight) #delete edge b-a and update prio weight while checking for cycles
                        # R4: a - d -> c -> b and a - - c ==> a -> b
                        elif (
                            G[a, d] == 1
                            and G[d, a] == 1
                            and G[c, d] == 0
                            and G[d, c] == 1
                            and G[b, c] == 0
                            and G[c, b] == 1
                            and (G[a, c] == 1 or G[c, a] == 1)
                            and ((prioweight := min(priomat[c,d], priomat[d,c], priomat[b,c], priomat[c,b])) > max(priomat[b,a], priomat[a,b])) #check that prio evidence (of oriented edges) is stronger than the to orient edge 
                        ):
                            print(f"meek R4: {node_names[a]} - {node_names[d]} -> {node_names[c]} -> {node_names[b]} and {node_names[a]} -> {node_names[c]} or {node_names[c]} -> {node_names[a]} orient to {node_names[a]} -> {node_names[b]} ")  
                            _orient_typeless_with_priomat_cycle_check_adjmat(G,a,b,priomat,prioweight) #delete edge b-a and update prio weight while checking for cycles

        if (previous_G == G).all():
            break
        if i >= iter_max:
            raise Exception(f"Typed Meek is stucked. More than {iter_max} iterations.")

        previous_G = np.copy(G)
        print("current adj. matrix: \n", G) 

    return G, priomat

# -------------------------------- algo-steps for weighted tag -------------------------------------

# tag_weight_step 1 type skeleton for every type and save adjacency matrices in list
def get_adjacency_matrices_for_all_types(skeleton, separating_sets, taglist, majority_rule_typed):
    """
    :param skeleton: nx.graph of the datas' skeleton (meaning undirected edges between all statistically connected nodes)
    :param separating_sets: np.ndarray where each entry (i, j) is a list of separating sets for nodes i and j
    :param taglist: list of list of int where every list is a typelist 
    :param majority_rule_typed: bool, majority rule from typed PC Algo: if true use majority rule to orient forks, if false use naive rule to orient forks (true is recommended)
    :return adjacency_mats: list of np.ndarray each containing the typed adjacency matrix for one type
    """

    # make adjacency mat for every type
    print("creating adjacency matrices for every type...\n")
    adjacency_mats = []
    current_type = 0
    for typelist in taglist:
        print("current_type: ", current_type) 
        
        #run typed PC to get adjacency result mat for all types
        current_skeleton = skeleton.copy()
        current_adjacency_mat = typed_pc_from_true_skeleton(skeleton=current_skeleton, separating_sets=separating_sets, typelist=typelist, current_tag_number=current_type, majority_rule_typed=majority_rule_typed)
        adjacency_mats.append(current_adjacency_mat)
        current_type += 1

    # generate weight matrices form the adjacency matrix by multiplying difference to skeleton -> more differences mean lower numbers
    weight_mats = []
    skeleton_mat = nx.adjacency_matrix(skeleton).todense()
    for adjacency_mat in adjacency_mats:
        weight_mats.append(adjacency_mat)

    return adjacency_mats

# tag_weight_step 2: multiply entries in matrix by (number of total entries - changed entries of skeleton), so that the larger the number, the fewer changed entries
def turn_adjacency_mats_to_weighted_prio_mats_depending_on_difference_to_skeleton(adjacency_mats_list, skeleton_mat):
    """
    :param adjacency_mats: list of np.ndarray each containing the typed adjacency matrix for one type
    :param skeleton_mat: np.ndarray - adjacency matrix of the datas' skeleton (meaning undirected edges between all statistically connected nodes)
    :return weight_mats: list of np.ndarray each containing the former adjacency matrix for one type, that now contains weighted entries representing how much the graph differs from the skeleton
    """

    # generate weight matrices from the adjacency matrix by multiplying difference to skeleton -> more differences mean lower numbers
    print("\nturning adjacency matrices into weight matrices....\n")
    weight_mats = []
    amount_elems = skeleton_mat.size
    for adjacency_mat in adjacency_mats_list:
        diff = get_amount_difference_two_mats(adjacency_mat=adjacency_mat, skeleton_mat=skeleton_mat)
        weight = (amount_elems - diff) #TODO wissenschaftlicher machen - alpha einführen
        weight_mat = adjacency_mat * weight #multiplied every entry by weight (meaning 1s become weight and 0s dont change)
        weight_mats.append(weight_mat)
    return weight_mats


# step 3: sum weight matrices (so that you have a balance between many and types and strong types), then make adjacency matrix for each stronger direction, while checking for cycles 
def get_adjacency_mat_from_tag_weight_mat_cycle_check(weight_mats_list, skeleton_mat, taglist, node_names=[]): #node names only needed for debugging (but very useful for that xD)):
    """
    this version iterates from highest to lowest value in weight_mats, it then checks for cycles before orienting, if cycles are found no orientation will be chosen
    :param weight_mats_list: list of np.ndarray each containing the former adjacency matrix for one type, that now contains weighted entries representing how much the graph differs from the skeleton
    :param skeleton_mat: np.ndarray - adjacency matrix of the datas' skeleton (meaning undirected edges between all statistically connected nodes)
    :optional param node_names: list of string of the node names in the correct order -> only needed for debugging, can be empty
    :return adjacency_mat: np.ndarray, adjacency matrix for combined graph, derived of the weight_mats
    :return priomat: np.ndarray, priority matrix for type evidence, derived of weight_mats but evidence goes in both direction (meaning for a directed edge the entries for both direction are the same)
    :return weight_mat_acc: single np.ndarray that is the sum of all weight_mats in weight_mats_list
    """
    print("\ngetting adjacency matrix from weight matrices.....\n")

    if len(node_names) < skeleton_mat[0].size: # make sure that code works if no node names are given -> if node names are empty just numerate nodes (for debugging)
        node_names = []
        for i in range(skeleton_mat.size):
            node_names.append(i)
        print(node_names)
    
    
    # add weight mats
    weight_mat_acc = np.zeros_like(weight_mats_list[0])
    for weight_mat in weight_mats_list:
        weight_mat_acc = weight_mat_acc + weight_mat
    
    # create priomat
    print("accumulated weight mats: \n", weight_mat_acc)
    priomat = weight_mat_acc
    max_weight = len(skeleton_mat)*len(skeleton_mat[0])*len(taglist) + 1 # make priomat absolutely unreadable (max number imageniable (amount of nodes for every tag))
    priomat = assign_deleted_edges_in_priomat_evidence(dag=skeleton_mat, priomat=weight_mat_acc, max_weight=max_weight) # init deleted edges with highest number in priomat for consistency


    # Get the indices sorted by the values in weight_mat_acc in descending order
    indices = np.dstack(np.unravel_index(np.arange(weight_mat_acc.size), weight_mat_acc.shape))[0]#flattens weight_mat_acc -> resorts by entries of highest value -> reformates into array of array containing the coordinates a,b of entry
    sorted_indices = sorted(indices, key=lambda idx: (-weight_mat_acc[idx[0], idx[1]], weight_mat_acc[idx[1], idx[0]])) # resort so that for same value in direction i,j, entrys are priotized for lower opposite entries (j,i)
    
    # get adjacency mat by setting 1s to dominant edges (entry in weight mat is bigger than for other direction)
    adjacency_mat = np.copy(skeleton_mat)
    # iterate over the sorted highest entries in weight_mat_acc and thereby iterating over the complete adjacency matrix
    for idx in sorted_indices:
        i, j = idx
        if (skeleton_mat[i,j] == 1): # entry is 1 (meaining opposite entry is also 1) -> we have undirected edge in skeleton
            print(f"{idx} first entry: {weight_mat_acc[i][j]}, other entry: {weight_mat_acc[j][i]}")
            # -> direct by stronger direction in weight_mat, if there is no stronger edge -> keep undirected
            if (weight_mat_acc[i][j] > weight_mat_acc[j][i]): # direct i -> j
                # print(f"directing graph from {node_names[i]} -> {node_names[j]}, because of prioweight {weight_mat_acc[i][j]} (> opposite direction: {weight_mat_acc[j][i]})")
                _orient_typeless_with_priomat_cycle_check_adjmat(adjacency_mat,i,j,priomat,weight_mat_acc[i][j]) # orient i -> j wit prioweight = weightmat[i->j]

    return adjacency_mat, priomat, weight_mat_acc


# ---------------------unused----------------------------------
#alt-step 3: sum weight matrices (so that you have a balance between many and types and strong types), then make adjacency matrix for each stronger direction 
def get_adjacency_mat_from_tag_weight_mat(weight_mats_list, skeleton_mat, taglist, node_names=[]): #node names only needed for debugging (but very useful for that xD)):
    """
    :param weight_mats_list: list of np.ndarray each containing the former adjacency matrix for one type, that now contains weighted entries representing how much the graph differs from the skeleton
    :param skeleton_mat: np.ndarray - adjacency matrix of the datas' skeleton (meaning undirected edges between all statistically connected nodes)
    :optional param node_names: list of string of the node names in the correct order -> only needed for debugging, can be empty
    :return adjacency_mat: np.ndarray, adjacency matrix for combined graph, derived of the weight_mats
    :return priomat: np.ndarray, priority matrix for type evidence, derived of weight_mats but evidence goes in both direction (meaning for a directed edge the entries for both direction are the same)
    :return weight_mat_acc: single np.ndarray that is the sum of all weight_mats in weight_mats_list
    """
    print("\ngetting adjacency matrix from weight matrices.....\n")

    if len(node_names) < skeleton_mat[0].size: # make sure that code works if no node names are given -> if node names are empty just numerate nodes (for debugging)
        node_names = []
        for i in range(skeleton_mat.size):
            node_names.append(i)
    print(node_names)
    
    # add weight mats
    weight_mat_acc = np.zeros_like(weight_mats_list[0])
    for weight_mat in weight_mats_list:
        weight_mat_acc = weight_mat_acc + weight_mat
    
    print("accumulated weight mats: \n", weight_mat_acc)
    priomat = weight_mat_acc
    max_weight = len(skeleton_mat)*len(skeleton_mat[0])*len(taglist) + 1 # make priomat absolutely unreadable (max number imageniable (amount of nodes for every tag))
    priomat = assign_deleted_edges_in_priomat_evidence(dag=skeleton_mat, priomat=weight_mat_acc, max_weight=max_weight) # init deleted edges with highest number in priomat for consistency

    # get adjacency mat by setting 1s to dominant edges (entry in weight mat is bigger than for other direction)
    adjacency_mat = np.copy(skeleton_mat)
    # iterate over (lower) triangular matrix of skeleton
    for i in range(len(skeleton_mat)):
        for j in range(0, i):
            if (skeleton_mat[i,j] == 1): # entry is 1 (meaining opposite entry is also 1) -> we have undirected edge in skeleton
                # -> direct by stronger direction in weight_mat, if there is no stronger edge -> keep undirected
                if (weight_mat_acc[i][j] > weight_mat_acc[j][i]): # direct i -> j
                    print(f"directing graph from {node_names[i]} -> {node_names[j]}, because of prioweight {weight_mat_acc[i][j]} (> opposite direction: {weight_mat_acc[j][i]})")
                    adjacency_mat[j][i] = 0
                    priomat[j][i] = weight_mat_acc[i][j] #set priomat evidence not needed?
                if (weight_mat_acc[j][i] > weight_mat_acc[i][j]): # direct j -> i
                    print(f"directing graph from {node_names[j]} -> {node_names[i]}, because of prioweight {weight_mat_acc[j][i]} (> opposite direction: {weight_mat_acc[i][j]})")
                    adjacency_mat[i][j] = 0
                    priomat[i][j] = weight_mat_acc[j][i]

    return adjacency_mat, priomat, weight_mat_acc
