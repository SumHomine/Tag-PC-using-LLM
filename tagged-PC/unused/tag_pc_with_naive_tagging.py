import logging
import networkx as nx
import numpy as np

from collections import Counter, defaultdict
from itertools import combinations, permutations
from typing import Tuple
from random import shuffle

from causallearn.utils.PCUtils import SkeletonDiscovery
from causallearn.utils.cit import *

import bnlearn as bn

from tag_pc_utils import exists_entry_in_forkevidencelist, get_taglist_of_int_from_text, get_typelist_from_text, get_undirect_graph, are_forks_clashing, set_types_as_int, set_tags_as_int, format_seperating_sets, _has_both_edges, _has_any_edge, _has_directed_edge, _orient_tagged, _orient_typeless_with_priomat, type_of_from_tag_single, type_of_from_tag_all, type_of_from_tag_all_from_taglist, _update_tedge_orientation, _orient_tedges, get_majority_tag_matrix, get_majority_tag_matrix_using_priomat_type_majority, orient_tagged_dag_after_majority_tag_matix_using_prio_mat, amount_of_matching_types_for_two_way_fork, amount_of_matching_types_for_two_way_fork_from_taglist, get_list_of_matching_types_for_two_way_fork, get_priomat_from_skeleton, get_separating_sets_using_true_skeleton
from visualization import plot_dag_state
# from test_tag_pc import load_skeleton, save_skeleton 

def tpc (data, tags, node_names, alpha=0.05, indep_test="fisherz", majority_rule_tagged=False, majority_rule_typed=True):
    """
    :param data: np.ndarray of datafile. instruction to get there are in run_tag
    :param tags: text of params and corresponding assigned tags with tags being seperated by comma in the form:
        Cloudy : Weather, Weather, NotWatering
        Sprinkler : Watering, NotWeather, Watering
        Rain : Weather, Weather, Watering
        Wet_Grass : Plant_Con, NotWeather, NotWatering  
    :param node_names: list of String of the column names of the dataset in the correct order, used to correct order of tags when mistakenly in wrong order
    :param alpha: significance level for skeleton discovery
    :param indep_test: str, type of the used independence test. default: fisherz, 
                other options are: "mv_fisherz", "mc_fisherz", "kci", "chisq", "gsq", "d_separation"
    :param majority_rule_tagged: bool, if true uses the majority of tags to orient forks, if false tag-order matters, orients with sequential tags
    :param majority_rule_typed: bool, majority rule from typed PC Algo: if true use majority rule to orient forks, if false use naive rule to orient forks
    :return: adjacency matrix of the DAG of type np.ndarray 
    :return: taglist transformed as list of list of Int where every list is a typelist

    step 1: infer skeleton
    step 2: orient v-structures & two type forks + type consistency
    step 3: t-propagation
    
    step 2 and 3 are seperated in tpc_tag_naive and tpc_tag_majority

    recommended usage regarding tag importance:
    if all tags have the same importance use tag majority (with type majority)
    if there are more important tags use them at TODO <start or finish> (with type naive) 
    """
    #step 1
    taglist = get_taglist_of_int_from_text(tags=tags, node_names_ordered=node_names) #get Tags as List of List of Int
    skeleton, separating_sets, stat_tests = create_skeleton_using_causallearn(data, taglist, alpha, indep_test) # get skeleton

    print("skeleton: \n", nx.adjacency_matrix(skeleton).todense()) #TODO to Debug.Log
#    save_skeleton(skeleton=skeleton, separating_sets=separating_sets, stat_tests=stat_tests)     # -> use for debugging when you want to save a new skeleton
#    skeleton, separating_sets, stat_tests = load_skeleton()                                 # -> use for debugging when you dont want to wait to create skeleton -> also comment out create skeleton/seperating_sets

    if (majority_rule_tagged):
        dag, stat_tests = tpc_tag_majority(taglist, skeleton, separating_sets, stat_tests, majority_rule_typed, node_names=node_names)
    else:
        dag, stat_tests = tpc_tag_naive(taglist, skeleton, separating_sets, stat_tests, majority_rule_typed, node_names=node_names)

    return dag, stat_tests, taglist

def tpc_from_true_skeleton (dataname, tags, majority_rule_tagged=True, majority_rule_typed=True, data=None):
    """
    :param tags: text of params and corresponding assigned tags with tags being seperated by comma in the form:
        Cloudy : Weather, Weather, NotWatering
        Sprinkler : Watering, NotWeather, Watering
        Rain : Weather, Weather, Watering
        Wet_Grass : Plant_Con, NotWeather, NotWatering  
    :param majority_rule_tagged: bool, if true uses the majority of tags to orient forks, if false tag-order matters, orients with sequential tags
    :param majority_rule_typed: bool, majority rule from typed PC Algo: if true use majority rule to orient forks, if false use naive rule to orient forks
    :optional param data: string of data with nodes seperated with whitespace and entries by line break
    :return: adjacency matrix of the DAG of type np.ndarray
    :return: stat_tests of the DAG of type np.ndarray  
    :return: node names of type list<String> 
    :return: taglist transformed as list of list of Int where every list is a typelist

    step 1: infer skeleton
    step 2: orient v-structures & two type forks + type consistency
    step 3: t-propagation
    
    step 2 and 3 are seperated in tpc_tag_naive and tpc_tag_majority

    recommended usage regarding tag importance:
    if all tags have the same importance use tag majority (with type majority)
    if there are more important tags use them at TODO <start or finish> (with type naive) 
    """
    #step 1 from ture skelton
    skeleton, separating_sets, stat_tests, node_names, taglist = get_true_skeleton(dataname=dataname, tags=tags, data=data)

    print("skeleton: \n", nx.adjacency_matrix(skeleton).todense()) #TODO to Debug.Log
    plot_dag_state(dag=nx.adjacency_matrix(skeleton).todense(), var_names=node_names, types=taglist[0], step_number=0, experiment_step="skeleton") # XXX comment out before publishing #comment in for intermediate dag states
#    save_skeleton(skeleton=skeleton, separating_sets=separating_sets, stat_tests=stat_tests)     # -> use for debugging when you want to save a new skeleton
#    skeleton, separating_sets, stat_tests = load_skeleton()                                 # -> use for debugging when you dont want to wait to create skeleton -> also comment out create skeleton/seperating_sets

    if majority_rule_tagged:
        dag, stat_tests = tpc_tag_majority(taglist, skeleton, separating_sets, stat_tests, majority_rule_typed, node_names=node_names)
    else:
        dag, stat_tests = tpc_tag_naive(taglist, skeleton, separating_sets, stat_tests, majority_rule_typed, node_names=node_names)

    return dag, stat_tests, node_names, taglist


def tpc_tag_naive(taglist, skeleton, separating_sets, stat_tests, majority_rule_typed, node_names=[]): #node names only needed for debugging
    
    #step 2
    if (majority_rule_typed):
        out = orient_forks_naive_tag_majority_top1(skeleton=skeleton, sep_sets=separating_sets)
    else:
        out = orient_forks_naive_type_naive_tag(skeleton=skeleton, sep_sets=separating_sets)
    
    #step 3
#    out = pc_meek_rules(out) #untyped meek

    out_matrix, t_matrix = typed_meek_naive_tag(cpdag=nx.adjacency_matrix(out).todense(), tags=taglist)

    print("adjacency_mat result: \n", out_matrix)
    print("stat_test result: \n", stat_tests)

    return out_matrix, stat_tests # return nx.adjacency_matrix(out).todense() when your graph isn't already an adjacency matrix

def tpc_tag_majority(taglist, skeleton, separating_sets, stat_tests, majority_rule_typed, node_names=[]): #node names only needed for debugging

    priomat = get_priomat_from_skeleton(nx.adjacency_matrix(skeleton).todense(), taglist)
    print("priomat \n", priomat) #TODO to Debug.Log

    #Step 2 - V Structues + Two Type Forks
    if (majority_rule_typed):
        dag, priomat = orient_forks_majority_tag_majority_top1(skeleton=skeleton, sep_sets=separating_sets, priomat=priomat, taglist=taglist)
    else:
        dag, priomat = orient_forks_majority_tag_naive_type(skeleton=skeleton, sep_sets=separating_sets, priomat=priomat, taglist=taglist)
    plot_dag_state(dag=nx.adjacency_matrix(dag).todense(), var_names=node_names, types=taglist[0], step_number=1, experiment_step="orienting_forks_v_structures") # XXX comment out before publishing #comment in for intermediate dag states

    #Step 3 - T-Propagation without type consistency
    adjacency_mat, priomat = typed_meek_majority_tag_without_typeconsistency(cpdag=nx.adjacency_matrix(dag).todense(), tags=taglist, priomat=priomat)
    plot_dag_state(dag=adjacency_mat, var_names=node_names, types=taglist[0], step_number=2, experiment_step="meek_no_consistency") # XXX comment out before publishing #comment in for intermediate dag states


    #Step 4 - T-Propagation with Tag_majority Consistency by using priomat
    adjacency_mat, priomat = typed_meek_majority_tag_with_consistency(cpdag=adjacency_mat, tags=taglist, priomat=priomat, majority_rule_typed=majority_rule_typed)
    plot_dag_state(dag=adjacency_mat, var_names=node_names, types=taglist[0], step_number=3, experiment_step="meek_type_consistency") # XXX comment out before publishing #comment in for intermediate dag states


    print("priomat result: \n", priomat)
    print("adjacency_mat result: \n", adjacency_mat)
    print("stat_test result: \n", stat_tests)
    return adjacency_mat, stat_tests


# step 1 - infer skeleton (using the causallearn library):
def create_skeleton_using_causallearn(data, taglist, alpha, indep_test):
    # get conditional independence test object and use it to create skeleton
    cit = CIT(data, method=indep_test)
    skeleton = SkeletonDiscovery.skeleton_discovery(data, alpha, indep_test=cit, stable=True,)
    # get seperating_sets from sekelton
    separating_sets = skeleton.sepset



    #get skeleton as nx.Graph
    skeleton.to_nx_skeleton()
    skeleton = skeleton.nx_skel

    node_ids = skeleton.number_of_nodes()
    for (i, j) in list(combinations(range(node_ids), 2)):
        print(f"seperating set of {i} and {j} is: {separating_sets[i][j]}")

    separating_sets = format_seperating_sets(separating_sets)
    
    print("seperating sets Formated \n") #TODO to Debug.Log
#    print("seperating sets: \n", separating_sets) #TODO to Debug.Log
    for (i, j) in list(combinations(range(node_ids), 2)):
        print(f"seperating set of {i} and {j} is: {separating_sets[i][j]}")

    stat_tests = nx.adjacency_matrix(skeleton).todense()

    skeleton = set_tags_as_int(skeleton, taglist)
    return skeleton, separating_sets, stat_tests

# step 1 - get correct skeleton from true graph (for assumption that skeleton is correct)
def get_true_skeleton(dataname : str, tags, data):
    """
    get the true skeleton (and trueish seperating sets), only works for forest example and the bnf dataset (you might have to download additional .bif files)
    """

    match dataname:
        # since we already get the true skeleton with the correct parameters for causallearn we do just that
        case "forest":
            if data is None:
                raise ValueError("since you are using non bnf data, you need give the function data ad np.ndarray")
            node_names = ["A","R","S","H","B","W","F"]
            taglist = get_taglist_of_int_from_text(tags, node_names)
            skeleton, separating_sets, stat_tests = create_skeleton_using_causallearn(data, taglist, alpha = 0.05, indep_test="fisherz") #TODO
            return skeleton, separating_sets, stat_tests, node_names, taglist
        #bnlearn directly supports those, we do not need a bif file
        case "asia" | 'sprinkler' | 'alarm' | 'andes' | 'sachs':
            path = dataname
        #we search for a .bif file in tagged-pc-using-LLM/additionalData 
        case _:
            path = os.path.join("tagged-pc-using-LLM/additionalData", (dataname + ".bif"))
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
        print(f"seperating set of {i} {node_names[i]} and {j} {node_names[j]} is: {separating_sets[i][j]}")

    separating_sets = format_seperating_sets(separating_sets)

    print("seperating sets Formated \n") #TODO to Debug.Log
#    print("seperating sets: \n", separating_sets) #TODO to Debug.Log
    for (i, j) in list(combinations(range(node_ids), 2)):
        print(f"seperating set of {i} {node_names[i]} and {j} {node_names[j]} is: {separating_sets[i][j]}")

    return skeleton, separating_sets, stat_tests, node_names, taglist

# the rest of the code is taken from https://github.com/ServiceNow/typed-dag and modified for tagged

# step 2 - orient v-structures & two type forks + type consistency:

#######
# This is the part where we orient all immoralities and two-type forks.
# The behavior used to orient t-edges depends on the chosen strategy:
#   * Naive: orient as first encountered orientation
#   * Majority: orient using the most frequent orientation
#######

def orient_forks_naive_type_naive_tag(skeleton, sep_sets):
    """
    Orient immoralities and two-type forks

    Strategy: naive -- orient as first encountered

    """

    dag = skeleton.to_directed()
    node_ids = skeleton.nodes()

    print("orienting forks")
    
    #iterate over all tags
    for current_type in range(len(skeleton.nodes[0])):
        print("current_type: ", current_type) #TODO to Debug.Log
        # Orient all immoralities and two-type forks
        # TODO: SERVICENOW DEBUG using shuffling to test hypothesis
        combos = list(combinations(node_ids, 2))
        shuffle(combos)
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
                    print(
                        f"S: orient immorality {i} (t{type_of_from_tag_single(dag, i, current_type)}) -> {k} (t{type_of_from_tag_single(dag, k, current_type)}) <- {j} (t{type_of_from_tag_single(dag, j, current_type)})"
                    ) #TODO to Debug.Log
                    _orient_tagged(dag, i, k, current_type)
                    _orient_tagged(dag, j, k, current_type)

                # Case: we have an orientable two-type fork, i.e., it is not an immorality, so i <- k -> j
                elif (
                    type_of_from_tag_single(dag, i, current_type) == type_of_from_tag_single(dag, j, current_type)
                    and type_of_from_tag_single(dag, i, current_type) != type_of_from_tag_single(dag, k, current_type)
                    and _has_both_edges(dag, i, k)
                    and _has_both_edges(dag, j, k)
                ):
                    print(
                        f"S: orient two-type fork {i} (t{type_of_from_tag_single(dag, i, current_type)}) <- {k} (t{type_of_from_tag_single(dag, k, current_type)}) -> {j} (t{type_of_from_tag_single(dag, j, current_type)})"
                    ) #TODO to Debug.Log
                    _orient_tagged(dag, k, i, current_type)  # No need to orient k -> j. Will be done in this call since i,j have the same type.
        print("current adj. matrix: \n", nx.adjacency_matrix(dag).todense()) #TODO to Debug.Log
    return dag

def orient_forks_naive_tag_majority_top1(skeleton, sep_sets):
    """
    Orient immoralities and two-type forks

    Strategy: majority -- orient using the most frequent orientation
    Particularity: Find the t-edge with most evidence, orient, repeat evidence collection.

    """
    print("orient forks")

    dag = skeleton.to_directed()
    node_ids = skeleton.nodes()
    #iterate over all tags
    for current_type in range(len(skeleton.nodes[0])):
        print("current_type", current_type) #TODO to Debug.Log

        n_types = len(np.unique([type_of_from_tag_single(dag, n, current_type) for n in dag.nodes()]))

        oriented_tedge = True
        while oriented_tedge:

            # Accumulator for evidence of t-edge orientation
            # We will count how often we see the t-edge in each direction and choose the most frequent one.
            tedge_evidence = np.zeros((n_types, n_types))
            oriented_tedge = False

            # Some immoralities will contain edges between variables of the same type. These will not be
            # automatically oriented when we decide on the t-edge orientations. To make sure that we orient
            # them correctly, we maintain a list of conditional orientations, i.e., how should an intra-type
            # edge be oriented if we make a specific orientation decision for the t-edges.
            conditional_orientations = defaultdict(list)

            # Step 1: Gather evidence of orientation for all immoralities and two-type forks that involve more than one type
            for (i, j) in combinations(node_ids, 2):
                adj_i = set(dag.successors(i))
                adj_j = set(dag.successors(j))

                # If j is a direct child of i, i is a direct child of j, or ij are directly connected
                if j in adj_i or i in adj_j or sep_sets[i][j] is None:
                    continue

                for k in adj_i & adj_j:  # Common direct children of i and j
                    # Case: we have an immorality i -> k <- j
                    if k not in sep_sets[i][j]:
                        # Check if already oriented
                        # if not _has_both_edges(dag, i, k) or not _has_both_edges(dag, j, k):
                        #     continue
                        if not _has_both_edges(dag, i, k) and not _has_both_edges(dag, j, k):
                            # Fully oriented
                            continue

                        # Ensure that we don't have only one type. We will orient these later.
                        if type_of_from_tag_single(dag, i, current_type) == type_of_from_tag_single(dag, j, current_type) == type_of_from_tag_single(dag, k, current_type):
                            continue

                        print(
                            f"Step 1: evidence of orientation {i} (t{type_of_from_tag_single(dag, i, current_type)}) -> {k} (t{type_of_from_tag_single(dag, k, current_type)}) <- {j} (t{type_of_from_tag_single(dag, j, current_type)})"
                        ) #TODO to Debug.Log
                        # Increment t-edge orientation evidence
                        print(tedge_evidence) #TODO to Debug.Log
                        print(type_of_from_tag_single(dag, i, current_type)) #TODO to Debug.Log
                        print(type_of_from_tag_single(dag, k, current_type)) #TODO to Debug.Log
                        tedge_evidence[type_of_from_tag_single(dag, i, current_type), type_of_from_tag_single(dag, k, current_type)] += 1
                        tedge_evidence[type_of_from_tag_single(dag, j, current_type), type_of_from_tag_single(dag, k, current_type)] += 1

                        # Determine conditional orientations
                        conditional_orientations[(type_of_from_tag_single(dag, j, current_type), type_of_from_tag_single(dag, k, current_type))].append((i, k))
                        conditional_orientations[(type_of_from_tag_single(dag, i, current_type), type_of_from_tag_single(dag, k, current_type))].append((j, k))

                    # Case: we have an orientable two-type fork, i.e., it is not an immorality, so i <- k -> j
                    elif type_of_from_tag_single(dag, i, current_type) == type_of_from_tag_single(dag, j, current_type) and type_of_from_tag_single(dag, i, current_type) != type_of_from_tag_single(dag, k, current_type):
                        # Check if already oriented
                        if not _has_both_edges(dag, i, k) or not _has_both_edges(dag, j, k):
                            continue

                        print(
                            f"Step 1: evidence of orientation {i} (t{type_of_from_tag_single(dag, i, current_type)}) <- {k} (t{type_of_from_tag_single(dag, k, current_type)}) -> {j} (t{type_of_from_tag_single(dag, j, current_type)})"
                        ) #TODO to Debug.Log
                        # Count evidence only once per t-edge
                        tedge_evidence[type_of_from_tag_single(dag, k, current_type), type_of_from_tag_single(dag, i, current_type)] += 2

            # Step 2: Orient t-edges based on evidence
            np.fill_diagonal(tedge_evidence, 0)
            ti, tj = np.unravel_index(tedge_evidence.argmax(), tedge_evidence.shape)
            if np.isclose(tedge_evidence[ti, tj], 0):
                continue

            # Orient!
            print("Evidence", tedge_evidence[ti, tj]) #TODO to Debug.Log
            print(conditional_orientations) #TODO to Debug.Log
            oriented_tedge = True
            first_ti = [n for n in dag.nodes() if type_of_from_tag_single(dag, n, current_type) == ti][0]
            first_tj = [n for n in dag.nodes() if type_of_from_tag_single(dag, n, current_type) == tj][0]
            print(
                f"Step 2: orienting t-edge according to max evidence. t{ti} -> t{tj} ({tedge_evidence[ti, tj]}) vs t{ti} <- t{tj} ({tedge_evidence[tj, ti]})"
            ) #TODO to Debug.Log
            _orient_tagged(dag, first_ti, first_tj, current_type)
            cond = Counter(conditional_orientations[ti, tj])
            for (n1, n2), count in cond.items():
                print(f"... conditional orientation {n1}->{n2} (count: {count}).")
                if (n2, n1) in cond and cond[n2, n1] > count:
                    print(
                        f"Skipping this one. Will orient its counterpart ({n2}, {n1}) since it's more frequent: {cond[n2, n1]}."
                    )
                else:
                    _orient_tagged(dag, n1, n2, current_type)
        print("Steps 1-2 completed. Moving to single-type forks.")

        # Step 3: Orient remaining immoralities (all variables of the same type)
        for (i, j) in combinations(node_ids, 2):
            adj_i = set(dag.successors(i))
            adj_j = set(dag.successors(j))

            # If j is a direct child of i, i is a direct child of j, ij are directly connected
            if j in adj_i or i in adj_j or sep_sets[i][j] is None:
                continue

            for k in adj_i & adj_j:  # Common direct children of i and j
                # Case: we have an immorality i -> k <- j
                if k not in sep_sets[i][j]:
                    # Only single-type immoralities
                    if not (type_of_from_tag_single(dag, i, current_type) == type_of_from_tag_single(dag, j, current_type) == type_of_from_tag_single(dag, k, current_type)):
                        continue
                    print(
                        f"Step 3: orient immorality {i} (t{type_of_from_tag_single(dag, i, current_type)}) -> {k} (t{type_of_from_tag_single(dag, k, current_type)}) <- {j} (t{type_of_from_tag_single(dag, j, current_type)})"
                    )
                    _orient_tagged(dag, i, k, current_type)
                    _orient_tagged(dag, j, k, current_type)
        
        print("current adj. matrix: \n", nx.adjacency_matrix(dag).todense()) 
    return dag

# Step 3 - t Propagation

# taken from https://github.com/ServiceNow/typed-dag/blob/main/typed_pc/tmec.py and modified for naive tag, by iterating over all tags
def typed_meek_naive_tag(cpdag: np.ndarray, tags: list, iter_max: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the Meek algorithm with the type consistency as described in Section 5 (from the CPDAG).

    :param cpdag: adjacency matrix of the CPDAG
    :param types: list of type of each node
    :param iter_max: The maximum number of iterations. If reached, an exception will be raised.
    """
    print("typed Meek")

    #iterate over all tags
    current_type = 0
    for types in tags:
        print("current_type: ", current_type) #TODO to Debug.Log

        n_nodes = cpdag.shape[0]
        types = np.asarray(types)
        n_types = len(np.unique(types))

        G = np.copy(cpdag)
        type_g = np.eye(n_types)  # Identity matrix to allow intra-type edges

        # repeat until the graph is not changed by the algorithm
        # or too high number of iteration
        previous_G = np.copy(G)
        i = 0
        while True and i < iter_max: #TODO while true entfernen
            """
            Each iteration is broken down into three stages:
            1) Update t-edge orientations based on the CPDAG
            2) Orient all edges that are part of the same t-edge consistently (if the t-edge is oriented)
            3) Apply the Meek rules (only one per iteration) to orient the remaining edges.

            Note: Edges are oriented one-by-one in step 3, but these orientations will be propagated to the whole
                t-edge once we return to step (1).

            """
            i += 1
            # Step 1: Determine the orientation of all t-edges based on t-edges (some may be unoriented)
            type_g = _update_tedge_orientation(G, type_g, types)

            # Step 2: Orient all edges of the same type in the same direction if their t-edge is oriented.
            # XXX: This will not change the orientation of oriented edges (i.e., if the CPDAG was not consistant) XXX: Guldan edit: I sure hope so
            G = _orient_tedges(G, type_g, types)

            # Step 3: Apply Meek's rules (R1, R2, R3, R4) and the two-type fork rule (R5)
            for a, b, c in permutations(range(n_nodes), 3):
                # Orient any undirected edge a - b as a -> b if any of the following rules is satisfied:
                if G[a, b] != 1 or G[b, a] != 1:
                    # Edge a - b is already oriented
                    continue

                # R1: c -> a - b ==> a -> b
                if G[a, c] == 0 and G[c, a] == 1 and G[b, c] == 0 and G[c, b] == 0:
                    print(f"meek R1: {c} -> {a} - {b}: orient to  {a} -> {b}")  #TODO to Debug.Log
                    G[b, a] = 0
                # R2: a -> c -> b and a - b ==> a -> b
                elif G[a, c] == 1 and G[c, a] == 0 and G[b, c] == 0 and G[c, b] == 1:
                    print(f"meek R2: {a} -> {c} -> {b} and {a} - {b}: orient to {a} -> {b}")  #TODO to Debug.Log
                    G[b, a] = 0
                # R5: b - a - c and t(c) = t(b) ==> a -> b and a -> c (two-type fork)
                elif (
                    G[a, c] == 1
                    and G[c, a] == 1
                    and G[b, c] == 0
                    and G[c, b] == 0
                    and types[b] == types[c]
                    and types[b] != types[a]  # Make sure there are at least two types
                ):
                    print(f"meek R5: {b} - {a} - {c} and t{c}: {type[c]} = t{b}: {type[b]}: orient to {a} -> {b} and {a} -> {c}")  #TODO to Debug.Log
                    G[b, a] = 0
                    G[c, a] = 0
                else:

                    for d in range(n_nodes):
                        if d != a and d != b and d != c:
                            # R3: a - c -> b and a - d -> b, c -/- d ==> a -> b, and a - b
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
                            ):
                                print(f"meek R3: {a} - {c} -> {b} and {a} - {d} -> {b} and {c} -/- {d} orient to {a} -> {b}")  #TODO to Debug.Log
                                G[b, a] = 0
                            # R4: a - d -> c -> b and a - - c ==> a -> b
                            elif (
                                G[a, d] == 1
                                and G[d, a] == 1
                                and G[c, d] == 0
                                and G[d, c] == 1
                                and G[b, c] == 0
                                and G[c, b] == 1
                                and (G[a, c] == 1 or G[c, a] == 1)
                            ):
                                print(f"meek R4: {a} - {d} -> {c} -> {b} and {a} -> {c} or {c} -> {a} orient to {a} -> {b}")  #TODO to Debug.Log
                                G[b, a] = 0

            if (previous_G == G).all():
                break
            if i >= iter_max:
                raise Exception(f"Typed Meek is stucked. More than {iter_max} iterations.")

            previous_G = np.copy(G)

        current_type += 1

        print("current adj. matrix: \n", G) #TODO to Debug.Log
    return G, type_g


# -------------------------------- algo-steps for tag majority -------------------------------------
# step 2 - orient v-strucutes and two type forks

def orient_forks_majority_tag_naive_type(skeleton, sep_sets, priomat, taglist):
    """
    Orient immoralities and two-type forks

    Type - Strategy: naive -- orient as first encountered
    Tag - Strategy: majority -- orient each edge taking all tags into consideration
    """

    dag = skeleton.to_directed()
    node_ids = skeleton.nodes()

    print("orient forks tag majority type naive")
    

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
                    f"V: orient immorality {i} (t{type_of_from_tag_all(dag, i)}) -> {k} (t{type_of_from_tag_all(dag, k)}) <- {j} (t{type_of_from_tag_all(dag, j)})"
                ) #TODO to Debug.Log
                # orient just v-structure ignore tags for now, but add strong entry in prio matrix (since v-strucutre comes from data it should not be overwritten easily later on)
                prio_weight = len(taglist)
                print("V-Structure_prio_weight: ", prio_weight)#TODO to Debug.Log
                _orient_typeless_with_priomat(dag, i, k, priomat, prio_weight)
                _orient_typeless_with_priomat(dag, j, k, priomat, prio_weight)
                # print("priomat: \n", priomat)  #TODO to Debug.Log
                

            # Case: we have an orientable two-type fork, i.e., it is not an immorality, so i <- k -> j 
            elif (
                ((_has_both_edges(dag, k, i)  and _has_both_edges(dag, k, j)) # we want to orient when both edges are unoriented
                or  (_has_both_edges(dag, k, i) and _has_directed_edge(dag, k, j)) # but also when already one edge points in the correct direction (going into k).
                or (_has_both_edges(dag, k, j) and _has_directed_edge(dag, k, i)) ) # For normal typed (or tag naive), type consistency would take care of that but here we would need meek first but since we are less confident in tag consistency here, we want to orient all possible two way forks before type consistency (changed for tagged naive)
                and ((prio_weight := amount_of_matching_types_for_two_way_fork(dag, i, j, k)) > 0) # check that there is a tag where typeof(i) == typeof(j) =! typeof(k)
            ):                
                print(
                    f"F: orient two-type fork {i} (t{type_of_from_tag_all(dag, i)}) <- {k} (t{type_of_from_tag_all(dag, k)}) -> {j} (t{type_of_from_tag_all(dag, j)})"
                ) #TODO to Debug.Log
                # orient both edges if they dont already have a higher prio
                if (priomat[k][i] <= prio_weight and priomat[k][j] <= prio_weight):
                    _orient_typeless_with_priomat(dag, k, i, priomat, prio_weight) 
                    _orient_typeless_with_priomat(dag, k, j, priomat, prio_weight) # Since we do not use type consistency yet, we need to orient both i and j to k
                else: 
                    print("ERROR THIS SHOULD NOT HAPPEN")
        
        #search for additional forks that are already partily oriented bevor applying type consistency in t-propagation. This is necessary because we do not use type consistency yet (which would oriented this forks before). 
        pred_i = set(dag.predecessors(i))
        pred_j = set(dag.predecessors(j))
        common_pred_k = pred_i & pred_j  # Common direct predecessor of i and j (used for forks)
        for k in common_pred_k:
            # Case: we have an immorality eventhough we have one edge oriented in the wrong direction i -> k <- j but we have (k->i or k->j)
            if k not in sep_sets[i][j]: # XXX: had to add the last two conditions in case k is no longer a child due to t-edge orientation
                # check if our previous orientation was for some reason more important than v-structure 
                prio_weight = len(taglist)
                if (priomat[k][i] <= prio_weight and priomat[k][j] <= prio_weight):
                    print(
                        f"V: orient PREVIOUSLY WORONG oriented immorality {i} (t{type_of_from_tag_all(dag, i)}) -> {k} (t{type_of_from_tag_all(dag, k)}) <- {j} (t{type_of_from_tag_all(dag, j)})"
                    ) #TODO to Debug.Log
                    # orient just v-structure ignore tags for now, but add strong entry in prio matrix (since v-strucutre comes from data it should not be overwritten easily later on)
                    print("V-Structure_prio_weight: ", prio_weight)#TODO to Debug.Log
                    _orient_typeless_with_priomat(dag, i, k, priomat, prio_weight)
                    _orient_typeless_with_priomat(dag, j, k, priomat, prio_weight)

            # Case: we have an orientable two-type fork, i.e., it is not an immorality, so i <- k -> j 
            elif (
                ((_has_both_edges(dag, k, i)  and _has_both_edges(dag, k, j)) # we want to orient when both edges are unoriented
                or  (_has_both_edges(dag, k, i) and _has_directed_edge(dag, k, j)) # but also when already one edge points in the correct direction (going into k).
                or (_has_both_edges(dag, k, j) and _has_directed_edge(dag, k, i)) ) # For normal typed (or tag naive), type consistency would take care of that but here we would need meek first but since we are less confident in tag consistency here, we want to orient all possible two way forks before type consistency (changed for tagged naive)
                and ((prio_weight := amount_of_matching_types_for_two_way_fork(dag, i, j, k)) > 0) # check that there is a tag where typeof(i) == typeof(j) =! typeof(k)
            ):                
                print(
                    f"F: orient two-type fork to predecesor {i} (t{type_of_from_tag_all(dag, i)}) <- {k} (t{type_of_from_tag_all(dag, k)}) -> {j} (t{type_of_from_tag_all(dag, j)})"
                ) #TODO to Debug.Log
                # orient both edges if they dont already have a higher prio
                if (priomat[k][i] <= prio_weight and priomat[k][j] <= prio_weight):
                    _orient_typeless_with_priomat(dag, k, i, priomat, prio_weight) 
                    _orient_typeless_with_priomat(dag, k, j, priomat, prio_weight) # Since we do not use type consistency yet, we need to orient both i and j to k
                else: 
                    print(f"Not enough Prio Evidence to orient {i} (t{type_of_from_tag_all(dag, i)}) <- {k} (t{type_of_from_tag_all(dag, k)}) -> {j} (t{type_of_from_tag_all(dag, j)})")
                        



    print("orienting forks finished - current adj. matrix: \n", nx.adjacency_matrix(dag).todense()) #TODO to Debug.Log
    print("current priomat: \n", priomat) #TODO to Debug.Log
    return dag, priomat


def orient_forks_majority_tag_majority_top1(skeleton, sep_sets, priomat, taglist):
    """
    Orient immoralities and two-type forks -> adjusted naive typed since classic majority algo uses integraly type consistence

    Type - Strategy: majority -- orient using the most frequent orientation
    Particularity: Find two-type forks in tags and only orient for edge triples, where more tags support a specific direction than tags support the other direction.
    also do not use type consistency yet (in contrast to majority top in typing algo)
    Tag - Strategy: majority -- orient each edge taking all tags into consideration

    """

    dag = skeleton.to_directed()
    node_ids = skeleton.nodes()

    print("orient forks tag majority type majority")
    

    # Orient all immoralities and two-type forks
    # TODO: SERVICENOW DEBUG using shuffling to test hypothesis
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
                # XXX: had to add the last two conditions in case k is no longer a child due to t-edge orientation
                print(
                    f"S: orient immorality {i} (t{type_of_from_tag_all(dag, i)}) -> {k} (t{type_of_from_tag_all(dag, k)}) <- {j} (t{type_of_from_tag_all(dag, j)})"
                ) #TODO to Debug.Log
                # orient just v-structure ignore tags for now, but add strong entry in prio matrix (since v-strucutre comes from data it should not be overwritten easily later on)
                prio_weight = len(taglist)
                _orient_typeless_with_priomat(dag, i, k, priomat, prio_weight)
                _orient_typeless_with_priomat(dag, j, k, priomat, prio_weight)
                # print("priomat after orienten v-strucutre: \n", priomat)  #TODO to Debug.Log #still comment out
                

            # Case: we have an orientable two-type fork, i.e., it is not an immorality, so i <- k -> j
            elif (
                ((_has_both_edges(dag, k, i)  and _has_both_edges(dag, k, j)) # we want to orient when both edges are unoriented
                or  (_has_both_edges(dag, k, i) and _has_directed_edge(dag, k, j)) # but also when already one edge points in the correct direction (going into k).
                or (_has_both_edges(dag, k, j) and _has_directed_edge(dag, k, i)) ) # For normal typed (or tag naive), type consistency would take care of that but here we would need meek first but since we are less confident in tag consistency here, we want to orient all possible two way forks before type consistency (changed for tagged naive)
                and ((prio_weight := amount_of_matching_types_for_two_way_fork(dag, i, j, k)) > 0) # check that there is a tag where typeof(i) == typeof(j) =! typeof(k)
            ):
                print(
                    f"S: found two-type fork {i} (t{type_of_from_tag_all(dag, i)}) <- {k} (t{type_of_from_tag_all(dag, k)}) -> {j} (t{type_of_from_tag_all(dag, j)} saving for orienting)"
                ) #TODO to Debug.Log
                # safe two way fork just for now
                two_way_evidence.append([prio_weight, i, k, j])

       #search for additional forks that are already partily oriented bevor applying type consistency in t-propagation This is necessary because we do not use type consistency yet (which would oriented this forks before). 
        pred_i = set(dag.predecessors(i))
        pred_j = set(dag.predecessors(j))
        common_pred_k = pred_i & pred_j  # Common direct predecessor of i and j (used for forks)
        for k in common_pred_k:
            # Case: we have an immorality eventhough we have one edge oriented in the wrong direction i -> k <- j but we have (k->i or k->j)
            if k not in sep_sets[i][j]: # XXX: had to add the last two conditions in case k is no longer a child due to t-edge orientation
                # check if our previous orientation was for some reason more important than v-structure 
                prio_weight = len(taglist)
                if (priomat[k][i] <= prio_weight and priomat[k][j] <= prio_weight):
                    print(
                        f"V: orient PREVIOUSLY WRONG oriented immorality {i} (t{type_of_from_tag_all(dag, i)}) -> {k} (t{type_of_from_tag_all(dag, k)}) <- {j} (t{type_of_from_tag_all(dag, j)})"
                    ) #TODO to Debug.Log
                    # orient just v-structure ignore tags for now, but add strong entry in prio matrix (since v-strucutre comes from data it should not be overwritten easily later on)
                    print("V-Structure_prio_weight: ", prio_weight)#TODO to Debug.Log
                    _orient_typeless_with_priomat(dag, i, k, priomat, prio_weight)
                    _orient_typeless_with_priomat(dag, j, k, priomat, prio_weight)

            # Case: we have an orientable two-type fork, i.e., it is not an immorality, so i <- k -> j 
            elif (
                ((_has_both_edges(dag, k, i)  and _has_both_edges(dag, k, j)) # we want to orient when both edges are unoriented
                or  (_has_both_edges(dag, k, i) and _has_directed_edge(dag, k, j)) # but also when already one edge points in the correct direction (going into k).
                or (_has_both_edges(dag, k, j) and _has_directed_edge(dag, k, i)) ) # For normal typed (or tag naive), type consistency would take care of that but here we would need meek first but since we are less confident in tag consistency here, we want to orient all possible two way forks before type consistency (changed for tagged naive)
                and ((prio_weight := amount_of_matching_types_for_two_way_fork(dag, i, j, k)) > 0) # check that there is a tag where typeof(i) == typeof(j) =! typeof(k)
            ):                
                # save fork if we have not encountered it already
                if not (exists_entry_in_forkevidencelist(two_way_evidence, prio_weight, i, k, j)):
                    print(
                        f"F: found two-type fork to predecesor {i} (t{type_of_from_tag_all(dag, i)}) <- {k} (t{type_of_from_tag_all(dag, k)}) -> {j} (t{type_of_from_tag_all(dag, j)} saving for orienting)"
                    ) #TODO to Debug.Log
                    two_way_evidence.append([prio_weight, i, k, j])
                else: 
                    print(f"two way fork {i} (t{type_of_from_tag_all(dag, i)}) <- {k} (t{type_of_from_tag_all(dag, k)}) -> {j} (t{type_of_from_tag_all(dag, j)}) found but already saved")


    # now orienting two way forks
    print("all evidence collected, now orienting two way forks")

    # iterate over all two way forks i <- k -> j to search for clashes
    for index, two_way_fork in enumerate(two_way_evidence):
        prio_weight, i, k, j = two_way_fork
        has_highest_prio_weight = True
        clash_found = False

        for other_index, other_fork in enumerate(two_way_evidence):
            if index != other_index:
                other_prio_weight, other_i, other_k, other_j = other_fork

                # TODO Condition testen
                # if any other two_way_fork has an opossed edge -> we have a clash meaning different oriented forks and have to decide which one to orient 
                if are_forks_clashing(two_way_fork, other_fork):
                    clash_found = True
                    # fork with higher (or same) prio_weight found -> we will not orient current fork, but wait in loop for the higher weight fork to orient it then (if other fork has no conflict that is)
                    # if there are only clashing forks with the same priority do not orient any of them.
                    if other_prio_weight >= prio_weight:
                        has_highest_prio_weight = False
                        break             
        
        if clash_found:
            if has_highest_prio_weight:
            # still check prioweight for (redundant) safety (in _orienting it will also be checked if edge was directed in other way previously (=checked for clas with immorality))
                if (priomat[k][i] <= prio_weight and priomat[k][j] <= prio_weight):
                    print(
                        f"Clash found -> orienting two-type fork {i} (t{type_of_from_tag_all(dag, i)}) <- {k} (t{type_of_from_tag_all(dag, k)}) -> {j} (t{type_of_from_tag_all(dag, j)}. despite of clash found with at least fork {other_i} <- {other_k} -> {other_j} with prio: {other_prio_weight} that is smaller than prio of orienting fork: {prio_weight}. )"
                    ) #TODO to Debug.Log
                    _orient_typeless_with_priomat(dag, k, i, priomat, prio_weight)
                    _orient_typeless_with_priomat(dag, k, j, priomat, prio_weight)
            else:
                print(f"Clash found -> No orientation of this fork: {i} <- {k} -> {j} with prio: {prio_weight} due to a clashing fork {other_i} <- {other_k} -> {other_j} with higher or prio_weight: with prio: {other_prio_weight}.")
        
        #k is not in i or j in any other two_way_fork, meaning we have no conflict and can orient fork safely
        else:
            # still check prioweight for (redundant) safety (in _orienting it will also be checked if edge was directed in other way previously (=checked for clas with immorality))
            if (priomat[k][i] <= prio_weight and priomat[k][j] <= prio_weight):
                print(
                    f"orienting two-type fork {i} (t{type_of_from_tag_all(dag, i)}) <- {k} (t{type_of_from_tag_all(dag, k)}) -> {j} (t{type_of_from_tag_all(dag, j)})"
                ) #TODO to Debug.Log
                _orient_typeless_with_priomat(dag, k, i, priomat, prio_weight)
                _orient_typeless_with_priomat(dag, k, j, priomat, prio_weight)

    print("orienting forks finished - current adj. matrix: \n", nx.adjacency_matrix(dag).todense()) #TODO to Debug.Log
    print("current priomat: \n", priomat) #TODO to Debug.Log
    return dag, priomat


#step 3 - t-propagation:

# taken from https://github.com/ServiceNow/typed-dag/blob/main/typed_pc/tmec.py changed for tag-algo
def typed_meek_majority_tag_without_typeconsistency(cpdag: np.ndarray, tags: list, priomat : np.matrix, iter_max: int = 100) -> Tuple[np.ndarray, np.matrix]:
    """
    Apply the Meek algorithm with the type consistency as described in Section 5 (from the CPDAG).

    :param cpdag: adjacency matrix of the CPDAG
    :param types: list of type of each node
    :param iter_max: The maximum number of iterations. If reached, an exception will be raised.
    """
    print("typed Meek without type consistency")
    # priomat checks are mostly redundant here, since a,b is undirected and should therefore have prioweight 0

    n_nodes = cpdag.shape[0]
    G = np.copy(cpdag)

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

            # R1: c -> a - b ==> a -> b
            if G[a, c] == 0 and G[c, a] == 1 and G[b, c] == 0 and G[c, b] == 0 and ((prioweight := min(priomat[a,c], priomat[c, a], priomat[b,c], priomat[c,b])) > max(priomat[b,a], priomat[a, b])): #check that prio evidence is stronger than the to orient edge
                print(f"meek R1: {c} -> {a} - {b}: orient to  {a} -> {b}")  #TODO to Debug.Log
                G[b, a] = 0
                priomat[a, b] = prioweight
                priomat[b, a] = prioweight #update prioweight of now oriented edge to lowest evidence of condition
            # R2: a -> c -> b and a - b ==> a -> b
            elif G[a, c] == 1 and G[c, a] == 0 and G[b, c] == 0 and G[c, b] == 1 and ((prioweight := min(priomat[a,c], priomat[c, a], priomat[b,c], priomat[c,b])) > max(priomat[b,a], priomat[a, b])): #check that prio evidence is stronger than the to orient edge
                print(f"meek R2: {a} -> {c} -> {b} and {a} - {b}: orient to {a} -> {b}")  #TODO to Debug.Log
                G[b, a] = 0
                priomat[a, b] = prioweight
                priomat[b, a] = prioweight #update prioweight of now oriented edge to lowest evidence of condition
            # R5: b - a - c and a-/-c and t(c) = t(b) ==> a -> b and a -> c (two-type fork)
            elif (
                G[a, c] == 1
                and G[c, a] == 1
                and G[b, c] == 0
                and G[c, b] == 0
                and ((prioweight := amount_of_matching_types_for_two_way_fork_from_taglist(G, tags, b, c, a)) > max(priomat[b,a], priomat[a, b])) #check if there are more tags where type matches for two way fork than evidence for current a-b edge
                and ((min(priomat[a,c], priomat[c, a], priomat[b,c], priomat[c,b])) >= max(priomat[b,a], priomat[a, b])) #check that prio evidence is stronger or equal than the to orient edge (need >= here since we use an undirected edge as evidence, which sholuld have evidence 0 meaing we only orient unoriented edges a,b and a,c (which is already or condition in the first place, so kinda useless))
            ):
                print(f"meek R5: {b} - {a} - {c} and t{c}: {type_of_from_tag_all_from_taglist(tags, c)} = t{b}: {type_of_from_tag_all_from_taglist(tags, b)}: orient to {a} -> {b} and {a} -> {c}")  #TODO to Debug.Log
                G[b, a] = 0
                G[c, a] = 0
                priomat[a, b] = prioweight
                priomat[b, a] = prioweight
                priomat[a, c] = prioweight
                priomat[c, a] = prioweight #update prioweight of now oriented edge to lowest evidence of condition
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
                            print(f"meek R3: {a} - {c} -> {b} and {a} - {d} -> {b} and {c} -/- {d} orient to {a} -> {b}")  #TODO to Debug.Log
                            G[b, a] = 0
                            priomat[a, b] = prioweight
                            priomat[b, a] = prioweight #update prioweight of now oriented edge to lowest evidence of condition
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
                            print(f"meek R4: {a} - {d} -> {c} -> {b} and {a} -> {c} or {c} -> {a} orient to {a} -> {b}")  #TODO to Debug.Log
                            G[b, a] = 0
                            priomat[a, b] = prioweight
                            priomat[b, a] = prioweight #update prioweight of now oriented edge to lowest evidence of condition

        if (previous_G == G).all():
            break
        if i >= iter_max:
            raise Exception(f"Typed Meek is stucked. More than {iter_max} iterations.")

        previous_G = np.copy(G)

    print("typed meek without type consistency finished - current adjacency matrix: \n", G) #TODO to Debug.Log
    print("current priomat: \n", priomat) #TODO to Debug.Log
    return G, priomat

# step 4 type-consistent t-propagation

# taken from https://github.com/ServiceNow/typed-dag/blob/main/typed_pc/tmec.py
def typed_meek_majority_tag_with_consistency(cpdag: np.ndarray, tags: list, priomat : np.matrix, majority_rule_typed : bool = True, iter_max: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the Meek algorithm with the type consistency as described in Section 5 (from the CPDAG).

    :param cpdag: adjacency matrix of the CPDAG
    :param types: list of type of each node
    :param iter_max: The maximum number of iterations. If reached, an exception will be raised.
    """
    print("typed Meek with type consistency")
    # priomat checks are mostly redundant here, since a,b is undirected and should therefore have prioweight 0 #TODO

    n_nodes = cpdag.shape[0]

    G = np.copy(cpdag)


    # repeat until the graph is not changed by the algorithm
    # or too high number of iteration
    previous_G = np.copy(G)
    i = 0
    while True and i < iter_max: #TODO while true entfernen
        """ #TODO desc updaten
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
        print(f"majority mat on t-propagation step: {i}: \n", majoritymat) #TODO to Debug.Log
        print(f"priomat before type consistency: \n", priomat) #TODO to Debug.Log
        print(f"adjacency mat before type consistency: \n", G) #TODO to Debug.Log

        # Step 2: Orient all Edges where there is a higher type consisitency evidence for the current Graph (entry in majoritymat) than already priority in priomat
        G, priomat = orient_tagged_dag_after_majority_tag_matix_using_prio_mat(G, tags, majoritymat, priomat)

        # Apply Meek's rules (R1, R2, R3, R4) and the two-type fork rule (R5)
        for a, b, c in permutations(range(n_nodes), 3):
            # Orient any undirected edge a - b as a -> b if any of the following rules is satisfied:
            if G[a, b] != 1 or G[b, a] != 1:
                # Edge a - b is already oriented
                continue

            # R1: c -> a - b ==> a -> b
            if G[a, c] == 0 and G[c, a] == 1 and G[b, c] == 0 and G[c, b] == 0 and ((prioweight := min(priomat[a,c], priomat[c, a], priomat[b,c], priomat[c,b])) > max(priomat[b,a], priomat[a, b])): #check that prio evidence is stronger than the to orient edge
                print(f"meek R1: {c} -> {a} - {b}: orient to  {a} -> {b}")  #TODO to Debug.Log
                G[b, a] = 0
                priomat[a, b] = prioweight
                priomat[b, a] = prioweight #update prioweight of now oriented edge to lowest evidence of condition
            # R2: a -> c -> b and a - b ==> a -> b
            elif G[a, c] == 1 and G[c, a] == 0 and G[b, c] == 0 and G[c, b] == 1 and ((prioweight := min(priomat[a,c], priomat[c, a], priomat[b,c], priomat[c,b])) > max(priomat[b,a], priomat[a, b])): #check that prio evidence is stronger than the to orient edge
                print(f"meek R2: {a} -> {c} -> {b} and {a} - {b}: orient to {a} -> {b}")  #TODO to Debug.Log
                G[b, a] = 0
                priomat[a, b] = prioweight
                priomat[b, a] = prioweight #update prioweight of now oriented edge to lowest evidence of condition
            # R5: b - a - c and a-/-c and t(c) = t(b) ==> a -> b and a -> c (two-type fork)
            elif (
                G[a, c] == 1
                and G[c, a] == 1
                and G[b, c] == 0
                and G[c, b] == 0
                and ((prioweight := amount_of_matching_types_for_two_way_fork_from_taglist(G, tags, b, c, a)) > max(priomat[b,a], priomat[a, b])) #check if there are more tags where type matches for two way fork than evidence for current a-b edge
                and ((min(priomat[a,c], priomat[c, a], priomat[b,c], priomat[c,b])) >= max(priomat[b,a], priomat[a, b])) #check that prio evidence is stronger or equal than the to orient edge (need >= here since we use an undirected edge as evidence, which sholuld have evidence 0 meaing we only orient unoriented edges a,b and a,c (which is already or condition in the first place, so kinda useless))
            ):
                print(f"meek R5: {b} - {a} - {c} and t{c}: {type_of_from_tag_all_from_taglist(tags, c)} = t{b}: {type_of_from_tag_all_from_taglist(tags, b)}: orient to {a} -> {b} and {a} -> {c}")  #TODO to Debug.Log
                G[b, a] = 0
                G[c, a] = 0
                priomat[a, b] = prioweight
                priomat[b, a] = prioweight
                priomat[a, c] = prioweight
                priomat[c, a] = prioweight #update prioweight of now oriented edge to lowest evidence of condition
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
                            print(f"meek R3: {a} - {c} -> {b} and {a} - {d} -> {b} and {c} -/- {d} orient to {a} -> {b}")  #TODO to Debug.Log
                            G[b, a] = 0
                            priomat[a, b] = prioweight
                            priomat[b, a] = prioweight #update prioweight of now oriented edge to lowest evidence of condition
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
                            print(f"meek R4: {a} - {d} -> {c} -> {b} and {a} -> {c} or {c} -> {a} orient to {a} -> {b}")  #TODO to Debug.Log
                            G[b, a] = 0
                            priomat[a, b] = prioweight
                            priomat[b, a] = prioweight #update prioweight of now oriented edge to lowest evidence of condition

        if (previous_G == G).all():
            break
        if i >= iter_max:
            raise Exception(f"Typed Meek is stucked. More than {iter_max} iterations.")

        previous_G = np.copy(G)
        print("current adj. matrix: \n", G) #TODO to Debug.Log

    return G, priomat
