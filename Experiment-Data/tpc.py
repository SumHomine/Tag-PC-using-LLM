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

from tpc_utils import get_separating_sets_using_true_skeleton, get_typelist_from_text, get_typelist_of_int_from_text, get_undirect_graph, set_types_as_int, format_seperating_sets, _has_both_edges, _has_any_edge, _orient, type_of, _update_tedge_orientation, _orient_tedges
# from test_tpc import load_skeleton,  save_skeleton


def tpc (data, types, node_names, alpha=0.05, indep_test="fisherz", majority_rule=False):
    """
    :param data: string of data with nodes seperated with whitespace and entries by line break
    :param types: text of params and corresponding assigned types in the form:
        Cloudy : Weather
        Sprinkler : Watering
        Rain : Weather
        Wet_Grass : Plant_Con 
    :param node_names: list of String of the column names of the dataset in the correct order, used to correct order of tags when mistakenly in wrong order and for debugging
    :param alpha: significance level for skeleton discovery
    :param indep_test: str, type of the used independence test. default: fisherz, 
                other options are: "mv_fisherz", "mc_fisherz", "kci", "chisq", "gsq", "d_separation"
    :param majority_rule: bool, if true use majority rule to orient forks, if false use naive rule to orient forks
    :return: adjacency matrix of the DAG of type np.ndarray 

    step 1: infer skeleton
    step 2: orient v-structures & two type forks + type consistency
    step 3: t-propagation
    """
    #step 1
    typelist = get_typelist_of_int_from_text(types=types, node_names_ordered=node_names)
    skeleton, separating_sets, stat_tests = create_skeleton_using_causallearn(data, typelist, alpha, indep_test)

#    save_skeleton(skeleton=skeleton, separating_sets=separating_sets, stat_tests=stat_tests)     # -> use for debugging when you want to save a new skeleton
#    skeleton, separating_sets, stat_tests = load_skeleton()                                 # -> use for debugging when you dont want to wait to create skeleton -> also comment out create skeleton/seperating_sets


    #step 2
    if (majority_rule):
        out = orient_forks_majority_top1(skeleton=skeleton, sep_sets=separating_sets)
    else:
        out = orient_forks_naive(skeleton=skeleton, sep_sets=separating_sets)
    
    #step 3
#    out = pc_meek_rules(out) #untyped meek

    out_matrix, t_matrix = typed_meek(cpdag=nx.adjacency_matrix(out).todense(), types=typelist)

    return out_matrix, stat_tests, typelist # return nx.adjacency_matrix(out).todense() when your graph isn't already an adjacency matrix


def tpc_from_true_skeleton (dataname, types, majority_rule=True, data=None):
    """
    :param types: text of params and corresponding assigned types in the form:
        Cloudy : Weather
        Sprinkler : Watering
        Rain : Weather
        Wet_Grass : Plant_Con 
    :param majority_rule: bool, if true use majority rule to orient forks, if false use naive rule to orient forks
    :optional param data: string of data with nodes seperated with whitespace and entries by line break
    :return: adjacency matrix of the DAG of type np.ndarray
    :return: stat_tests of the DAG of type np.ndarray  
    :return: node names of type list<String> 

    step 1: infer skeleton
    step 2: orient v-structures & two type forks + type consistency
    step 3: t-propagation
    
    """
    #step 1 from true skelton
    skeleton, separating_sets, stat_tests, node_names, typelist = get_true_skeleton(dataname=dataname, types=types, data=data)

    print("skeleton: \n", nx.adjacency_matrix(skeleton).todense()) #TODO to Debug.Log
#    save_skeleton(skeleton=skeleton, separating_sets=separating_sets, stat_tests=stat_tests)     # -> use for debugging when you want to save a new skeleton
#    skeleton, separating_sets, stat_tests = load_skeleton()                                 # -> use for debugging when you dont want to wait to create skeleton -> also comment out create skeleton/seperating_sets

    #step 2
    if (majority_rule):
        out = orient_forks_majority_top1(skeleton=skeleton, sep_sets=separating_sets)
    else:
        out = orient_forks_naive(skeleton=skeleton, sep_sets=separating_sets)

    print("graph after forks-v: \n", nx.adjacency_matrix(out).todense())
 
    #step 3
    out_matrix, t_matrix = typed_meek(cpdag=nx.adjacency_matrix(out).todense(), types=typelist)

    print("resulting graph: \n", out_matrix)
    return out_matrix, stat_tests, node_names, typelist


# step 1 - infer skeleton (using the causallearn library):
def create_skeleton_using_causallearn(data, typelist, alpha=0.05, indep_test="fisherz"):
    # get conditional independence test object and use it to create skeleton
    cit = CIT(data, method=indep_test)
    skeleton = SkeletonDiscovery.skeleton_discovery(data, alpha, indep_test=cit, stable=True,)
    # get seperating_sets from sekelton
    separating_sets = skeleton.sepset
    separating_sets = format_seperating_sets(separating_sets)
    # print(separating_sets)
    #get skeleton as nx.Graph
    skeleton.to_nx_skeleton()
    skeleton = skeleton.nx_skel
    stat_tests = nx.adjacency_matrix(skeleton).todense()

    skeleton = set_types_as_int(skeleton, typelist)
    return skeleton, separating_sets, stat_tests


# step 1 - get correct skeleton from true graph (for assumption that skeleton is correct)
def get_true_skeleton(dataname : str, types, data):
    """
    get the true skeleton (and trueish seperating sets), only works for forest example and the bnf dataset (you might have to download additional .bif files)
    """

    match dataname:
        # since we already get the true skeleton with the correct parameters for causallearn we do just that
        case "forest":
            if (data == None):
                assert ValueError("since you are using non bnf data, you need give the function data ad np.ndarray")
            node_names = ["A","R","S","H","B","W","F"]
            skeleton, separating_sets, stat_tests = create_skeleton_using_causallearn(data, typelist, alpha = 0.05, indep_test="fisherz")
            return skeleton, separating_sets, stat_tests, node_names
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
    adjacency_mat = model['adjmat']
    adjacency_mat = adjacency_mat.astype(int) #get int representation
    print("true graph from bnf: \n", adjacency_mat) #to debug
    node_names = adjacency_mat.columns.tolist()
    adjacency_mat = adjacency_mat.values #delete headers
    skeleton_adjacency_mat = get_undirect_graph(adjacency_mat)
    print("skeleton: \n", skeleton_adjacency_mat) #to debug
    stat_tests = skeleton_adjacency_mat

    #get skeleton as nx.Graph
    skeleton = nx.from_numpy_array(skeleton_adjacency_mat)
    typelist = get_typelist_of_int_from_text(types=types, node_names_ordered=node_names)
    skeleton = set_types_as_int(skeleton, typelist)
    
    # Get separating sets
    separating_sets = get_separating_sets_using_true_skeleton(skeleton, adjacency_mat)
    separating_sets = format_seperating_sets(separating_sets)
    print("seperating sets: \n", separating_sets) #TODO to Debug.Log

    return skeleton, separating_sets, stat_tests, node_names, typelist

# the rest of the code is shamelessly stolen from https://github.com/ServiceNow/typed-dag


# step 2 - orient v-structures & two type forks + type consistency:

#######
# This is the part where we orient all immoralities and two-type forks.
# The behavior used to orient t-edges depends on the chosen strategy:
#   * Naive: orient as first encountered orientation
#   * Majority: orient using the most frequent orientation
#######


def orient_forks_naive(skeleton, sep_sets):
    """
    Orient immoralities and two-type forks

    Strategy: naive -- orient as first encountered

    """
    dag = skeleton.to_directed()
    node_ids = skeleton.nodes()

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
                # XXX: had to add the last two conditions in case k is no longer a child due to t-edge orientation
                logging.debug(
                    f"S: orient immorality {i} (t{type_of(dag, i)}) -> {k} (t{type_of(dag, k)}) <- {j} (t{type_of(dag, j)})"
                )
                _orient(dag, i, k)
                _orient(dag, j, k)

            # Case: we have an orientable two-type fork, i.e., it is not an immorality, so i <- k -> j
            elif (
                type_of(dag, i) == type_of(dag, j)
                and type_of(dag, i) != type_of(dag, k)
                and _has_both_edges(dag, i, k)
                and _has_both_edges(dag, j, k)
            ):
                logging.debug(
                    f"S: orient two-type fork {i} (t{type_of(dag, i)}) <- {k} (t{type_of(dag, k)}) -> {j} (t{type_of(dag, j)})"
                )
                _orient(dag, k, i)  # No need to orient k -> j. Will be done in this call since i,j have the same type.

    return dag


def orient_forks_majority_top1(skeleton, sep_sets):
    """
    Orient immoralities and two-type forks

    Strategy: majority -- orient using the most frequent orientation
    Particularity: Find the t-edge with most evidence, orient, repeat evidence collection.

    """
    dag = skeleton.to_directed()
    node_ids = skeleton.nodes()
    n_types = len(np.unique([type_of(dag, n) for n in dag.nodes()]))

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
                    if type_of(dag, i) == type_of(dag, j) == type_of(dag, k):
                        continue

                    logging.debug(
                        f"Step 1: evidence of orientation {i} (t{type_of(dag, i)}) -> {k} (t{type_of(dag, k)}) <- {j} (t{type_of(dag, j)})"
                    )
                    # Increment t-edge orientation evidence
                    print(tedge_evidence)
                    print(type_of(dag, i))
                    print(type_of(dag, k))
                    tedge_evidence[type_of(dag, i), type_of(dag, k)] += 1
                    tedge_evidence[type_of(dag, j), type_of(dag, k)] += 1

                    # Determine conditional orientations
                    conditional_orientations[(type_of(dag, j), type_of(dag, k))].append((i, k))
                    conditional_orientations[(type_of(dag, i), type_of(dag, k))].append((j, k))

                # Case: we have an orientable two-type fork, i.e., it is not an immorality, so i <- k -> j
                elif type_of(dag, i) == type_of(dag, j) and type_of(dag, i) != type_of(dag, k):
                    # Check if already oriented
                    if not _has_both_edges(dag, i, k) or not _has_both_edges(dag, j, k):
                        continue

                    logging.debug(
                        f"Step 1: evidence of orientation {i} (t{type_of(dag, i)}) <- {k} (t{type_of(dag, k)}) -> {j} (t{type_of(dag, j)})"
                    )
                    # Count evidence only once per t-edge
                    tedge_evidence[type_of(dag, k), type_of(dag, i)] += 2

        # Step 2: Orient t-edges based on evidence
        np.fill_diagonal(tedge_evidence, 0)
        ti, tj = np.unravel_index(tedge_evidence.argmax(), tedge_evidence.shape)
        if np.isclose(tedge_evidence[ti, tj], 0):
            continue

        # Orient!
        print("Evidence", tedge_evidence[ti, tj])
        print(conditional_orientations)
        oriented_tedge = True
        first_ti = [n for n in dag.nodes() if type_of(dag, n) == ti][0]
        first_tj = [n for n in dag.nodes() if type_of(dag, n) == tj][0]
        logging.debug(
            f"Step 2: orienting t-edge according to max evidence. t{ti} -> t{tj} ({tedge_evidence[ti, tj]}) vs t{ti} <- t{tj} ({tedge_evidence[tj, ti]})"
        )
        _orient(dag, first_ti, first_tj)
        cond = Counter(conditional_orientations[ti, tj])
        for (n1, n2), count in cond.items():
            logging.debug(f"... conditional orientation {n1}->{n2} (count: {count}).")
            if (n2, n1) in cond and cond[n2, n1] > count:
                logging.debug(
                    f"Skipping this one. Will orient its counterpart ({n2}, {n1}) since it's more frequent: {cond[n2, n1]}."
                )
            else:
                _orient(dag, n1, n2)
    logging.debug("Steps 1-2 completed. Moving to single-type forks.")

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
                if not (type_of(dag, i) == type_of(dag, j) == type_of(dag, k)):
                    continue
                logging.debug(
                    f"Step 3: orient immorality {i} (t{type_of(dag, i)}) -> {k} (t{type_of(dag, k)}) <- {j} (t{type_of(dag, j)})"
                )
                _orient(dag, i, k)
                _orient(dag, j, k)

    return dag


#step 3 - t-propagation:

# taken from https://github.com/ServiceNow/typed-dag/blob/main/typed_pc/tmec.py
def typed_meek(cpdag: np.ndarray, types: list, iter_max: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the Meek algorithm with the type consistency as described in Section 5 (from the CPDAG).

    :param cpdag: adjacency matrix of the CPDAG
    :param types: list of type of each node
    :param iter_max: The maximum number of iterations. If reached, an exception will be raised.
    """
    n_nodes = cpdag.shape[0]
    types = np.asarray(types)
    n_types = len(np.unique(types))

    G = np.copy(cpdag)
    type_g = np.eye(n_types)  # Identity matrix to allow intra-type edges

    # repeat until the graph is not changed by the algorithm
    # or too high number of iteration
    previous_G = np.copy(G)
    i = 0
    while True and i < iter_max:
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
        # XXX: This will not change the orientation of oriented edges (i.e., if the CPDAG was not consistant)
        G = _orient_tedges(G, type_g, types)

        # Step 3: Apply Meek's rules (R1, R2, R3, R4) and the two-type fork rule (R5)
        for a, b, c in permutations(range(n_nodes), 3):
            # Orient any undirected edge a - b as a -> b if any of the following rules is satisfied:
            if G[a, b] != 1 or G[b, a] != 1:
                # Edge a - b is already oriented
                continue

            # R1: c -> a - b ==> a -> b
            if G[a, c] == 0 and G[c, a] == 1 and G[b, c] == 0 and G[c, b] == 0:
                G[b, a] = 0
            # R2: a -> c -> b and a - b ==> a -> b
            elif G[a, c] == 1 and G[c, a] == 0 and G[b, c] == 0 and G[c, b] == 1:
                G[b, a] = 0
            # R5: b - a - c and a-/-c and t(c) = t(b) != t(a) ==> a -> b and a -> c (two-type fork)
            elif (
                G[a, c] == 1
                and G[c, a] == 1
                and G[b, c] == 0
                and G[c, b] == 0
                and types[b] == types[c]
                and types[b] != types[a]  # Make sure there are at least two types
            ):
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
                            G[b, a] = 0

        if (previous_G == G).all():
            break
        if i >= iter_max:
            raise Exception(f"Typed Meek is stucked. More than {iter_max} iterations.")

        previous_G = np.copy(G)

    return G, type_g


# Unused -> for PC Algo using normal Meek
# Note: This code is a snippet taken from the pcalg package.
def pc_meek_rules(dag):
    """
    Step 3: Meek rules portion of the PC algorithm

    """
    node_ids = dag.nodes()

    # For all the combination of nodes i and j, apply the following
    # rules.
    old_dag = dag.copy()
    while True:
        for (i, j) in permutations(node_ids, 2):
            # Rule 1: Orient i-j into i->j whenever there is an arrow k->i
            # such that k and j are nonadjacent.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Look all the predecessors of i.
                for k in dag.predecessors(i):
                    # Skip if there is an arrow i->k.
                    if dag.has_edge(i, k):
                        continue
                    # Skip if k and j are adjacent.
                    if _has_any_edge(dag, k, j):
                        continue
                    # Make i-j into i->j
                    logging.debug("R1: remove edge (%s, %s)" % (j, i))
                    _orient(dag, i, j)
                    break

            # Rule 2: Orient i-j into i->j whenever there is a chain
            # i->k->j.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Find nodes k where k is i->k.
                succs_i = set()
                for k in dag.successors(i):
                    if not dag.has_edge(k, i):
                        succs_i.add(k)
                # Find nodes j where j is k->j.
                preds_j = set()
                for k in dag.predecessors(j):
                    if not dag.has_edge(j, k):
                        preds_j.add(k)
                # Check if there is any node k where i->k->j.
                if len(succs_i & preds_j) > 0:
                    # Make i-j into i->j
                    logging.debug("R2: remove edge (%s, %s)" % (j, i))
                    _orient(dag, i, j)

            # Rule 3: Orient i-j into i->j whenever there are two chains
            # i-k->j and i-l->j such that k and l are nonadjacent.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Find nodes k where i-k.
                adj_i = set()
                for k in dag.successors(i):
                    if dag.has_edge(k, i):
                        adj_i.add(k)
                # For all the pairs of nodes in adj_i,
                for (k, l) in combinations(adj_i, 2):
                    # Skip if k and l are adjacent.
                    if _has_any_edge(dag, k, l):
                        continue
                    # Skip if not k->j.
                    if dag.has_edge(j, k) or (not dag.has_edge(k, j)):
                        continue
                    # Skip if not l->j.
                    if dag.has_edge(j, l) or (not dag.has_edge(l, j)):
                        continue
                    # Make i-j into i->j.
                    logging.debug("R3: remove edge (%s, %s)" % (j, i))
                    _orient(dag, i, j)
                    break

            # Rule 4: Orient i-j into i->j whenever there are two chains
            # i-k->l and k->l->j such that k and j are nonadjacent.
            # TODO: validate me
            if _has_both_edges(dag, i, j):
                # Find nodes k where i-k.
                adj_i = set()
                for k in dag.successors(i):
                    if dag.has_edge(k, i):
                        adj_i.add(k)

                # Find nodes l where l -> j
                preds_j = set()
                for l in dag.predecessors(j):
                    if not dag.has_edge(j, l):
                        preds_j.add(l)

                # Find nodes where k -> l
                for k in adj_i:
                    for l in preds_j:
                        if dag.has_edge(k, l) and not dag.has_edge(l, k):
                            _orient(dag, i, j)
                            break

        if nx.is_isomorphic(dag, old_dag):
            break
        old_dag = dag.copy()

    return dag