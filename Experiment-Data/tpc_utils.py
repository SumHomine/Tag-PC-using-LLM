
import logging
from random import shuffle

import networkx as nx
from itertools import permutations, combinations

import numpy as np

def set_types_as_int(dag, typeslist):
    current_node = 0
    for current_node in range(dag.number_of_nodes()):
        dag.nodes[current_node]["type"] = typeslist[current_node]
    return dag

def get_typelist_from_text(types):
    """
    :param types: text of params and corresponding assigned types in the form:
    A : T1
    B : T2
    :returns: list of int where each entry is a param that mapps to int representation of its type (i.e. [0, 1, 1, 2, 2, 3, 3] )
    """
    typelist = []
    lines = types.strip().split('\n')
    type_to_int = {}
    current_type_number = 0
    for line in lines:
        param, type = map(str.strip, line.split(':'))
        # make new number for new types
        if type not in type_to_int:
            type_to_int[type] =  current_type_number
            current_type_number += 1
        #append type from current param
        typelist.append(type_to_int[type])        
    return typelist

# the following methods are crudely copied from tag, thats why they are extremly bad performance wise, but I have a Deadline to meet!
def get_typelist_of_int_from_text(types, node_names_ordered):
    """
    :param tags: text of params and corresponding assigned types in the form:
        Cloudy : Weather
        Sprinkler : Watering
        Rain : Weather
        Wet_Grass : Plant_Con
    :param node_names_ordered: list of node names in true order of the data: 
    :returns: list of int where each entry is the int of the type representation of the node:
        [0, 1, 0, 2],  # Intuitive self-made
    """
    typelist_string = get_typelist_of_string_from_text(types, node_names_ordered)
    print(typelist_string)
    return turn_typelist_of_string_to_int(typelist_string)

def get_typelist_of_string_from_text(types, node_names_ordered):
    """
    :param types: text of params and corresponding assigned types in the form:
        Cloudy : Weather
        Sprinkler : Watering
        Rain : Weather
        Wet_Grass : Plant_Con 
    :param node_names_ordered: list of node names in true order of the data:
    :returns: list of string where each entry is the type for a node
        ["Weather", "Watering", "Weather", "Plant_Con"],  # Intuitive self-made
    """

    # Split input text into lines
    lines = types.strip().split("\n")

    # Initialize dict for node to tags
    type_dict = {}
    
    # Parse each line into dict
    for line in lines:
        node_name, node_type = line.split(":")
        type_dict[node_name.strip()] = node_type.strip()
    
    # Reorder the taglines according to node_names_ordered
    reordered_type_dict = reorder_typelines_in_node_name_order(type_dict, node_names_ordered)

    # Convert the reordered dict into a list of types in the correct order
    reordered_transposed_type_lists = [reordered_type_dict[node_name] for node_name in node_names_ordered]
    # reordered_transposed_type_lists = list(map(list, zip(*reordered_type_dict.values()))) # in tag

    return reordered_transposed_type_lists

def reorder_typelines_in_node_name_order(typelines, node_names_ordered):
    ordered_typelines = {}
    current_pos = 0 # For error recovering
    typelines_keys = list(typelines.keys()) # For error recovering
    for node_name in node_names_ordered:
        if not (node_name in typelines): # When user misspells node name in type
            positional_typeline_key = typelines_keys[current_pos]
            type_value = typelines[positional_typeline_key]
            ordered_typelines[node_name] = type_value
            print(f"WARNING: no fitting type has been found for \"{node_name}\", please check your nodes to make sure that you use the correct node names.") 
            print(f"For continued operation, node \"{node_name}\" got tagged with {type_value} based on current position {current_pos}, meaning node \"{node_name}\" got the type of \"{positional_typeline_key}\"")
        else:
            type_value = typelines[node_name]
            ordered_typelines[node_name] = type_value
        current_pos += 1
    return ordered_typelines


def turn_typelist_of_string_to_int(typelist):
    """
    :param taglist: list of list of string where each entry is a typelist of with one tag for all nodes
    [
        ["Weather", "Watering", "Weather", "Plant_Con"],  # Intuitive self-made
        ["Weather", "NotWeather", "Weather", "NotWeather"],  # One vs all approach
        ["NotWatering", "Watering", "Watering", "NotWatering"], # 2nd One vs all approach
    ]
    :returns: list of list of int where each entry is a typelist of int with one one tag representation as int for all nodes:
    [
        [0, 1, 0, 2],  # Intuitive self-made
        [0, 1, 0, 1],  # One vs all approach
        [0, 1, 1, 0], # 2nd One vs all approach
    ]
        )
    """

    typelist_int = []
    type_to_int = {}
    current_type_number = 0
    for type in typelist:
        if type not in type_to_int:
            type_to_int[type] =  current_type_number
            current_type_number += 1
        #append type from current param
        typelist_int.append(type_to_int[type])  
      

    return typelist_int

# currently Unused
def set_types(dag, types):
    """
    :param dag: nx.diGraph()thingy
    :param types: text of params and corresponding assigned types in the form:
    A : T1
    B : T2
    :returns: dag that where each param can be mapped to its corresponding assigned type
    """
    lines = types.strip().split('\n')
    current_node = 0
    for line in lines:
        param, type = map(str.strip, line.split(':'))
        dag.nodes[current_node]["type"] = type
        current_node += 1
    return dag

def format_seperating_sets(separating_sets):
    """
    :param separating_sets: np.ndarray matrix of separating sets, with each entry being either None or a list of tuples. Each tuple contains nodes that form a separating set.
    :returns: np.ndarray - matrix of same shape as `separating_sets`, where each entry is either None or set of nodes that form the separating set.
    """
    sep_sets = np.empty_like(separating_sets, dtype=object)

    # Iterate over the separating_sets and populate the sep_sets matrix
    for i in range(separating_sets.shape[0]):
        for j in range(separating_sets.shape[1]):
            if separating_sets[i, j] is not None:
                # Combine all tuples into a single set of nodes
                sep_sets[i, j] = set()
                for sep_set in separating_sets[i, j]:
                    sep_sets[i, j].update(sep_set)
            else:
                sep_sets[i, j] = None
    
    return(sep_sets)

def get_undirect_graph(true_adjacency_matrix):    
    # get shape
    n = true_adjacency_matrix.shape[0]
    
    # Initialize undirected adjacency matrix as a copy of input
    undirected_matrix = true_adjacency_matrix.copy()
    
    # Loop through the upper triangle of the matrix
    for i in range(n):
        for j in range(i + 1, n):
            if true_adjacency_matrix[i, j] == 1 or true_adjacency_matrix[j, i] == 1:
                undirected_matrix[i, j] = 1
                undirected_matrix[j, i] = 1
    
    return undirected_matrix

def get_separating_sets_using_true_skeleton(skeleton: nx.Graph, true_adjacency_mat: np.ndarray):
    """
    Get the true separating sets for each pair of nodes in the graph that are not directly connected.
    
    Parameters:
    - skeleton: NetworkX graph representing the true skeleton of the graph.
    - data: The dataset used for conditional independence testing.
    - alpha: Significance level for conditional independence tests.
    - indep_test: Method for conditional independence tests (e.g., 'fisherz').
    
    Returns:
    - separating_sets: A numpy array where each entry (i, j) is a list of separating sets for nodes i and j.
    """
    node_ids = skeleton.number_of_nodes()
    separating_sets = np.empty((node_ids, node_ids), dtype=object)
    combos = list(combinations(range(node_ids), 2))
    shuffle(combos)
    # Iterate over all Edges
    for (i, j) in combos:
        # If i and j are directly connected skip seperating sets ( separating_sets[i, j] and [j,i] will be NONE)
        if _has_any_edge(skeleton, i, j):
            continue

        neighbors_i = set(skeleton.neighbors(i))
        neighbors_j = set(skeleton.neighbors(j))
        possible_separators = neighbors_i.union(neighbors_j)
        possible_separators.discard(i)
        possible_separators.discard(j)

        separating_sets[i, j] = []
        separating_sets[j, i] = []
        for size in range(len(possible_separators) + 1):
            for subset in combinations(possible_separators, size):
                subset = list(subset)
                append_subset = True
                # iterate over subset to search for immorality
                for k in subset:
                   # found immorality i -> k <- j in subset -> not adding subset
                    if (true_adjacency_mat[i,k] == 1 and true_adjacency_mat[j,k] == 1 and true_adjacency_mat[k,i] == 0 and true_adjacency_mat[k,j] == 0): 
                        # print(f"breaking for: {i} and {j} on {subset} because of immorality {i} -> {k} <- {j}")
                        append_subset = False
                        break
    
                if (append_subset):
                    # print(f"appending: {i} and {j} on {subset}")
                    separating_sets[i, j].append(subset)
                    separating_sets[j, i].append(subset)
                    break  # We only need one separating set, break on finding the first

    return separating_sets

# the following dag parameters should be of type nx.DiGraph()
# copied from https://github.com/ServiceNow/typed-dag
def _has_both_edges(dag, i: int, j: int):
    """
    Check if edge i-j is unoriented
    """
    return dag.has_edge(i, j) and dag.has_edge(j, i)

# copied from https://github.com/ServiceNow/typed-dag
def _has_any_edge(dag, i: int, j: int):
    """
    Check if i and j are connected (irrespective of direction)
    """
    return dag.has_edge(i, j) or dag.has_edge(j, i)

# copied from https://github.com/ServiceNow/typed-dag
def type_of(dag, node: int):
    """
    Get the type of a node

    """
    return dag.nodes[node]["type"]

# copied from https://github.com/ServiceNow/typed-dag
def _orient(dag, n1: int, n2: int):
    """
    Orients all edges from type(node1) to type(node2). If types are the same, simply orient the edge between the nodes.

    """
    t1 = type_of(dag, n1)
    t2 = type_of(dag, n2)

    # Case: orient intra-type edges (as if not typed)
    if t1 == t2:
        if not _has_both_edges(dag, n1, n2):
            print(f"Edge {n1}-{n2} is already oriented. Not touching it.")
        else:
            logging.debug(f"... Orienting {n1} (t{t1}) -> {n2} (t{t2}) (intra-type)")
            dag.remove_edge(n2, n1)

    # Case: orient t-edge
    else:
        logging.debug(f"Orienting t-edge: {t1} --> {t2}")
        for _n1, _n2 in permutations(dag.nodes(), 2):
            if type_of(dag, _n1) == t1 and type_of(dag, _n2) == t2 and _has_both_edges(dag, _n1, _n2):
                logging.debug(f"... Orienting {_n1} (t{t1}) -> {_n2} (t{t2})")
                dag.remove_edge(_n2, _n1)
            elif (
                type_of(dag, _n1) == t1
                and type_of(dag, _n2) == t2
                # CPDAG contains at least one edge with t2 -> t1, while it should be t1 -> t2.
                and dag.has_edge(_n2, _n1)
                and not dag.has_edge(_n1, _n2)
            ):
                raise Exception(
                    f"State of inconsistency. CPDAG contains edge {_n2} (t{t2}) -> {_n1} (t{t1}), while the t-edge should be t{t1} -> t{t2}."
                )


# taken from https://github.com/ServiceNow/typed-dag/blob/main/typed_pc/tmec.py
def _update_tedge_orientation(G, type_g, types):
    """
    Detects which t-edges are oriented and unoriented and updates the type compatibility graph

    """
    type_g = np.copy(type_g)

    for a, b in permutations(range(G.shape[0]), 2):
        # XXX: No need to consider the same-type case, since the type matrix starts at identity.
        if types[a] == types[b]:
            continue
        # Detect unoriented t-edges
        if G[a, b] == 1 and G[b, a] == 1 and not (type_g[types[a], types[b]] + type_g[types[b], types[a]] == 1):
            type_g[types[a], types[b]] = 1
            type_g[types[b], types[a]] = 1
        # Detect oriented t-edges
        if G[a, b] == 1 and G[b, a] == 0:
            type_g[types[a], types[b]] = 1
            type_g[types[b], types[a]] = 0

    return type_g

# taken from https://github.com/ServiceNow/typed-dag/blob/main/typed_pc/tmec.py
def _orient_tedges(G, type_g, types):
    """
    Ensures that edges that belong to oriented t-edges are consistently oriented.

    Note: will not change the orientation of edges that are already oriented, even if they clash with the direction
          of the t-edge. This can happen if the CPDAG was not type consistant at the start of t-Meek.

    """
    G = np.copy(G)
    for a, b in permutations(range(G.shape[0]), 2):
        if type_g[types[a], types[b]] == 1 and type_g[types[b], types[a]] == 0 and G[a, b] == 1 and G[b, a] == 1:
            G[a, b] = 1
            G[b, a] = 0
    return G


