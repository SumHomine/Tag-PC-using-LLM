
from collections import Counter, defaultdict
import logging
from random import shuffle
from typing import Tuple

import networkx as nx
from itertools import permutations, combinations

from causallearn.utils.cit import *

import numpy as np

from cdt.metrics import SHD
from cdt.metrics import SID

def set_types_as_int(dag, typeslist):
    current_node = 0
    for current_node in range(dag.number_of_nodes()):
        dag.nodes[current_node]["type"] = typeslist[current_node]
    return dag

def set_tags_as_int(dag, taglist):
    current_type = 0
    for typelist in taglist:
        current_node = 0
        for current_node in range(dag.number_of_nodes()):
            dag.nodes[current_node]["type" + str(current_type)] = typelist[current_node]
        current_type += 1
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

def reorder_taglines_in_node_name_order(taglines, node_names_ordered):
    ordered_taglines = {}
    current_pos = 0 #for error recovering
    taglines_keys = list(taglines.keys()) #for error recovering
    for node_name in node_names_ordered:
        if not (node_name in taglines): #when user misspells nodename in tag
            positional_tagline_key = taglines_keys[current_pos]
            tags = taglines[positional_tagline_key]
            ordered_taglines[node_name] = tags
            print(f"WARNING: no fitting node has been found for \"{node_name}\", please check your tags to make sure that you use the correct node names.") 
            print(f"For Continued Operation node \"{node_name}\" got tagged with {tags} based on current position {current_pos}, meaning node \"{node_name}\" got the tags of \"{positional_tagline_key}\"")
        else:
            tags = taglines[node_name] #fix?
            ordered_taglines[node_name] = tags
        current_pos += 1
    return ordered_taglines

def get_taglist_of_string_from_text(tags, node_names_ordered):
    """
    :param tags: text of params and corresponding assigned tags with tags being seperated by comma in the form:
        Cloudy : Weather, Weather, NotWatering
        Sprinkler : Watering, NotWeather, Watering
        Rain : Weather, Weather, Watering
        Wet_Grass : Plant_Con, NotWeather, NotWatering   
    :param node_names_ordered: list of node names in true order of the data:
    :returns: list of list of string where each entry is a typelist of with one tag for all nodes
    [
        ["Weather", "Watering", "Weather", "Plant_Con"],  # Intuitive self-made
        ["Weather", "NotWeather", "Weather", "NotWeather"],  # One vs all approach
        ["NotWatering", "Watering", "Watering", "NotWatering"], # 2nd One vs all approach
    ] 
        )
    """

    # Split input text into lines
    lines = tags.strip().split("\n")

    # Initialize dict for node to tags
    tag_dict = {}
    
    # Parse each line into dict
    for line in lines:
        node_name, tag_str = line.split(":")
        tags = tag_str.strip().split(", ")
        tag_dict[node_name.strip()] = tags
    
    # Reorder the taglines according to node_names_ordered
    reordered_tag_dict = reorder_taglines_in_node_name_order(tag_dict, node_names_ordered)

    # Convert and transpose reordered dict into list of lists so that every list entry functions as a typelist
    reordered_transposed_tag_lists = list(map(list, zip(*reordered_tag_dict.values())))

    # Transpose taglist so that every list entry functions as a typelist
    # transposed_tags = list(map(list, zip(*tag_lists_node)))
    
    return reordered_transposed_tag_lists

def turn_taglist_of_string_to_int(taglist):
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
    taglist_int = []
    for typelist in taglist:
        typelist_int = []
        type_to_int = {}
        current_type_number = 0
        for type in typelist:
            if type not in type_to_int:
                type_to_int[type] =  current_type_number
                current_type_number += 1
        #append type from current param
            typelist_int.append(type_to_int[type])  
        taglist_int.append(typelist_int)          

    return taglist_int

def get_taglist_of_int_from_text(tags, node_names_ordered):
    """
    :param tags: text of params and corresponding assigned tags with tags being seperated by comma in the form:
        Cloudy : Weather, Weather, NotWatering
        Sprinkler : Watering, NotWeather, Watering
        Rain : Weather, Weather, Watering
        Wet_Grass : Plant_Con, NotWeather, NotWatering  
    :param node_names_ordered: list of node names in true order of the data: 
    :returns: list of list of int where each entry is a typelist of int with one one tag representation as int for all nodes:
    [
        [0, 1, 0, 2],  # Intuitive self-made
        [0, 1, 0, 1],  # One vs all approach
        [0, 1, 1, 0], # 2nd One vs all approach
    ]
        )
    """
    taglist_string = get_taglist_of_string_from_text(tags, node_names_ordered)
    return turn_taglist_of_string_to_int(taglist_string)

    #llm util function:
def get_taglist_from_llm_output(tag_list_string):
    taglist = tag_list_string.split(',')
    taglist = list(map(str.lstrip, taglist)) #remove leading whitespaces for all tags
    return taglist

    #llm util function:
def recover_tag_string_onevsall_using_taglist(tags_string : str, tag_list : list, node_names = []): #node_names is optional but helps alot
    # turn tags temporarely into dict (I know that this is mega ineficient, but I want to protect modularity)    
    # split input text into lines
    lines = tags_string.strip().split("\n")

    # initialize dict for node to tags
    tag_dict = {}
    
    # parse each line into dict
    for line in lines:
        # skip lines in wrong format that are properly just unnecessary words (since LLama just cant seem to shut up)
        if ":" not in line:
            continue

        node_name, tag_str = line.split(":")
        tags = tag_str.strip().split(", ")
        tag_dict[node_name.strip()] = tags
    
    # fill node_names using tag dict, if no argument was given
    if not node_names:
        node_names = list(tag_dict.keys())
    
    # recover string by fixing ordering and adding not<tags> for missing (under one vs all tagging assumption)
    print("Recovering Tags...") #Somehow hier nach falsch geschriebenen tags checken
    recovered_tag_string = "\n"
    for node in node_names:
        if node in tag_dict: #if node is in dict iterate normaly
            dict_tag_list = tag_dict[node]
        else: #if node is not in dict LLM forgot to print it because it does that sometimes ffs, just give only no tag
            dict_tag_list = [] 
        tag_counter = 0
        recovered_tag_string += (node + " : ") # write node name in resulting string
        while tag_counter < len(tag_list): #iterate over all true tags
            reference_tag = tag_list[tag_counter]
            if (reference_tag in dict_tag_list): # tag is attributed in our dict
                dict_tag_list.remove(reference_tag) #remove tag from dict (for debug reasons)
                recovered_tag_string += reference_tag  #write tag into string
                if (tag_counter < len(tag_list) - 1):
                    recovered_tag_string += ", " #write comma if we are not on the last tag
            elif (("Tag" + str(tag_counter + 1)) in dict_tag_list): #LLM fucked up and just wrote the tag + its tagnumber (LLM will start counting at 1 therefore +1) instead of its name (happens way to often)
                dict_tag_list.remove(("Tag" + str(tag_counter + 1))) #remove tag from dict (for debug reasons)
                recovered_tag_string += reference_tag  #write tag into string
                if (tag_counter < len(tag_list) - 1):
                    recovered_tag_string += ", " #write comma if we are not on the last tag
            else:  # no idea what tag is used, we assume a not<tag>
                recovered_tag_string += ("not" + reference_tag)   #write not<tag> into string
                if (tag_counter < len(tag_list) - 1):
                    recovered_tag_string += ", " #write comma if we are not on the last tag
                
            tag_counter += 1

        recovered_tag_string += "\n" #we iterated over all tags for this node -> end line
        #test if there are still elems in dict that were ignored because LLM wanted to share some bs again
        if (dict_tag_list): print(f"unidentified tags: {dict_tag_list} for node: <{node}>")
    

    return recovered_tag_string

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
    
    :param skeleton: NetworkX graph representing the true skeleton of the graph.
    :param data: Dataset used for performing conditional independence tests.
    :param alpha: float, significance level for the conditional independence tests.
    :param indep_test: str, method for conditional independence tests (e.g., 'fisherz').

    :return: separating_sets of type np.ndarray where each entry (i, j) is a list of separating sets for nodes i and j.
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

def get_priomat_from_skeleton(dag, taglist):
    """
    param dag: nx.adjacency matrix of Skeleton
    return priomat: matrix where every edge is 0 and no edge is number of tags + 1
    """
    priomat = np.where(dag == 1, 0, len(taglist) + 1)
    return  priomat

def assign_deleted_edges_in_priomat_evidence(dag, priomat, max_weight):
    """
    param dag: nx.adjacency matrix of dag
    return priomat: matrix of evidence
    return priomat: updated matrix where every no edge is number of tags + 1
    """
    for i in range(len(dag)):
        for j in range(len(dag[i])):
            if dag[i][j] == 0 and dag[j][i] == 0: #deleted edge
                priomat[i][j] = max_weight
                priomat[j][i] = max_weight
    return priomat

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

def _has_directed_edge(dag, i: int, j: int):
    """
    Check if there is an edge from i to j
    """
    return dag.has_edge(i, j)

def _has_only_directed_edge(dag, i: int, j: int):
    """
    Check if there is an edge from i to j but not j to i
    """
    return dag.has_edge(i, j) and not dag.has_edge(j,i)



def exists_entry_in_forkevidencelist(two_way_evidence, prio_weight, i, k, j):
    for entry in two_way_evidence:
        if entry == [prio_weight, i, k, j]:
            return True
    return False

def fork_clashes_immorality(dag, fork):
    prio_weight, i, k, j = fork # i <- k -> j
    if _has_only_directed_edge(dag,i,k):
        print(f"fork {i} <- {k} -> {j} clashes with immorality at {i} -> {k}, deleting it!") #TODO to logging.debug
        return True
    if _has_only_directed_edge(dag,j,k):
        print(f"fork {i} <- {k} -> {j} clashes with immorality at {j} -> {k}, deleting it!") #TODO to logging.debug
        return True
    return False

def are_forks_clashing(fork1, fork2):
    prio_weight1, i1, k1, j1 = fork1 # i1 <- k1 -> j1
    prio_weight2, i2, k2, j2 = fork2 # i2 <- k2 -> j2

    # there is a clash when one forks has an edge (either i <- k or k -> j) that is direct in the other direction of the other fork
    # case i-k switched with j-k: i1=k2 <1-2> k1=j2 and case i2=k1 <2-1> k2=j1
    if ((i1 == k2 and k1 == j2) or (i2 == k1 and k2 == j1)):
        return True
    # case i-k switched with itself: ik1=i2 <2-1> i1=k2 and case k2=i1 <1-2> i2=k1
    if ((k1 == i2 and i1 == k2) or (k2 == i1 and i2 == k1)):
        return True
    # case j-k switched with itself: k1=j2 <2-1> j1=k2 and case k2=j1 <1-2> j2=k1
    if ((k1 == j2 and j1 == k2) or (k1 == j2 and j1 == k2)):
        return True

    return False

# copied from https://github.com/ServiceNow/typed-dag -> adjustated to Tag
def type_of_from_tag_single(dag, node: int, current_tag: int):
    """
    Get the type of a node from a specific tag

    """
    return dag.nodes[node]["type" + str(current_tag)]

def type_of_from_tag_all(dag, node: int):
    """
    Get all types of a node

    """
    tags = []
    node_data = dag.nodes[node]
    for type, tag in node_data.items():
        if type.startswith("type"):
            tags.append(tag)
    return tags

def type_of_from_tag_all_from_taglist(tags : np.ndarray, node: int):
    """
    Get all types of a node from taglist

    """
    tags_node = []
    for typelist in tags:
        tags_node.append(typelist[node])
    return tags_node

def amount_of_matching_types_for_two_way_fork(dag, i : int, j : int, k : int):
    """
    get amount of matching tags to get a two way fork: i <- k -> j for tags with same type: i,j, different tag k
    return: times that tags give us a two ay fork. with 0 being that there is no two way fork for any tag
    """
    matching_types = 0
    # iterate over all types
    for current_type in range(len(dag.nodes[0])):
        if(dag.nodes[i]["type" + str(current_type)] == dag.nodes[j]["type" + str(current_type)] # i j have same type
           and dag.nodes[i]["type" + str(current_type)] != dag.nodes[k]["type" + str(current_type)] # k has diferent type than i 
           ):
            matching_types += 1
    return matching_types

# used for t-propagation when we use adjacency matrix instead of nx.graph
def amount_of_matching_types_for_two_way_fork_from_taglist(dag, taglist, i : int, j : int, k : int):
    """
    get amount of matching tags to get a two way fork: i <- k -> j for tags with same type: i,j, different tag k
    return: times that tags give us a two way fork. with 0 being that there is no two way fork for any tag
    """
    matching_types = 0
    # iterate over all types
    for typelist in taglist:
        if (typelist[i] == typelist[j] # i j have same type
            and typelist[i] != typelist[k] # k has diferent type than i 
            ):
            matching_types += 1
#            print(f"found matching fork: t1 {typelist[i]} == t2 {typelist[j]} # i j have same type  and t1 {typelist[i]} != t2 {typelist[k]}")
    return matching_types

def get_list_of_matching_types_for_two_way_fork(dag, i : int, j : int, k : int):
    """
    get amount of matching tags to get a two way fork: i <- k -> j for tags with same type: i,j, different tag k
    return: times that tags give us a two ay fork. with 0 being that there is no two way fork for any tag
    """
    matching_types_list = []
    # iterate over all types
    for current_type in range(len(dag.nodes[0])):
        if(dag.nodes[i]["type" + str(current_type)] == dag.nodes[j]["type" + str(current_type)] # i j have same type
           and dag.nodes[i]["type" + str(current_type)] != dag.nodes[k]["type" + str(current_type)] # k has ddiferent type than i 
           ):
            matching_types_list.append(current_type)
    return matching_types_list
    
 
# copied from https://github.com/ServiceNow/typed-dag -> adjustated to Tag
def _orient_tagged(dag, n1: int, n2: int, current_type: int):
    """
    Orients all edges from type(node1) to type(node2). If types are the same, simply orient the edge between the nodes.

    """
    t1 = type_of_from_tag_single(dag, n1, current_type)
    t2 = type_of_from_tag_single(dag, n2, current_type)

    # Case: orient intra-type edges (as if not typed)
    if t1 == t2:
        if not _has_both_edges(dag, n1, n2):
            print(f"Edge {n1}-{n2} is already oriented. Not touching it.")
        else:
            logging.debug(f"... Orienting {n1} (t{t1}) -> {n2} (t{t2}) (intra-type)")
            dag.remove_edge(n2, n1)

    # Case: orient t-edge
    else:
        print(f"Orienting t-edge: {t1} --> {t2}") #TODO to logging.debug
        for _n1, _n2 in permutations(dag.nodes(), 2):
            if type_of_from_tag_single(dag, _n1, current_type) == t1 and type_of_from_tag_single(dag, _n2, current_type) == t2 and _has_both_edges(dag, _n1, _n2):
                logging.debug(f"... Orienting {_n1} (t{t1}) -> {_n2} (t{t2})")
                dag.remove_edge(_n2, _n1)
            elif (
                type_of_from_tag_single(dag, _n1, current_type) == t1
                and type_of_from_tag_single(dag, _n2, current_type) == t2
                # CPDAG contains at least one edge with t2 -> t1, while it should be t1 -> t2.
                and dag.has_edge(_n2, _n1)
                and not dag.has_edge(_n1, _n2)
            ):
                
                    print(f"State of inconsistency. CPDAG contains edge {_n2} (t{t2}) -> {_n1} (t{t1}), while the t-edge should be t{t1} -> t{t2}.")
                    print("ignoring edge for now")


# copied from https://github.com/ServiceNow/typed-dag -> adjustated to Tag
def _orient_typeless_with_priomat(dag, n1: int, n2: int, priomat, prio_weight):
    """
    Orients the edge n1 to n2. 

    """

    # Case: orient intra-type edges (as if not typed)
    if not _has_both_edges(dag, n1, n2):
        if (_has_directed_edge(dag, n1, n2)):
            print(f"Edge {n1} -> {n2} is already correctly directed not touching it")
        elif(_has_directed_edge(dag, n2, n1)):
            if (priomat[n1][n2] < prio_weight and priomat[n2][n1] < prio_weight):
                print(f"REORIENTING EDGE {n2} -> {n1} TO {n1} -> {n2} BECAUSE OF PRIO EVIDENCE")
                dag.add_edge(n1, n2)
                dag.remove_edge(n2, n1)
                priomat[n1][n2] = prio_weight
                priomat[n2][n1] = prio_weight
            else:
                print(f"Edge {n2} -> {n1} is oriented in the other direction, but prio evidence is to low, NOT TOUCHING IT")
        
    else:
        # remove edge in reverse direction, add priomat entry for both direction
        logging.debug(f"... Orienting {n1} -> {n2}")
        dag.remove_edge(n2, n1)
        priomat[n1][n2] = prio_weight
        priomat[n2][n1] = prio_weight

def _orient_typeless_with_priomat_cycle_check_nx(dag, n1: int, n2: int, priomat, prio_weight):
    """
    Orients the edge n1 to n2 on an nx.graph. while checking if that introduces cycles and updating priomatrix
    """

    # Case: orient intra-type edges (as if not typed)
    if not _has_both_edges(dag, n1, n2):
        if (_has_directed_edge(dag, n1, n2)):
            print(f"Edge {n1} -> {n2} is already correctly directed, updating priomat only")
            priomat[n1][n2] = max(prio_weight, priomat[n1][n2])
            priomat[n2][n1] = max(prio_weight, priomat[n2][n1])
        elif(_has_directed_edge(dag, n2, n1)):
            if (priomat[n1][n2] < prio_weight and priomat[n2][n1] < prio_weight):
                print(f"REORIENTING EDGE {n2} -> {n1} TO {n1} -> {n2} BECAUSE OF PRIO EVIDENCE: {prio_weight}")
                dag.add_edge(n1, n2)
                dag.remove_edge(n2, n1)
                if (contains_cycle(nx.adjacency_matrix(dag).todense())): # cycle check
                    print(f"CYCLE FOUND AFTER REORIENTING {n1} -> {n2}, DEORIENTING EDGE!")
                    dag.add_edge(n2, n1) #rereorient
                    dag.remove_edge(n1, n2)
                    return
                priomat[n1][n2] = prio_weight
                priomat[n2][n1] = prio_weight
            else:
                print(f"Edge {n2} -> {n1} is oriented in the other direction, but prio evidence is too low, NOT TOUCHING IT")
        
    else:
        # remove edge in reverse direction, add priomat entry for both direction
        logging.debug(f"... Orienting {n1} -> {n2}")
        dag.remove_edge(n2, n1)
        if (contains_cycle(nx.adjacency_matrix(dag).todense())): # cycle check
            print(f"CYCLE FOUND AFTER ORIENTING {n1} -> {n2}, DEORIENTING EDGE!")
            dag.add_edge(n2, n1) #reorient
            return
        priomat[n1][n2] = prio_weight
        priomat[n2][n1] = prio_weight


def _orient_typeless_with_priomat_cycle_check_adjmat(dag, n1: int, n2: int, priomat, prio_weight):
    """
    Orients the edge n1 to n2 on an adacency matrix. while checking if that introduces cycles and updating priomatrix
    """

    if dag[n1][n2] == 1 and dag[n2][n1] == 1:    # unoriented edge
        # remove edge in reverse direction, add priomat entry for both direction
        logging.debug(f"... Orienting {n1} -> {n2}")
        dag[n2][n1] = 0
        if (contains_cycle(dag)): # cycle check
            print(f"CYCLE FOUND AFTER ORIENTING {n1} -> {n2}, DEORIENTING EDGE!") 
            dag[n2][n1] = 1 #reorient
            return
        priomat[n1][n2] = prio_weight
        priomat[n2][n1] = prio_weight

    elif dag[n1][n2] == 1 and dag[n2][n1] == 0: # already correctly oriented
        print(f"Edge {n1} -> {n2} is already correctly directed, updating priomat only")
        priomat[n1][n2] = max(prio_weight, priomat[n1][n2])
        priomat[n2][n1] = max(prio_weight, priomat[n2][n1])

    elif dag[n1][n2] == 0 and dag[n2][n1] == 1: # opposite direction of edge
        if (priomat[n1][n2] < prio_weight and priomat[n2][n1] < prio_weight):
            print(f"REORIENTING EDGE {n2} -> {n1} TO {n1} -> {n2} BECAUSE OF PRIO EVIDENCE: {prio_weight}")
            dag[n1][n2] = 1
            dag[n2][n1] = 0
            if (contains_cycle(dag)): # cycle check
                print(f"CYCLE FOUND AFTER REORIENTING {n1} -> {n2}, DEORIENTING EDGE!")
                dag[n1][n2] = 0 #rereorienting
                dag[n2][n1] = 1
                return
            priomat[n1][n2] = prio_weight
            priomat[n2][n1] = prio_weight
        else:
            print(f"Edge {n2} -> {n1} is oriented in the other direction, but prio evidence is too low, NOT TOUCHING IT")
    



def has_cycle_dfs(node, visited, rec_stack, adj_matrix, cycle=[]):
    # mark current node as visited
    visited[node] = True
    rec_stack[node] = True
    
    # explore all neighbors
    for neighbor in range(len(adj_matrix)):
        if adj_matrix[node][neighbor] == 1:  # edge from node to neighbor
            if not visited[neighbor]:
                if has_cycle_dfs(neighbor, visited, rec_stack, adj_matrix, cycle=cycle): # recursion step to find further edges
                    cycle.append(neighbor) # for debugging
                    if node in cycle: cycle.append(node) # for debugging
                    return True
            elif rec_stack[neighbor]: #recursion anchor: ends recursion when an edge links to a node that was already encountered in the current path
                cycle.append(neighbor) # for debugging
                return True  # Cycle detected
    
    # Remove the node from the recursion stack
    rec_stack[node] = False
    return False

def contains_cycle(dag):
    # Remove undirected edges first
    adj_matrix = _get_matrix_with_only_oriented_edges_of_dag(dag)
    
    n = len(adj_matrix)
    visited = [False] * n
    rec_stack = [False] * n
    cycle = []

    # Check for cycles in the graph
    for node in range(n):
        if not visited[node]:
            if has_cycle_dfs(node, visited, rec_stack, adj_matrix, cycle=cycle):
                print(f"CYCLE FOUND: {cycle}") # for debugging
                return True
    
    return False


def get_majority_tag_matrix(cpdag: np.ndarray, taglist: list):
    majoritymat = np.zeros_like(cpdag)
    for typelist in taglist:
        current_adjacency_mat = np.copy(cpdag)
        n_types = len(np.unique(typelist))
        type_g = np.eye(n_types)
        #make current_adjacency_mat type consistent for the current typelist (will only orient undirected edges (not type consistent edges stay the same))
        # Step 1: Determine the orientation of all t-edges based on t-edges (some may be unoriented)
        type_g = _update_tedge_orientation(current_adjacency_mat, type_g, typelist)

        # Step 2: Orient all edges of the same type in the same direction if their t-edge is oriented.
        # XXX: This will not change the orientation of oriented edges (i.e., if the CPDAG was not consistant) XXX: Guldan edit: I sure hope so
        current_adjacency_mat = _orient_tedges(current_adjacency_mat, type_g, typelist)

        # Step 3: add oriented edges of current_adjacency_mat to majoritymat -> high numbers mean that they are often directed that way
        orientmat = _get_matrix_with_only_oriented_edges_of_dag(current_adjacency_mat)
        majoritymat += orientmat

    return majoritymat


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
            if (type_g[types[b], types[a]] == 1):
                print(f"reorienting type", types[a], "to", types[b], "meaning Graph was type insconsistent!") #TODO to debug.log for publishing
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


def _get_matrix_with_only_oriented_edges_of_dag(adjacency_mat: np.ndarray):
    # helper function for get_majority_tag_matrix
    orientmat = np.zeros_like(adjacency_mat)
    for a, b in permutations(range(adjacency_mat.shape[0]), 2):
        if (adjacency_mat[a, b] == 1 and adjacency_mat[b, a] == 0):
            orientmat[a,b] = 1

    return orientmat


def get_majority_tag_matrix_using_priomat_type_majority(cpdag: np.ndarray, taglist: list, priomat: np.ndarray):
    """
    Generates a majority tag matrix using type majority consistency and priority matrix.

    :param cpdag: np.ndarray, adjacency matrix of graph.
    :param taglist: list of list of int, where each entry is a typelist mapping nodes to their types.
    :param priomat: np.ndarray, priority matrix for type evidence.
    :return: np.ndarray, majority tag matrix with oriented edges.
    """
    print("getting majority matrix by typing over current graph with each type")
    majoritymat = np.zeros_like(cpdag).astype(float)
    for typelist in taglist:
        current_adjacency_mat = np.copy(cpdag)
        n_types = len(np.unique(typelist))
        type_g = np.eye(n_types)
        print("calculating with type:", typelist)
        # print("type_g before: \n", type_g)
        #make current_adjacency_mat type consistent for the current typelist (will only orient undirected edges (not type consistent edges stay the same))
        # Step 1: Determine the orientation of all t-edges based on t-edges (some may be unoriented)
        type_g = _update_tedge_orientation_majority_prio(current_adjacency_mat, type_g, typelist, priomat)
        print("type_g (relations between typs): \n", type_g)

        # Step 2: Orient all edges of the same type in the same direction if their t-edge is oriented.
        # XXX: This CAN CHANGE the orientation of edges via type consistency in this current_adjacency_matrix
        current_adjacency_mat = _orient_tedges_using_priomat_type_majority(current_adjacency_mat, type_g, typelist, priomat)
        # print("current adjacency mat: \n", current_adjacency_mat)

        # Step 3: add oriented edges of current_adjacency_mat to majoritymat -> high numbers mean that they are often directed that way
        orientmat = _get_matrix_with_only_oriented_edges_of_dag_using_priomat(current_adjacency_mat, taglength=len(taglist))
        # print("oritentmat for current type: \n", orientmat)
        # print("majoritymat: \n", majoritymat)
        majoritymat += orientmat

    return majoritymat

# taken from https://github.com/ServiceNow/typed-dag/blob/main/typed_pc/tmec.py
def _update_tedge_orientation_majority_prio(G, type_g, types, priomat):
    """
    Detects which t-edges are oriented and unoriented and updates the type compatibility graph.
    
    :param G: np.ndarray, adjacency matrix of the graph.
    :param type_g: np.ndarray, type compatibility graph matrix (len(type) * len(type)).
    :param types: list of int, type list mapping nodes to their types.
    :param priomat: np.ndarray, priority matrix for type evidence.
    :return: np.ndarray, updated type compatibility graph matrix where types[a, b] > types[b, a] indicates type consistency for type a -> type b.
    """

    amount_nodes = G.shape[0] - 1
    type_g = np.copy(type_g)

    for a, b in permutations(range(G.shape[0]), 2):
        # XXX: No need to consider the same-type case, since the type matrix starts at identity.
        if types[a] == types[b]:
            continue
        # Detect unoriented t-edges
        if G[a, b] == 1 and G[b, a] == 1 and not (type_g[types[a], types[b]] + type_g[types[b], types[a]] == 1):
            continue
        # Detect oriented t-edges
        if G[a, b] == 1 and G[b, a] == 0:
            if (type_g[types[b], types[a]] == 1):
                logging.debug(f"adding to type", types[a], "to", types[b], "Graph is not type consistent for this tag!") #TODO netter schreiben, dass soll ja passieren können
            logging.debug(f"type g adding {priomat[a,b]/amount_nodes} to types {types[a]} -> {types[b]} because edge {a} -> {b}") 
            type_g[types[a], types[b]] += (priomat[a,b]/amount_nodes) #normalized with graphlength so that max (every evidence is v-structure) = weight of v-structure
            type_g[types[b], types[a]] += 0
        
        # Detect oriented t-edges (other direction)
        if G[b, a] == 1 and G[a, b] == 0:
            if (type_g[types[a], types[b]] == 1):
                logging.debug(f"adding to type", types[b], "to", types[a], "Graph is not type consistent for this tag!") #TODO netter schreiben, dass soll ja passieren können
            logging.debug(f"type g adding {priomat[b,a]/amount_nodes} to types {types[b]} -> {types[a]} because edge {b} -> {a}") 
            type_g[types[b], types[a]] += (priomat[b,a]/amount_nodes) #normalized with graphlength so that max (every evidence is v-structure) = weight of v-structure
            type_g[types[a], types[b]] += 0

    # print("type_g_: \n", type_g)
    return type_g

# taken from https://github.com/ServiceNow/typed-dag/blob/main/typed_pc/tmec.py
def _orient_tedges_using_priomat_type_majority(G, type_g, types, priomat):
    """
    Ensures that edges that belong to oriented t-edges are consistently oriented.

    Note: Will change orientation of edges that are already oriented, even if they clash with the direction
          of the t-edge, when type constency evidence is higher than current evidence in prio matrix

    :param G: np.ndarray, adjacency matrix of the graph.
    :param type_g: np.ndarray, type compatibility graph matrix.
    :param types: list of int, type list mapping nodes to their types.
    :param priomat: np.ndarray, priority matrix for type evidence.
    :return: np.ndarray, updated adjacency matrix with t-edges oriented based on type evidence for current type.
    """

    G = np.copy(G).astype(float)
    for a, b in permutations(range(G.shape[0]), 2):
        #only orient edges
        if(G[a,b] > 0 or G[a,b] > 0):
            # give edge the value of the prio_weighted type consistency, do not check for priomat value as we will do that before orienting the real complete graph
            if type_g[types[a], types[b]] > type_g[types[b], types[a]]:
                if(G[a,b] == 0):
                    logging.debug(f"Reorinting to {a} -> {b}")
                prio_weight = type_g[types[a], types[b]]
                G[a, b] = prio_weight #use prio_weigth instead of actual true direction for now, this will be used to orient later on
                G[b, a] = 0
            elif  type_g[types[b], types[a]] > type_g[types[a], types[b]]: # check for other direction
                if(G[b,a] == 0):
                    logging.debug(f"Reorinting to {b} -> {a}")
                prio_weight = type_g[types[b], types[a]]
                G[b, a] = prio_weight #use prio_weigth instead of actual true direction for now, this will be used to orient later on
                G[a, b] = 0
                #if typing weight is the same in both directions - do not do anything
    return G


def _get_matrix_with_only_oriented_edges_of_dag_using_priomat(adjacency_mat: np.ndarray, taglength : int):
    """
    Generates matrix with only oriented edges of DAG using priority matrix.

    :param adjacency_mat: np.ndarray, adjacency matrix of graph.
    :param taglength: int, length of tag list for normalization.
    :return: np.ndarray, matrix with only oriented edges.
    """
    orientmat = np.zeros_like(adjacency_mat).astype(float)
    for a, b in permutations(range(adjacency_mat.shape[0]), 2):
        if (adjacency_mat[a, b] > adjacency_mat[b, a]):
            orientmat[a,b] = (adjacency_mat[a, b])/taglength #normalize over amount of tag

    # print("getting singles: \n", orientmat)
    return orientmat

def orient_tagged_dag_according_majority_tag_matrix_using_prio_mat_cycle_check(cpdag: np.ndarray, taglist: list, majoritymat: np.ndarray, priomat: np.ndarray, node_names = []): #node names only needed for debugging
    """ 
    Orients the DAG based on the majority tag matrix and the priority matrix. and checks for cycles
    in order to orient the edges with most evidence when in conflict, we iterate from highest to lowest value
    :param cpdag: np.ndarray, CPDAG adjacency matrix.
    :param taglist: list of list of int, each entry is a typelist mapping nodes to their types.
    :param majoritymat: np.ndarray, majority tag matrix with oriented edges.
    :param priomat: np.ndarray, priority matrix for type evidence.
    :param node_names: list of node_names as strings, not necessary only used for debugging
    :return: np.ndarray-tuple: updated adjacency matrix and priority matrix.
    """
    print("Orient according to majority-matrix") # TODO loging.Debug
    if (len(node_names) < cpdag.shape[0]): # make sure that code works if no node names are given -> if node names are empty just numerate nodes
        node_names = []
        for i in range(cpdag.shape[0]):
            node_names.append(i)
    
    # Get the indices sorted by the values in majoritymat in descending order
    indices = np.dstack(np.unravel_index(np.argsort(-majoritymat.ravel()), majoritymat.shape))[0] #flattens majoritymat -> resorts by entries of highest value -> reformates into array of array containing the coordinates a,b of entry
    sorted_indices = sorted(indices, key=lambda idx: (-majoritymat[idx[0], idx[1]], majoritymat[idx[1], idx[0]])) # resort so that for same value in direction i,j, entrys are priotized for lower opposite entries (j,i)
   
    # Iterate over the sorted indices
    for idx in sorted_indices:
        a, b = idx
        #ignore all non skeleton edges:
        if (majoritymat[a, b] == 0 and majoritymat[b, a] == 0):
            continue
        logging.debug(f"{idx} {node_names[a]} to {node_names[b]} -> entry: {majoritymat[a, b]}, opposite entry: {majoritymat[b, a]}, max priomat = {max(priomat[a, b], priomat[b, a])}") # loging.Debug
        #ignore all edges, where there is the same amount of type evidence for both direction
        if (majoritymat[a, b] == majoritymat[b, a]):
            continue

        # time for proper type evidence
        if (majoritymat[a, b] > majoritymat[b, a]):
            type_weight = majoritymat[a, b]
            # orient edge if typing has more evidence than already obtained
            if (type_weight > max(priomat[a, b], priomat[b, a])):
                if (cpdag[b, a] == 1): # debug help -> only print out orientation when edge isnt already oriented
                    print(f"orient edge {node_names[a]} to {node_names[b]} because of {type_weight} type_weight/instances of typing") #TODO to debug.log for publishing
                else:
                    print(f"Update evidence for edge {node_names[a]} to {node_names[b]} because of {type_weight} type_weight/instances of typing") #TODO to debug.log for publishing
                _orient_typeless_with_priomat_cycle_check_adjmat(cpdag, a, b, priomat, type_weight)
        
        # same as above but for inverted edges -> should be rare since matrix is sorted by highest entries [a,b]
        if (majoritymat[b, a] > majoritymat[a, b]):
            type_weight = majoritymat[b, a]
            # orient edge if typing has more evidence than already obtained
            if (type_weight > max(priomat[a, b], priomat[b, a])):
                if (cpdag[a, b] == 1): # debug help -> only print out orientation when edge isnt already oriented
                    print(f"orient edge {node_names[b]} to {node_names[a]} because of {type_weight} type_weight/instances of typing") #TODO to debug.log for publishing
                else:
                    print(f"Update evidence for edge {node_names[b]} to {node_names[a]} because of {type_weight} type_weight/instances of typing") #TODO to debug.log for publishing
                _orient_typeless_with_priomat_cycle_check_adjmat(cpdag, b, a, priomat, type_weight)

        

    return cpdag, priomat

# currently unused because cycle ckeing version exists, but this method is far more efficent when not dealing with cycles
def orient_tagged_dag_according_majority_tag_matrix_using_prio_mat(cpdag: np.ndarray, taglist: list, majoritymat: np.ndarray, priomat: np.ndarray, node_names = []): #node names only needed for debugging
    """ 
    Orients the DAG based on the majority tag matrix and the priority matrix.

    :param cpdag: np.ndarray, CPDAG adjacency matrix.
    :param taglist: list of list of int, each entry is a typelist mapping nodes to their types.
    :param majoritymat: np.ndarray, majority tag matrix with oriented edges.
    :param priomat: np.ndarray, priority matrix for type evidence.
    :param node_names: list of node_names as strings, not necessary only used for debugging
    :return: np.ndarray-tuple: updated adjacency matrix and priority matrix.
    """

    if (len(node_names) < cpdag.shape[0]): # make sure that code works if no node names are given -> if node names are empty just numerate nodes
        node_names = []
        for i in range(cpdag.shape[0]):
            node_names.append(i)

    #iterate over all edges in majority mat
    for a, b in permutations(range(majoritymat.shape[0]), 2):
        #ignore all non skeleton edges:
        if (majoritymat[a, b] == 0 and majoritymat[b, a] == 0):
            continue
        #ignore all edges, where there is the same amount of type evidence for both direction
        if (majoritymat[a, b] == majoritymat[b, a]):
            continue

        # time for proper type evidence
        if (majoritymat[a, b] > majoritymat[b, a]):
            type_weight = majoritymat[a, b]
            # orient edge if typing has more evidence than already obtained
            if (type_weight > max(priomat[a, b], priomat[b, a])):
                if (cpdag[b, a] == 1): # debug help -> only print out orientation when edge isnt already oriented
                    print(f"orient edge {node_names[a]} to {node_names[b]} because of {type_weight} type_weight/instances of typing") #TODO to debug.log for publishing
                else:
                    print(f"Update evidence for edge {node_names[a]} to {node_names[b]} because of {type_weight} type_weight/instances of typing") #TODO to debug.log for publishing
                cpdag[a, b] = 1 # should already be true
                cpdag[b, a] = 0
                priomat[a, b] = type_weight
                priomat[b, a] = type_weight
        
        # same as above but for inverted edges
        if (majoritymat[b, a] > majoritymat[a, b]):
            type_weight = majoritymat[b, a]
            # orient edge if typing has more evidence than already obtained
            if (type_weight > max(priomat[a, b], priomat[b, a])):
                if (cpdag[a, b] == 1): # debug help -> only print out orientation when edge isnt already oriented
                    print(f"orient edge {node_names[b]} to {node_names[a]} because of {type_weight} type_weight/instances of typing") #TODO to debug.log for publishing
                else:
                    print(f"Update evidence for edge {node_names[b]} to {node_names[a]} because of {type_weight} type_weight/instances of typing") #TODO to debug.log for publishing
                cpdag[a, b] = 0 
                cpdag[b, a] = 1 # should already be true
                priomat[a, b] = type_weight
                priomat[b, a] = type_weight
        

    return cpdag, priomat


# for tag weight
# tag_weight_step 2: count differences between adjacency matrix and skeleton = SHD (I think)
def get_amount_difference_two_mats(adjacency_mat, skeleton_mat):
    difference_counter = 0
    for i in range(len(skeleton_mat)):
        for j in range(len(skeleton_mat[i])):
            if (skeleton_mat[i][j] != adjacency_mat[i][j]):
                difference_counter += 1
    return difference_counter
                   
# normal typing algo (see tagged-pc-using-LLM/typed-PC/tpc.py) used for tag-weight (and slightly modified)
def typed_pc_from_true_skeleton (skeleton, separating_sets, typelist, current_tag_number, majority_rule_typed):
    """
    :param typelist: list of int where each entry is a param that mapps to int representation of its type (i.e. [0, 1, 1, 2, 2, 3, 3] ),
                can be generated using tpc_utils.get_typelist_from_text
    :param majority_rule_typed: bool, if true use majority rule to orient forks, if false use naive rule to orient forks
    :optional param data: string of data with nodes seperated with whitespace and entries by line break
    :return: adjacency matrix of the DAG of type np.ndarray
    :return: stat_tests of the DAG of type np.ndarray  
    :return: node names of type list<String> 

    step 1: infer skeleton
    step 2: orient v-structures & two type forks + type consistency
    step 3: t-propagation
    
    """
    #step 1 from true skelton
    # skeleton, separating_sets, stat_tests, node_names = get_true_skeleton(dataname=dataname, typelist=typelist, *data)
    # wie already have true skeleton -> skip step 1

    print("skeleton: \n", nx.adjacency_matrix(skeleton).todense()) #TODO to Debug.Log
#    save_skeleton(skeleton=skeleton, separating_sets=separating_sets, stat_tests=stat_tests)     # -> use for debugging when you want to save a new skeleton
#    skeleton, separating_sets, stat_tests = load_skeleton()                                 # -> use for debugging when you dont want to wait to create skeleton -> also comment out create skeleton/seperating_sets

    #step 2
    if (majority_rule_typed):
        out = orient_forks_majority_top1(skeleton=skeleton, sep_sets=separating_sets, current_tag_number=current_tag_number)
    else:
        out = orient_forks_naive(skeleton=skeleton, sep_sets=separating_sets, current_tag_number=current_tag_number)

    #step 3
    out_matrix, t_matrix = typed_meek(cpdag=nx.adjacency_matrix(out).todense(), types=typelist)

    return out_matrix


def orient_forks_naive(skeleton, sep_sets, current_tag_number):
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
                    f"S: orient immorality {i} (t{type_of_from_tag_single(dag, i, current_tag_number)}) -> {k} (t{type_of_from_tag_single(dag, k, current_tag_number)}) <- {j} (t{type_of_from_tag_single(dag, j, current_tag_number)})"
                )
                _orient_tagged(dag, i, k, current_tag_number)
                _orient_tagged(dag, j, k, current_tag_number)

            # Case: we have an orientable two-type fork, i.e., it is not an immorality, so i <- k -> j
            elif (
                type_of_from_tag_single(dag, i, current_tag_number) == type_of_from_tag_single(dag, j, current_tag_number)
                and type_of_from_tag_single(dag, i, current_tag_number) != type_of_from_tag_single(dag, k, current_tag_number)
                and _has_both_edges(dag, i, k)
                and _has_both_edges(dag, j, k)
            ):
                logging.debug(
                    f"S: orient two-type fork {i} (t{type_of_from_tag_single(dag, i, current_tag_number)}) <- {k} (t{type_of_from_tag_single(dag, k, current_tag_number)}) -> {j} (t{type_of_from_tag_single(dag, j, current_tag_number)})"
                )
                _orient_tagged(dag, k, i, current_tag_number)  # No need to orient k -> j. Will be done in this call since i,j have the same type.

    return dag


def orient_forks_majority_top1(skeleton, sep_sets, current_tag_number):
    """
    Orient immoralities and two-type forks

    Strategy: majority -- orient using the most frequent orientation
    Particularity: Find the t-edge with most evidence, orient, repeat evidence collection.

    """
    dag = skeleton.to_directed()
    node_ids = skeleton.nodes()
    n_types = len(np.unique([type_of_from_tag_single(dag, n, current_tag_number) for n in dag.nodes()]))

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
                    if type_of_from_tag_single(dag, i, current_tag_number) == type_of_from_tag_single(dag, j, current_tag_number) == type_of_from_tag_single(dag, k, current_tag_number):
                        continue

                    logging.debug(
                        f"Step 1: evidence of orientation {i} (t{type_of_from_tag_single(dag, i, current_tag_number)}) -> {k} (t{type_of_from_tag_single(dag, k, current_tag_number)}) <- {j} (t{type_of_from_tag_single(dag, j, current_tag_number)})"
                    )
                    # Increment t-edge orientation evidence
                    print(tedge_evidence)
                    print(type_of_from_tag_single(dag, i, current_tag_number))
                    print(type_of_from_tag_single(dag, k, current_tag_number))
                    tedge_evidence[type_of_from_tag_single(dag, i, current_tag_number), type_of_from_tag_single(dag, k, current_tag_number)] += 1
                    tedge_evidence[type_of_from_tag_single(dag, j, current_tag_number), type_of_from_tag_single(dag, k, current_tag_number)] += 1

                    # Determine conditional orientations
                    conditional_orientations[(type_of_from_tag_single(dag, j, current_tag_number), type_of_from_tag_single(dag, k, current_tag_number))].append((i, k))
                    conditional_orientations[(type_of_from_tag_single(dag, i, current_tag_number), type_of_from_tag_single(dag, k, current_tag_number))].append((j, k))

                # Case: we have an orientable two-type fork, i.e., it is not an immorality, so i <- k -> j
                elif type_of_from_tag_single(dag, i, current_tag_number) == type_of_from_tag_single(dag, j, current_tag_number) and type_of_from_tag_single(dag, i, current_tag_number) != type_of_from_tag_single(dag, k, current_tag_number):
                    # Check if already oriented
                    if not _has_both_edges(dag, i, k) or not _has_both_edges(dag, j, k):
                        continue

                    logging.debug(
                        f"Step 1: evidence of orientation {i} (t{type_of_from_tag_single(dag, i, current_tag_number)}) <- {k} (t{type_of_from_tag_single(dag, k, current_tag_number)}) -> {j} (t{type_of_from_tag_single(dag, j, current_tag_number)})"
                    )
                    # Count evidence only once per t-edge
                    tedge_evidence[type_of_from_tag_single(dag, k, current_tag_number), type_of_from_tag_single(dag, i, current_tag_number)] += 2

        # Step 2: Orient t-edges based on evidence
        np.fill_diagonal(tedge_evidence, 0)
        ti, tj = np.unravel_index(tedge_evidence.argmax(), tedge_evidence.shape)
        if np.isclose(tedge_evidence[ti, tj], 0):
            continue

        # Orient!
        print("Evidence", tedge_evidence[ti, tj])
        print(conditional_orientations)
        oriented_tedge = True
        first_ti = [n for n in dag.nodes() if type_of_from_tag_single(dag, n, current_tag_number) == ti][0]
        first_tj = [n for n in dag.nodes() if type_of_from_tag_single(dag, n, current_tag_number) == tj][0]
        logging.debug(
            f"Step 2: orienting t-edge according to max evidence. t{ti} -> t{tj} ({tedge_evidence[ti, tj]}) vs t{ti} <- t{tj} ({tedge_evidence[tj, ti]})"
        )
        _orient_tagged(dag, first_ti, first_tj, current_tag_number)
        cond = Counter(conditional_orientations[ti, tj])
        for (n1, n2), count in cond.items():
            logging.debug(f"... conditional orientation {n1}->{n2} (count: {count}).")
            if (n2, n1) in cond and cond[n2, n1] > count:
                logging.debug(
                    f"Skipping this one. Will orient its counterpart ({n2}, {n1}) since it's more frequent: {cond[n2, n1]}."
                )
            else:
                _orient_tagged(dag, n1, n2, current_tag_number)
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
                if not (type_of_from_tag_single(dag, i, current_tag_number) == type_of_from_tag_single(dag, j, current_tag_number) == type_of_from_tag_single(dag, k, current_tag_number)):
                    continue
                logging.debug(
                    f"Step 3: orient immorality {i} (t{type_of_from_tag_single(dag, i, current_tag_number)}) -> {k} (t{type_of_from_tag_single(dag, k, current_tag_number)}) <- {j} (t{type_of_from_tag_single(dag, j, current_tag_number)})"
                )
                _orient_tagged(dag, i, k, current_tag_number)
                _orient_tagged(dag, j, k, current_tag_number)

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
            # R5: b - a - c and a-/-c and t(c) = t(b) ==> a -> b and a -> c (two-type fork)
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

# only used for Experiment
def calculate_shd_sid(dag, true_dag):
    """
    calculates SHD and SID and returns in that order. graph, true graph can be numpy.ndarray (of the adjacency matrix) or networkx.DiGraph
    """
    shd = SHD(true_dag, dag)
    sid = SID(true_dag, dag)

    return shd, sid

# only used for Experiment
def calculate_shd(dag, true_dag):
    """
    calculates ONLY SHD and returns it. graph, true graph can be numpy.ndarray (of the adjacency matrix) or networkx.DiGraph
    """
    shd = SHD(true_dag, dag)

    return shd