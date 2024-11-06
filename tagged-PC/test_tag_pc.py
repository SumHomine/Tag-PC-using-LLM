import pickle
import networkx as nx
import bnlearn as bn
import numpy as np
import os
import pandas as pd
from itertools import combinations, permutations
from random import shuffle

import time #TODO ggf. entfernen

from tag_pc_utils import _has_both_edges, _has_directed_edge, _orient_typeless_with_priomat_cycle_check_nx, _orient_typeless_with_priomat_cycle_check_adjmat, calculate_shd_sid, contains_cycle, get_taglist_from_llm_output, recover_tag_string_onevsall_using_taglist, set_types, set_types_as_int, set_tags_as_int, get_typelist_from_text, get_taglist_of_string_from_text, turn_taglist_of_string_to_int, get_taglist_of_int_from_text, type_of_from_tag_single, type_of_from_tag_all, amount_of_matching_types_for_two_way_fork, get_priomat_from_skeleton, _orient_typeless_with_priomat, reorder_taglines_in_node_name_order  
from tag_pc import create_skeleton_using_causallearn, get_true_skeleton, orient_forks_majority_tag_majority_top1, orient_forks_majority_tag_naive_type, typed_meek_majority_tag_with_consistency, meek_majority_tag_without_typeconsistency
from visualization import create_graph_viz, create_graph_viz_colorless, create_graph_viz_colorless_all_undirected, show_data
from llm_interface import load_model_pipeline, text_generation, text_reduction_taglist, text_reduction_variable_tags  
from run_llm_tag_engineering import run_llm, run_llm_generic_prompt
# This File is for testing purpose only and not necessary for tpc

def test_set_types():
    # Create a simple DAG with 3 nodes
    dag = nx.DiGraph()
    dag.add_nodes_from([0, 1, 2])

    # Define the types string
    types = """
    A : T1
    B : T2
    C : T3
    """

    # Apply the set_types function
    set_types(dag, types)

    # Check that the types are correctly assigned
    assert dag.nodes[0]['type'] == 'T1', "Node 0 type should be T1"
    assert dag.nodes[1]['type'] == 'T2', "Node 1 type should be T2"
    assert dag.nodes[2]['type'] == 'T3', "Node 2 type should be T3"

    print("All tests passed.")

def test_set_types_as_int():
    # Create testdata
    dag = nx.DiGraph()
    dag.add_nodes_from([0, 1, 2, 3])
    typeslist = [0, 1, 2, 0]
    dag = set_types_as_int(dag, typeslist)
    
    # Check if types are correctly assigned
    success = True
    for i, expected_type in enumerate(typeslist):
        if dag.nodes[i]["type"] != expected_type:
            success = False
            print(f"Test failed for node {i}. Expected {expected_type}, got {dag.nodes[i]['type']}")
    
    if success:
        print("All tests passed!")


def test_set_tags_as_int():
    # Create testdata
    dag = nx.DiGraph()
    dag.add_nodes_from([0, 1, 2, 3])
    taglist = [
        ["Weather", "Rest", "Weather", "Rest"],  # One vs all approach
        ["Weather", "Object", "Weather", "Plant_Con"],  # Intuitive self-made
        ["Rest", "Watering", "Watering", "Rest"], # 2nd One vs all approach
    ]
    dag = set_tags_as_int(dag, taglist)
 
    # Assertions to check if the tags were correctly set
    expected_tags = {
        0: {"type0": "Weather", "type1": "Weather", "type2": "Rest"},
        1: {"type0": "Rest", "type1": "Object", "type2": "Watering"},
        2: {"type0": "Weather", "type1": "Weather", "type2": "Watering"},
        3: {"type0": "Rest", "type1": "Plant_Con", "type2": "Rest"}
    }
    
    print(taglist[0][0])
    print(taglist[2])
    print(len(dag.nodes[2]))
    print(dag.nodes[2]["type0"])
    print(type_of_from_tag_single(dag, 2, 0))
    print(dag.nodes[2]["type1"])
    print(dag.nodes[2]["type2"])
    print(type_of_from_tag_single(dag, 2, 2))


    for node in dag.nodes:
        for tag, value in expected_tags[node].items():
            assert dag.nodes[node][tag] == value, f"Node {node} - Expected {tag} to be {value} but got {dag.nodes[node][tag]}"

    print("All tests passed!")

def test_get_taglist_as_int_from_taglist():
    tags = """
    Cloudy : Weather, Weather, NotWatering
    Sprinkler : Watering, NotWeather, Watering
    Rain : Weather, Weather, Watering
    Wet_Grass : Plant_Con, NotWeather, NotWatering
    """
    taglist_sprinkler_string =  [
        ["Weather", "Watering", "Weather", "Plant_Con"],  # Intuitive self-made
        ["Weather", "NotWeather", "Weather", "NotWeather"],  # One vs all approach
        ["NotWatering", "Watering", "Watering", "NotWatering"], # 2nd One vs all approach
        ] 

    taglist_sprinkler_int = [
        [0, 1, 0, 2],  # Intuitive self-made
        [0, 1, 0, 1],  # One vs all approach
        [0, 1, 1, 0], # 2nd One vs all approach
    ]


    print(get_taglist_of_string_from_text(tags))

    assert get_taglist_of_string_from_text(tags) == taglist_sprinkler_string, f"Test case 1 failed: {get_taglist_of_string_from_text(tags)}"

    print(turn_taglist_of_string_to_int(taglist_sprinkler_string))

    assert (turn_taglist_of_string_to_int(taglist_sprinkler_string)) == taglist_sprinkler_int, f"Test case 1 failed: {turn_taglist_of_string_to_int(taglist_sprinkler_string)}"

    assert (get_taglist_of_int_from_text(tags)) == taglist_sprinkler_int, f"Test case 1 failed: {get_taglist_of_int_from_text(tags)}"
    
    # Additional test case for animals
    tags_animals = """
    Lion : Carnivore, Big, Fast
    Elephant : Herbivore, Big, Slow
    Cheetah : Carnivore, Big, Fast
    Gazelle : Herbivore, Small, Fast
    """

    # Expected taglists in string format for animals
    taglist_animals_string = [
        ["Carnivore", "Herbivore", "Carnivore", "Herbivore"],  # Intuitive self-made
        ["Big", "Big", "Big", "Small"],  # One vs all approach
        ["Fast", "Slow", "Fast", "Fast"],  # 2nd One vs all approach
    ]

    # Expected taglists in integer format for animals
    taglist_animals_int = [
        [0, 1, 0, 1],  # Intuitive self-made
        [0, 0, 0, 1],  # One vs all approach
        [0, 1, 0, 0],  # 2nd One vs all approach
    ]

    print(get_taglist_of_string_from_text(tags_animals))
    assert get_taglist_of_string_from_text(tags_animals) == taglist_animals_string, f"Test case 4 failed: {get_taglist_of_string_from_text(tags_animals)}"

    print(turn_taglist_of_string_to_int(taglist_animals_string))
    assert turn_taglist_of_string_to_int(taglist_animals_string) == taglist_animals_int, f"Test case 5 failed: {turn_taglist_of_string_to_int(taglist_animals_string)}"

    assert get_taglist_of_int_from_text(tags_animals) == taglist_animals_int, f"Test case 6 failed: {get_taglist_of_int_from_text(tags_animals)}"
 
    print("tests passed!")

    

def test_get_typelist_from_text():
    types = """
    A : Producer
    R : 1st Consumer
    S : 1st Consumer
    H : 2nd Consumer
    B : 2nd Consumer
    W : 3rd Consumer
    F : 3rd Consumer
    """
    expected = [0, 1, 1, 2, 2, 3, 3] 
    assert get_typelist_from_text(types) == expected, f"Test case 1 failed: {get_typelist_from_text(types)}"

    types = """
    A : T-1
    B : T_2
    C : T-1
    D : T_3
    """
    expected = [0, 1, 0, 2]
    assert get_typelist_from_text(types) == expected, f"Test case 2 failed: {get_typelist_from_text(types)}"
    
    print("All test cases passed!")

def test_type_of_from_tag_all():
    # Example usage:
    # Create a simple DAG
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])

    # Set tags for the nodes
    taglist = [[1, 2, 1, 4], [10, 10, 30, 20], [100, 105, 100, 100]]
    set_tags_as_int(G, taglist)

    # Get all tags of a specific node
    node_tags = type_of_from_tag_all(G, 1)
    print(node_tags)  # Output should be [2, 10, 105] for node 1

def test_amount_of_matching_types_for_two_way_fork():
    # Example usage:
    # Create a simple DAG
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])

    # Set tags for the nodes
    taglist = [[1, 2, 1, 4], [10, 10, 10, 20], [100, 105, 100, 100], [1,1,3,4]]
    set_tags_as_int(G, taglist)
    i = 0
    print(f"i tags: ", type_of_from_tag_all(G, i))
    j = 2
    print(f"j tags: ", type_of_from_tag_all(G, j))
    k = 1
    print(f"k tags: ", type_of_from_tag_all(G, k))

    print(f"matching types for two way fork: ", amount_of_matching_types_for_two_way_fork(G, i, j, k))

def test_tag_step_by_step():
    # bn.import_example() is not deterministic
    path_bnlearn = "/home/ml-stud19/tagged-pc-using-LLM/additionalData" 
    bnfdata = "insurance"  # change depending on example
    # data, node_names, tags = get_data_from_csv_string(path_to_folder=path_bnlearn,bnfdata=bnfdata)
    type_insurance = """
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

    
    tags = type_insurance
    taglist = get_taglist_of_int_from_text(tags)


    iterations = 1
    matrices = []
#    for i in range(iterations):
    skeleton, separating_sets, stat_tests, node_names = get_true_skeleton(dataname=bnfdata, taglist=taglist)
    # skeleton and seperatins_sets are deterministic
    priomat = get_priomat_from_skeleton(nx.adjacency_matrix(skeleton).todense(), taglist)
    print("priomat \n", priomat) #TODO to Debug.Log
    # testing for errors when orienting v-structures:
    # dag, priomat = _orient_only_immoralites(skeleton=skeleton, sep_sets=separating_sets, priomat=priomat, taglist=taglist) 


    #testing orienting forks naive tag
    # dag = orient_forks_naive_type_naive_tag(skeleton=skeleton, sep_sets=separating_sets) # not deterministic -> deterministic for correct input -> enforces type consistency
    # dag = orient_forks_naive_tag_majority_top1(skeleton=skeleton, sep_sets=separating_sets) # deterministic!

    # meek naive tag
    # adjacency, t_matrix = typed_meek_naive_tag(cpdag=nx.adjacency_matrix(dag).todense(), tags=taglist) # works the same for both majority and naive typed

    # testing orienting forks majority tag
    # dag, priomat = orient_forks_majority_tag_naive_type(skeleton=skeleton, sep_sets=separating_sets, priomat=priomat, taglist=taglist) # not deterministic -> deterministic for correct input does not enforces type consistency
    dag, priomat = orient_forks_majority_tag_majority_top1(skeleton=skeleton, sep_sets=separating_sets, priomat=priomat, taglist=taglist) # not determinisitisch

    #meek without type consistency (majority tag)
    adjacency, priomat = meek_majority_tag_without_typeconsistency(cpdag=nx.adjacency_matrix(dag).todense(), tags=taglist, priomat=priomat)
    #meek with type consistency (majority tag)
    adjacency, priomat = typed_meek_majority_tag_with_consistency(cpdag=adjacency, tags=taglist, priomat=priomat, majority_rule_typed=True) # Adjust here for test above

    # saving 
    current_mat = nx.adjacency_matrix(dag).todense()
    current_mat = adjacency      #if you use meek
    matrices.append(current_mat)
#    print("seperation Age, Socio Econ:", separating_sets[2][0])
#    print("seperation Socio Econ, Age (should be same as above):", separating_sets[0][2])
#    print(current_mat)
    print(node_names)

    #save as pic
    dir = "tagged-pc-using-LLM/tagged-PC"   
    fname = "test_" + bnfdata + "_" + "true_skeleton_" + "majority_type_majority_tag" + "_0" #naive_type_majority_tag
    create_graph_viz(dag=current_mat, var_names=node_names, stat_tests=stat_tests, types=taglist[0], save_to_dir=dir, fname=fname) #print using first tag

        # print("skeleton: \n", skeleton_mat)
#    for matrix in matrices:
#        print("matrix: \n", matrix)
#    martix_equal, compare_matrix, uneqal_matrix = all_matrices_equal(matrices)
    martix_equal, compare_matrix, uneqal_matrix = compare_adjacencymat_to_last_dag(current_mat)
    print("matrices equal?:", martix_equal)
#    if (not martix_equal):
#        print("\nunequal old matrix: \n", compare_matrix, "\nunequal new matrix: \n: ", uneqal_matrix)

def _orient_only_immoralites(skeleton, sep_sets, priomat, taglist):
    """
    Orient immoralities only -> adjusted naive typed since classic majority algo uses integraly type consistence

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
                
                
    return dag, priomat

def test_order_tags():
    type_insurance_true = """
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

    type_insurance_false = """
    SocioEcon : First
    RiskAversion : First
    GoodStudent : First
    Age : First
    DrivQuality : Fourth
    DrivingSkill : Third
    SeniorTrain : Second
    ThisCarCost : Fifth
    CarValue : Third
    Theft : Fourth
    AntiTheft : Second
    RuggedAuto : Third
    MakeModel : Second
    VehicleYear : Second
    Accident : Fourth
    ThisCarDam : Fourth
    Antilock : Third
    Mileage : First
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


    path = os.path.join("tagged-pc-using-LLM/additionalData", ("insurance" + ".bif"))
    model = bn.import_DAG(path)
    adjacency_mat = model['adjmat']
    node_names = adjacency_mat.columns.tolist()
    print("node names:", node_names)

    taglist_insurance_true = get_taglist_of_string_from_text(type_insurance_true, node_names)
    print("taglist from ordered taglist:", taglist_insurance_true)

    taglist_insurance_false = get_taglist_of_string_from_text(type_insurance_false, node_names)
    print("taglist from missordered taglist:", taglist_insurance_false)

    #check if both taglists are equal
    assert len(taglist_insurance_true) == len(taglist_insurance_false), "test failed"

    # Check if each sublist is equal
    for sublist1, sublist2 in zip(taglist_insurance_true, taglist_insurance_false):
        assert sublist1 == sublist2, "test failed"



    # test multitags with sprinkler

    tag_sprinkler_true = """
    Cloudy : Weather, Weather, NotWatering
    Sprinkler : Watering, NotWeather, Watering
    Rain : Weather, Weather, Watering
    Wet_Grass : Plant_Con, NotWeather, NotWatering   
    """

    tag_sprinkler_false = """
    Sprinkler : Watering, NotWeather, Watering
    Rain : Weather, Weather, Watering
    Wet_Grass : Plant_Con, NotWeather, NotWatering   
    Cloudy : Weather, Weather, NotWatering
    """

    model = bn.import_DAG("sprinkler")
    adjacency_mat = model['adjmat']
    node_names = adjacency_mat.columns.tolist()
    print("node names:", node_names)


    taglist_sprinkler_true = get_taglist_of_string_from_text(tag_sprinkler_true, node_names)
    print("taglist from ordered taglist:", taglist_sprinkler_true)

    taglist_sprinkler_false = get_taglist_of_string_from_text(tag_sprinkler_false, node_names)
    print("taglist from missordered taglist:", taglist_sprinkler_false)

    #check if both taglists are equal
    assert len(tag_sprinkler_true) == len(tag_sprinkler_false), "test failed"

    # Check if each sublist is equal
    for sublist1, sublist2 in zip(tag_sprinkler_true, tag_sprinkler_false):
        assert sublist1 == sublist2, "test failed"


    print("test passed")


def all_matrices_equal(matrices):
    compare_matrix = matrices[0]
    for matrix in matrices:
        if not (compare_matrix == matrix).all():
            return False, compare_matrix, matrix
    return True, None, None

def compare_adjacencymat_to_last_dag(matrix):
    true_skeleton_insurance = np.array([
    [0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])
    naive_naive_insurance_true_skeleton = np.array([
    [0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

    last_matrix = naive_naive_insurance_true_skeleton
    if (last_matrix == matrix).all():
            return True, None, None
    return False, last_matrix, matrix

#copied from main for testing - TODO DELETE
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

def test_llm():

    generation_prompt = """We have found 27 Factors that impact whether a person gets insurance money after a traffic accident. Your job is now to assign those factors with multiple tags. Think of a handful of recurring characteristika to use as tags and then iteratively assign to each tag all fitting variables, so that each variable can have multiple fitting tags. The variables are the following: 
    'SocioEcon', 'GoodStudent', 'Age', 'RiskAversion', 'VehicleYear', 'Accident', 'ThisCarDam', 'RuggedAuto', 'MakeModel', 'Antilock', 'Mileage', 'DrivQuality', 'DrivingSkill', 'SeniorTrain', 'ThisCarCost', 'CarValue', 'Theft', 'AntiTheft', 'HomeBase', 'OtherCarCost', 'PropCost', 'OtherCar', 'MedCost', 'Cushioning', 'Airbag', 'ILiCost', 'DrivHist'"""

    pipeline = load_model_pipeline()
    out = text_generation(pipeline, generation_prompt)
    tags = text_reduction_variable_tags(pipeline, out)
    taglist = text_reduction_taglist(pipeline, out)

def test_get_taglist_from_llm_output():
    out_string = "Demographics, Vehicle Characteristics, Accident-Related Factors, Financial Factors, Risk Factors"
    out_list = get_taglist_from_llm_output(out_string)
    print(out_list)
    for tag in out_list:
        print(tag)
    return

def test_recover_tag_string_onevsall_using_taglist():
    
#--------------------------------test 1--------------------------------
    #check wrong order
    print("--------------------------------test 1--------------------------------")
    taglist = ["Demographic", "notVehicle Characteristics", "Driving Habits", "Accident-Related", "Financial", "Risk Factors"]
    tags_wrong_order = """
Correct Order : Demographic, Risk Factors
Wrong Order : Risk Factors, Demographic, notVehicle Characteristics
"""
    tags_reordered = recover_tag_string_onevsall_using_taglist(tag_list=taglist, tags_string=tags_wrong_order)
    print(f"reordering test: \n{tags_reordered}")

    true_reordered_tags = """
Correct Order : Demographic, notnotVehicle Characteristics, notDriving Habits, notAccident-Related, notFinancial, Risk Factors
Wrong Order : Demographic, notVehicle Characteristics, notDriving Habits, notAccident-Related, notFinancial, Risk Factors
"""
#    print(f"tags_reordered (repr): {repr(tags_reordered)}")
#    print(f"true_reordered (repr): {repr(true_reordered_tags)}")

    assert tags_reordered == true_reordered_tags, "test ordering failed"

#--------------------------------test 2--------------------------------
    print("--------------------------------test 2--------------------------------")
    #check completly wrong tag
    taglist = ["Demographic", "Vehicle Characteristics", "Driving Habits", "Accident-Related", "Financial", "Risk Factors"]
    tags_wrong_tag = """
Correct Names : Demographic, Risk Factors
Wrong Names 1 : Demographhhhhhhh, Risk Factors
Wrong Names 2 : Demographic, Risk Facktor
"""
    tags_corrected = recover_tag_string_onevsall_using_taglist(tag_list=taglist, tags_string=tags_wrong_tag)
    print(f"unidentified tags test: \n{tags_corrected}")

    true_wrong_tags = """
Correct Names : Demographic, notVehicle Characteristics, notDriving Habits, notAccident-Related, notFinancial, Risk Factors
Wrong Names 1 : notDemographic, notVehicle Characteristics, notDriving Habits, notAccident-Related, notFinancial, Risk Factors
Wrong Names 2 : Demographic, notVehicle Characteristics, notDriving Habits, notAccident-Related, notFinancial, notRisk Factors
"""

    assert tags_corrected == true_wrong_tags, "test ordering failed"


#--------------------------------test 3--------------------------------
    #full realistic test (wrong tags, different ordering of tags and nodes, different format) using interpretion method from the algorithm
    print("--------------------------------test 3--------------------------------")
    # get insurance node names:
    node_names = ['SocioEcon', 'GoodStudent', 'Age', 'RiskAversion', 'VehicleYear', 'Accident', 'ThisCarDam', 'RuggedAuto', 'MakeModel', 'Antilock', 'Mileage', 'DrivQuality', 'DrivingSkill', 'SeniorTrain', 'ThisCarCost', 'CarValue', 'Theft', 'AntiTheft', 'HomeBase', 'OtherCarCost', 'PropCost', 'OtherCar', 'MedCost', 'Cushioning', 'Airbag', 'ILiCost', 'DrivHist']

    tags_full_test = """
    Age : Demographhhhhhhh
    SocioEcon : Demographic
    GoodStudent : Demographic, Risk Factors
    RiskAversion : Risk Factors, Demographic
    VehicleYear : Vehicle Characteristics
    ThisCarDam : Accident-Related
    RuggedAuto : Vehicle Characteristics
    Accident : Accident-Related
    MakeModel : Vehicle Characteristics
    DrivQuality : Driving Habits
    Mileage : Vehicle Characteristics
    Antilock : Vehicle Characteristics
    DrivingSkill : Driving Habits
    SeniorTrain : Demographic
    ThisCarCost : Vehicle Characteristics
    Theft : Risk Factors, Financial
    CarValue : Vehicle Characteristics
    AntiTheft : Vehicle Characteristics
    PropCost : Financial
    OtherCarCost : Financial
    OtherCar : Financial, Risk Factors
    MedCost : Financial
    ILiCost : Financial
    DrivHist : Driving Habits
    Cushioning : Accident-Related
    Airbag : Accident-Related
    """

    full_recovered_tags = recover_tag_string_onevsall_using_taglist(tag_list=taglist, tags_string=tags_full_test, node_names=node_names)
    print(full_recovered_tags)

    true_recovered_tags_differently_formated = """
    SocioEcon : Demographic, notVehicle Characteristics, notDriving Habits, notAccident-Related, notFinancial, notRisk Factors
    GoodStudent : Demographic, notVehicle Characteristics, notDriving Habits, notAccident-Related, notFinancial, Risk Factors
    Age : notDemographic, notVehicle Characteristics, notDriving Habits, notAccident-Related, notFinancial, notRisk Factors
    RiskAversion : Demographic, notVehicle Characteristics, notDriving Habits, notAccident-Related, notFinancial, Risk Factors
    VehicleYear : notDemographic, Vehicle Characteristics, notDriving Habits, notAccident-Related, notFinancial, notRisk Factors
    Accident : notDemographic, notVehicle Characteristics, notDriving Habits, Accident-Related, notFinancial, notRisk Factors
    ThisCarDam : notDemographic, notVehicle Characteristics, notDriving Habits, Accident-Related, notFinancial, notRisk Factors
    RuggedAuto : notDemographic, Vehicle Characteristics, notDriving Habits, notAccident-Related, notFinancial, notRisk Factors
    MakeModel : notDemographic, Vehicle Characteristics, notDriving Habits, notAccident-Related, notFinancial, notRisk Factors
    Antilock : notDemographic, Vehicle Characteristics, notDriving Habits, notAccident-Related, notFinancial, notRisk Factors
    Mileage : notDemographic, Vehicle Characteristics, notDriving Habits, notAccident-Related, notFinancial, notRisk Factors
    DrivQuality : notDemographic, notVehicle Characteristics, Driving Habits, notAccident-Related, notFinancial, notRisk Factors
    DrivingSkill : notDemographic, notVehicle Characteristics, Driving Habits, notAccident-Related, notFinancial, notRisk Factors
    SeniorTrain : Demographic, notVehicle Characteristics, notDriving Habits, notAccident-Related, notFinancial, notRisk Factors
    ThisCarCost : notDemographic, Vehicle Characteristics, notDriving Habits, notAccident-Related, notFinancial, notRisk Factors
    CarValue : notDemographic, Vehicle Characteristics, notDriving Habits, notAccident-Related, notFinancial, notRisk Factors
    Theft : notDemographic, notVehicle Characteristics, notDriving Habits, notAccident-Related, Financial, Risk Factors
    AntiTheft : notDemographic, Vehicle Characteristics, notDriving Habits, notAccident-Related, notFinancial, notRisk Factors
    HomeBase : notDemographic, notVehicle Characteristics, notDriving Habits, notAccident-Related, notFinancial, notRisk Factors
    OtherCarCost : notDemographic, notVehicle Characteristics, notDriving Habits, notAccident-Related, Financial, notRisk Factors
    PropCost : notDemographic, notVehicle Characteristics, notDriving Habits, notAccident-Related, Financial, notRisk Factors
    OtherCar : notDemographic, notVehicle Characteristics, notDriving Habits, notAccident-Related, Financial, Risk Factors
    MedCost : notDemographic, notVehicle Characteristics, notDriving Habits, notAccident-Related, Financial, notRisk Factors
    Cushioning : notDemographic, notVehicle Characteristics, notDriving Habits, Accident-Related, notFinancial, notRisk Factors
    Airbag : notDemographic, notVehicle Characteristics, notDriving Habits, Accident-Related, notFinancial, notRisk Factors
    ILiCost : notDemographic, notVehicle Characteristics, notDriving Habits, notAccident-Related, Financial, notRisk Factors
    DrivHist : notDemographic, notVehicle Characteristics, Driving Habits, notAccident-Related, notFinancial, notRisk Factors
    """

    tested_taglist = get_taglist_of_int_from_text(full_recovered_tags, node_names)
    true_taglist = get_taglist_of_int_from_text(true_recovered_tags_differently_formated, node_names)
    
    assert tested_taglist == true_taglist, "test ordering failed"


#--------------------------------test 4--------------------------------
    print("--------------------------------test 4--------------------------------")
    #check tags where LLM printed some tag+Number for some fucking reason
    node_names = ['SocioEcon', 'GoodStudent', 'Age', 'RiskAversion', 'VehicleYear', 'Accident', 'ThisCarDam', 'RuggedAuto', 'MakeModel', 'Antilock', 'Mileage', 'DrivQuality', 'DrivingSkill', 'SeniorTrain', 'ThisCarCost', 'CarValue', 'Theft', 'AntiTheft', 'HomeBase', 'OtherCarCost', 'PropCost', 'OtherCar', 'MedCost', 'Cushioning', 'Airbag', 'ILiCost', 'DrivHist']
    taglist = ["Demographics", "Vehicle Characteristics", "Safety Features", "Financial Situation", "Accident-Related Factors", "Driving Habits", "Risk Perception"]
    tags_tagnumber = """
Age : Demographics, Tag2
GoodStudent : Demographics, Tag2
SeniorTrain : Demographics, Tag2
VehicleYear : Vehicle Characteristics, Tag3
MakeModel : Vehicle Characteristics, Tag5
Mileage : Vehicle Characteristics, Tag4
DrivQuality : Vehicle Characteristics, Tag3
CarValue : Vehicle Characteristics, Tag4
OtherCarCost : Vehicle Characteristics, Tag5
OtherCar : Vehicle Characteristics, Tag5
Antilock : Safety Features, Tag5
Airbag : Safety Features, Tag6
Cushioning : Safety Features, Tag7
AntiTheft : Safety Features, Tag4
SocioEcon : Financial Situation, Tag2
ThisCarCost : Financial Situation, Tag5
PropCost : Financial Situation, Tag6
ILiCost : Financial Situation, Tag7
MedCost : Financial Situation, Tag5
Accident : Accident-Related Factors, Tag4
ThisCarDam : Accident-Related Factors, Tag3
RuggedAuto : Accident-Related Factors, Tag4
Theft : Accident-Related Factors, Tag4
DrivingSkill : Driving Habits, Tag4
DrivHist : Driving Habits, Tag2
RiskAversion : Risk Perception, Tag1
"""

    full_recovered_tags = recover_tag_string_onevsall_using_taglist(tag_list=taglist, tags_string=tags_tagnumber, node_names=node_names)
    print(full_recovered_tags)

    true_recovered_tags_tags_tagnumber = """
SocioEcon : notDemographics, Vehicle Characteristics, notSafety Features, Financial Situation, notAccident-Related Factors, notDriving Habits, notRisk Perception
GoodStudent : Demographics, Vehicle Characteristics, notSafety Features, notFinancial Situation, notAccident-Related Factors, notDriving Habits, notRisk Perception
Age : Demographics, Vehicle Characteristics, notSafety Features, notFinancial Situation, notAccident-Related Factors, notDriving Habits, notRisk Perception
RiskAversion : Demographics, notVehicle Characteristics, notSafety Features, notFinancial Situation, notAccident-Related Factors, notDriving Habits, Risk Perception
VehicleYear : notDemographics, Vehicle Characteristics, Safety Features, notFinancial Situation, notAccident-Related Factors, notDriving Habits, notRisk Perception
Accident : notDemographics, notVehicle Characteristics, notSafety Features, Financial Situation, Accident-Related Factors, notDriving Habits, notRisk Perception
ThisCarDam : notDemographics, notVehicle Characteristics, Safety Features, notFinancial Situation, Accident-Related Factors, notDriving Habits, notRisk Perception
RuggedAuto : notDemographics, notVehicle Characteristics, notSafety Features, Financial Situation, Accident-Related Factors, notDriving Habits, notRisk Perception
MakeModel : notDemographics, Vehicle Characteristics, notSafety Features, notFinancial Situation, Accident-Related Factors, notDriving Habits, notRisk Perception
Antilock : notDemographics, notVehicle Characteristics, Safety Features, notFinancial Situation, Accident-Related Factors, notDriving Habits, notRisk Perception
Mileage : notDemographics, Vehicle Characteristics, notSafety Features, Financial Situation, notAccident-Related Factors, notDriving Habits, notRisk Perception
DrivQuality : notDemographics, Vehicle Characteristics, Safety Features, notFinancial Situation, notAccident-Related Factors, notDriving Habits, notRisk Perception
DrivingSkill : notDemographics, notVehicle Characteristics, notSafety Features, Financial Situation, notAccident-Related Factors, Driving Habits, notRisk Perception
SeniorTrain : Demographics, Vehicle Characteristics, notSafety Features, notFinancial Situation, notAccident-Related Factors, notDriving Habits, notRisk Perception
ThisCarCost : notDemographics, notVehicle Characteristics, notSafety Features, Financial Situation, Accident-Related Factors, notDriving Habits, notRisk Perception
CarValue : notDemographics, Vehicle Characteristics, notSafety Features, Financial Situation, notAccident-Related Factors, notDriving Habits, notRisk Perception
Theft : notDemographics, notVehicle Characteristics, notSafety Features, Financial Situation, Accident-Related Factors, notDriving Habits, notRisk Perception
AntiTheft : notDemographics, notVehicle Characteristics, Safety Features, Financial Situation, notAccident-Related Factors, notDriving Habits, notRisk Perception
HomeBase : notDemographics, notVehicle Characteristics, notSafety Features, notFinancial Situation, notAccident-Related Factors, notDriving Habits, notRisk Perception
OtherCarCost : notDemographics, Vehicle Characteristics, notSafety Features, notFinancial Situation, Accident-Related Factors, notDriving Habits, notRisk Perception
PropCost : notDemographics, notVehicle Characteristics, notSafety Features, Financial Situation, notAccident-Related Factors, Driving Habits, notRisk Perception
OtherCar : notDemographics, Vehicle Characteristics, notSafety Features, notFinancial Situation, Accident-Related Factors, notDriving Habits, notRisk Perception
MedCost : notDemographics, notVehicle Characteristics, notSafety Features, Financial Situation, Accident-Related Factors, notDriving Habits, notRisk Perception
Cushioning : notDemographics, notVehicle Characteristics, Safety Features, notFinancial Situation, notAccident-Related Factors, notDriving Habits, Risk Perception
Airbag : notDemographics, notVehicle Characteristics, Safety Features, notFinancial Situation, notAccident-Related Factors, Driving Habits, notRisk Perception
ILiCost : notDemographics, notVehicle Characteristics, notSafety Features, Financial Situation, notAccident-Related Factors, notDriving Habits, Risk Perception
DrivHist : notDemographics, Vehicle Characteristics, notSafety Features, notFinancial Situation, notAccident-Related Factors, Driving Habits, notRisk Perception
"""

    tested_taglist = get_taglist_of_int_from_text(full_recovered_tags, node_names)
    print(tested_taglist)
    true_taglist = get_taglist_of_int_from_text(true_recovered_tags_tags_tagnumber, node_names)
    
    assert tested_taglist == true_taglist, "test tag tagnumber failed"

    print("all tests passed!")

def test_llm_generic_prompt():
    node_names_sprinkler = ["Cloudy", "Sprinkler", "Rain", "Wet_Grass"]
    node_names_asia =      ["asia", "tub", "smoke", "lung", "bronc", "either", "xray", "dysp"] 
    node_names_insurance = ['SocioEcon', 'GoodStudent', 'Age', 'RiskAversion', 'VehicleYear', 'Accident', 'ThisCarDam', 'RuggedAuto', 'MakeModel', 'Antilock', 'Mileage', 'DrivQuality', 'DrivingSkill', 'SeniorTrain', 'ThisCarCost', 'CarValue', 'Theft', 'AntiTheft', 'HomeBase', 'OtherCarCost', 'PropCost', 'OtherCar', 'MedCost', 'Cushioning', 'Airbag', 'ILiCost', 'DrivHist']
    node_names_andes = ['DISPLACEM0', 'RApp1', 'SNode_3', 'GIVEN_1', 'RApp2', 'SNode_8', 'SNode_16', 'SNode_20', 'NEED1', 'SNode_21', 'GRAV2', 'SNode_24', 'VALUE3', 'SNode_15', 'SNode_25', 'SLIDING4', 'SNode_11', 'SNode_26', 'CONSTANT5', 'SNode_47', 'VELOCITY7', 'KNOWN6', 'RApp3', 'KNOWN8', 'RApp4', 'SNode_27', 'GOAL_2', 'GOAL_48', 'COMPO16', 'TRY12', 'TRY11', 'SNode_5', 'GOAL_49', 'SNode_6', 'GOAL_50', 'CHOOSE19', 'SNode_17', 'SNode_51', 'SYSTEM18', 'SNode_52', 'KINEMATI17', 'GOAL_53', 'IDENTIFY10', 'SNode_28', 'IDENTIFY9', 'TRY13', 'TRY14', 'TRY15', 'SNode_29', 'VAR20', 'SNode_31', 'SNode_10', 'SNode_33', 'GIVEN21', 'SNode_34', 'GOAL_56', 'APPLY32', 'GOAL_57', 'CHOOSE35', 'SNode_7', 'SNode_59', 'MAXIMIZE34', 'SNode_60', 'AXIS33', 'GOAL_61', 'WRITE31', 'GOAL_62', 'WRITE30', 'GOAL_63', 'RESOLVE37', 'SNode_64', 'NEED36', 'SNode_9', 'SNode_41', 'SNode_42', 'SNode_43', 'IDENTIFY39', 'GOAL_66', 'RESOLVE38', 'SNode_67', 'SNode_54', 'IDENTIFY41', 'GOAL_69', 'RESOLVE40', 'SNode_70', 'SNode_55', 'IDENTIFY43', 'GOAL_72', 'RESOLVE42', 'SNode_73', 'SNode_74', 'KINE29', 'SNode_4', 'SNode_75', 'VECTOR44', 'GOAL_79', 'EQUATION28', 'VECTOR27', 'RApp5', 'GOAL_80', 'RApp6', 'GOAL_81', 'TRY25', 'TRY24', 'GOAL_83', 'GOAL_84', 'CHOOSE47', 'SNode_86', 'SYSTEM46', 'SNode_156', 'NEWTONS45', 'GOAL_98', 'DEFINE23', 'SNode_37', 'IDENTIFY22', 'TRY26', 'SNode_38', 'SNode_40', 'SNode_44', 'SNode_46', 'SNode_65', 'NULL48', 'SNode_68', 'SNode_71', 'GOAL_87', 'FIND49', 'SNode_88', 'NORMAL50', 'NORMAL52', 'INCLINE51', 'SNode_91', 'SNode_12', 'SNode_13', 'STRAT_90', 'HORIZ53', 'BUGGY54', 'SNode_92', 'SNode_93', 'IDENTIFY55', 'SNode_94', 'WEIGHT56', 'SNode_95', 'WEIGHT57', 'SNode_97', 'GOAL_99', 'FIND58', 'SNode_100', 'IDENTIFY59', 'SNode_102', 'FORCE60', 'GOAL_103', 'APPLY61', 'GOAL_104', 'CHOOSE62', 'SNode_106', 'SNode_152', 'GOAL_107', 'WRITE63', 'GOAL_108', 'WRITE64', 'GOAL_109', 'GOAL_110', 'GOAL65', 'GOAL_111', 'GOAL66', 'NEED67', 'RApp7', 'RApp8', 'SNode_112', 'GOAL_113', 'GOAL68', 'GOAL_114', 'SNode_115', 'SNode_116', 'VECTOR69', 'SNode_117', 'SNode_118', 'VECTOR70', 'SNode_119', 'EQUAL71', 'SNode_120', 'GOAL_121', 'GOAL72', 'SNode_122', 'SNode_123', 'VECTOR73', 'SNode_124', 'NEWTONS74', 'SNode_125', 'SUM75', 'GOAL_126', 'GOAL_127', 'RApp9', 'RApp10', 'SNode_128', 'GOAL_129', 'GOAL_130', 'SNode_131', 'SNode_132', 'SNode_133', 'SNode_134', 'SNode_135', 'SNode_154', 'SNode_136', 'SNode_137', 'GOAL_142', 'GOAL_143', 'GOAL_146', 'RApp11', 'RApp12', 'RApp13', 'GOAL_147', 'GOAL_149', 'TRY76', 'GOAL_150', 'APPLY77', 'SNode_151', 'GRAV78', 'GOAL_153', 'SNode_155', 'SNode_14', 'SNode_18', 'SNode_19']


    #test sprinkler
    tags_sprinkler, node_names_sprinkler = run_llm_generic_prompt(node_names=node_names_sprinkler, determinstic=True)
    print(tags_sprinkler)

    #test asia
    tags_asia, node_names_asia = run_llm_generic_prompt(node_names=node_names_asia, determinstic=True)
    print(tags_asia)

    #test insurance
    tags_insurance, node_names_insurance = run_llm_generic_prompt(node_names=node_names_insurance, determinstic=True)
    print(tags_insurance)

    #test andes
    # tags_andes, node_names_andes = run_llm_generic_prompt(node_names=node_names_andes, determinstic=True)
    # print(tags_andes)

def test_run_llm_tag_engineering():
    tag, nodes = run_llm("insurance", deterministic=True)
    print(nodes)
    print(tag)


def test_finding_cycles_adjacency_mat():
    dag = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])
    
    
    print(contains_cycle(dag))
    assert not contains_cycle(dag)

    dag = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ])

    print(contains_cycle(dag))
    assert contains_cycle(dag)  

    dag = np.array([
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ])

    print(contains_cycle(dag))
    assert not contains_cycle(dag)  

    cycle_mat = np.array([
        [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    print(contains_cycle(cycle_mat))
    assert contains_cycle(cycle_mat)  


    print("tests passed")

def test_cycle_add_edge_nx_graph():
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3,2), (0, 3), (3, 0)])
    dag_adj_matrix = nx.adjacency_matrix(G).todense()
    print("adjacency start: \n",dag_adj_matrix)

    # Convert the DAG adjacency matrix
    taglist = [[0,2,3],[1,0,2],[2,0,5]]
    priomat = get_priomat_from_skeleton(dag_adj_matrix,taglist)
    print("priomat start: \n",priomat)

    # orient first edge no cycle
    _orient_typeless_with_priomat_cycle_check_nx(G, 1, 0 , priomat, 2)
    print("after first orientation: \n",nx.adjacency_matrix(G).todense())

    # roerient first edge no cycle
    _orient_typeless_with_priomat_cycle_check_nx(G, 0, 1 , priomat, 3)
    print("after reorienting first orientation: \n",nx.adjacency_matrix(G).todense())

    # orient second edge no cycle
    _orient_typeless_with_priomat_cycle_check_nx(G, 1, 2 , priomat, 2)
    print("after second orientation: \n",nx.adjacency_matrix(G).todense())

    # try to reorient with too little evidence
    _orient_typeless_with_priomat_cycle_check_nx(G, 2, 1, priomat, 1)
    print("after failed orientation: \n",nx.adjacency_matrix(G).todense())

    # orient third edge no cycle
    _orient_typeless_with_priomat_cycle_check_nx(G, 2, 3 , priomat, 2)
    print("after third orientation: \n",nx.adjacency_matrix(G).todense())

    # try to orient edge but there is cycle
    _orient_typeless_with_priomat_cycle_check_nx(G, 3, 0 , priomat, 2)
    print("should not change because cycle orientation: \n",nx.adjacency_matrix(G).todense())

    # then orient edge in other direction
    _orient_typeless_with_priomat_cycle_check_nx(G, 0, 3 , priomat, 2)
    print("after last orientation: \n",nx.adjacency_matrix(G).todense())

def test_cycle_add_edge_adjacency():
    dag = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])

    print("adjacency start: \n",dag)

    # Convert the DAG adjacency matrix
    taglist = [[0,2,3],[1,0,2],[2,0,5]]
    priomat = get_priomat_from_skeleton(dag,taglist)
    print("priomat start: \n",priomat)

    # orient first edge no cycle
    _orient_typeless_with_priomat_cycle_check_adjmat(dag, 1, 0, priomat, 2)
    print("after first orientation: \n",dag)

    # roerient first edge no cycle
    _orient_typeless_with_priomat_cycle_check_adjmat(dag, 0, 1, priomat, 3)
    print("after reorienting first orientation: \n",dag)

    # orient second edge no cycle
    _orient_typeless_with_priomat_cycle_check_adjmat(dag, 1, 2, priomat, 2)
    print("after second orientation: \n",dag)

    # try to reorient with too little evidence
    _orient_typeless_with_priomat_cycle_check_adjmat(dag, 2, 1, priomat, 1)
    print("after failed orientation: \n",dag)

    # orient third edge no cycle
    _orient_typeless_with_priomat_cycle_check_adjmat(dag, 2, 3, priomat, 2)
    print("after third orientation: \n",dag)

    # try to orient edge but there is cycle
    _orient_typeless_with_priomat_cycle_check_adjmat(dag, 3, 0, priomat, 2)
    print("should not change because cycle orientation: \n",dag)

    # then orient edge in other direction
    _orient_typeless_with_priomat_cycle_check_adjmat(dag, 0, 3, priomat, 2)
    print("after last orientation: \n",dag)

    

#taken from https://github.com/ServiceNow/typed-dag/blob/main/typed_pc/main.py
def sanity_type_consistency(cpdag: np.ndarray, types: np.ndarray) -> bool:
    # Collect t-edge orientations (some may be unoriented)
    n_types = len(set(types))
    t_edges = np.zeros((n_types, n_types))
    for i, j in permutations(range(cpdag.shape[0]), 2):
        if cpdag[i, j] and not cpdag[j, i]:  # For every oriented edge
            t_edges[types[i], types[j]] = 1

    # Check if some oriented edges caused t-edges to be oriented in both directions
    for i, j in permutations(range(n_types), 2):
        if t_edges[i, j] and t_edges[j, i]:
            return False

    return True

def type_consistency_test():
    # put here your dag as adjacency matrix
    dag = np.array([
        [0, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0]])
    typelist = [0, 1, 1, 2, 2, 3, 3] 
    print(sanity_type_consistency(cpdag=dag, types=typelist ))

def save_skeleton(skeleton, separating_sets, stat_tests):
    dir = "/home/ml-stud19/typed-pc-using-LLM/typed-PC/debugdump/"
    os.makedirs(dir, exist_ok=True)

    print("saving the following skeleton:", skeleton)
    with open(os.path.join(dir, 'skeleton.pkl'), 'wb') as f:
        pickle.dump(skeleton, f)
    with open(os.path.join(dir, 'separating_sets.pkl'), 'wb') as f:
        pickle.dump(separating_sets, f)
    with open(os.path.join(dir, 'stat_tests.pkl'), 'wb') as f:
        pickle.dump(stat_tests, f)

def load_skeleton():
    dir = "/home/ml-stud19/typed-pc-using-LLM/typed-PC/debugdump/"
    with open(os.path.join(dir, 'skeleton.pkl'), 'rb') as f:
        skeleton = pickle.load(f)
    with open(os.path.join(dir, 'separating_sets.pkl'), 'rb') as f:
        separating_sets = pickle.load(f)
    with open(os.path.join(dir, 'stat_tests.pkl'), 'rb') as f:
        stat_tests = pickle.load(f)
    
    print("Loading the following skeleton:", skeleton)
    return skeleton, separating_sets, stat_tests

def test_skeleton_saving():
    skeleton = nx.Graph()
    skeleton.add_node(3)
    skeleton.add_node(4)
    skeleton.add_edge(1,2)
    stat_tests = np.zeros((7, 7), dtype=bool)
    separating_sets = [[set() for _ in range(7)] for _ in range(7)]
    print(skeleton)
    save_skeleton(skeleton=skeleton, separating_sets=separating_sets, stat_tests=stat_tests)
    skeleton, separating_sets, stat_tests = load_skeleton()
    print(skeleton)



def get_true_dag():
    true_dag = np.array([
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ])
    return true_dag

def compare_to_true_dag(dag: np.array):
    true_dag = get_true_dag()

    result = dag == true_dag
    return result

def test_printing_dag():
        
    # Read CSV file and extract node names from first line
    data = np.genfromtxt("/home/ml-stud19/typed-pc-using-LLM/generated_forestdata.csv", delimiter=",", dtype=(str, int))
    node_names = data[0]
    data = data[1:].astype(int) #convert back to int

    types = """
        A : Producer
        R : 1st Consumer
        S : 1st Consumer
        H : 2nd Consumer
        B : 2nd Consumer
        W : 3rd Consumer
        F : 3rd Consumer
        """

    #TODO das automatisiert von irgendwo auslesehen

    typelist = [0, 1, 1, 2, 2, 3, 3] 

    # got the following from running tpc
    dag = np.array([
        [0, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 0]
    ])
    stat_tests = np.array([
    [False,  True,  True,  True,  True,  True,  True],
    [ True, False,  True,  True,  True,  True,  True],
    [ True,  True, False,  True,  True,  True,  True],
    [ True,  True,  True, False,  True, False,  True],
    [ True,  True,  True,  True, False,  True,  True],
    [ True,  True,  True, False,  True, False,  True],
    [ True,  True,  True,  True,  True,  True, False]
])
    dir = "/home/ml-stud19/typed-pc-using-LLM/typed-PC"
    create_graph_viz(dag=dag, var_names=node_names, stat_tests=stat_tests, types=typelist, save_to_dir=dir)

def print_standard_pc_true_skeleton():
    types_sprinkler = """
        Cloudy : Uses Watervapor
        Sprinkler : Uses Watervapor
        Rain : NotUses Watervapor
        Wet_Grass : NotUses Watervapor   
    """
    
    
    tags_asia =  """           
        asia : first
        tub : first
        smoke : second
        lung : second
        bronc : third
        either : third
        xray : fourth
        dysp : fourth
    """

    tag_insurance = """
        SocioEcon : person
        GoodStudent : person
        Age : person
        RiskAversion : person
        VehicleYear : notperson
        Accident : notperson
        ThisCarDam : notperson
        RuggedAuto : notperson
        MakeModel : notperson
        Antilock : notperson
        Mileage : notperson
        DrivQuality : person
        DrivingSkill : person
        SeniorTrain : person
        ThisCarCost : notperson
        CarValue : notperson
        Theft : notperson
        AntiTheft : notperson
        HomeBase : notperson
        OtherCarCost : notperson
        PropCost : notperson
        OtherCar : notperson
        MedCost : notperson
        Cushioning : notperson
        Airbag : notperson
        ILiCost : notperson
        DrivHist : person
    """


    dataname = "sprinkler"
    tags = types_sprinkler
    

    skeleton, separating_sets, stat_tests, node_names, taglist = get_true_skeleton(dataname, tags=tags)
    priomat = get_priomat_from_skeleton(nx.adjacency_matrix(skeleton).todense(), taglist)
    dag, priomat = _orient_only_immoralites(skeleton=skeleton, sep_sets=separating_sets, priomat=priomat, taglist=taglist) #v-strucuteres
    adjacency_mat, priomat = meek_majority_tag_without_typeconsistency(cpdag=nx.adjacency_matrix(dag).todense(), tags=taglist, priomat=priomat, node_names=node_names)# Meek
    dir = "tagged-pc-using-LLM/tagged-PC"
    fname = "tdag_" + dataname + "_" + "true_skeleton_only_immoralities_and_meek" 
    create_graph_viz_colorless(dag=nx.adjacency_matrix(skeleton).todense(), var_names=node_names, save_to_dir=dir, fname=fname)

def print_true_graph():
    dataname = "insurance"

    # get Dag from bnf data
    path = os.path.join("tagged-pc-using-LLM/additionalData", (dataname + ".bif"))
    model = bn.import_DAG(path)
    adjacency_mat = model['adjmat']
    adjacency_mat = adjacency_mat.astype(int) #get int representation
    node_names = adjacency_mat.columns.tolist()
    adjacency_mat = adjacency_mat.values #delete headers
    dir = "tagged-pc-using-LLM/tagged-PC"
    fname = "tdag_" + dataname + "_" + "true_graph" 
    create_graph_viz_colorless(dag=adjacency_mat, var_names=node_names, save_to_dir=dir, fname=fname)

def print_graph():
    types_sprinkler = """
        Cloudy : Uses Watervapor
        Sprinkler : Uses Watervapor
        Rain : NotUses Watervapor
        Wet_Grass : NotUses Watervapor   
    """


    dataname = "sprinkler"
    tags = types_sprinkler
    

    skeleton, separating_sets, stat_tests, node_names, taglist = get_true_skeleton(dataname, tags=tags)
    priomat = get_priomat_from_skeleton(nx.adjacency_matrix(skeleton).todense(), taglist)
    dag, priomat = _orient_only_immoralites(skeleton=skeleton, sep_sets=separating_sets, priomat=priomat, taglist=taglist) #v-strucuteres
    adjacency_mat, priomat = meek_majority_tag_without_typeconsistency(cpdag=nx.adjacency_matrix(dag).todense(), tags=taglist, priomat=priomat, node_names=node_names)# Meek
    dir = "tagged-pc-using-LLM/tagged-PC"
    fname = "tdag_" + dataname + "_" + "true_skeleton_only_immoralities_and_meek" 
    create_graph_viz_colorless_all_undirected(dag=nx.adjacency_matrix(dag).todense(), var_names=node_names, save_to_dir=dir, fname=fname)


def get_priomat_from_skeleton_test():
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 0), (2, 3), (3,2)])
    dag_adj_matrix = nx.adjacency_matrix(G).todense()
    print(dag_adj_matrix)

    # Convert the DAG adjacency matrix
    taglist = [[0,2,3],[1,0,2],[2,0,5]]
    priomat = get_priomat_from_skeleton(dag_adj_matrix,taglist)
    print(priomat)

def test_calculate_shd_sid():
    true_dag = np.array([
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ])
    predicted_dag_unoriented = np.array([
        [0, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0]
    ])
    predicted_dag_false = np.array([
        [0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0]
    ])

    shd1, sid1 = calculate_shd_sid(dag=predicted_dag_unoriented, true_dag=true_dag)
    print("unoriented edges:")
    print(shd1)
    print(sid1)
    shd2, sid2 = calculate_shd_sid(dag=predicted_dag_false, true_dag=true_dag)
    print("wrong oriented edges:", )
    print(shd2)
    print(sid2)

# comment out test you like
# test_set_types()
# test_set_types_as_int()
# test_set_tags_as_int()
# test_get_typelist_from_text()
# test_get_taglist_as_int_from_taglist()
# get_type_test()
# type_consistency_test()
# test_printing_dag()
# test_skeleton_saving()
# get_priomat_from_skeleton_test()
# test_type_of_from_tag_all()
# test_amount_of_matching_types_for_two_way_fork()
# test_tag_step_by_step()
# test_order_tags()
test_llm()
# test_get_taglist_from_llm_output()
# test_recover_tag_string_onevsall_using_taglist()
# test_run_llm_tag_engineering()
# test_llm_generic_prompt()
# test_finding_cycles_adjacency_mat()
# test_cycle_add_edge_nx_graph()
# test_cycle_add_edge_adjacency()
# print_standard_pc_true_skeleton()
# print_true_graph()
# test_calculate_shd_sid()
# print_graph()