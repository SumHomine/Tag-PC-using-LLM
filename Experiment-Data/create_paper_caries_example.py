from itertools import combinations
from tag_pc import get_adjacency_mat_from_tag_weight_mat_cycle_check, get_adjacency_matrices_for_all_types, meek_majority_tag_without_typeconsistency, orient_forks_majority_tag_majority_top1, turn_adjacency_mats_to_weighted_prio_mats_depending_on_difference_to_skeleton, typed_meek_majority_tag_with_consistency
from tag_pc_utils import format_seperating_sets, get_priomat_from_skeleton, get_separating_sets_using_true_skeleton, get_taglist_of_int_from_text, get_undirect_graph, set_tags_as_int  
from visualization_experiment import create_graph_viz, create_graph_viz_all_undirected, create_graph_viz_colorless_all_undirected, create_graph_viz_typed_colors_edges_colorless

import networkx as nx
import numpy as np

def print_graph(): 
    node_names = ["Toothbrush access", "Brushing Teeth", "Sugar Intake", "Caries", "Obesity", "Tooth Loss", "Blood Pressure", "Heart Issues"]

    tags = """
    Toothbrush access: Socioeconomic, not Symptom
    Brushing Teeth: not Socioeconomic, not Symptom
    Sugar Intake: Socioeconomic, not Symptom
    Caries: not Socioeconomic, not Symptom
    Obesity: not Socioeconomic, not Symptom
    Tooth Loss: not Socioeconomic, Symptom
    Blood Pressure: not Socioeconomic, Symptom
    Heart Issues: not Socioeconomic, Symptom
    """

    true_dag_adjacency_mat = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])


    skeleton_adjacency_mat = get_undirect_graph(true_dag_adjacency_mat)

    
    # Skeleton
    dir = "Tag-PC-using-LLM/Experiment-Data/caries_example"
    fname = "Skeleton_Caries" 
    taglist = get_taglist_of_int_from_text(tags=tags, node_names_ordered=node_names)
    print(taglist)
    skeleton = nx.from_numpy_array(skeleton_adjacency_mat)
    skeleton = set_tags_as_int(skeleton, taglist)
    separating_sets = get_separating_sets_using_true_skeleton(skeleton=skeleton, true_adjacency_mat=true_dag_adjacency_mat)
    separating_sets = format_seperating_sets(separating_sets)
    print(separating_sets)
    node_ids = skeleton.number_of_nodes()
    for (i, j) in list(combinations(range(node_ids), 2)):
        print(f"seperating set of {i} {node_names[i]} and {j} {node_names[j]} is: {separating_sets[i][j]}")


    priomat = get_priomat_from_skeleton(nx.adjacency_matrix(skeleton).todense(), taglist)
    # create_graph_viz_typed_colors_edges_colorless(dag=nx.adjacency_matrix(skeleton).todense(), var_names=node_names, types=taglist[0], save_to_dir=dir, fname=fname)
    create_graph_viz_colorless_all_undirected(dag=true_dag_adjacency_mat, var_names=node_names, types=taglist[0], save_to_dir=dir, fname=fname)
    fname = "Skeleton_Caries_Tagged" 
    create_graph_viz_all_undirected(dag=true_dag_adjacency_mat, var_names=node_names, types=taglist[0], save_to_dir=dir, fname=fname)

    # Tag Majority
    dir = "Tag-PC-using-LLM/Experiment-Data/caries_example/Tag-Majority"
    fname = "V-Structure-Forks_Caries"
    dag, priomat = orient_forks_majority_tag_majority_top1(skeleton=skeleton, sep_sets=separating_sets, priomat=priomat, taglist=taglist, node_names=node_names)
    create_graph_viz_typed_colors_edges_colorless(dag=nx.adjacency_matrix(dag).todense(), var_names=node_names, types=taglist[0], save_to_dir=dir, fname=fname)

    fname = "Meek_Caries"
    adjacency_mat, priomat = meek_majority_tag_without_typeconsistency(cpdag=nx.adjacency_matrix(dag).todense(), tags=taglist, priomat=priomat, node_names=node_names)
    create_graph_viz_typed_colors_edges_colorless(dag=adjacency_mat, var_names=node_names, types=taglist[0], save_to_dir=dir, fname=fname)

    fname = "Tag-Propagation_Caries"
    adjacency_mat, priomat = typed_meek_majority_tag_with_consistency(cpdag=adjacency_mat, tags=taglist, priomat=priomat, majority_rule_typed=True, node_names=node_names)
    create_graph_viz_typed_colors_edges_colorless(dag=adjacency_mat, var_names=node_names, types=taglist[0], save_to_dir=dir, fname=fname)



    # Tag Weighted
    dir = "Tag-PC-using-LLM/Experiment-Data/caries_example/Tag-Weighted"
    fname = "Typed-DAGs_Caries"
    adjacency_mats_list = get_adjacency_matrices_for_all_types(skeleton=skeleton, separating_sets=separating_sets, taglist=taglist, majority_rule_typed=True)
    for i in range(len(adjacency_mats_list)): #plot adjacency mat for all types: #not necessary -> comment out if needed
        if i == 0:
            fname = "Typed-DAGs_Caries-Socioeconomic"
        if i == 1:
            fname = "Typed-DAGs_Caries-Symptom"
        create_graph_viz_typed_colors_edges_colorless(dag=adjacency_mats_list[i], var_names=node_names, types=taglist[i], save_to_dir=dir, fname=fname) 
    
    fname = "Compose-Typed-DAGs_Caries"
    weight_mats_list = turn_adjacency_mats_to_weighted_prio_mats_depending_on_difference_to_skeleton(adjacency_mats_list=adjacency_mats_list, skeleton_mat=nx.adjacency_matrix(skeleton).todense())
    adjacency_mat, priomat, weight_mat = get_adjacency_mat_from_tag_weight_mat_cycle_check(weight_mats_list=weight_mats_list, skeleton_mat=nx.adjacency_matrix(skeleton).todense(), taglist=taglist, node_names=node_names)
    create_graph_viz_typed_colors_edges_colorless(dag=adjacency_mat, var_names=node_names, types=taglist[0], save_to_dir=dir, fname=fname)

    fname = "Meek_Caries"
    meek_majority_tag_without_typeconsistency(cpdag=adjacency_mat, tags=taglist, priomat=priomat, node_names=node_names)
    create_graph_viz_typed_colors_edges_colorless(dag=adjacency_mat, var_names=node_names, types=taglist[0], save_to_dir=dir, fname=fname)

    # True DAG
    dir = "Tag-PC-using-LLM/Experiment-Data/caries_example"
    fname = "True-DAG_Caries" 
    create_graph_viz_typed_colors_edges_colorless(dag=true_dag_adjacency_mat, var_names=node_names, types=taglist[0], save_to_dir=dir, fname=fname)
    fname = "True-DAG-Green_Caries"
    create_graph_viz(dag=true_dag_adjacency_mat, var_names=node_names, types=taglist[0], save_to_dir=dir, fname=fname)


print_graph()