from tag_pc import get_true_skeleton, meek_majority_tag_without_typeconsistency, orient_forks_majority_tag_majority_top1, typed_meek_majority_tag_with_consistency
from tag_pc_utils import get_priomat_from_skeleton  
from visualization_experiment import create_graph_viz_typed_colors_edges_colorless

import networkx as nx

def print_graph():
    types_sprinkler = """
        Cloudy : Uses Watervapor
        Sprinkler : Uses Watervapor
        Rain : NotUses Watervapor
        Wet_Grass : NotUses Watervapor   
    """


    dataname = "sprinkler"
    tags = types_sprinkler
    
    dir = "Tag-PC-using-LLM/tagged-PC"
    fname = "tdag_" + dataname + "_" + "true_skeleton_caries" 

    skeleton, separating_sets, stat_tests, node_names, taglist = get_true_skeleton(dataname, tags=tags)
    priomat = get_priomat_from_skeleton(nx.adjacency_matrix(skeleton).todense(), taglist)
    dag, priomat = orient_forks_majority_tag_majority_top1(skeleton=skeleton, sep_sets=separating_sets, priomat=priomat, taglist=taglist, node_names=node_names)

    adjacency_mat, priomat = meek_majority_tag_without_typeconsistency(cpdag=nx.adjacency_matrix(dag).todense(), tags=taglist, priomat=priomat, node_names=node_names)

    adjacency_mat, priomat = typed_meek_majority_tag_with_consistency(cpdag=adjacency_mat, tags=taglist, priomat=priomat, majority_rule_typed=True, node_names=node_names)


    create_graph_viz_typed_colors_edges_colorless(dag=nx.adjacency_matrix(dag).todense(), var_names=node_names, types=types_sprinkler[0], save_to_dir=dir, fname=fname)


print_graph()