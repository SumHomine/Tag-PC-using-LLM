import networkx as nx
from itertools import combinations, permutations
from random import shuffle


from tag_pc import get_true_skeleton, meek_majority_tag_without_typeconsistency
from tag_pc_utils import _orient_typeless_with_priomat, get_priomat_from_skeleton, type_of_from_tag_all

def standard_pc_deconstructed(dataname, tags):
    skeleton, separating_sets, stat_tests, node_names, taglist = get_true_skeleton(dataname, tags=tags)
    priomat = get_priomat_from_skeleton(nx.adjacency_matrix(skeleton).todense(), taglist)
    dag, priomat = _orient_only_immoralites(skeleton=skeleton, sep_sets=separating_sets, priomat=priomat, taglist=taglist) #v-strucuteres
    adjacency_mat, priomat = meek_majority_tag_without_typeconsistency(cpdag=nx.adjacency_matrix(dag).todense(), tags=taglist, priomat=priomat, node_names=node_names)# Meek
    return adjacency_mat




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
                )
                # orient just v-structure ignore tags for now, but add strong entry in prio matrix (since v-strucutre comes from data it should not be overwritten easily later on)
                prio_weight = len(taglist)
                _orient_typeless_with_priomat(dag, i, k, priomat, prio_weight)
                _orient_typeless_with_priomat(dag, j, k, priomat, prio_weight)
                # print("priomat after orienten v-strucutre: \n", priomat)
                
                
    return dag, priomat