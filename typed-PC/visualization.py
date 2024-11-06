# stolen from https://github.com/ServiceNow/typed-dag/blob/main/typed_pc/graph_utils.py

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from graphviz import Graph, Digraph


def create_skeleton_viz(graph, coeff, var_names, stat_tests, types=None):
    """ Create a graph from a skeleton in graphviz for visualization """
    colors = ["green", "red"]
    g = Graph("G", filename="skeleton.gv")

    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if graph[i, j] == 1:
                if stat_tests[i, j]:  # == 1 if statistically significative
                    g.edge(var_names[i], var_names[j], color=colors[0], label=str(coeff[i, j]))
                else:
                    g.edge(var_names[i], var_names[j], color=colors[0])

    return g


def rgb2hex(rgb: list) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


# used for plotting at the end #TODO mit farben herumspielen ggf. undirected andere farbe als directed
def create_graph_viz(dag: np.ndarray, var_names: list, types: list, save_to_dir: str, fname="tdag"): #added save_to_dir parameter and changed default name, changed edge addition
    """ Create a graph from a CPDAG in graphviz for visualization """
    edge_colors = ["green", "red"]

    if len(np.unique(types)) <= 12:
        cmap = matplotlib.cm.get_cmap("Set3", 12)
        node_colors = cmap(np.linspace(0, 1, 12))
    else:
        node_colors = plt.get_cmap("hsv")(np.linspace(0, 1.0, len(np.unique(types))))
    node_colors = [rgb2hex(c) for c in node_colors]

    g = Digraph("G", filename=f"figures/{fname}", format="png")

    for i in range(dag.shape[0]):
        color = node_colors[types[i]]
        g.node(var_names[i], fillcolor=color, style="filled")

    #XXX Gulfan edit: removed stat_tests and added undirected edges
    for i in range(dag.shape[0]):
        for j in range(i):  # iterate only over lower matrix
            if dag[i, j] == 1 and dag[j, i] == 1:  # undirected edge
                g.edge(var_names[i], var_names[j], color=edge_colors[0], label="", dir="none") #change dir to none or both depending on preference
                g.edge(var_names[j], var_names[i], color=edge_colors[0], label="", dir="none", style="invis") #invisible edge to keep nice looking distance
            elif dag[i, j] == 1:  # directed edge from i to j
                g.edge(var_names[i], var_names[j], color=edge_colors[0], label="",)
            elif dag[j, i] == 1:  # directed edge from j to i 
                g.edge(var_names[j], var_names[i], color=edge_colors[0], label="",)


    g.render(directory=save_to_dir, overwrite_source=True, cleanup=True) #XXX Guldan edit: added cleanup=True so that no unnecessary dot file ist saved

    return g

# used for plotting at the end #TODO mit farben herumspielen ggf. undirected andere farbe als directed
def create_graph_viz_colorless(dag: np.ndarray, var_names: list, types: list, save_to_dir: str, fname="tdag"): #added save_to_dir parameter and changed default name, changed edge addition
    """ Create a graph from a CPDAG in graphviz for visualization """
    edge_colors = ["green", "red"]
    edge_colors = ["black", "green", "red"] # comment out for normal green edges

    g = Digraph("G", filename=f"figures/{fname}", format="png")

    for i in range(dag.shape[0]):
        g.node(var_names[i])

    #XXX Gulfan edit: removed stat_tests and added undirected edges
    for i in range(dag.shape[0]):
        for j in range(i):  # iterate only over lower matrix
            if dag[i, j] == 1 and dag[j, i] == 1:  # undirected edge
                g.edge(var_names[i], var_names[j], color=edge_colors[0], label="", dir="none") #change dir to none or both depending on preference
                g.edge(var_names[j], var_names[i], color=edge_colors[0], label="", dir="none", style="invis") #invisible edge to keep nice looking distance
            elif dag[i, j] == 1:  # directed edge from i to j
                g.edge(var_names[i], var_names[j], color=edge_colors[0], label="",)
            elif dag[j, i] == 1:  # directed edge from j to i 
                g.edge(var_names[j], var_names[i], color=edge_colors[0], label="",)


    g.render(directory=save_to_dir, overwrite_source=True, cleanup=True) #XXX Guldan edit: added cleanup=True so that no unnecessary dot file ist saved

    return g



# unused, old graph_viz implementation (TODO delete)
def create_single_edges_graph_viz(dag: np.ndarray, var_names: list, stat_tests: np.ndarray, types: list, save_to_dir: str, fname="tdag"): #added save_to_dir parameter and changed defauolt name
    """ Create a graph from a CPDAG in graphviz for visualization """
    edge_colors = ["green", "red"]

    if len(np.unique(types)) <= 12:
        cmap = matplotlib.cm.get_cmap("Set3", 12)
        node_colors = cmap(np.linspace(0, 1, 12))
    else:
        node_colors = plt.get_cmap("hsv")(np.linspace(0, 1.0, len(np.unique(types))))
    node_colors = [rgb2hex(c) for c in node_colors]

    g = Digraph("G", filename=f"figures/{fname}", format="png")

    for i in range(dag.shape[0]):
        color = node_colors[types[i]]
        g.node(var_names[i], fillcolor=color, style="filled")

    for i in range(dag.shape[0]):
        for j in range(dag.shape[1]):
            if dag[i, j] == 1:
                if stat_tests[i, j]:  # == 1 if statistically significative
                    g.edge(
                        var_names[i],
                        var_names[j],
                        color=edge_colors[0],
                        label="",
                    )
                else:
                    g.edge(var_names[i], var_names[j], color=edge_colors[1])

    g.render(directory=save_to_dir, overwrite_source=True, cleanup=True) #XXX Guldan edit: added cleanup=True so that no unnecessary dot file ist saved

    return g

def show_data(data, dag, only_child=True):
    n_nodes = data.shape[1]

    fig, axs = plt.subplots(n_nodes, n_nodes, figsize=(15, 15))

    for i in range(n_nodes):
        for j in range(n_nodes):
            if only_child and dag[i, j] == 1:
                axs[i, j].scatter(data[:, i], data[:, j], s=1)
            elif not only_child:
                axs[i, j].scatter(data[:, i], data[:, j], s=1)


#used to plot intermediate states, works the same as create_graph_viz
def plot_dag_state(dag: np.ndarray, var_names: list, types: list, experiment_step : str, step_number = 0, addiational_name = ""):
    dir_path = "tagged-pc-using-LLM/tagged-PC/intermediate_tagPC_dag_states"
    fname = "tdag_intermediate" + "_step_" + str(step_number) + "_" + experiment_step  + "_" + addiational_name

    #copied and changed from create_graph_viz
    """ Create a graph from a CPDAG in graphviz for visualization """
    if len(np.unique(types)) <= 12:
        cmap = matplotlib.cm.get_cmap("Set3", 12)
        node_colors = cmap(np.linspace(0, 1, 12))
    else:
        node_colors = plt.get_cmap("hsv")(np.linspace(0, 1.0, len(np.unique(types))))
    node_colors = [rgb2hex(c) for c in node_colors]

    g = Digraph("G", filename=f"figures/{fname}", format="png")

    for i in range(dag.shape[0]):
        color = node_colors[types[i]]
        g.node(var_names[i], fillcolor=color, style="filled")

    for i in range(dag.shape[0]):
        for j in range(i):  # iterate only over lower matrix
            if dag[i, j] == 1 and dag[j, i] == 1:  # undirected edge
                g.edge(var_names[i], var_names[j], color="green", label="", dir="none") #change dir to none or both depending on preference
                g.edge(var_names[j], var_names[i], color="green", label="", dir="none", style="invis") #invisible edge to keep nice looking distance
            elif dag[i, j] == 1:  # directed edge from i to j
                g.edge(var_names[i], var_names[j], color="green", label="",)
            elif dag[j, i] == 1:  # directed edge from j to i 
                g.edge(var_names[j], var_names[i], color="green", label="",)

    g.render(directory=dir_path, overwrite_source=True, cleanup=True)

    return g