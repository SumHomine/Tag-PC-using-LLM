import pickle
import networkx as nx
import numpy as np
import os
from itertools import permutations
import bnlearn as bn

from tpc_utils import get_typelist_of_string_from_text, set_types, set_types_as_int, get_typelist_from_text
from tpc import create_skeleton_using_causallearn
from visualization import create_graph_viz, show_data
from llm_interface_typed import run_llm_generic_prompt_typed

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

def test_skeleton_creation():
    # bn.import_example() is not deterministic
    alpha = 0.05
    indep_test = "fisherz"
    data = bn.import_example("asia")
    node_names = data.columns.tolist()
    data = data.values
    types = """
        asia : first
        tub : first
        smoke : second
        lung : second
        bronc : third
        either : third
        xray : fourth
        dysp : fourth
            """
    typelist = get_typelist_from_text(types)
    iterations = 5
    matrices = []
    for i in range(iterations):
        skeleton, separating_sets, stat_tests = create_skeleton_using_causallearn(data, typelist, alpha, indep_test)
        skeleton_mat = nx.adjacency_matrix(skeleton).todense()
        matrices.append(skeleton_mat)
        # print("skeleton: \n", skeleton_mat)
    for i in range(iterations):
        print("skeleton: \n", skeleton_mat)
    martix_equal, compare_matrix, uneqal_matrix = all_matrices_equal(matrices)
    print("matrices equal?:", martix_equal)
    if (not martix_equal):
        print("\nunequal matrix 1: \n", compare_matrix, "\nunequal matrix 2: \n: ", uneqal_matrix)
    
def all_matrices_equal(matrices):
    compare_matrix = matrices[0]
    for matrix in matrices:
        if not (compare_matrix == matrix).all():
            return False, compare_matrix, matrix
    return True, None, None

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

def test_order_types():
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

    path = os.path.join("Tag-PC-using-LLM/additionalData", ("insurance" + ".bif"))
    model = bn.import_DAG(path)
    adjacency_mat = model['adjmat']
    node_names = adjacency_mat.columns.tolist()
    print("node names:", node_names)

    typelist_insurance_true = get_typelist_of_string_from_text(type_insurance_true, node_names)
    print("typelist from ordered typelist:", typelist_insurance_true)

    typelist_insurance_false = get_typelist_of_string_from_text(type_insurance_false, node_names)
    print("typelist from missordered typelist:", typelist_insurance_false)

    #check if both taglists are equal
    assert len(typelist_insurance_true) == len(typelist_insurance_false), "test failed"

    # Check if each sublist is equal
    for sublist1, sublist2 in zip(typelist_insurance_true, typelist_insurance_false):
        assert sublist1 == sublist2, "test failed"




def test_llm_output():
    # node names for generic prompt LLM Typing
    node_names_sprinkler = ["Cloudy", "Sprinkler", "Rain", "Wet_Grass"]
    node_names_asia = ["asia", "tub", "smoke", "lung", "bronc", "either", "xray",  "dysp"]
    node_names_alarm = ['LVFAILURE', 'HISTORY', 'LVEDVOLUME', 'CVP', 'PCWP', 'HYPOVOLEMIA', 'STROKEVOLUME', 'ERRLOWOUTPUT', 'HRBP', 'HR', 'ERRCAUTER', 'HREKG', 'HRSAT', 'ANAPHYLAXIS', 'TPR', 'ARTCO2', 'EXPCO2', 'VENTLUNG', 'INTUBATION', 'MINVOL', 'FIO2', 'PVSAT', 'VENTALV', 'SAO2', 'SHUNT', 'PULMEMBOLUS', 'PAP', 'PRESS', 'KINKEDTUBE', 'VENTTUBE', 'MINVOLSET', 'VENTMACH', 'DISCONNECT', 'CATECHOL', 'INSUFFANESTH', 'CO', 'BP']
    node_names_insurance = ['SocioEcon', 'GoodStudent', 'Age', 'RiskAversion', 'VehicleYear', 'Accident', 'ThisCarDam', 'RuggedAuto', 'MakeModel', 'Antilock', 'Mileage', 'DrivQuality', 'DrivingSkill', 'SeniorTrain', 'ThisCarCost', 'CarValue', 'Theft', 'AntiTheft', 'HomeBase', 'OtherCarCost', 'PropCost', 'OtherCar', 'MedCost', 'Cushioning', 'Airbag', 'ILiCost', 'DrivHist']

    types, node_names = run_llm_generic_prompt_typed(node_names=node_names_alarm, determinstic=True)

# comment out test you like
# test_set_types()
# test_set_types_as_int()
# test_get_typelist_from_text()
# get_type_test()
# type_consistency_test()
# test_printing_dag()
# test_skeleton_saving()
# test_skeleton_creation()
# test_order_types()
test_llm_output()