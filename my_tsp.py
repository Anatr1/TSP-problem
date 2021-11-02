import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from math import sqrt
from sys import argv

MAX_MUTATIONS = 10
NUM_CITIES = 23
NUM_CHILDREN = 1000
GRAPH = nx.DiGraph()
np.random.seed(int(argv[1]))

def get_pairs(sequence):
    seq = sequence[:]
    e_2 = seq.pop(0)
    seq.append(e_2)
    while seq:
        e_1 = e_2
        e_2 = seq.pop(0)
        yield (e_1, e_2)

def plot_solution(solution=None):
    if solution:
        GRAPH.remove_edges_from(list(GRAPH.edges))
        for n1, n2 in get_pairs(solution):
            GRAPH.add_edge(n1, n2)

    plt.figure(figsize=(20, 10))
    nx.draw(GRAPH,
            pos=nx.get_node_attributes(GRAPH, 'pos'),
            with_labels=True,
            node_color='pink')
    plt.show()


def single_swap_mutation(solution):
    swapped_solution = solution.copy()
    loc_a = np.random.randint(0, NUM_CITIES - 1)
    loc_b = np.random.randint(0, NUM_CITIES - 1)

    if loc_a != loc_b:
        gene_a = solution[loc_a]
        gene_b = solution[loc_b]
        swapped_solution[loc_a] = gene_b
        swapped_solution[loc_b] = gene_a

    return swapped_solution

def swap_mutate(solution, gen_n):
    mutated_solution = solution.copy()
    times = np.random.randint(1, max(MAX_MUTATIONS - gen_n, 3))
    for i in range(times):
        mutated_solution = single_swap_mutation(mutated_solution)

    return mutated_solution


def get_distance(node1, node2):
    return round( 1_000_000 / NUM_CITIES *
        sqrt(pow(node1[0] - node2[0], 2) + pow(node1[1] - node2[1], 2))
    )


def evaluate_solution(solution):
    total_cost = 0
    for n1, n2 in get_pairs(solution):
        n1 = GRAPH.nodes[n1]['pos']
        n2 = GRAPH.nodes[n2]['pos']
        total_cost += get_distance(n1, n2)
    return total_cost


def evolve_single_gen(parent_solution, gen_n):
    current_cost = evaluate_solution(parent_solution)
    best_so_far = parent_solution.copy()
    for c in range(NUM_CHILDREN):
        mutated_solution = swap_mutate(parent_solution, gen_n)
        cost = evaluate_solution(mutated_solution)
        print(f"Mutated solution n° {c}: \n\t{mutated_solution}\nCost: {round(cost,2)}\n")
        if cost <= current_cost:
            best_so_far = mutated_solution.copy()
            current_cost = cost
            print(f"Current best cost: {current_cost} from {best_so_far}")

    print(f"Final best cost: {current_cost} from {best_so_far}")

    return best_so_far


def evolve_solution(init_solution, gen_n):
    current_cost = evaluate_solution(init_solution)
    best_child = evolve_single_gen(init_solution, gen_n)
    child_cost = evaluate_solution(best_child)
    print(f"------------------\nGeneration n° {gen_n}\nBest solution: {best_child}\nCost: {round(child_cost,2)}\n------------------")
    if child_cost >= current_cost:
        return list(best_child)
        
    return evolve_solution(best_child, gen_n + 1)
    

for c in range(NUM_CITIES):
    GRAPH.add_node(c, pos=(np.random.random(), np.random.random()))

initial_solution = list(range(NUM_CITIES))
final_solution = evolve_solution(initial_solution, 0)

plot_solution(initial_solution)
print(f"Solution cost: {evaluate_solution(initial_solution):,}")
plot_solution(final_solution)
print(f"{evaluate_solution(final_solution)}")
