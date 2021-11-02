import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from math import sqrt
from sys import argv

MAX_MUTATIONS = 10
NUM_CITIES = 23
NUM_CHILDREN = 1000

class TSP:
    def __init__(self) -> None:
        if len(argv) > 1:
            np.random.seed(int(argv[1]))
        else:
            np.random.seed(NUM_CITIES)
        self._graph = nx.DiGraph()
        for c in range(NUM_CITIES):
            self._graph.add_node(c, pos=(np.random.random(), np.random.random()))


    def get_pairs(self, sequence):
        seq = sequence[:]
        e_2 = seq.pop(0)
        seq.append(e_2)
        while seq:
            e_1 = e_2
            e_2 = seq.pop(0)
            yield (e_1, e_2)

    def plot_solution(self, solution=None):
        if solution:
            self._graph.remove_edges_from(list(self._graph.edges))
            for n1, n2 in self.get_pairs(solution):
                self._graph.add_edge(n1, n2)

        plt.figure(figsize=(20, 10))
        nx.draw(self._graph,
                pos=nx.get_node_attributes(self._graph, 'pos'),
                with_labels=True,
                node_color='pink')
        plt.show()


    def single_swap_mutation(self, solution):
        swapped_solution = solution.copy()
        loc_a = np.random.randint(0, NUM_CITIES - 1)
        loc_b = np.random.randint(0, NUM_CITIES - 1)

        if loc_a != loc_b:
            gene_a = solution[loc_a]
            gene_b = solution[loc_b]
            swapped_solution[loc_a] = gene_b
            swapped_solution[loc_b] = gene_a

        return swapped_solution

    def swap_mutate(self, solution, gen_n):
        mutated_solution = solution.copy()
        times = np.random.randint(1, max(MAX_MUTATIONS - gen_n, 3))
        for _ in range(times):
            mutated_solution = self.single_swap_mutation(mutated_solution)

        return mutated_solution


    def get_distance(self, node1, node2):
        return round( 1_000_000 / NUM_CITIES *
            sqrt(pow(node1[0] - node2[0], 2) + pow(node1[1] - node2[1], 2))
        )


    def evaluate_solution(self, solution):
        total_cost = 0
        for n1, n2 in self.get_pairs(solution):
            n1 = self._graph.nodes[n1]['pos']
            n2 = self._graph.nodes[n2]['pos']
            total_cost += self.get_distance(n1, n2)
        return total_cost


    def evolve_single_gen(self, parent_solution, gen_n):
        current_cost = self.evaluate_solution(parent_solution)
        best_so_far = parent_solution.copy()
        for c in range(NUM_CHILDREN):
            mutated_solution = self.swap_mutate(parent_solution, gen_n)
            cost = self.evaluate_solution(mutated_solution)
            print(f"Mutated solution n° {c}: \n\t{mutated_solution}\nCost: {round(cost,2)}\n")
            if cost <= current_cost:
                best_so_far = mutated_solution.copy()
                current_cost = cost
                print(f"Current best cost: {current_cost} from {best_so_far}")

        print(f"Final best cost: {current_cost} from {best_so_far}")

        return best_so_far


    def evolve_solution(self, init_solution, gen_n):
        current_cost = self.evaluate_solution(init_solution)
        best_child = self.evolve_single_gen(init_solution, gen_n)
        child_cost = self.evaluate_solution(best_child)
        print(f"------------------\nGeneration n° {gen_n}\nBest solution: {best_child}\nCost: {round(child_cost,2)}\n------------------")
        if child_cost >= current_cost:
            return list(best_child)
            
        return self.evolve_solution(best_child, gen_n + 1)
        
def main():
    problem = TSP()
    initial_solution = list(range(NUM_CITIES))
    final_solution = problem.evolve_solution(initial_solution, 0)
    problem.plot_solution(initial_solution)
    print(f"Initial path cost: {problem.evaluate_solution(initial_solution):,}")
    problem.plot_solution(final_solution)
    print(f"Final path cost:{problem.evaluate_solution(final_solution):,}")


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().setLevel(level=logging.INFO)
    main()
