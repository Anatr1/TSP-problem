# TSP PROBLEM

My personal implementation for the TSP problem for the Computational Intelligence course by Prof. Squillero at Politecnico di Torino.

## Usage

```
python my_tsp_classes.py [random seed]
```
If the random seed is not given it will automatically use ```23``` as random seed

## About
I already had a working implementation which I uploaded as ```my_tsp.py```.```my_tsp_classes.py``` only differs for the use of the class ```TSP```.

The implementation works by generating ```NUM_CHILDREN``` new paths from a parent path. Each of the children has been mutated, whereas for mutation I mean swapping two of its elements, a random number of times between 1 and ```MAX_MUTATIONS```. The actual number of maximum possible mutation a child can possibly have is decreasing as the number of explored generations grows. For each generation the children with the lesser cost is takes ad the next parent path. The program stops executing when it's unable to generate a children with a lesser cost than its parent. 

I already tested my implementation against the one given by the prof.
I ran the two implementations a hundred times using random seeds from 1 to 100.
As can be seen here my implementation was able to find a shorter path in 53 cases, resulting in an average of 0.6% shorter distances on the overall test cases.
