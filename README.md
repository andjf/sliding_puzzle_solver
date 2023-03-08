# sliding_puzzle_solver

## Usage
To run the solver on a given puzzle, first create the puzzle file in the following format:

`puzzle1.in`
```
3 3
3 8 1
5 6 0
4 7 2
```

The project can be run with:
```bash
$ > python3 solver.py puzzle1.in
```

This will print out the trace of how to optimally solve the given board in the following format:
```
 3  8  1
 5  6   
 4  7  2
--------
 3  8  1
 5  6  2
 4  7   
--------
 3  8  1
 5  6  2
 4     7
--------
 3  8  1
 5     2
 4  6  7
--------
 3     1
 5  8  2
 4  6  7

    .
    .
    .

 1  2  3
 4  5  6
    7  8
--------
 1  2  3
 4  5  6
 7     8
--------
 1  2  3
 4  5  6
 7  8   
--------
Solved in 25 moves
```

Alternatively, to calculate and display the distribution of number of moves for all reachable boards:
```bash
> $ python3 solve.py
```

The output will show the number of boards that can be optimally solved in a given number of moves
```
 0:     1
 1:     2
 2:     4
 3:     8
 4:    16
 5:    20
 6:    39
 7:    62
 8:   116
 9:   152
10:   286
11:   396
12:   748
13:  1024
14:  1893
15:  2512
16:  4485
17:  5638
18:  9529
19: 10878
20: 16993
21: 17110
22: 23952
23: 20224
24: 24047
25: 15578
26: 14560
27:  6274
28:  3910
29:   760
30:   221
31:     2
```

This means that there are 2 boards that can be solved optimally in 31 moves. They are:
```
---------
 6  4  7
 8  5
 3  2  1
--------
 8  6  7
 2  5  4
 3     1
--------
```
