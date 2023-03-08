from __future__ import annotations
from sys import argv
from functools import lru_cache
from collections import deque, defaultdict
from typing import List, Tuple, Generator, Dict, DefaultDict, Deque, Union


class Puzzle(object):
    def __init__(
        self,
        puzzle: List[List[int]],
        x: Union[int, None] = None,
        y: Union[int, None] = None,
    ) -> None:
        """
        Constructor for the Puzzle class. Requires the puzzle 2D list
        that represents the puzzle. Optionally takes the x and y index
        positions of the empty space in the puzzle. If not given, they
        will be calculated.
        """
        self.puzzle = puzzle
        if x and y:
            self.x: int = x
            self.y: int = y
            return
        # Loop through the puzzle and search for the empty space
        for Y, row in enumerate(puzzle):
            for X, v in enumerate(row):
                if v == 0:
                    self.y: int = Y
                    self.x: int = X
                    return
        # If the given puzzle does not have an empty space signified by a 0
        raise Exception(f"Puzzle {puzzle} does not have any empty space (0)")

    @staticmethod
    def read_from_file(filename: str) -> Puzzle:
        """Read and generate a puzzle from a file"""
        with open(filename, "r") as f:
            w, h = map(int, f.readline().split())
            return Puzzle([[int(v) for v in f.readline().split()] for _ in range(h)])

    @staticmethod
    def solved_of_size(n: int) -> Puzzle:
        """Generate a solved puzzle of size (n x n)"""
        answer: List[List[int]] = []
        for i in range(n - 1):
            answer.append(list(range(i * n + 1, (i + 1) * n + 1)))
        answer.append(list(range((n - 1) * n + 1, n * n)) + [0])
        return Puzzle(answer)

    def to_tuple(self) -> Tuple[Tuple[int]]:
        """Convert the puzzle object into a tuple of tuples (for use in a dict)"""
        return tuple(tuple(row) for row in self.puzzle)

    def move_into_empty(self, x: int, y: int) -> None:
        """Move the tile at a given position into the empty space"""
        self.puzzle[self.y][self.x], self.puzzle[y][x] = (
            self.puzzle[y][x],
            self.puzzle[self.y][self.x],
        )
        self.y, self.x = y, x

    def next_moves(self) -> Generator[Puzzle]:
        """Given the current state of the puzzle, find all possible moves"""
        around: List[Tuple[int]] = [
            (self.x - 1, self.y),
            (self.x + 1, self.y),
            (self.x, self.y - 1),
            (self.x, self.y + 1),
        ]
        # Look around at all valid neighbors
        for nx, ny in around:
            if (0 <= nx < len(self.puzzle[0])) and (0 <= ny < len(self.puzzle)):
                # Make a copy of the base puzzle
                next_puzzle = self.__copy__()
                # Perform the appropriate swap
                next_puzzle.move_into_empty(nx, ny)
                # Yield the newly created puzzle
                yield next_puzzle

    def __copy__(self) -> Puzzle:
        """Create a deep copy of a Puzzle object"""
        return Puzzle([row.copy() for row in self.puzzle], x=self.x, y=self.y)

    def __str__(self) -> str:
        """to string method for the Puzzle class"""
        return "\n".join(
            " ".join(map(lambda v: f"{v:2d}" if v else "  ", row))
            for row in self.puzzle
        )


class PuzzleSolver(object):
    def __init__(self, n: int) -> None:
        """Constructor for the PuzzleSolver class"""
        # Generate a solved puzzle of size (n x n)
        solved: Puzzle = Puzzle.solved_of_size(n)

        # Initialize the solver dictionary.
        # The value at a given puzzle is it's parent puzzle.
        # This parent puzzle is the puzzle is the result of the transformation
        # that should be made on the current puzzle to optimally solve.
        self.solver: Dict[Tuple[Tuple[int]] : Tuple[Tuple[int]]] = dict()
        # The solved puzzle has no parent. It is already solved
        self.solver[solved.to_tuple()] = None

        # Start the search from the solved puzzle
        q: Deque[Puzzle] = deque()
        q.append(solved)

        # While there are more puzzles to be inspected, continue searching
        while len(q):
            # Get the current puzzle
            current_puzzle: Puzzle = q.popleft()
            # Go through every possible next move for the current puzzle
            for next_puzzle in current_puzzle.next_moves():
                # If there is a faster way to get to this puzzle state, skip it
                if next_puzzle.to_tuple() not in self.solver:
                    # If this is the first time we're seeing it, record it's parent
                    self.solver[next_puzzle.to_tuple()] = current_puzzle.to_tuple()
                    # And add it to the queue for future inspection
                    q.append(next_puzzle)

    @lru_cache(maxsize=None)
    def solve_steps(self, puzzle_tuple: Union[Tuple[Tuple[int]], None]) -> int:
        """Given a puzzle, get the number of steps to optimally solve that puzzle"""
        if puzzle_tuple not in self.solver:
            return -1
        next_puzzle = self.solver[puzzle_tuple]
        return 1 + self.solve_steps(next_puzzle) if next_puzzle else 0

    def show_required_moves(self, puzzle_tuple: Tuple[Tuple[int]]) -> None:
        """Given a puzzle, print the exact moves required to optimally solve"""
        if puzzle_tuple not in self.solver:
            print("Given puzzle is unsolvable.")
            return
        current_puzzle = puzzle_tuple
        next_puzzle = self.solver[current_puzzle]
        while next_puzzle != None:
            print(get_diff(current_puzzle, next_puzzle))
            current_puzzle, next_puzzle = next_puzzle, self.solver[next_puzzle]

    def show_solve_trace(self, puzzle_tuple: Union[Tuple[Tuple[int]], None]) -> int:
        """Given a puzzle, print the steps to optimally solve that puzzle"""
        # If this puzzle is not reachable
        if puzzle_tuple not in self.solver:
            print("Given puzzle is unsolvable.")
            return -1

        # Print the puzzle out and a line separator
        print(
            "\n".join(
                " ".join(map(lambda v: f"{v:2d}" if v else "  ", row))
                for row in puzzle_tuple
            )
        )
        print("-" * (len(puzzle_tuple) * 3 - 1))

        next_puzzle = self.solver[puzzle_tuple]
        # If puzzle_tuple describes the solved puzzle
        if next_puzzle == None:
            # No moves need to be made
            return 0

        # Otherwise, search this puzzle's parent and count a move
        return 1 + self.show_solve_trace(next_puzzle)


def get_diff(old_puzzle: Tuple[Tuple[int]], new_puzzle: Tuple[Tuple[int]]) -> int:
    for old_row, new_row in zip(old_puzzle, new_puzzle):
        for old_value, new_value in zip(old_row, new_row):
            if old_value == 0:
                return new_value
            if new_value == 0:
                return old_value
    return -1


def main(filename: str, show_moves_only: bool) -> None:
    puzzle = Puzzle.read_from_file(filename)
    solver = PuzzleSolver(len(puzzle.puzzle))
    if show_moves_only:
        solver.show_required_moves(puzzle.to_tuple())
    else:
        num_moves = solver.show_solve_trace(puzzle.to_tuple())
        print(f"Solved in {num_moves} move{'s' * (num_moves != 1)}")


def display_frequencies():
    frequencies: DefaultDict[List:int] = defaultdict(list)

    puzzle_solver: PuzzleSolver = PuzzleSolver(3)
    for puzzle_tuple in puzzle_solver.solver:
        frequencies[puzzle_solver.solve_steps(puzzle_tuple)].append(puzzle_tuple)

    total_puzzles: int = 0
    for frequency in frequencies:
        print(f"{frequency:2d}: {len(frequencies[frequency]):5d}")
        total_puzzles += len(frequencies[frequency])

    print("Total:", total_puzzles)


if __name__ == "__main__":
    if 2 <= len(argv) <= 3:
        main(argv[1], len(argv) == 3 and argv[2] in ["-m", "--moves", "--moves-only"])
    else:
        display_frequencies()
