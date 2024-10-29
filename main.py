import numpy as np
import copy
import random

# Load sudokus
#sudoku = np.load("data/very_easy_puzzle.npy")
#print("very_easy_puzzle.npy has been loaded into the variable sudoku")
#print(f"sudoku.shape: {sudoku.shape}, sudoku[0].shape: {sudoku[0].shape}, sudoku.dtype: {sudoku.dtype}")

# Load solutions for demonstration
#solutions = np.load("data/very_easy_solution.npy")


class PartialSudokuState:
    def __init__(self, puzzle):
        self.n = 9
        self.puzzle = puzzle
        self.possible_values = self.initialize_possible_values()

    def initialize_possible_values(self):
        possible_values = np.empty((self.n, self.n), dtype=object)
        for row in range(self.n):
            for column in range(self.n):
                if self.puzzle[row][column] == 0:
                    possible_values[row][column] = list(range(1, self.n + 1))
                else:
                    possible_values[row][column] = [self.puzzle[row][column]]
        return possible_values

    def is_goal(self):
        """This partial state is a goal state if every cell has a final value"""

        return np.all(self.puzzle != 0)

    def is_invalid(self):
        """This partial state is invalid if any cell has no possible values"""

        return any(len(values) == 0 for row in self.possible_values for values in row)

    def get_possible_values(self, row, column):
        return self.possible_values[row][column].copy()

    def get_final_state(self):
        if self.is_goal():
            return self.puzzle
        else:
            return -1

    def get_singleton_cells(self):
        """Returns the cells which have no final value but exactly 1 possible value"""
        return [(i, j) for i in range(self.n) for j in range(self.n)
                if len(self.possible_values[i][j]) == 1 and self.puzzle[i][j] == 0]

    def set_value(self, row, column, value):
        """Returns a new state with this cell set to this value, and the change propagated to other domains"""
        if value not in self.possible_values[row][column]:
            raise ValueError(f"{value} is not a valid choice for cell ({row}, {column})")

        # create a deep copy: the method returns a new state, does not modify the existing one
        state = copy.deepcopy(self)

        # update this cell
        state.possible_values[row][column] = [value]
        state.puzzle[row][column] = value

        # now update all other cells possible values
        # update same row and column
        for i in range(self.n):
            if value in state.possible_values[i][column] and i != row:
                state.possible_values[i][column].remove(value)
            if value in state.possible_values[row][i] and i != column:
                state.possible_values[row][i].remove(value)

        # update same box
        start_row, start_col = row - row % 3, column - column % 3
        for i in range(3):
            for j in range(3):
                if value in state.possible_values[start_row + i][start_col + j] and (start_row + i, start_col + j) != (
                row, column):
                    state.possible_values[start_row + i][start_col + j].remove(value)

        # if any other cells with no final value only have 1 possible value, make them final
        singleton_cells = state.get_singleton_cells()
        while len(singleton_cells) > 0:
            cell = singleton_cells[0]
            state = state.set_value(cell[0], cell[1], state.possible_values[cell[0]][cell[1]][0])
            singleton_cells = state.get_singleton_cells()

        return state


def pick_next_cell(sudoku_state):
    """Chooses the next cell to be searched based on the number of possible values the cell has - will pick the minimum"""
    cell_index = None
    minLen = 10
    for row in range(sudoku_state.n):
        for col in range(sudoku_state.n):
            if len(sudoku_state.possible_values[row][col]) > 1 and sudoku_state.puzzle[row][col] == 0:
                if len(sudoku_state.possible_values[row][col]) < minLen:
                    minLen = len(sudoku_state.possible_values[row][col])
                    cell_index = (row, col)
    return cell_index


def order_values(sudoku_state, row_index, col_index):
    values = sudoku_state.get_possible_values(row_index, col_index)
    random.shuffle(values)
    return values


def depth_first_search(partial_sudoku_state):
    cell_index = pick_next_cell(partial_sudoku_state)
    values = order_values(partial_sudoku_state, cell_index[0], cell_index[1])

    for value in values:
        new_state = partial_sudoku_state.set_value(cell_index[0], cell_index[1], value)
        if new_state.is_goal():
            return new_state
        if not new_state.is_invalid():
            deep_state = depth_first_search(new_state)
            if deep_state is not None and deep_state.is_goal():
                return deep_state
    return None


def sudoku_solver(sudoku):
    """
    Solves a Sudoku puzzle and returns its unique solution.

    Input
        sudoku : 9x9 numpy array
            Empty cells are designated by 0.

    Output
        9x9 numpy array of integers
            It contains the solution, if there is one. If there is no solution, all array entries should be -1.
    """
    partial_sudoku_state = PartialSudokuState(sudoku)
    solved_sudoku_puzzle = None
    noSolution = False

    # Adjusts the possible moves for the empty squares based on the squares that are already there

    anchors = [(i, j) for i in range(partial_sudoku_state.n) for j in range(partial_sudoku_state.n)
               if partial_sudoku_state.puzzle[i][j] != 0]

    for cell in anchors:
        row = cell[0]
        col = cell[1]
        if not partial_sudoku_state.set_value(row, col, partial_sudoku_state.puzzle[row][col]).is_invalid():
            partial_sudoku_state = partial_sudoku_state.set_value(row, col, partial_sudoku_state.puzzle[row][col])
        else:
            noSolution = True
            break

    if not partial_sudoku_state.is_goal() and not noSolution:
        solved_sudoku_puzzle = depth_first_search(partial_sudoku_state)
    elif noSolution:
        solved_sudoku_puzzle = None
    else:
        solved_sudoku_puzzle = partial_sudoku_state

    if solved_sudoku_puzzle == None:
        solved_sudoku = np.full((9, 9), -1)
    else:
        solved_sudoku = solved_sudoku_puzzle.get_final_state()
    return solved_sudoku


sudoku = np.array([[0,0,3,0,0,0,0,6,0],
                   [9,0,0,4,0,0,0,7,5],
                   [0,4,0,0,8,0,0,0,0],
                   [0,0,0,5,0,0,6,0,0],
                   [0,0,7,0,2,0,0,5,1],
                   [1,0,0,0,0,0,8,0,0],
                   [0,0,0,0,0,0,1,0,0],
                   [0,0,2,0,0,3,0,0,0],
                   [7,0,0,8,0,0,0,4,9]])

solution = sudoku_solver(sudoku)
#print(sudoku[0])
#print("")
print(solution)
#print("")
#print(solutions[0])
#print("")
#print(np.array_equal(solution, solutions[0]))