"""
Lab 3 - Search and A*.

Implements:
  * A* search with Manhattan heuristic (Task 2, 4, 6)
  * BFS for blind-search comparison (Task 5)
  * Four test maps: original, trivial, no-solution, alternative-paths (Task 3)
  * Heuristic experiments: h=0, Euclidean, Manhattan*2 (Task 6)

Every search returns a SearchResult holding:
  solution_found, path (list of cells), path_length (moves),
  states_expanded (# pops from frontier), algorithm, heuristic.
"""

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

Cell = Tuple[int, int]
Heuristic = Callable[[Cell, Cell], float]


# ---------------------------------------------------------------------------
# Warehouse map (Task 4 of lab2 is reused)
# ---------------------------------------------------------------------------
WAREHOUSE = [
    "##################",
    "#S....#..........#",
    "#.###.#.########.#",
    "#...#.#........#.#",
    "###.#.########.#.#",
    "#...#..........#.#",
    "#.############.#.#",
    "#..............#G#",
    "##################",
]


ACTIONS = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    algorithm: str
    heuristic: str
    solution_found: bool
    path: List[Cell] = field(default_factory=list)
    path_length: int = 0            # number of moves = len(path) - 1
    states_expanded: int = 0        # pops from the frontier

    def summary(self) -> str:
        status = "FOUND" if self.solution_found else "NO SOLUTION"
        return (f"{self.algorithm:5s} h={self.heuristic:11s}  "
                f"{status:11s}  len={self.path_length:3d}  "
                f"expanded={self.states_expanded:4d}")


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------
def find(grid: List[str], ch: str) -> Cell:
    for r, row in enumerate(grid):
        c = row.find(ch)
        if c != -1:
            return (r, c)
    raise ValueError(f"{ch!r} not in grid")


def passable(grid: List[str], cell: Cell) -> bool:
    r, c = cell
    if not (0 <= r < len(grid) and 0 <= c < len(grid[0])):
        return False
    return grid[r][c] != "#"


def neighbours(grid: List[str], cell: Cell):
    for _, (dr, dc) in ACTIONS.items():
        nxt = (cell[0] + dr, cell[1] + dc)
        if passable(grid, nxt):
            yield nxt


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------
def h_manhattan(a: Cell, b: Cell) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def h_zero(a: Cell, b: Cell) -> float:
    return 0.0


def h_euclidean(a: Cell, b: Cell) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def h_manhattan_x2(a: Cell, b: Cell) -> float:
    return 2.0 * h_manhattan(a, b)


HEURISTICS = {
    "zero":        h_zero,
    "manhattan":   h_manhattan,
    "euclidean":   h_euclidean,
    "manhattan*2": h_manhattan_x2,
}


# ---------------------------------------------------------------------------
# A* search  --  frontier = min-heap keyed by f(n) = g(n) + h(n)
# ---------------------------------------------------------------------------
def a_star(grid: List[str], start: Cell, goal: Cell,
           heuristic: Heuristic = h_manhattan,
           heuristic_name: str = "manhattan") -> SearchResult:
    came_from = {start: None}
    g_score   = {start: 0}
    # heap entry: (f, counter, cell) -- counter breaks ties FIFO
    counter = 0
    frontier = [(heuristic(start, goal), counter, start)]
    expanded = 0

    while frontier:
        f, _, cell = heapq.heappop(frontier)
        expanded += 1
        if cell == goal:
            return _result("A*", heuristic_name, True,
                           _reconstruct(came_from, goal), expanded)
        for nxt in neighbours(grid, cell):
            tentative = g_score[cell] + 1
            if tentative < g_score.get(nxt, math.inf):
                came_from[nxt] = cell
                g_score[nxt] = tentative
                counter += 1
                heapq.heappush(frontier, (tentative + heuristic(nxt, goal),
                                          counter, nxt))

    return _result("A*", heuristic_name, False, [], expanded)


# ---------------------------------------------------------------------------
# BFS  --  frontier = FIFO queue, uniform-cost shortest path
# ---------------------------------------------------------------------------
def bfs(grid: List[str], start: Cell, goal: Cell) -> SearchResult:
    came_from = {start: None}
    frontier = deque([start])
    expanded = 0

    while frontier:
        cell = frontier.popleft()
        expanded += 1
        if cell == goal:
            return _result("BFS", "-", True,
                           _reconstruct(came_from, goal), expanded)
        for nxt in neighbours(grid, cell):
            if nxt not in came_from:
                came_from[nxt] = cell
                frontier.append(nxt)

    return _result("BFS", "-", False, [], expanded)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------
def _reconstruct(came_from, goal: Cell) -> List[Cell]:
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path


def _result(algo, h, found, path, expanded) -> SearchResult:
    return SearchResult(
        algorithm=algo, heuristic=h, solution_found=found,
        path=path, path_length=max(0, len(path) - 1),
        states_expanded=expanded,
    )


def render(grid: List[str], path: List[Cell]) -> str:
    out = [list(row) for row in grid]
    for r, c in path:
        if out[r][c] not in ("S", "G"):
            out[r][c] = "*"
    return "\n".join("".join(row) for row in out)


# ---------------------------------------------------------------------------
# Test maps for Task 3
# ---------------------------------------------------------------------------
TRIVIAL = ["#####",
           "#SG##",
           "#####"]

NO_SOLUTION = ["#######",
               "#S...##",
               "###.###",
               "#.####",            # this will become length-mismatched; see below
               "#.#.#G#",
               "#######"]

# Make every row the same length -- block the goal properly
NO_SOLUTION = ["########",
               "#S.....#",
               "########",          # wall of shelves separates start from goal
               "#.....G#",
               "########"]

# Two paths of different lengths -- A* / BFS should both pick the shorter.
ALT_PATHS = ["##########",
             "#S.......#",
             "#.######.#",
             "#.#......#",
             "#.#.####.#",
             "#.#.#....#",
             "#.#.#.##.#",
             "#.#.#.#G.#",
             "##########"]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_on(grid: List[str], label: str):
    print("\n" + "=" * 68)
    print(f"Map: {label}")
    print("=" * 68)
    for row in grid: print(row)
    s = find(grid, "S"); g = find(grid, "G")

    res_astar = a_star(grid, s, g)
    res_bfs   = bfs(grid, s, g)

    print()
    print(res_astar.summary())
    print(res_bfs.summary())

    if res_astar.solution_found:
        print("\nA* path overlaid on map:")
        print(render(grid, res_astar.path))


def heuristic_experiment():
    print("\n" + "=" * 68)
    print("Task 6: Heuristic experiment on the original warehouse")
    print("=" * 68)
    s = find(WAREHOUSE, "S"); g = find(WAREHOUSE, "G")
    for name, h in HEURISTICS.items():
        res = a_star(WAREHOUSE, s, g, h, name)
        print(res.summary())

    print("""
Notes
  zero         -- A* with h=0 is equivalent to BFS (uniform cost).
  manhattan    -- admissible AND consistent for 4-connected unit-cost grids;
                  A* returns a shortest path and expands fewer states.
  euclidean    -- admissible (underestimates the true cost in 4-connected
                  grids) but weaker than Manhattan, so expands more states
                  than Manhattan.
  manhattan*2  -- INADMISSIBLE (overestimates true cost): A* may expand
                  fewer states, but the returned path is NOT guaranteed
                  to be a shortest path.
""")


if __name__ == "__main__":
    run_on(WAREHOUSE,    "original warehouse")
    run_on(TRIVIAL,      "trivial (goal next to start)")
    run_on(NO_SOLUTION,  "no solution (walled off)")
    run_on(ALT_PATHS,    "alternative paths")
    heuristic_experiment()
