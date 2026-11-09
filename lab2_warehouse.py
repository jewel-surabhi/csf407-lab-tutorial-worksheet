"""
Lab 2 - Agents: Constructing a Goal-Based Agent using an LLM.

Problem: a warehouse vehicle must find a collision-free path from S to G
on a 2-D grid using Up / Down / Left / Right moves.  Shelving units (#)
cannot be crossed.

Search algorithm: Breadth-First Search.
Reason: every move has unit cost, so BFS returns a provably shortest
path, is complete (always finds a path if one exists), and runs in
O(|cells|) time -- ideal for a small, unweighted grid world.
"""

from collections import deque

# ---------------------------------------------------------------------------
# Warehouse map  (S = start, G = goal, # = shelf, . = free)
# ---------------------------------------------------------------------------
WAREHOUSE = [
    "####################",
    "#S....#...........G#",
    "#.##....#########..#",
    "#....##............#",
    "#.######.###.#.###.#",
    "#........#.........#",
    "####################",
]


# ---------------------------------------------------------------------------
# Goal-based agent
# ---------------------------------------------------------------------------
ACTIONS = {
    "Up":    (-1, 0),
    "Down":  ( 1, 0),
    "Left":  ( 0,-1),
    "Right": ( 0, 1),
}


class WarehouseAgent:
    """Goal-based agent: percepts the grid, holds the goal, chooses actions
    by planning a shortest path with BFS, then executes that plan."""

    def __init__(self, grid):
        self.grid  = [list(row) for row in grid]
        self.rows  = len(self.grid)
        self.cols  = len(self.grid[0])
        self.start = self._find("S")
        self.goal  = self._find("G")
        self.plan  = None                       # list of (action, cell) pairs

    # ---- percept ----------------------------------------------------------
    def _find(self, ch):
        for r, row in enumerate(self.grid):
            for c, v in enumerate(row):
                if v == ch:
                    return (r, c)
        raise ValueError(f"{ch!r} not in grid")

    def _passable(self, cell):
        r, c = cell
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        return self.grid[r][c] != "#"

    # ---- deliberation: BFS -----------------------------------------------
    def plan_path(self):
        """BFS over grid cells.  Returns a list of (action, cell) pairs
        describing the shortest path from start to goal, or None."""
        frontier = deque([self.start])
        came_from = {self.start: (None, None)}  # cell -> (prev cell, action used)

        while frontier:
            cell = frontier.popleft()
            if cell == self.goal:
                return self._reconstruct(came_from)
            for action, (dr, dc) in ACTIONS.items():
                nxt = (cell[0] + dr, cell[1] + dc)
                if self._passable(nxt) and nxt not in came_from:
                    came_from[nxt] = (cell, action)
                    frontier.append(nxt)
        return None

    def _reconstruct(self, came_from):
        path = []
        cell = self.goal
        while came_from[cell][0] is not None:
            prev, action = came_from[cell]
            path.append((action, cell))
            cell = prev
        path.reverse()
        return path

    # ---- execution (acts out the plan one step at a time) ----------------
    def run(self):
        self.plan = self.plan_path()
        if self.plan is None:
            return None
        trail = [self.start] + [cell for _, cell in self.plan]
        return trail


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def render_path(grid, trail):
    """Overlay the path onto the grid with '*' (keeping S and G visible)."""
    out = [list(row) for row in grid]
    for r, c in trail:
        if out[r][c] not in ("S", "G"):
            out[r][c] = "*"
    return "\n".join("".join(row) for row in out)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_known_map():
    print("=" * 60)
    print("Warehouse map")
    print("=" * 60)
    print("\n".join(WAREHOUSE))

    agent = WarehouseAgent(WAREHOUSE)
    trail = agent.run()
    assert trail is not None, "BFS failed to find a path on the given map"

    actions = [a for a, _ in agent.plan]
    print(f"\nStart: {agent.start}   Goal: {agent.goal}")
    print(f"Path length (moves): {len(actions)}")
    print(f"Action sequence: {' '.join(actions)}")

    print("\nPath overlaid on the grid ('*' = step, S = start, G = goal):")
    print(render_path(WAREHOUSE, trail))


def test_no_path():
    """Walled-off goal -- the agent must report that no path exists."""
    blocked = [
        "########",
        "#S..#.G#",
        "#...#..#",
        "#...#..#",
        "########",
    ]
    agent = WarehouseAgent(blocked)
    trail = agent.run()
    print("\n" + "=" * 60)
    print("No-path test (goal walled off by a column of #)")
    print("=" * 60)
    print("\n".join(blocked))
    if trail is None:
        print("Agent correctly reports: no path exists.")
    else:
        print("Unexpectedly found a path:", trail)


def test_shortest():
    """BFS must return a path no longer than Manhattan distance + detours."""
    simple = [
        "#######",
        "#S...G#",
        "#######",
    ]
    agent = WarehouseAgent(simple)
    agent.run()
    assert len(agent.plan) == 4, f"expected 4 moves, got {len(agent.plan)}"
    print("\nShortest-path sanity check on a straight corridor: 4 moves  OK")


if __name__ == "__main__":
    test_known_map()
    test_no_path()
    test_shortest()
