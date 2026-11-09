# Lab 2 — Agents: Constructing a Goal-Based Agent using an LLM

**Author:** Surabhi Agarwal
**Code:** [lab2_warehouse.py](lab2_warehouse.py)

---

## Task 1 — Understanding the Problem

1. **Environment.** A 2-D rectangular grid representing the warehouse floor. Each cell is one of:
   - `S` the vehicle's starting position,
   - `G` the dispatch area (goal),
   - `#` a shelving unit (obstacle, impassable),
   - `.` free floor space.
   The environment is **static** (shelves don't move), **fully observable** (the whole map is given), **deterministic** (each move changes the vehicle's position by exactly one cell in the chosen direction), **discrete** in time and space, and **single-agent**.

2. **Goal.** Transport the vehicle from `S` to `G` along a *collision-free* path — i.e. a sequence of moves that never lands on a `#` cell.

3. **Available actions.** `Up`, `Down`, `Left`, `Right`. Each moves the vehicle one grid cell in that direction.

4. **Information the agent must maintain.** The map (walls and free cells), its current position, the goal position, the set of already-visited cells (to avoid cycles during planning), and the planned action sequence that will take it to the goal.

5. **Why goal-based, not simple reflex.** A reflex agent would react only to the immediate percept (e.g. "there is a wall to my right, turn left"), with no notion of *where it is going*. Here the agent must choose its next step based on which step will *bring it closer to a particular goal cell `G`* — this requires an explicit goal and a planning / search procedure to work out a path to it.

**Think About It (size doubles).** BFS still works — it is complete and optimal for unweighted grids — but its memory grows with the number of reachable cells, which quadruples if the warehouse doubles in each dimension. For very large warehouses the appropriate upgrade is **A\*** with a Manhattan-distance heuristic: it keeps BFS's optimality (the heuristic is admissible) but expands far fewer cells.

---

## Task 2 — Agent Design

**Components**

| Component          | What it holds                                                                 |
|--------------------|-------------------------------------------------------------------------------|
| Environment        | The grid (`WAREHOUSE` 2-D list of characters).                                |
| Current state      | The vehicle's `(row, col)` position.                                          |
| Goal state         | The `(row, col)` of `G`.                                                      |
| Available actions  | `{Up, Down, Left, Right}`, each a unit vector `(dr, dc)`.                     |
| Decision component | **BFS planner** that searches for a shortest action sequence from state → goal. |
| Executor           | Walks the plan, cell by cell, emitting the action sequence.                   |

**Block diagram**

```
          ┌──────────────────┐
          │   Environment    │   (grid, obstacles)
          │  (percepts: S,G, │
          │   walls, cells)  │
          └────────┬─────────┘
                   │  percept
                   ▼
          ┌──────────────────┐
          │  Current state   │  ◄── updated after each action
          │  (position)      │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐         ┌─────────────┐
          │  Deliberation /  │◄────────│    Goal     │
          │  Planner (BFS)   │         │ (G position)│
          └────────┬─────────┘         └─────────────┘
                   │  action sequence
                   ▼
          ┌──────────────────┐
          │    Executor /    │
          │     Actuator     │ ──► Up / Down / Left / Right
          └──────────────────┘
```

**Decision-making component — why BFS?**

- Every action has the *same* unit cost, so "fewest moves" = "shortest path".
- BFS over the grid graph is **complete** (finds a path if one exists), **optimal** (returns the fewest-move path), and runs in `O(|cells|)` time and memory — trivially cheap on a 20 × 7 map.
- It needs no heuristic, no priority queue, no weights — minimal machinery for a minimal problem.

---

## Task 3 — Prompt Engineering

**Prompt I gave the LLM**

> Write a well-documented Python program implementing a goal-based agent
> for a warehouse navigation problem. The warehouse is a 2-D grid of
> characters: `S` = start, `G` = goal, `#` = obstacle, `.` = free.
> The vehicle may move Up / Down / Left / Right, one cell per move.
> The program should: represent the warehouse as a list-of-strings grid,
> locate `S` and `G`, find a collision-free path with **breadth-first
> search** (optimal for unit-cost moves), print the action sequence and
> the path overlaid on the grid, and print a clear message if no path
> exists. Include a second test case in which the goal is walled off and
> verify that the no-path branch is reached. Do not import pathfinding
> libraries; use only the standard library (`collections.deque` is fine).

**Reflection questions**

1. **Did the LLM generate a working program on the first attempt?**
   Mostly yes — the BFS and reconstruction were correct. However, the first "no-path" test case I wrote actually *had* a path (a single column of `#` did not fully enclose the goal), so the test reported a false positive. I corrected the test map rather than the search code: the algorithm was right; the test input was wrong.

2. **How could the prompt be improved?**
   - State the grid dimensions explicitly so the LLM does not invent a different size.
   - Ask for a self-check after the main run (shortest-path sanity test on a straight corridor), which catches off-by-one bugs.
   - Specify the exact output format for the action sequence (`"Right Right Down …"`) so the grader can diff easily.

3. **What search algorithm did the LLM choose?**
   BFS over grid cells, using a FIFO queue (`collections.deque`), with a `came_from` dictionary for path reconstruction.

4. **Why that algorithm?**
   Because every move has equal cost (unit step) and the grid is small and unweighted: BFS is the simplest algorithm that is both **complete** and **optimal** in this setting. A\* would also work but would need a heuristic; Dijkstra reduces to BFS when all edge weights are equal; depth-first search would find *a* path but not the shortest and could wander arbitrarily.

---

## Results

With the warehouse map in the lab PDF, the agent finds a 19-move shortest path:

```
####################
#S***.#***********G#
#.##****#########..#
#....##............#
#.######.###.#.###.#
#........#.........#
####################
```

Action sequence:
```
Right Right Right Down Right Right Right Up Right Right Right Right Right Right Right Right Right Right Right
```

The "goal walled off" test correctly reports *no path exists*, and a straight-corridor sanity test confirms BFS returns exactly 4 moves for a 4-cell gap.

---

## Reflection on LLM-Assisted Software Engineering

**Strengths observed.**
- The LLM produced the BFS skeleton, the direction table, and a reasonable file structure in one shot. Typing-time saved: minutes.
- When asked, it explained *why* BFS fits this problem (unit-cost, unweighted, small graph) — useful as documentation.

**Limitations / what needed human judgement.**
- I had to specify the algorithm (`breadth-first search`) explicitly — left open, the LLM sometimes reaches for A\* or recursive DFS that doesn't track visited cells.
- The LLM's first "no-path" test map was not actually blocking. Only running the code and reading the output revealed this. **The LLM cannot verify that its own test is a valid test** — only execution does.
- Any claim the LLM makes about performance ("O(V+E)") or correctness ("this will always find the shortest path") is **data to verify**, not evidence. I checked both empirically on the shortest-corridor sanity test.

**Take-away.** The LLM is a productive drafting partner for boilerplate (data structures, I/O, docstrings) and for routine algorithms it has clearly seen in training. The *specification*, the *choice of algorithm for the task*, and the *validation that the implementation does what the spec says* remain the engineer's responsibility.
