# Lab 3 — Search and A\*

**Author:** Surabhi Agarwal
**Code:** [lab3_search.py](lab3_search.py)

---

## Task 0 — Understanding the Search Problem

The warehouse-robot problem fits the tuple `P = (S, A, T, s₀, G, c)`:

| Component       | My specification                                                                 |
|-----------------|----------------------------------------------------------------------------------|
| **State `S`**   | A cell `(row, col)` that the robot currently occupies, where the cell is not `#`. |
| **Actions `A`** | `{Up, Down, Left, Right}`, each a unit vector.                                   |
| **Transition `T`** | `T(s, a) = s + Δ(a)` if the resulting cell is inside the grid and not `#`; otherwise `a` is disallowed in `s`. |
| **Initial state `s₀`** | The position of `S` in the map.                                          |
| **Goal `G`**    | The singleton `{position of G}`.                                                 |
| **Cost `c`**    | `c(s, a, s') = 1` for every legal move (uniform cost).                           |

**(a) What specifies a state?**
Just the robot's `(row, col)`. The grid, obstacles, and goal position are fixed, not part of the state.

**(b) What makes an action invalid?**
Either it would move the robot off the grid, or it would move into a `#` cell.

**(c) Is it deterministic?**
Yes. Each `(state, action)` pair has exactly one successor — no randomness, no slip probability.

**(d) What is a solution?**
A finite sequence of actions whose successive application starting from `s₀` ends in `G`, visiting only legal states. In this lab we also want the *shortest* such sequence (fewest moves).

---

## Task 1 — Plan the Agent

| Design decision                   | Choice                                                                             |
|-----------------------------------|------------------------------------------------------------------------------------|
| State representation              | tuple `(row, col)` — hashable, cheap to compare.                                   |
| Warehouse representation          | `list[str]` — one string per row; `grid[r][c]` gives the cell.                     |
| Valid actions                     | Compute `s' = s + Δ(a)` for each action, keep those with `grid[r'][c'] != '#'`.    |
| Goal recognition                  | `cell == goal`.                                                                    |
| Frontier contents                 | For A\*: priority queue of `(f, counter, cell)` with parallel `g_score` and `came_from` dicts. For BFS: a FIFO queue of cells with a `came_from` dict. |
| Path reconstruction               | Walk `came_from` from `goal` back to `s₀` and reverse.                             |

**Program reports.** `SearchResult` with `solution_found`, `path`, `path_length`, `states_expanded`, `algorithm`, `heuristic`.

---

## Task 2 — Prompt for A\*

> I am implementing a goal-based search agent in Python. The environment is a
> grid given as a list of strings. `S` is the start, `G` the goal, `#` an
> obstacle, `.` a free cell. Actions are Up/Down/Left/Right with unit cost.
> Implement **A\* search** with the **Manhattan-distance** heuristic
> `h(n) = |x − x_G| + |y − y_G|`. Represent grid positions as `(row, col)`
> tuples. Use a `heapq` min-heap keyed by `f(n) = g(n) + h(n)` with a FIFO
> tiebreaker counter. Keep `g_score` and `came_from` dicts. Avoid
> re-expanding states whose `g` cannot be improved. After the search, report
> whether a solution was found, the path, its length, and the number of
> states expanded. Do not use any search library.

**What I accepted from the LLM.** The heap entry format, `came_from`
reconstruction, the `g_score`-based re-opening check.

**What I changed / added.** (i) Factored the three heuristics (zero,
Manhattan, Euclidean, Manhattan×2) into a dict so experiments are cheap;
(ii) wrapped all results in a `SearchResult` dataclass so BFS and A\* are
directly comparable; (iii) added a path-rendering helper so I can eyeball
whether the returned path is sensible.

---

## Task 3 — Test the Program (results)

All four test maps are included in the script.

| Test                              | Result                                                               |
|-----------------------------------|----------------------------------------------------------------------|
| **1. Original warehouse**         | A\* FOUND, length **43**, expanded **69**.                           |
| **2. Trivial (`#SG##`)**          | A\* FOUND, length **1**, expanded 2.                                 |
| **3. No solution (walled off)**   | A\* **NO SOLUTION**, expanded 6, terminated (no infinite loop).      |
| **4. Alternative paths**          | A\* FOUND, length **14**, expanded 23. The returned path matches the optimal detour length (BFS also returns 14), so A\* returns a shortest path. |

A\* path on the original warehouse:

```
##################
#S****#**********#
#.###*#*########*#
#...#*#********#*#
###.#*########*#*#
#...#**********#*#
#.############.#*#
#..............#G#
##################
```

---

## Task 4 — Inspect the A\* Algorithm

Mapping the AI concepts to concrete lines in [lab3_search.py](lab3_search.py):

| Concept           | Where in the code                                                             |
|-------------------|-------------------------------------------------------------------------------|
| **State**         | A `Cell = Tuple[int, int]`; variables `start`, `goal`, `cell`, `nxt`.         |
| **Action**        | `ACTIONS` dict mapping each name to a `(dr, dc)` offset; applied in `neighbours`. |
| **Transition**    | `nxt = (cell[0] + dr, cell[1] + dc)` guarded by `passable(grid, nxt)`.        |
| **Goal test**     | `if cell == goal:` in both `a_star` and `bfs`.                                |
| **`g(n)`**        | `g_score[cell]` — best known cost from `start` to `cell`.                      |
| **`h(n)`**        | `heuristic(cell, goal)` — supplied as a parameter.                            |
| **`f(n)`**        | `tentative + heuristic(nxt, goal)` pushed onto the heap.                      |
| **Frontier**      | `heapq` min-heap of `(f, counter, cell)` triples (counter is a FIFO tiebreaker). |
| **Visited states**| Implicit in `g_score`: a state is "closed" when we find no shorter `g` for it. |
| **Path reconstruction** | `_reconstruct(came_from, goal)` walks the parent pointers and reverses. |

**(a) Data structure for the frontier.** A binary min-heap (`heapq`).
Popping the lowest-`f` node is `O(log n)`.

**(b) How is the next state chosen?** `heapq.heappop(frontier)` returns the
cell with the smallest `f(n) = g(n) + h(n)` currently on the frontier.

**(c) Where is `h` calculated?** On every push, in the term
`heuristic(nxt, goal)`.

**(d) Is `f = g + h` explicit?** Yes — see the `heapq.heappush` call
inside `a_star`: the first element of the heap tuple is literally
`tentative + heuristic(nxt, goal)`.

**(e) How are repeated expansions avoided?** Before pushing a successor we
check `tentative < g_score.get(nxt, inf)`. A cell is only re-added to the
heap when we have strictly improved its `g`. Because the Manhattan
heuristic is consistent, a cell is actually popped only once with its final
`g` value.

---

## Task 5 — Compare A\* with Blind Search (BFS)

| Measure            | BFS      | A\* (Manhattan)  |
|--------------------|----------|------------------|
| Solution found     | yes      | yes              |
| Path length        | 43       | 43               |
| States expanded (original warehouse) | 68 | 69 |
| States expanded (alternative-paths map) | 27 | **23** |

**(a) Did both find a solution?** Yes.
**(b) Same length?** Yes — both are optimal for unit-cost grids (BFS by construction; A\* because Manhattan is admissible).
**(c) Which expanded fewer?** On the alternative-paths map, A\* expanded 23 vs BFS 27 — the expected advantage. On the original warehouse the counts were almost identical (69 vs 68). This is itself the lesson of Think-About-It: A\* is not automatically faster; it is only faster when the heuristic can actually steer search *around* dead ends.
**(d) Why would A\* generally expand fewer?** BFS explores all cells at distance `k` before moving to distance `k+1`, regardless of direction. A\* orders by `g + h`, so a cell that is close *and* already looks promising towards the goal is popped before a cell that is close but clearly off-direction. Nodes far from the goal line are deferred, often never reached.

**Why A\* was not a win on the original warehouse.** The map has long
narrow corridors with few branching points. In that topology the Manhattan
heuristic is almost immediately defeated by walls: the heuristic value drops
monotonically along the only passable route, so A\* and BFS wind up popping
nearly the same set of cells. Alternative-paths topology (open space, choice
of detours) is where Manhattan actually prunes.

---

## Task 6 — Heuristic Investigation

A\* on the original warehouse with four heuristics:

| Heuristic          | Path length | States expanded | Admissible? | Guarantees shortest path? |
|--------------------|-------------|-----------------|-------------|---------------------------|
| `h(n) = 0`         | 43          | 68              | yes         | yes (this is uniform-cost search ≡ BFS) |
| Manhattan          | 43          | 69              | yes         | yes                       |
| Euclidean          | 43          | 69              | yes (weaker than Manhattan for 4-connected grids) | yes |
| Manhattan × 2      | 43          | 73              | **no** (overestimates) | **no guarantee** — it happened to find 43 here, but can return a longer path on other maps |

**(1) `h = 0`.** A\* degenerates into uniform-cost search, which on
unit-cost edges is BFS. Still optimal, still complete; no guidance.

**(2) Euclidean.** `√((Δx)² + (Δy)²) ≤ |Δx| + |Δy|`, so Euclidean
underestimates Manhattan and is admissible. It is a *weaker* bound, so A\*
expects less work to still look worthwhile and ends up expanding at least
as many cells as Manhattan. Here it ties Manhattan.

**(3) `2 × Manhattan`.** This overestimates the true distance — it is
**inadmissible**. A\* with an inadmissible heuristic becomes a greedier
search: it is willing to abandon partially-explored paths if some new node
"looks closer", which can expand fewer or more nodes but loses the
optimality guarantee. In this experiment it expanded more (73) and still
happened to return a length-43 path, but that is a property of the map, not
of the algorithm.

**Think About It — admissibility.**
If `h(n) ≤ h\*(n)` (true remaining cost), A\* is **optimal**: it will never
prematurely commit to a suboptimal path, because the lower bound rules out
better alternatives honestly. If the heuristic becomes too optimistic in
the wrong sense — i.e. overestimates — the lower-bound property breaks and
A\* can pop and commit to a suboptimal goal, returning a longer path.

---

## Task 7 — Evaluate the LLM-Generated Agent

1. **Immediately correct.** The `heapq` wiring, the `came_from`
   reconstruction, the Manhattan formula.
2. **Bugs / design problems.** Early drafts of the "no-solution" test map
   accidentally left a corridor open (it was the same bug as in Lab 2);
   also the LLM initially omitted the FIFO tiebreaker in heap entries,
   which made `heapq` compare tuples by cell and occasionally raise
   `TypeError` on ties.
3. **How discovered.** By *running* the tests. The no-path test reported
   a path; the tiebreaker bug surfaced as an intermittent exception on the
   alternative-paths map.
4. **Unfamiliar structures?** No — `heapq`, `deque`, dataclasses, and dict
   lookups are all standard-library.
5. **Modifications.** Added the tiebreaker counter, extracted a
   `SearchResult` dataclass, consolidated heuristics into a dict, wrote the
   `render` helper to overlay paths on the grid for visual inspection.
6. **Most useful tests.** The no-solution and alternative-paths tests —
   both expose correctness failures that the single warehouse map hides.
7. **Could you have trusted the program without testing it?** No. The code
   *looked* fine the first time. Only the no-solution test and the
   alternative-paths comparison distinguished a correct A\* from one that
   happens to work on one input.
8. **What did I understand about A\* after implementing it?** That the
   *representation* of the frontier is what defines the algorithm:
   swap the min-heap for a FIFO queue and A\* becomes BFS; swap `g + h` for
   just `h` and it becomes greedy best-first. A\* is a particular *policy*
   for ordering an otherwise generic graph search.

**Division of responsibility.**

| Who did it                     | What                                                          |
|--------------------------------|---------------------------------------------------------------|
| I designed                     | The search-problem tuple, the test matrix, the heuristic experiment menu. |
| LLM suggested                  | A\* skeleton with `heapq`, `came_from` reconstruction, Manhattan formula. |
| I accepted                     | The core algorithm shape and the standard-library choices.    |
| I changed                      | Added FIFO tiebreaker, dataclass result, heuristics dict, path-rendering helper, fixed the no-solution test map. |
| I tested                       | All four maps + all four heuristics, numerically and visually. |

---

## Final Reflection

1. **Why formulate the search problem first?** Because the algorithm is a
   generic engine over any `(S, A, T, s₀, G, c)`. Without pinning those
   down, you can't tell whether a bug is in the formulation (wrong goal,
   wrong neighbour function) or in the algorithm (wrong frontier policy).
   Formulating first also makes the test suite obvious: you test each
   component (goal test, transition, cost) separately.

2. **In what sense is A\* "informed"?** It uses the heuristic `h(n)` —
   knowledge *external* to the graph structure — to estimate remaining
   cost. BFS/DFS only use the graph itself. The heuristic is where
   problem-specific knowledge enters.

3. **Why does the heuristic choice matter?** Because `h` controls both
   optimality (via admissibility) and efficiency (via how tightly it
   bounds `h\*`). An admissible, tight heuristic prunes aggressively and
   remains optimal. A loose but admissible heuristic (`h = 0`,
   Euclidean on a 4-connected grid) stays optimal but expands more
   states. A tight but inadmissible heuristic (Manhattan × 2) may
   prune further but returns suboptimal paths on general maps.

4. **What did the LLM contribute?** Speed of first draft — minutes rather
   than tens of minutes. Idiomatic Python (e.g. heap-based priority
   queue, dataclass for results). A sensible default structure for
   `came_from` / `g_score`. It did not contribute the problem
   formulation, the test matrix, the heuristic analysis, or the
   diagnosis of the no-solution bug — all human work.

5. **What could go wrong if LLM code were accepted without testing?**
   - The no-path test can silently have a path (tested data is as much a
     hypothesis as code).
   - A heap without a tiebreaker throws `TypeError` only sometimes,
     passing most runs.
   - A function that *looks* like A\* but uses `h` alone is actually
     greedy search — syntactically indistinguishable, semantically
     suboptimal.
   - An inadmissible heuristic can return "a path" that scores well on
     a single map but silently violates optimality elsewhere.
   The pattern: *working output ≠ validated algorithm*. The engineer's job
   is to design the validation, not to delegate correctness to the
   generator.
