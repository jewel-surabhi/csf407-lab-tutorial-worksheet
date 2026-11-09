# Lab 4 — Logical Reasoning for Planning

**Author:** Surabhi Agarwal
**Code:** [lab4_planner.py](lab4_planner.py)

**Central lesson:** `Logic + Search = Planning`.

---

## Task 0 — Understand the Planning Problem

| Component | Specification |
|---|---|
| **Initial state `I`** | `{ At(Robot, A), At(Package, A) }` |
| **Goal `G`** | `{ At(Package, C) }` |
| **Locations** | `A, B, C` (fully connected: A↔B, B↔C, and in the extended test also A↔C) |
| **Actions `A`** | `Move(x, y)` for each ordered pair of directly connected locations; `PickUp(Package, x)`; `Drop(Package, x)` |

**Action schemas.**

| Action | Positive preconditions | Positive effects | Negative effects |
|---|---|---|---|
| `Move(x, y)` | `At(Robot, x)` | `At(Robot, y)` | `At(Robot, x)` |
| `PickUp(Package, x)` | `At(Robot, x), At(Package, x)` | `Holding(Package)` | `At(Package, x)` |
| `Drop(Package, x)` | `At(Robot, x), Holding(Package)` | `At(Package, x)` | `Holding(Package)` |

**Which actions are applicable in `I`?** Using `S ⊨ Preconditions(a)` (all positive preconditions are in `S`, all negative preconditions are absent from `S`):

- `Move(A, B)` — ✅ `At(Robot, A) ∈ I`.
- `Move(A, C)` — ✅ `At(Robot, A) ∈ I` (only when the schema includes an A↔C edge).
- `PickUp(Package, A)` — ✅ both `At(Robot, A)` and `At(Package, A)` are in `I`.
- `Drop(Package, C)` — ❌ needs `At(Robot, C)` and `Holding(Package)`; **neither** holds in `I`.

My program verifies this at runtime: `PickUp(Package,A) applicable? True`, `Drop(Package,C) applicable? False`.

**Think About It.** An action's presence in the catalogue is not evidence that it is applicable *now*. The planner must check `S ⊨ Preconditions(a)` against the *current* state before firing it. That check is where **logical reasoning** enters the planning loop.

---

## Task 1 — Construct a Plan by Hand

A valid 4-step plan for `I = {At(Robot,A), At(Package,A)}` → `G = {At(Package,C)}`:

| step | action | state after |
|---|---|---|
| `S₀` | — | `{At(Robot,A), At(Package,A)}` |
| `S₁` | `PickUp(Package, A)` | `{At(Robot,A), Holding(Package)}` |
| `S₂` | `Move(A, B)` | `{At(Robot,B), Holding(Package)}` |
| `S₃` | `Move(B, C)` | `{At(Robot,C), Holding(Package)}` |
| `S₄` | `Drop(Package, C)` | `{At(Robot,C), At(Package,C)}` |

`G = {At(Package,C)} ⊆ S₄` ✓

**Precondition check at every step:**
- Step 1: `PickUp(Package,A)` needs `At(Robot,A), At(Package,A)` — both in `S₀`. ✓
- Step 2: `Move(A,B)` needs `At(Robot,A)` — in `S₁`. ✓
- Step 3: `Move(B,C)` needs `At(Robot,B)` — in `S₂`. ✓
- Step 4: `Drop(Package,C)` needs `At(Robot,C), Holding(Package)` — both in `S₃`. ✓

---

## Task 2 — LLM-Generated Planner

**Prompt used** (close to the one on p. 7 of the PDF):

> I want to implement a simple STRIPS planner in Python. Represent a state
> as a `frozenset` of ground atoms (tuples). Represent an action as a
> dataclass with `name`, `pos_pre`, `neg_pre`, `pos_eff`, `neg_eff`, all
> `frozenset`s. An action is applicable in `S` iff `pos_pre ⊆ S` and
> `neg_pre` is disjoint from `S`. Applying it returns
> `(S − neg_eff) ∪ pos_eff`. Use breadth-first search over states to find
> the shortest plan achieving a goal set. Detect "no plan". Print the
> action sequence and the state after each action. Use only the standard
> library.

**Where the specification lives in the code** (Think-About-It mapping):

| Idea in spec           | Where in `lab4_planner.py` |
|------------------------|-----------------------------|
| **Preconditions** → when applicable | `Action.applicable()` — `pos_pre <= state and neg_pre.isdisjoint(state)` |
| **Effects** → how the state changes  | `Action.apply()` — `(state - neg_eff) | pos_eff` |
| **Goal** → when planning terminates   | `if goal <= succ:` inside `plan_bfs` (and the `if goal <= initial:` early-exit at the top) |
| **BFS** → how alternatives are explored | `deque([initial])` + `frontier.popleft()` + `came_from` dict for parent pointers |

---

## Task 3 — Test the Generated Planner

| Test | Expectation | Result | Valid? |
|---|---|---|---|
| **A — Solvable** (original warehouse) | A 4-step plan delivering `At(Package,C)` | Returned `PickUp(A) → Move(A,B) → Move(B,C) → Drop(C)`, 6 states expanded | ✅ precondition-checked every step |
| **B — Impossible** (remove all `PickUp` actions) | "no plan found" | **NO PLAN FOUND**, 3 states expanded, no crash, no invented action | ✅ |
| **C — Irrelevant actions** (add `Move(A,C)`, `Move(C,A)` shortcuts) | Plan still delivers the *package*, not just the robot, to `C` | Returned `PickUp(A) → Move(A,C) → Drop(C)` — now 3 steps because the shortcut is legitimately shorter. Final state contains `At(Package,C)`, confirming the planner did not confuse "robot at C" with "package at C". | ✅ |

For every test, the printed state trace lets me re-verify by hand that each action's preconditions are present in the previous state and its effects produce the next state.

---

## Task 4 — Logic and Search

The planner weaves two ideas together:

```
Current state
     ↓
Check action preconditions   ← logical reasoning:  S ⊨ Preconditions(a)
     ↓
Apply action's effects       ← logical reasoning:  S' = (S − neg_eff) ∪ pos_eff
     ↓
Generate successor state
     ↓
Search over alternatives     ← BFS: which applicable sequence to try next
     ↓
Goal reached?                ← logical reasoning:  G ⊆ S'
```

**In words.** Logic answers two local questions — *is this action allowed in this state?* and *what state does it produce?* Search answers one global question — *which sequence of locally-legal actions, out of exponentially many, reaches the goal?* Neither works without the other: logic alone tells you what you *could* do next, not what you *should*; search alone would generate nonsense if it ignored preconditions. BFS specifically guarantees the shortest plan (fewest actions) because all actions have unit cost in this problem.

**Think About It.** The slogan "logic determines what is possible; search determines what to try" is literal here: without preconditions, BFS would wander into illegal transitions; without BFS, logic would produce a cloud of valid actions with no notion of ordering toward the goal.

---

## Task 5 — Optional: Can the LLM Verify Its Own Plan?

I asked the LLM to explain why its plan is valid by walking the preconditions against the state after each step. The explanation it produced matched the state trace that my Python program computed — in this small warehouse both are easy to verify independently.

**Who should you trust more — the LLM's explanation or the executed state transitions?** The **executed transitions.** The Python planner applies the same four set operations at every step: subset check for preconditions, difference/union for effects, subset check for the goal. There is no room for a persuasive-sounding but incorrect explanation to slip through; either the set operations succeed or they fail. An LLM's narrative explanation, however coherent, is generated text with no mechanical guarantee of fidelity to the semantics. On this toy example they agreed, but **a generated explanation is not an independent verification**: the former re-uses the same reasoning the model used to produce the plan, while the latter is a separate program with no shared failure mode.

---

## Reflection Questions

1. **Why specify preconditions and effects before asking an LLM to write the planner?** Because those are the mathematical content of the planning problem. The LLM can *implement* set-subset and set-difference correctly; it cannot *invent* which atoms you consider part of a state or which atoms an action changes. Writing the spec first means the LLM's job is a translation from a formal definition to Python, which is easy to audit; left to invent them, the LLM may plausibly but wrongly conflate `At(Robot, x)` with `At(Package, x)` and produce a planner that looks right but is unsound.

2. **Example error from skipping preconditions.** A planner that applies `Drop(Package, C)` from the initial state would set `At(Package, C)` and delete `Holding(Package)` — producing a "plan" with a single action and a final state that *satisfies the goal*, yet the robot never moved and the package was never picked up. The returned trace is coherent, the goal literal appears, and only a human looking at the physics realises it is nonsense. Checking `S ⊨ Preconditions(a)` makes the illegal step impossible.

3. **Why a "reasonable looking" plan may still be invalid.** Validity is not a property of the sequence of action *names* — it is a property of the trajectory of *states*. Two things can go wrong: (i) an action in the middle of the plan has an unmet precondition in the state the planner thinks it has; (ii) the final state does not actually entail the goal. Both are invisible unless you re-execute the plan against the state transitions and verify precondition- and goal-subset tests step by step.

4. **What did the LLM contribute?** Boilerplate scaffolding: the `Action` dataclass layout, the `frozenset`-based state, the `heapq`-vs-`deque` choice (`deque` here because BFS, not best-first), the `came_from` + reversal reconstruction pattern. These are idiomatic Python patterns the model has seen many times; writing them by hand is tedious and error-prone (off-by-one in the reversal, forgetting to wrap sets as `frozenset` so they are hashable, forgetting that `set` difference does not mutate).

5. **What I verified independently.** Every decision that is specific to *this* planning problem: the exact preconditions of `PickUp` and `Drop`, that `Move` has `At(Robot,x)` as a negative effect (otherwise the robot is in both places at once), that the Test B map genuinely removes *all* PickUp actions (not just one), and that the Test C plan actually ends with `At(Package,C)` rather than `At(Robot,C)`.

6. **Where is logical reasoning used here?** In three places — all of them set membership / subset tests against the current state:
   - `Action.applicable`: `pos_pre ⊆ S` and `neg_pre ∩ S = ∅` — i.e. `S ⊨ Preconditions(a)`.
   - `Action.apply`: `S' = (S − neg_eff) ∪ pos_eff` — the state update rule.
   - Goal test: `G ⊆ S'` — i.e. `S' ⊨ G`.

7. **Relation to Lab 3 search.** Planning *is* search — the state space is the set of all reachable states, nodes are states, edges are applicable actions, and the goal test is `G ⊆ S`. BFS here is exactly the BFS of Lab 2 / Lab 3, but instead of a grid of `(row, col)` cells the graph is implicit: successors are generated by iterating over the action set, filtering by `applicable`, and applying their effects. Lab 3 had a geometric heuristic (Manhattan); planning typically does not, which is why basic planners use blind search (BFS / IDS) or build weaker heuristics from the problem structure (e.g. count of unmet goal atoms).

---

## Optional Section 7 — Prolog as a Logical Verifier (short note)

I did not re-implement the planner in Prolog, but the mapping is tight and worth recording:

| Our Python                                   | Prolog analogue                               |
|----------------------------------------------|-----------------------------------------------|
| `("At", "Robot", "A")` in state              | `at(robot, a).` fact                          |
| `Action.applicable(state)` set-subset check  | Rule body `at(robot, X), at(package, X).`     |
| `Action.apply` producing a new state         | `retract/assert` on the dynamic database      |
| BFS plan loop                                | Backtracking via SLD-resolution               |
| Goal subset test                             | Query `?- at(package, c).`                    |

Prolog's value as a *verifier* is that it does not share any code with the Python planner: if you load the plan's state trace as `at/2` facts, Prolog can independently prove that each action's precondition and the final goal hold. That is an **independent check**, which is exactly what the Task 5 Think-About-It demands — and exactly what a second explanation from the same LLM is not.

---

## Division of Responsibility

| Who            | What                                                                 |
|----------------|----------------------------------------------------------------------|
| I designed     | The planning-problem tuple (`I`, `G`, action schemas), the test matrix (solvable / impossible / irrelevant actions), the state-trace format I wanted printed. |
| LLM suggested  | The `Action` dataclass, `frozenset` state representation, `deque`-based BFS skeleton, `came_from` reconstruction. |
| I accepted     | The data structures and the BFS shape.                               |
| I changed      | Made the trace include every intermediate state (not just the plan), named actions by their ground signature for readability, added the Task 0 precondition sanity check. |
| I tested       | Three scenarios; verified each action's preconditions hold in the preceding state; confirmed the goal literal is in the final state in every "found" case. |
