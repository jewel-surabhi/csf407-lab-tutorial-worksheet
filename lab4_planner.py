"""
Lab 4 - Logical Reasoning for Planning.

A STRIPS-style planning agent for a small warehouse:

  Locations: A, B, C
  Robot starts at A; Package starts at A; goal = At(Package, C).
  Actions: Move(x, y), PickUp(Package, x), Drop(Package, x).

The state is a frozenset of ground atoms.  An action is applicable in S iff
its positive preconditions are a subset of S and its negative preconditions
are disjoint from S.  The successor is  (S  negative_effects) +
positive_effects.  We search over states with BFS and return a shortest
plan.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Tuple

Atom  = Tuple[str, ...]              # e.g. ("At", "Robot", "A")
State = FrozenSet[Atom]


# ---------------------------------------------------------------------------
# Action schema
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    name:           str
    pos_pre:        FrozenSet[Atom] = field(default_factory=frozenset)
    neg_pre:        FrozenSet[Atom] = field(default_factory=frozenset)
    pos_eff:        FrozenSet[Atom] = field(default_factory=frozenset)
    neg_eff:        FrozenSet[Atom] = field(default_factory=frozenset)

    def applicable(self, state: State) -> bool:
        return self.pos_pre <= state and self.neg_pre.isdisjoint(state)

    def apply(self, state: State) -> State:
        return (state - self.neg_eff) | self.pos_eff


# ---------------------------------------------------------------------------
# Action factory for the warehouse
# ---------------------------------------------------------------------------
def move(x: str, y: str) -> Action:
    return Action(
        name   = f"Move({x},{y})",
        pos_pre = frozenset({("At", "Robot", x)}),
        pos_eff = frozenset({("At", "Robot", y)}),
        neg_eff = frozenset({("At", "Robot", x)}),
    )

def pick_up(x: str) -> Action:
    return Action(
        name   = f"PickUp(Package,{x})",
        pos_pre = frozenset({("At", "Robot", x), ("At", "Package", x)}),
        pos_eff = frozenset({("Holding", "Package")}),
        neg_eff = frozenset({("At", "Package", x)}),
    )

def drop(x: str) -> Action:
    return Action(
        name   = f"Drop(Package,{x})",
        pos_pre = frozenset({("At", "Robot", x), ("Holding", "Package")}),
        pos_eff = frozenset({("At", "Package", x)}),
        neg_eff = frozenset({("Holding", "Package")}),
    )


# ---------------------------------------------------------------------------
# BFS planner
# ---------------------------------------------------------------------------
@dataclass
class PlanResult:
    found:     bool
    plan:      List[Action]
    trace:     List[State]            # state after each action (S_0 .. S_n)
    expanded:  int


def plan_bfs(initial: State, goal: FrozenSet[Atom],
             actions: List[Action]) -> PlanResult:
    """BFS over states.  Returns the shortest plan achieving every atom in
    `goal` as a subset of the final state."""
    if goal <= initial:
        return PlanResult(True, [], [initial], 0)

    came_from = {initial: (None, None)}     # state -> (prev_state, action)
    frontier  = deque([initial])
    expanded  = 0

    while frontier:
        state = frontier.popleft()
        expanded += 1
        for a in actions:
            if not a.applicable(state):
                continue
            succ = a.apply(state)
            if succ in came_from:
                continue
            came_from[succ] = (state, a)
            if goal <= succ:
                return _reconstruct(came_from, succ, expanded, initial)
            frontier.append(succ)

    return PlanResult(False, [], [initial], expanded)


def _reconstruct(came_from, final, expanded, initial) -> PlanResult:
    plan, trace = [], [final]
    s = final
    while came_from[s][0] is not None:
        prev, action = came_from[s]
        plan.append(action)
        trace.append(prev)
        s = prev
    plan.reverse()
    trace.reverse()
    return PlanResult(True, plan, trace, expanded)


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------
def fmt_state(s: State) -> str:
    return "{" + ", ".join(sorted(f"{a[0]}({','.join(a[1:])})" for a in s)) + "}"

def show(result: PlanResult, label: str):
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    if not result.found:
        print(f"NO PLAN FOUND   (expanded {result.expanded} states)")
        return

    print(f"Plan of {len(result.plan)} actions   (expanded {result.expanded} states):")
    for i, a in enumerate(result.plan, 1):
        print(f"  {i}. {a.name}")

    print("\nState trace:")
    print(f"  S_0 = {fmt_state(result.trace[0])}")
    for i, (a, s) in enumerate(zip(result.plan, result.trace[1:]), 1):
        print(f"  S_{i} = {fmt_state(s)}      (after {a.name})")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_a_solvable():
    """Original warehouse: goal is At(Package, C)."""
    I = frozenset({("At", "Robot", "A"), ("At", "Package", "A")})
    G = frozenset({("At", "Package", "C")})
    A = [
        move("A", "B"), move("B", "A"),
        move("B", "C"), move("C", "B"),
        pick_up("A"), pick_up("B"), pick_up("C"),
        drop("A"), drop("B"), drop("C"),
    ]
    show(plan_bfs(I, G, A), "Test A - solvable warehouse")


def test_b_impossible():
    """Remove PickUp entirely: the package cannot be moved."""
    I = frozenset({("At", "Robot", "A"), ("At", "Package", "A")})
    G = frozenset({("At", "Package", "C")})
    A = [
        move("A", "B"), move("B", "A"),
        move("B", "C"), move("C", "B"),
        drop("A"), drop("B"), drop("C"),         # Drop needs Holding(Package)
    ]
    show(plan_bfs(I, G, A), "Test B - impossible (no PickUp action)")


def test_c_irrelevant_actions():
    """Introduce actions that move the robot but do not help the package.
    The planner must still deliver the package, not confuse Robot-at-C
    with Package-at-C."""
    I = frozenset({("At", "Robot", "A"), ("At", "Package", "A")})
    G = frozenset({("At", "Package", "C")})
    A = [
        move("A", "B"), move("B", "A"),
        move("B", "C"), move("C", "B"),
        move("A", "C"), move("C", "A"),          # NEW: tempting shortcut
        pick_up("A"), pick_up("B"), pick_up("C"),
        drop("A"), drop("B"), drop("C"),
    ]
    show(plan_bfs(I, G, A), "Test C - with irrelevant Move(A,C) shortcut")


def precondition_question():
    """Task 0 sanity check -- is PickUp(Package,A) applicable in I?
    Is Drop(Package,C)?"""
    I = frozenset({("At", "Robot", "A"), ("At", "Package", "A")})
    pu_A = pick_up("A")
    dr_C = drop("C")
    print("\n" + "=" * 70)
    print("Task 0 preconditions check from I = {At(Robot,A), At(Package,A)}")
    print("=" * 70)
    print(f"  PickUp(Package,A) applicable? {pu_A.applicable(I)}")
    print(f"     pos_pre = {pu_A.pos_pre}")
    print(f"  Drop(Package,C)   applicable? {dr_C.applicable(I)}")
    print(f"     pos_pre = {dr_C.pos_pre}")


if __name__ == "__main__":
    precondition_question()
    test_a_solvable()
    test_b_impossible()
    test_c_irrelevant_actions()
