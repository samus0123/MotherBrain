"""An agent: something that pursues a goal instead of finishing a sentence.

A language model produces the next token. That is all it does, and no amount
of prompting changes it into something that takes actions. An agent is the
loop built around it:

    goal -> plan -> act -> observe -> did that work? -> act again -> stop

The parts that matter are the ones that are *not* the model. Tools are real
functions with real effects. Observations come back from whatever actually
happened, not from what the model expected. The stopping condition is checked
against the world, so the loop cannot talk itself into being finished.

Two kinds of planning live here, and the difference is worth being clear
about:

* **Searched plans.** When the goal is a state of the world model, the plan
  is found by breadth-first search over actions whose preconditions and
  effects are written down (see `world.py`). This is real planning: the plan
  is correct before it is run, and if no plan exists the agent says so
  instead of trying anyway. No language model is involved and none is needed.

* **Chosen tools.** When the goal is a question, the agent picks a tool by
  matching the question against what each tool declares it handles, runs it,
  and reads the answer. The model is consulted last, for the things only a
  language model can do, and its output is never treated as an observation
  about the world.

Every step is recorded with what was tried, what came back, and whether it
helped. The trace is the agent's account of itself and it is written in the
past tense - things done and observed, not claims being made.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from . import world as W

MAX_STEPS = 12          # a loop that cannot end is not an agent
MAX_SEARCH = 20000      # states expanded before planning gives up


# ---- tools ------------------------------------------------------------------

@dataclass
class Tool:
    """One thing the agent can do, and how it says whether it applies.

    `handles` returns a score: 0 means "not for me", higher means "this is
    my sort of question". A tool that overclaims will be picked wrongly, so
    the scores are deliberately conservative and ties fall through to the
    next tool rather than being broken at random.
    """

    name: str
    help: str
    run: Callable[..., str]
    handles: Callable[[str], float] = lambda goal: 0.0
    changes_world: bool = False

    def score(self, goal: str) -> float:
        try:
            return float(self.handles(goal))
        except Exception:
            return 0.0


@dataclass
class Registry:
    """The tools an agent has. Nothing is available that is not in here."""

    tools: dict = field(default_factory=dict)

    def add(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str):
        return self.tools.get(name)

    def best(self, goal: str, skip=()):
        """The tool most likely to help, or None. Ties go to the first added,
        which is the order the tools were declared in, deliberately."""
        ranked = [(t.score(goal), i, t)
                  for i, t in enumerate(self.tools.values())
                  if t.name not in skip]
        ranked = [r for r in ranked if r[0] > 0]
        if not ranked:
            return None
        ranked.sort(key=lambda r: (-r[0], r[1]))
        return ranked[0][2]

    def catalogue(self) -> str:
        width = max((len(n) for n in self.tools), default=0)
        return "\n".join(f"  {name.ljust(width)}  {tool.help}"
                         for name, tool in self.tools.items())

    def __len__(self) -> int:
        return len(self.tools)


# ---- the record -------------------------------------------------------------

@dataclass
class Act:
    """One step the agent took."""

    tool: str
    argument: str
    observation: str
    ok: bool = True
    seconds: float = 0.0


@dataclass
class Run:
    """Everything that happened while pursuing one goal."""

    goal: str
    acts: list = field(default_factory=list)
    answer: str = ""
    done: bool = False
    why: str = ""

    def add(self, tool: str, argument: str, observation: str,
            ok: bool = True, seconds: float = 0.0) -> None:
        self.acts.append(Act(tool, argument, observation, ok, seconds))

    def render(self, width: int = 68) -> str:
        lines = [f"Goal: {self.goal}", ""]
        for i, act in enumerate(self.acts, start=1):
            mark = "ok " if act.ok else "no "
            head = f"{i}. {mark}{act.tool}"
            if act.argument:
                head += f" ({act.argument})"
            lines.append(head[:width])
            for line in act.observation.splitlines():
                while len(line) > width - 5:
                    lines.append("     " + line[:width - 5])
                    line = line[width - 5:]
                lines.append("     " + line)
        lines.append("")
        lines.append(f"Outcome: {self.why or ('done' if self.done else 'gave up')}")
        if self.answer:
            lines.append("")
            lines.append(self.answer)
        return "\n".join(lines)

    def steps(self) -> int:
        return len(self.acts)


# ---- planning in the world model --------------------------------------------

@dataclass(frozen=True)
class Goal:
    """A state of the world to be brought about.

    Kept to the three relations that can be checked exactly, because a goal
    that cannot be checked cannot end a loop.
    """

    relation: str           # "on", "in", "held", "clear", "open", "closed"
    thing: str
    target: str = ""

    def met(self, world: W.World) -> bool:
        item = world.get(self.thing)
        if item is None:
            return False
        if self.relation == "on":
            return item.on == self.target
        if self.relation == "in":
            return item.inside == self.target
        if self.relation == "held":
            return item.held
        if self.relation == "clear":
            return world.clear(self.thing)
        if self.relation == "open":
            return bool(item.open)
        if self.relation == "closed":
            return item.is_container and not item.open
        return False

    def __str__(self) -> str:
        if self.target:
            return f"{self.thing} {self.relation} {self.target}"
        return f"{self.thing} {self.relation}"


def _moves(world: W.World):
    """Every action worth trying from here.

    Only legal moves are generated - the world model knows its own
    preconditions - so the search never wastes a state on something that
    could not happen.
    """
    names = [t.name for t in world.things]
    for name in names:
        thing = world.get(name)
        if thing.held:
            continue
        yield ("take", name)
        if thing.shape == "plank":
            yield ("pull-out", name)
    for held in world.held:
        yield ("put-on", held, "floor")
        for name in names:
            if name != held:
                yield ("put-on", held, name)
                if world.get(name).is_container:
                    yield ("put-in", held, name)
    for thing in world.things:
        if thing.is_container:
            yield ("open" if not thing.open else "close", thing.name)


def _key(world: W.World):
    """Two worlds are the same state if the same things are in the same
    places. Step count and history are not part of the state, or the search
    would never close a loop."""
    return (tuple(sorted((t.name, t.on, t.inside, t.held, t.open)
                         for t in world.things)),
            tuple(sorted(world.held)))


def plan(start: W.World, goals, limit: int = MAX_SEARCH):
    """Find a shortest sequence of actions that makes every goal true.

    Breadth-first, so the plan is the shortest one that exists. Returns
    (plan, note): plan is None when no sequence of legal actions reaches the
    goal, and the note says which - "no plan exists" is a real answer and a
    useful one, and it is the answer a text model can never give you.
    """
    goals = list(goals)
    if all(g.met(start) for g in goals):
        return [], "already true"

    seen = {_key(start)}
    queue = deque([(start, [])])
    expanded = 0

    while queue:
        world, path = queue.popleft()
        expanded += 1
        if expanded > limit:
            return None, f"gave up after {expanded - 1} states"
        for move in _moves(world):
            outcome = W.act(world, *move)
            if not outcome:
                continue
            after, _ = W.settle(outcome.world)
            marker = _key(after)
            if marker in seen:
                continue
            seen.add(marker)
            step = path + [move]
            if all(g.met(after) for g in goals):
                return step, f"found in {len(step)} steps ({expanded} states)"
            if len(step) < MAX_STEPS:
                queue.append((after, step))
    return None, f"no plan exists ({expanded} states searched)"


# ---- the agent --------------------------------------------------------------

class Agent:
    """A goal, a set of tools, and a loop with a way to stop.

    Given a `world`, world goals are planned and executed. Given a question,
    tools are chosen and run. Either way the answer is what came back from
    doing something, not what a model said would come back.
    """

    def __init__(self, registry: Registry, world: W.World | None = None,
                 max_steps: int = MAX_STEPS) -> None:
        self.tools = registry
        self.world = world
        self.max_steps = max_steps

    # -- acting in the world --

    def achieve(self, goals, run: Run | None = None) -> Run:
        """Bring about a state of the world, or say why it cannot be."""
        goals = list(goals)
        label = " and ".join(str(g) for g in goals)
        run = run or Run(goal=label)

        if self.world is None:
            run.add("plan", label, "there is no world to act in", ok=False)
            run.why = "no world"
            return run

        began = time.perf_counter()
        steps, note = plan(self.world, goals)
        run.add("plan", label, note, ok=steps is not None,
                seconds=time.perf_counter() - began)

        if steps is None:
            run.why = note
            run.answer = f"I cannot get to: {label}. {note}."
            return run

        for move in steps:
            outcome = W.act(self.world, *move)
            said = outcome.said
            if outcome:
                self.world, fell = W.settle(outcome.world)
                if fell:
                    said += "; " + "; ".join(fell)
            run.add("act", " ".join(move), said, ok=bool(outcome))
            if not outcome:
                run.why = f"the plan broke: {outcome.said}"
                return run

        met = [g for g in goals if g.met(self.world)]
        run.done = len(met) == len(goals)
        run.why = "done" if run.done else "the plan ran but the goal is not met"
        run.answer = (f"Done in {len(steps)} step(s)." if run.done
                      else "The plan ran and the goal is still not met.")
        return run

    # -- answering a question by using tools --

    def pursue(self, goal: str) -> Run:
        """Work at a goal with tools until it is met or the budget is spent."""
        run = Run(goal=goal)
        remaining = goal
        spent: set[str] = set()          # tried on *this* goal, not removed

        for _ in range(self.max_steps):
            tool = self.tools.best(remaining, skip=spent)
            if tool is None:
                break
            began = time.perf_counter()
            try:
                observation = tool.run(remaining)
                ok = bool(observation)
            except Exception as error:                    # a tool may fail
                observation = f"{type(error).__name__}: {error}"
                ok = False
            run.add(tool.name, "", observation or "(nothing)", ok=ok,
                    seconds=time.perf_counter() - began)

            if ok and observation:
                run.answer = observation
                run.done = True
                run.why = f"answered by {tool.name}"
                return run

            # A tool that did not answer is not tried again on this goal.
            # It stays in the registry: the next goal gets the full set.
            spent.add(tool.name)

        if not run.acts:
            run.why = "no tool applies"
            run.answer = ("I have no tool for that. What I can do:\n"
                          + self.tools.catalogue())
        else:
            run.why = "every tool that applied came back empty"
            run.answer = "I tried what I have and none of it answered that."
        return run


# ---- the standard tools -----------------------------------------------------

def standard(run_dir=None, world: W.World | None = None,
             model=None, tok=None, device: str = "cpu") -> Registry:
    """The tools MotherBrain has. Exact ones first, the model last.

    The order is the whole argument: arithmetic goes to a calculator, facts
    go to the rule engine, the world goes to the planner, and the language
    model is asked only for the thing that is genuinely a language task. A
    tool that can be exact is never replaced by one that is merely fluent.
    """
    from . import logic

    registry = Registry()

    registry.add(Tool(
        name="calculate",
        help="exact arithmetic, conversion, statistics - never estimated",
        run=lambda goal: (lambda r: r.render() if r else "")(
            logic.solve(goal)),
        handles=lambda goal: 3.0 if logic.solve(goal) else 0.0,
    ))

    if run_dir is not None:
        from . import knowledge

        def recall(goal: str) -> str:
            store = knowledge.Knowledge(run_dir, create=False)
            asked = knowledge.parse_question(goal)
            if asked:
                answer = store.ask(*asked)
                return answer.render() if answer.holds else ""
            name = knowledge.parse_about(goal)
            if name:
                answer = store.about(name)
                return answer.render() if answer.facts else ""
            return ""

        registry.add(Tool(
            name="recall",
            help="what I have been told, and what follows from it",
            run=recall,
            handles=lambda goal: 2.0 if (
                knowledge.parse_question(goal) or knowledge.parse_about(goal)
            ) else 0.0,
        ))

    if world is not None:
        def look(goal: str) -> str:
            return W.describe(world, full=True)

        registry.add(Tool(
            name="look",
            help="what is in the world right now",
            run=look,
            handles=lambda goal: 2.0 if any(
                word in goal.lower()
                for word in ("look", "what is there", "what do you see",
                             "describe the world", "the world")
            ) else 0.0,
        ))

    from . import aware

    def introspect(goal: str) -> str:
        found = aware.introspect(goal)
        return found.render() if found else ""

    registry.add(Tool(
        name="myself",
        help="how I am built, read from my own source",
        run=introspect,
        handles=lambda goal: 2.0 if aware.introspect(goal) else 0.0,
    ))

    if run_dir is not None:
        from . import nlp

        def read(goal: str) -> str:
            index = nlp.corpus_index(run_dir)
            hits = nlp.search(index, goal, limit=1)
            return hits[0].text.strip() if hits else ""

        registry.add(Tool(
            name="read",
            help="find the passage in what I have read that bears on this",
            run=read,
            handles=lambda goal: 1.0 if len(goal.split()) > 2 else 0.0,
        ))

    return registry
