"""A world model: causes and effects, not statistics about sentences.

A language model trained on text learns which words follow which words. It
does not learn that a block on top of another block falls when you take
the lower one away - it learns that people who write about blocks tend to
write that. Those look identical until you ask about a situation nobody
has written down.

So this is the other thing: a small world with state, actions that have
preconditions and effects, and rules that hold whether or not anyone has
described them. It is a world model in the strict sense - it predicts what
happens next from causes, and it can be wrong in a way you can check.

Deliberately small and deliberately physical:

* Objects have a place, a colour, a shape, a size and a weight.
* Things rest on other things. Take away what a thing rests on and it
  falls to whatever is underneath.
* Containers hold things. A closed container's contents are out of reach
  and out of sight, which are two different facts.
* An agent has one pair of hands and a carrying limit.

Every action states what has to be true before it and what is true after
it, so a plan can be checked before it is run and a failure names the
precondition that stopped it. That is what makes the world usable for
planning rather than only for narration.

It renders, too: `picture()` draws the state so the perception tower can
be shown the same world the planner is reasoning about. A world model that
the senses cannot see is two systems, not one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

# The vocabulary of the world. Small on purpose: a closed world is one you
# can be exhaustively right or wrong about, which is what makes measuring
# possible.
COLOURS = ("red", "green", "blue", "yellow", "grey")
SHAPES = ("cube", "ball", "pyramid", "plank")
SIZES = {"small": 1, "medium": 2, "large": 3}

# What can hold what. A ball supports nothing - anything put on it rolls
# off - and a pyramid has a point. This is the sort of rule that is obvious
# to a person, invisible in text statistics, and decisive in a plan.
SUPPORTS = {"cube": True, "plank": True, "ball": False, "pyramid": False}

CARRY_LIMIT = 3          # in size units, in the agent's two hands


@dataclass(frozen=True)
class Thing:
    """One object. Immutable: the world moves by making a new world."""

    name: str
    shape: str = "cube"
    colour: str = "grey"
    size: str = "medium"
    on: str | None = None          # what it rests on: a thing, or "floor"
    inside: str | None = None      # which container holds it
    open: bool | None = None       # containers only
    held: bool = False

    @property
    def weight(self) -> int:
        return SIZES.get(self.size, 2)

    @property
    def is_container(self) -> bool:
        return self.open is not None

    def describe(self) -> str:
        return f"the {self.size} {self.colour} {self.shape} called {self.name}"


@dataclass(frozen=True)
class World:
    """Everything that is the case, and nothing that is not."""

    things: tuple[Thing, ...] = ()
    held: tuple[str, ...] = ()             # what the agent is carrying
    step: int = 0

    # -- looking at it --

    def get(self, name: str) -> Thing | None:
        for thing in self.things:
            if thing.name == name:
                return thing
        return None

    def on_top_of(self, name: str) -> list[Thing]:
        return [t for t in self.things if t.on == name]

    def contents(self, name: str) -> list[Thing]:
        return [t for t in self.things if t.inside == name]

    def clear(self, name: str) -> bool:
        """Nothing on it, so something can be put there or it can be lifted."""
        return not self.on_top_of(name)

    def reachable(self, name: str) -> bool:
        """In the open, or in an open container. Not under anything."""
        thing = self.get(name)
        if thing is None or thing.held:
            return False
        if thing.inside is not None:
            box = self.get(thing.inside)
            return bool(box and box.open)
        return self.clear(name)

    def visible(self, name: str) -> bool:
        """Reachable is not the same as visible: a lid hides, a stack does not."""
        thing = self.get(name)
        if thing is None:
            return False
        if thing.inside is not None:
            box = self.get(thing.inside)
            return bool(box and box.open)
        return True

    def carrying(self) -> int:
        return sum(self.get(n).weight for n in self.held if self.get(n))

    def _replace_thing(self, name: str, **changes) -> "World":
        things = tuple(replace(t, **changes) if t.name == name else t
                       for t in self.things)
        return replace(self, things=things)


@dataclass(frozen=True)
class Outcome:
    """What an action did, or the precondition that stopped it."""

    world: World
    ok: bool
    said: str

    def __bool__(self) -> bool:
        return self.ok


# ---- the actions ------------------------------------------------------------

def take(world: World, name: str) -> Outcome:
    """Pick something up. Hands, reach and weight all have to allow it."""
    thing = world.get(name)
    if thing is None:
        return Outcome(world, False, f"there is no {name}")
    if thing.held:
        return Outcome(world, False, f"{name} is already held")
    if len(world.held) >= 2:
        return Outcome(world, False, "both hands are full")
    if not world.clear(name):
        blocker = world.on_top_of(name)[0].name
        return Outcome(world, False, f"{blocker} is on top of {name}")
    if thing.inside is not None:
        box = world.get(thing.inside)
        if box is not None and not box.open:
            return Outcome(world, False, f"{box.name} is closed")
    if world.carrying() + thing.weight > CARRY_LIMIT:
        return Outcome(world, False,
                       f"{name} is too heavy to carry as well")

    after = world._replace_thing(name, held=True, on=None, inside=None)
    after = replace(after, held=world.held + (name,), step=world.step + 1)
    return Outcome(after, True, f"took {name}")


def put_on(world: World, name: str, target: str) -> Outcome:
    """Put a held thing on something. Some shapes do not hold things up."""
    thing = world.get(name)
    if thing is None or not thing.held:
        return Outcome(world, False, f"{name} is not being held")
    if target != "floor":
        base = world.get(target)
        if base is None:
            return Outcome(world, False, f"there is no {target}")
        if not SUPPORTS.get(base.shape, True):
            return Outcome(world, False,
                           f"a {base.shape} will not hold anything up")
        if not world.clear(target):
            return Outcome(world, False, f"{target} already has something on it")
        if base.held:
            return Outcome(world, False, f"{target} is in the air")
        if SIZES[thing.size] > SIZES[base.size]:
            return Outcome(world, False,
                           f"{name} is bigger than {target} and would topple")

    after = world._replace_thing(name, held=False, on=target, inside=None)
    after = replace(after, held=tuple(n for n in world.held if n != name),
                    step=world.step + 1)
    return Outcome(after, True, f"put {name} on {target}")


def put_in(world: World, name: str, container: str) -> Outcome:
    """Put a held thing into an open container."""
    thing = world.get(name)
    box = world.get(container)
    if thing is None or not thing.held:
        return Outcome(world, False, f"{name} is not being held")
    if box is None or not box.is_container:
        return Outcome(world, False, f"{container} is not a container")
    if not box.open:
        return Outcome(world, False, f"{container} is closed")
    if SIZES[thing.size] >= SIZES[box.size]:
        return Outcome(world, False, f"{name} will not fit in {container}")

    after = world._replace_thing(name, held=False, on=None, inside=container)
    after = replace(after, held=tuple(n for n in world.held if n != name),
                    step=world.step + 1)
    return Outcome(after, True, f"put {name} in {container}")


def open_box(world: World, name: str) -> Outcome:
    box = world.get(name)
    if box is None or not box.is_container:
        return Outcome(world, False, f"{name} is not a container")
    if box.open:
        return Outcome(world, False, f"{name} is already open")
    if not world.clear(name):
        return Outcome(world, False,
                       f"{world.on_top_of(name)[0].name} is on the lid")
    after = world._replace_thing(name, open=True)
    return Outcome(replace(after, step=world.step + 1), True, f"opened {name}")


def close_box(world: World, name: str) -> Outcome:
    box = world.get(name)
    if box is None or not box.is_container:
        return Outcome(world, False, f"{name} is not a container")
    if not box.open:
        return Outcome(world, False, f"{name} is already closed")
    after = world._replace_thing(name, open=False)
    return Outcome(replace(after, step=world.step + 1), True, f"closed {name}")


def pull_out(world: World, name: str) -> Outcome:
    """Pull a flat thing out from under whatever is stacked on it.

    This is the action that makes gravity matter. Everything else has a
    `clear` precondition, so nothing can lose its support - which would
    leave `settle` as dead code dressed up as physics. A plank can be
    slid out from under a stack, and then the stack has nothing to rest
    on and comes down.
    """
    thing = world.get(name)
    if thing is None:
        return Outcome(world, False, f"there is no {name}")
    if thing.held:
        return Outcome(world, False, f"{name} is already held")
    if thing.shape != "plank":
        return Outcome(world, False,
                       f"only a flat thing slides out; {name} is a "
                       f"{thing.shape}")
    if len(world.held) >= 2:
        return Outcome(world, False, "both hands are full")
    if world.carrying() + thing.weight > CARRY_LIMIT:
        return Outcome(world, False, f"{name} is too heavy to carry as well")

    after = world._replace_thing(name, held=True, on=None, inside=None)
    after = replace(after, held=world.held + (name,), step=world.step + 1)
    return Outcome(after, True, f"pulled {name} out")


ACTIONS = {
    "take": (take, 1),
    "pull-out": (pull_out, 1),
    "put-on": (put_on, 2),
    "put-in": (put_in, 2),
    "open": (open_box, 1),
    "close": (close_box, 1),
}


def act(world: World, action: str, *arguments: str) -> Outcome:
    """One action, by name. Unknown actions fail rather than doing nothing."""
    entry = ACTIONS.get(action)
    if entry is None:
        return Outcome(world, False, f"there is no action called {action}")
    function, arity = entry
    if len(arguments) != arity:
        return Outcome(world, False,
                       f"{action} takes {arity} thing(s), got {len(arguments)}")
    return function(world, *arguments)


def simulate(world: World, plan) -> tuple[World, list[str], str]:
    """Run a plan forward. Returns (world after, what happened, why it stopped).

    This is the whole point of a world model: you can find out what a plan
    does before doing it, and when it fails you are told which step failed
    and which precondition it broke.
    """
    log: list[str] = []
    for i, step in enumerate(plan, start=1):
        outcome = act(world, step[0], *step[1:])
        log.append(f"{i}. {' '.join(step)} - {outcome.said}")
        if not outcome:
            return world, log, f"stopped at step {i}: {outcome.said}"
        world = outcome.world
    return world, log, ""


# ---- gravity ----------------------------------------------------------------

def settle(world: World) -> tuple[World, list[str]]:
    """Let go of what is unsupported, and say what fell.

    Called after anything is removed. A thing resting on something that is
    no longer there falls to whatever was under that, which is the sort of
    consequence a text model has no way to work out and a world model gets
    for nothing.
    """
    said: list[str] = []
    changed = True
    while changed:
        changed = False
        for thing in world.things:
            if thing.on in (None, "floor") or thing.held:
                continue
            base = world.get(thing.on)
            if base is None or base.held:
                landing = base.on if base is not None else "floor"
                world = world._replace_thing(thing.name,
                                             on=landing or "floor")
                said.append(f"{thing.name} fell to "
                            f"{landing or 'the floor'}")
                changed = True
    return world, said


# ---- saying what is the case ------------------------------------------------

def describe(world: World, full: bool = False) -> str:
    """The world in sentences, for anything that reads sentences."""
    lines: list[str] = []
    floor = [t for t in world.things if t.on == "floor" and not t.held]
    for thing in sorted(floor, key=lambda t: t.name):
        lines.append(f"{thing.describe()} is on the floor.")
        for above in _stack(world, thing.name):
            lines.append(f"{above.describe()} is on {above.on}.")
    for box in [t for t in world.things if t.is_container]:
        inside = world.contents(box.name)
        state = "open" if box.open else "closed"
        if inside:
            names = ", ".join(t.name for t in inside)
            lines.append(f"{box.name} is {state} and holds {names}.")
        else:
            lines.append(f"{box.name} is {state} and empty.")
    if world.held:
        lines.append("You are holding " + ", ".join(world.held) + ".")
    else:
        lines.append("Your hands are empty.")
    if full:
        hidden = [t.name for t in world.things if not world.visible(t.name)]
        if hidden:
            lines.append("Out of sight: " + ", ".join(hidden) + ".")
    return "\n".join(lines)


def _stack(world: World, base: str) -> list[Thing]:
    out, current = [], base
    while True:
        above = world.on_top_of(current)
        if not above:
            return out
        out.append(above[0])
        current = above[0].name


def facts(world: World) -> set[tuple]:
    """The world as ground facts, for the symbolic side to reason over."""
    out: set[tuple] = set()
    for thing in world.things:
        out.add(("colour", thing.name, thing.colour))
        out.add(("shape", thing.name, thing.shape))
        out.add(("size", thing.name, thing.size))
        if thing.on:
            out.add(("on", thing.name, thing.on))
        if thing.inside:
            out.add(("in", thing.name, thing.inside))
        if thing.held:
            out.add(("held", thing.name))
        if thing.is_container:
            out.add(("open" if thing.open else "closed", thing.name))
        if world.clear(thing.name):
            out.add(("clear", thing.name))
    return out


# ---- drawing it -------------------------------------------------------------

PALETTE = {
    "red": (220, 60, 60), "green": (60, 190, 90), "blue": (70, 110, 230),
    "yellow": (240, 210, 70), "grey": (150, 150, 155),
}


def picture(world: World, size: int = 64):
    """Draw the world, so the perception tower can be shown what the planner
    is reasoning about. A world the senses cannot see is two systems."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    image = Image.new("RGB", (size, size), (24, 26, 34))
    draw = ImageDraw.Draw(image)
    floor = size - 6
    draw.rectangle((0, floor, size, size), fill=(60, 56, 50))

    columns = [t for t in world.things
               if t.on == "floor" and not t.held and t.inside is None]
    if not columns:
        return image
    width = max(8, size // max(1, len(columns) + 1))

    for i, base in enumerate(sorted(columns, key=lambda t: t.name)):
        x = int((i + 0.5) * size / len(columns))
        y = floor
        for thing in [base] + _stack(world, base.name):
            height = 4 + SIZES[thing.size] * 4
            half = 3 + SIZES[thing.size] * 3
            box = (x - half, y - height, x + half, y)
            colour = PALETTE.get(thing.colour, (150, 150, 150))
            if thing.shape == "ball":
                draw.ellipse(box, fill=colour)
            elif thing.shape == "pyramid":
                draw.polygon([(x, y - height), (x - half, y), (x + half, y)],
                             fill=colour)
            elif thing.shape == "plank":
                draw.rectangle((x - half - 3, y - 4, x + half + 3, y),
                               fill=colour)
                height = 4
            else:
                draw.rectangle(box, fill=colour)
            y -= height + 1
    return image


# ---- making one -------------------------------------------------------------

def build(seed: int = 0, things: int = 4, container: bool = True) -> World:
    """A random but legal world. The same seed is the same world, always."""
    rng = random.Random(seed)
    made: list[Thing] = []
    names = ["a", "b", "c", "d", "e", "f"][:things]
    for name in names:
        made.append(Thing(
            name=name,
            shape=rng.choice(SHAPES),
            colour=rng.choice(COLOURS),
            size=rng.choice(list(SIZES)),
            on="floor",
        ))
    if container:
        made.append(Thing(name="box", shape="cube", colour="grey",
                          size="large", on="floor", open=rng.random() < 0.5))

    world = World(things=tuple(made))

    # Stack one thing on another where the rules allow it, so the starting
    # world is not always flat.
    for _ in range(things):
        top = rng.choice(names)
        base = rng.choice(names)
        if top == base:
            continue
        held = take(world, top)
        if not held:
            continue
        placed = put_on(held.world, top, base)
        if placed:
            world = placed.world
    return replace(world, step=0)
