"""A body: sensors, actuators, and a loop that closes through perception.

What makes intelligence embodied is not having arms. It is that the only way
to find out what happened is to look again. A disembodied planner is handed
the state of the world; an embodied one has to perceive it, and perception is
lossy, so it acts on a belief that can be wrong and has to notice.

That distinction is implementable at this scale, and it is implemented here:

    sense    the world is rendered to a picture and the perception tower
             reads it - the same tower trained in v4 and v5
    believe  what the model thinks is there, which is not what is there
    plan     the planner works on the belief, not on the truth
    act      the action goes to the world, which has its own rules
    sense    look again, because the only way to know is to look

`Body.truth` exists so tests and reports can compare belief against reality,
and nothing in the loop is allowed to read it. There is a test that asserts
that, because a loop that peeks is just the planner with extra steps.

What this is not
----------------
There is no robot. `Hardware` below is the interface a real one would
implement - four methods - and nothing implements it. I am not going to ship
a module that looks like it drives a servo and does not.

And the honest measurement: MotherBrain's sight is 22.7% accurate on held-out
images against a 3.1% chance baseline. That is far above chance and wrong most
of the time. So the embodied loop here *works* in the sense that it runs, and
it is *bad* in the sense that acting on what it sees goes wrong often. Both
halves of that are reported by `Episode.render()` rather than one of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from . import world as W

MAX_LOOPS = 8          # look-plan-act rounds before giving up


class Hardware(Protocol):
    """What a real body would have to provide. Nothing implements this.

    It is here to make the shape explicit - and to make it obvious that the
    simulated body below is standing in for something absent, rather than
    being quietly presented as the thing itself.
    """

    def photograph(self):
        """Return an image of what is in front of the sensor."""

    def holding(self) -> tuple:
        """Return what is currently gripped."""

    def do(self, action: str, *arguments: str) -> tuple[bool, str]:
        """Attempt an action. Return (did it work, what happened)."""

    def stop(self) -> None:
        """Come to rest. Called when a loop ends, including on failure."""


@dataclass
class Belief:
    """What the body thinks is in front of it, and how sure.

    Deliberately separate from `World`. Collapsing them is the bug this whole
    module exists to avoid: if the belief *is* the world, perception is
    perfect and nothing about embodiment has been modelled.
    """

    seen: dict = field(default_factory=dict)      # name -> what it looked like
    confidence: dict = field(default_factory=dict)
    from_picture: bool = False

    def wrong_about(self, truth: W.World) -> list:
        """Where belief and reality disagree. For reports, never for planning.

        Only visual guesses count. "held" comes from proprioception, which
        is a different sense and is not a claim about shape - scoring it as
        one invents a misperception that never happened.
        """
        out = []
        for name, guess in self.seen.items():
            if guess not in W.SHAPES:        # not a claim about shape
                continue
            thing = truth.get(name)
            if thing is None:
                out.append(f"believed in {name}, which is not there")
            elif guess != thing.shape:
                out.append(f"{name} looked like a {guess}, is a {thing.shape}")
        return out


@dataclass
class Step:
    """One round of the loop."""

    looked: str = ""
    planned: str = ""
    acted: str = ""
    happened: str = ""
    ok: bool = True


@dataclass
class Episode:
    """One attempt at a goal, through a body."""

    goal: str
    steps: list = field(default_factory=list)
    done: bool = False
    why: str = ""
    mistakes: list = field(default_factory=list)
    blind: bool = False          # ran on proprioception alone, no camera
    refuted: set = field(default_factory=set)   # actions the world said no to

    def render(self, width: int = 68) -> str:
        lines = [f"Goal: {self.goal}", ""]
        if self.blind:
            lines.append("No perception tower was available, so this ran on")
            lines.append("the body's own sense of position only.")
            lines.append("")
        for i, step in enumerate(self.steps, start=1):
            mark = "" if step.ok else "  (failed)"
            lines.append(f"{i}. looked:   {step.looked}"[:width])
            if step.planned:
                lines.append(f"   planned:  {step.planned}"[:width])
            if step.acted:
                lines.append(f"   did:      {step.acted}{mark}"[:width])
            if step.happened:
                lines.append(f"   happened: {step.happened}"[:width])
            lines.append("")
        lines.append(f"Outcome: {self.why or ('done' if self.done else 'gave up')}")
        if self.mistakes:
            lines.append("")
            lines.append("What it misperceived:")
            for mistake in self.mistakes:
                lines.append(f"  {mistake}")
            lines.append("")
            lines.append("That is the point of the exercise, not a bug in it:")
            lines.append("acting on what you see means sometimes being wrong.")
        return "\n".join(lines)


class Simulated:
    """A body in the world model. The only body there is.

    Satisfies `Hardware` in shape, so when something real does turn up the
    loop below does not change. `truth` is the actual world and the loop is
    forbidden from reading it.
    """

    def __init__(self, world: W.World) -> None:
        self.truth = world

    def photograph(self):
        return W.picture(self.truth)

    def picture_tensor(self):
        """The same view as a tensor, for when PIL is not installed."""
        image = self.photograph()
        if image is None:
            return None
        import numpy as np
        import torch
        return torch.from_numpy(
            np.asarray(image, dtype="float32") / 255.0).permute(2, 0, 1)

    def holding(self) -> tuple:
        return tuple(self.truth.held)

    def fixate(self) -> list:
        """The objects a look could land on: visible, and not in hand.

        A closed box's contents are not among them, because you cannot see
        through a lid - which is a fact the world model already holds and
        this does not get to override.
        """
        return [t for t in self.truth.things
                if self.truth.visible(t.name) and not t.held]

    def silhouette(self, name: str):
        """One object, drawn the way the perception tower expects to see it.

        The tower learned single centred shapes on a plain ground, so this
        renders that. Choosing WHICH object is being looked at uses the
        world's own state - a body knows where it pointed its camera - and
        what the object is called is left entirely to the model.
        """
        import random

        from motherbrain.imagedata import COLOURS, render

        thing = self.truth.get(name)
        if thing is None:
            return None
        shape = SILHOUETTE.get(thing.shape)
        if shape is None:
            return None                     # a plank has no caption here
        colour = thing.colour if thing.colour in COLOURS else "grey"
        # render() already hands back a (3, size, size) tensor scaled to 0..1,
        # which is exactly what the tower was trained on. A fixed seed keeps
        # looking twice at an unchanged object from producing two answers.
        return render(shape, colour, 64, random.Random(0))

    def do(self, action: str, *arguments: str) -> tuple[bool, str]:
        outcome = W.act(self.truth, action, *arguments)
        said = outcome.said
        if outcome:
            self.truth, fell = W.settle(outcome.world)
            if fell:
                said += "; " + "; ".join(fell)
        return bool(outcome), said

    def stop(self) -> None:
        """Nothing to power down. A real body would need this to mean something."""
        return None


# The perception tower learned flat shapes - circle, square, triangle,
# diamond - and the world is made of blocks. Three of the four have a real
# silhouette correspondence; a plank seen flat-on is a rectangle, which this
# tower would have to call a square, and that would conflate it with a cube.
# So a plank is something MotherBrain can look at and cannot name, and that is
# reported rather than guessed at.
SILHOUETTE = {"ball": "circle", "cube": "square", "pyramid": "triangle"}
UNNAMEABLE = tuple(s for s in W.SHAPES if s not in SILHOUETTE)

#: Back the other way, for turning what the tower said into a world shape.
FROM_SILHOUETTE = {v: k for k, v in SILHOUETTE.items()}


def name_it(model, tok, image, device: str = "cpu") -> tuple[str, float]:
    """What the perception tower calls one image, by forced choice.

    The same method the sight accuracy was measured with: score every caption
    this world admits and take the one the model finds least surprising. No
    sampling, so no temperature and no luck - and the margin between the best
    and second-best caption is the only honest thing to call confidence.
    """
    import torch

    from motherbrain.imagedata import COLOURS, caption
    from motherbrain.sight import encode_batch

    shapes = tuple(SILHOUETTE.values())
    candidates = [caption(shape, colour)
                  for shape in shapes for colour in COLOURS]

    was_training = model.training
    model.eval()
    losses = []
    try:
        with torch.no_grad():
            batch = image.unsqueeze(0).to(device)
            for text in candidates:
                idx, targets = encode_batch(tok, [text], device)
                logits, _ = model(idx, targets=targets, images=batch)
                per_token = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1), ignore_index=-100, reduction="none")
                mask = (targets.reshape(-1) != -100).float()
                losses.append(float((per_token * mask).sum() / mask.sum()))
    finally:
        if was_training:
            model.train()

    order = sorted(range(len(losses)), key=lambda i: losses[i])
    best = candidates[order[0]]
    # Margin over the runner-up, squashed into 0..1. A tower that finds two
    # captions equally likely has told you nothing, and this says so.
    margin = losses[order[1]] - losses[order[0]] if len(order) > 1 else 0.0
    return best, max(0.0, min(1.0, margin))


def look(body, model=None, tok=None, device: str = "cpu") -> Belief:
    """Form a belief by perceiving, not by being told.

    With a perception tower the body fixates on one object at a time - the
    tower was trained on single centred shapes, so a whole-scene photograph
    is off its distribution and the answer would be noise - renders that
    object's silhouette, and the belief is whatever the tower calls it.
    Wrong included: that is the point.

    What this measures is **naming, not segmentation**. Which object is being
    looked at comes from the body; what it is called comes from the model. A
    64x64 single-object classifier cannot pick objects out of a scene, and
    pretending otherwise by feeding it the scene would produce a number that
    looked like perception and was not.

    Without a tower the body still knows what it is holding: proprioception
    is a different sense, and losing sight does not cost it.
    """
    belief = Belief()

    for name in body.holding():
        belief.seen[name] = "held"
        belief.confidence[name] = 1.0      # you know what is in your hand

    if model is None or getattr(model, "vision", None) is None or tok is None:
        return belief

    for thing in body.fixate():
        if thing.name in belief.seen:      # already in hand, already known
            continue
        image = body.silhouette(thing.name)
        if image is None:
            continue
        said, margin = name_it(model, tok, image, device)
        word = said.split()[-1]            # "a red circle" -> "circle"
        belief.from_picture = True
        belief.seen[thing.name] = FROM_SILHOUETTE.get(word, word)
        belief.confidence[thing.name] = margin
    return belief


def drive(body, goals, model=None, tok=None, device: str = "cpu",
          rounds: int = MAX_LOOPS) -> Episode:
    """Pursue a goal through a body: look, plan, act, look again.

    The loop never reads `body.truth`. It plans against a world assembled
    from what it believes, acts, and then looks again to find out what
    actually happened - which is the whole difference between this and
    `agent.achieve`.
    """
    from . import agent

    goals = list(goals)
    episode = Episode(goal=" and ".join(str(g) for g in goals))

    try:
        for _ in range(rounds):
            belief = look(body, model, tok, device)
            episode.blind = not belief.from_picture

            # The world as believed. Shapes come from perception, so a
            # misread cube-for-ball produces a plan that cannot work - and
            # the loop finds that out by acting, the same as anything with a
            # body does.
            believed = _assemble(body, belief)
            step = Step(looked=_describe_belief(belief))

            plan, note = agent.plan(believed, goals)
            step.planned = note
            if plan is None:
                step.ok = False
                episode.steps.append(step)
                episode.why = f"no plan from what it could see: {note}"
                break
            if not plan:
                episode.steps.append(step)
                episode.done = True
                episode.why = "already done"
                break

            # One action per round. Acting out a whole plan without looking
            # again is exactly the disembodied thing this avoids.
            action = plan[0]

            # An action the world has already refused is not worth refusing
            # again. Looking is not the only way to learn: what happened when
            # you tried is evidence too, and a loop that ignores it will sit
            # there repeating one impossible move until its budget runs out.
            if action in episode.refuted:
                step.planned = (f"{note}, but the world has already refused "
                                f"{' '.join(action)}")
                step.ok = False
                episode.steps.append(step)
                episode.why = ("what it sees and what happens disagree: the "
                               "only plan its eyes allow is one the world "
                               "refuses")
                break

            step.acted = " ".join(action)
            ok, said = body.do(*action)
            step.ok, step.happened = ok, said
            if not ok:
                episode.refuted.add(action)
            episode.steps.append(step)

            if all(goal.met(body.truth) for goal in goals):
                episode.done = True
                episode.why = "done"
                break
        else:
            episode.why = f"gave up after {rounds} rounds"
    finally:
        body.stop()

    # Only now, for the report: how wrong was it?
    final = look(body, model, tok, device)
    episode.mistakes = final.wrong_about(body.truth)
    return episode


def _assemble(body, belief: Belief) -> W.World:
    """Build the world the body believes it is in.

    Positions are not visible to a 64x64 classifier, so this keeps the
    world's arrangement and substitutes only what perception actually
    reported: the shapes. That is the honest boundary - inventing a believed
    layout the tower never produced would be making up a result.
    """
    things = []
    for thing in body.truth.things:
        guess = belief.seen.get(thing.name)
        if guess in W.SHAPES:
            thing = type(thing)(**{**thing.__dict__, "shape": guess})
        things.append(thing)
    return W.World(things=tuple(things), held=tuple(body.holding()))


def _describe_belief(belief: Belief) -> str:
    if not belief.seen:
        return "nothing"
    parts = []
    for name, guess in sorted(belief.seen.items()):
        score = belief.confidence.get(name)
        parts.append(f"{name}={guess}"
                     + (f" ({score:.0%})" if score is not None else ""))
    source = "from the camera" if belief.from_picture else "from position only"
    return ", ".join(parts) + f"  [{source}]"


def explain() -> str:
    """What is and is not connected, in the words it would use on screen."""
    return (
        "I have a body only in simulation.\n\n"
        "The loop is real: I render the world, read the picture with the\n"
        "same perception tower I grew in v4 and v5, plan against what I\n"
        "believe rather than what is there, act, and look again. Acting on\n"
        "what I see means I am sometimes wrong, and the report says where.\n\n"
        "What is not there is hardware. The interface a robot would\n"
        "implement is four methods - photograph, holding, do, stop - and\n"
        "nothing implements it. My sight is 22.7% accurate against a 3.1%\n"
        "chance baseline: far above chance, and wrong most of the time.\n"
        "I would not put that in charge of anything that can move."
    )
