# What MotherBrain gained in version 6

MotherBrain is a BBSLLM: a bulletin board system with a language model in it.
It answers telnet on port 23, it versions everything it learns, and every
answer it gives says where that answer came from. Version 6 is the version in
which it gained a world model, a planner, a warrant on every answer, a guard
against forgetting, and a body in simulation.

## A world model

MotherBrain has a world model. A world model is not a store of sentences about
the world; it is a set of causes and effects that hold whether or not anyone
has written them down. Objects have a place, a colour, a shape, a size and a
weight. Things rest on other things. Containers hold things, and a closed
container's contents are out of reach and also out of sight, which are two
different facts. The agent has one pair of hands and a carrying limit of three
size units.

Some rules in that world are obvious to a person and invisible in text
statistics. A ball supports nothing: anything placed on a ball rolls off. A
pyramid has a point, so nothing rests on it. A large thing placed on a small
thing topples. These are decisive when making a plan and no amount of fluency
substitutes for them.

Gravity is real in that world. A plank can be pulled out from under a stack,
and then whatever rested on it falls to whatever was underneath. Every other
action requires its target to be clear, so pulling a plank out is the one
action that can remove a support, and without it gravity would never fire.

## A planner

MotherBrain plans by breadth-first search over actions whose preconditions and
effects are written down. A plan found this way is correct before it is run.
When a goal cannot be reached, MotherBrain says no plan exists and says how
many states it searched. That is an answer a language model cannot give: asked
to balance a cube on a ball, MotherBrain reports that no plan exists after
searching three states, rather than producing plausible text.

The planner found, without being told, that pulling a shelf out from under a
cup is a one-step way to put the cup on the floor. Nobody wrote that down. It
follows from the rules.

## A warrant on every answer

MotherBrain answers with the most checkable method that fits the question, and
labels which one answered. There are five kinds of grounds.

An answer is CALCULATED when it was computed exactly, so the same input always
gives it. It is DERIVED when it follows from stated facts by stated rules. It
is PLANNED when it was checked by simulating it in the world model. It is
QUOTED when it was taken from something MotherBrain was given to read. It is
GENERATED when the language model produced it and nothing checked it.

The first three can be run again from nothing to confirm them. A generated
answer cannot be, and MotherBrain says so rather than nodding it through. This
ordering is the whole design: a fluent wrong answer costs more than a blunt
refusal.

## Learning after deployment, and a guard against forgetting

MotherBrain learns after it is deployed. New information becomes a new version
while the board stays up, and no caller is disconnected. Nothing is
overwritten: a version is the base weights plus a stack of patch files, so an
earlier version is always one command away.

The danger in learning after deployment is catastrophic forgetting: a network
trained on something new will overwrite what it knew, and the loss on the new
material looks excellent exactly while that happens. Measuring the new
material cannot see it.

So MotherBrain keeps a retention probe: a sample of older material chosen once
and never changed. It measures that probe before a patch and again afterwards.
If the probe gets worse by more than 0.15 nats per token, the patch is rolled
back automatically and the earlier version is served again. The refused patch
is kept on disk with its measurements, because a system that silently
discarded the attempt would learn the same bad lesson twice.

The probe measures forgetting, not generalisation. Its text was trained on, so
a low loss means "still knows this", not "learned to reason". It is a canary.

## A body, in simulation only

What makes intelligence embodied is not having arms. It is that the only way
to find out what happened is to look again.

MotherBrain has a body in simulation. It renders the world, reads the picture
with the same perception tower it grew in versions 4 and 5, plans against what
it believes rather than against what is there, acts, and looks again. It acts
once per look, because acting out a whole plan without looking again is the
disembodied thing this avoids.

Acting on what it sees means being wrong sometimes. Shown a red ball,
MotherBrain's perception tower says cube. Set the goal of putting something on
that ball, it plans the stack, acts, is refused by the world because a ball
holds nothing up, and stops saying that what it sees and what happens
disagree. It does not try the refused action again: what happened when you
tried is evidence too.

There is no robot. The interface a real body would implement has four methods
- photograph, holding, do, stop - and nothing implements it. MotherBrain's
sight is 22.7 per cent accurate against a 3.1 per cent chance baseline: far
above chance, and wrong most of the time. It would not put that in charge of
anything that can move.

The perception tower learned flat shapes: circle, square, triangle, diamond.
The world is made of blocks. A ball looks like a circle, a cube looks like a
square and a pyramid looks like a triangle, so those three can be named. A
plank seen flat on is a rectangle, which this tower would have to call a
square, confusing it with a cube. So a plank is something MotherBrain can look
at and cannot name, and it reports that rather than guessing.
