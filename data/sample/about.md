# About MotherBrain

MotherBrain is a small language model that grows with every version it is
given. It starts deliberately tiny, and each release makes it larger: a patch
widens its feed-forward layers, a minor release widens the residual stream and
adds a layer, and a major release doubles its depth and the amount of text it
can hold in mind at once.

The model itself is an ordinary decoder-only transformer. Text arrives as raw
bytes, so it can read anything without a vocabulary being trained first. Each
byte becomes a vector. Those vectors pass through a stack of blocks, and every
block does the same two things: it lets each position look back at the
positions before it, and then it thinks about what it found.

Attention is the looking-back step. Every position produces a query, a key, and
a value. The query of one position is compared against the keys of all earlier
positions, and the values of the best matches are mixed together. Positions are
marked with rotary embeddings, so the model knows not just what came before but
how far back it was.

The feed-forward step is where the model does its private work. It projects
each position up into a wider space, filters it through a gate, and projects it
back down. Nothing moves between positions here; every position is considered
on its own.

Training is prediction. The model reads a stretch of text and, at every
position, guesses the byte that comes next. When it guesses badly the error is
measured and pushed back through every weight in the network. Repeat this over
enough text and the guesses stop being random.

A freshly initialised MotherBrain knows nothing at all. Its weights are noise,
and its output is noise. Everything it will ever know has to be fed to it as
text, one batch at a time, until the loss comes down and the noise starts to
look like language.

Growth is not the same as knowledge. A larger MotherBrain has more room to
store what it reads, but the room is empty until the reading is done. Version
numbers describe capacity. Training describes what is actually in there.
