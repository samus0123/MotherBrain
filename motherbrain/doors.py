"""Door games.

On a real board a "door" was a separate program the BBS shelled out to,
handing it your serial port. You left the board, played, and came back.
These do not shell out - the board is one process and a door that forked
would be a remote shell with extra steps - but they keep the shape: you
pick one, it takes the screen, and it hands you back to the menu.

Five of them. Three are the games a 1980s board actually had. Two exist
because the board is running a language model and it would be a waste not
to: one lets you ask it things, and one asks whether you can tell its
honest answers from its generated ones. That last is the most useful thing
on the board, and it is a game.
"""

from __future__ import annotations

import random

from motherbrain import ansi as A

# What the door menu shows, in the order it shows them.
CATALOGUE = [
    ("H", "HAMURABI", "Rule Sumer for ten years. 1968. Still hard."),
    ("G", "GUESS", "The number is between 1 and 100. Eight tries."),
    ("Z", "THE MAZE", "Walk out of it. Drawn fresh every time."),
    ("O", "THE ORACLE", "Ask MotherBrain anything. Watch what it does."),
    ("T", "TURING", "Two answers. One is honest. Which is which?"),
]


async def play(caller, key: str) -> bool:
    """Run the door named by `key`. Returns False if there is no such door."""
    door = {"H": hamurabi, "G": guess, "Z": maze,
            "O": oracle, "T": turing}.get(key.upper())
    if door is None:
        return False
    await door(caller)
    return True


async def _banner(caller, title: str, subtitle: str) -> None:
    await caller.cls()
    await caller.art(A.gradient(f"  {title}  ", (A.HY, A.Y, A.HR, A.R)))
    await caller.line(f"  {A.GREY}{subtitle}{A.RESET}")
    await caller.line("")


# ---- HAMURABI ---------------------------------------------------------------

async def hamurabi(caller) -> None:
    """The 1968 BASIC original, rules intact.

    Kept faithful because the game is not really about grain: it is about a
    plague year arriving with no warning and no way to have planned for it,
    which is the thing everyone remembers about it forty years on.
    """
    await _banner(caller, "H A M U R A B I",
                  "Ten years of Sumer. Try not to starve them.")

    rng = random.Random()
    population, grain, acres = 100, 2800, 1000
    starved_total, immigrants, plague = 0, 5, False
    year = 1

    while year <= 10:
        price = rng.randint(17, 26)
        await caller.line(f"{A.HC}O great Hamurabi!{A.RESET}")
        await caller.line(f"  You are in year {A.HY}{year}{A.RESET} of your "
                          f"ten year rule.")
        if starved_total:
            await caller.line(f"  {A.HR}{starved_total} people starved last "
                              f"year.{A.RESET}")
        await caller.line(f"  {immigrants} people came to the city.")
        if plague:
            await caller.line(f"  {A.HR}A horrible plague struck! Half the "
                              f"people died.{A.RESET}")
        await caller.line(f"  Population is now {A.HW}{population}{A.RESET}.")
        await caller.line(f"  The city owns {A.HW}{acres}{A.RESET} acres.")
        await caller.line(f"  You harvested {A.HW}{grain // max(acres, 1)}"
                          f"{A.RESET} bushels per acre.")
        await caller.line(f"  You have {A.HW}{grain}{A.RESET} bushels in store.")
        await caller.line(f"  Land is trading at {A.HY}{price}{A.RESET} "
                          f"bushels per acre.")
        await caller.line("")

        buy = await _number(caller, "How many acres to buy? ", 0)
        if buy * price > grain:
            await caller.line(f"  {A.HR}You have only {grain} bushels. "
                              f"Not enough.{A.RESET}\n")
            continue
        if buy:
            acres += buy
            grain -= buy * price
        else:
            sell = await _number(caller, "How many acres to sell? ", 0)
            if sell > acres:
                await caller.line(f"  {A.HR}You own only {acres} acres."
                                  f"{A.RESET}\n")
                continue
            acres -= sell
            grain += sell * price

        feed = await _number(caller, "How many bushels to feed the people? ", 0)
        if feed > grain:
            await caller.line(f"  {A.HR}You have only {grain} bushels."
                              f"{A.RESET}\n")
            continue
        grain -= feed

        plant = await _number(caller, "How many acres to plant with seed? ", 0)
        if plant > acres:
            await caller.line(f"  {A.HR}You own only {acres} acres.{A.RESET}\n")
            continue
        if plant // 2 > grain:
            await caller.line(f"  {A.HR}You have only {grain} bushels of "
                              f"seed.{A.RESET}\n")
            continue
        if plant > 10 * population:
            await caller.line(f"  {A.HR}Only {population} people to tend the "
                              f"fields; ten acres each.{A.RESET}\n")
            continue
        grain -= plant // 2

        yield_per_acre = rng.randint(1, 5)
        harvest = plant * yield_per_acre
        eaten = 0
        if rng.randint(1, 5) <= 2:                     # rats, two years in five
            eaten = grain // rng.randint(2, 4)
        grain = grain + harvest - eaten

        fed = feed // 20
        starved = max(0, population - fed)
        if starved > 0.45 * population:
            await caller.line("")
            await caller.line(f"{A.HR}You starved {starved} people in one "
                              f"year!{A.RESET}")
            await caller.line("Due to this extreme mismanagement you have "
                              "not only been impeached and thrown out of "
                              "office, but you have also been declared "
                              "national fink.")
            await caller.pause()
            return
        starved_total = starved
        population -= starved
        immigrants = (starved // 2 + (5 - yield_per_acre) * grain // 600 + 1)
        immigrants = max(0, min(50, immigrants))
        population += immigrants
        plague = rng.randint(1, 100) <= 15
        if plague:
            population //= 2
        if eaten:
            await caller.line(f"  {A.HY}Rats ate {eaten} bushels.{A.RESET}")
        await caller.line("")
        year += 1

    per_head = acres / max(population, 1)
    await caller.line(f"{A.HC}In your ten year term {starved_total} people "
                      f"starved.{A.RESET}")
    await caller.line(f"You finished with {population} people and {acres} "
                      f"acres: {per_head:.1f} acres each.")
    if per_head > 10:
        verdict = "A fantastic performance. Charlemagne, Disraeli and " \
                  "Jefferson combined could not have done better."
    elif per_head > 7:
        verdict = "Your heavy-handed performance smacks of Nero and Ivan IV. " \
                  "The people remain, but they do not love you."
    else:
        verdict = "Your performance could have been somewhat better."
    await caller.line(verdict)
    await caller.pause()


async def _number(caller, prompt: str, default: int) -> int:
    raw = await caller.ask(f"  {A.HW}{prompt}{A.RESET}", limit=12)
    try:
        return max(0, int(raw.strip() or default))
    except ValueError:
        return default


# ---- GUESS ------------------------------------------------------------------

async def guess(caller) -> None:
    """The one every board had, with the thermometer it never had."""
    await _banner(caller, "G U E S S   T H E   N U M B E R",
                  "1 to 100. Eight guesses. Binary search wins in seven.")

    rng = random.Random()
    secret = rng.randint(1, 100)
    low, high = 1, 100

    for attempt in range(1, 9):
        await caller.line(f"  {A.GREY}it is somewhere in "
                          f"{A.HW}{low}-{high}{A.RESET}")
        await caller.line("  " + _thermometer(low, high))
        raw = await caller.ask(f"  {A.HY}guess {attempt}/8: {A.RESET}", limit=8)
        try:
            value = int(raw.strip())
        except ValueError:
            await caller.line(f"  {A.HR}that is not a number.{A.RESET}")
            continue
        if value == secret:
            await caller.line("")
            await caller.line(f"  {A.HG}{A.BOLD}GOT IT in {attempt}."
                              f"{A.RESET} The number was {secret}.")
            if attempt <= 7:
                await caller.line("  That is optimal play or better. "
                                  "Well done.")
            await caller.pause()
            return
        if value < secret:
            low = max(low, value + 1)
            await caller.line(f"  {A.HB}higher.{A.RESET}")
        else:
            high = min(high, value - 1)
            await caller.line(f"  {A.HR}lower.{A.RESET}")
        await caller.line("")

    await caller.line(f"  {A.HR}Out of guesses. It was {secret}.{A.RESET}")
    await caller.pause()


def _thermometer(low: int, high: int, width: int = 50) -> str:
    """The range still in play, drawn as a bar. 1-100 maps onto the width."""
    a = int((low - 1) / 99 * width)
    b = int((high - 1) / 99 * width)
    out = []
    for i in range(width):
        out.append(f"{A.HG}█" if a <= i <= b else f"{A.GREY}░")
    return "".join(out) + A.RESET


# ---- THE MAZE ---------------------------------------------------------------

def build_maze(width: int, height: int, seed: int | None = None) -> list[list[int]]:
    """A perfect maze by recursive backtracking: one path between any two cells.

    Returns a grid of 0 (wall) and 1 (floor), sized (2*width+1, 2*height+1),
    which is the standard way to hold a maze whose walls are cells too.
    """
    rng = random.Random(seed)
    grid = [[0] * (2 * width + 1) for _ in range(2 * height + 1)]
    seen = [[False] * width for _ in range(height)]
    stack = [(0, 0)]
    seen[0][0] = True
    grid[1][1] = 1

    while stack:
        x, y = stack[-1]
        options = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and not seen[ny][nx]:
                options.append((nx, ny, dx, dy))
        if not options:
            stack.pop()
            continue
        nx, ny, dx, dy = rng.choice(options)
        seen[ny][nx] = True
        grid[2 * y + 1 + dy][2 * x + 1 + dx] = 1
        grid[2 * ny + 1][2 * nx + 1] = 1
        stack.append((nx, ny))
    return grid


def draw_maze(grid: list[list[int]], player: tuple[int, int],
              exit_at: tuple[int, int]) -> str:
    """The maze as ANSI. Walls are solid blocks, which is what made it read."""
    lines = []
    for y, row in enumerate(grid):
        out = []
        for x, cell in enumerate(row):
            if (x, y) == player:
                out.append(f"{A.HY}{A.BOLD}@@{A.RESET}")
            elif (x, y) == exit_at:
                out.append(f"{A.HG}{A.BOLD}><{A.RESET}")
            elif cell:
                out.append("  ")
            else:
                out.append(f"{A.HB}██{A.RESET}")
        lines.append("".join(out))
    return "\n".join(lines)


async def maze(caller) -> None:
    """Walk out. Drawn fresh, so nobody has the map."""
    width = max(6, min(18, (caller.columns - 4) // 4))
    height = max(4, min(9, (caller.rows - 8) // 2))
    grid = build_maze(width, height)
    player = (1, 1)
    exit_at = (2 * width - 1, 2 * height - 1)
    moves = 0

    while True:
        await caller.cls()
        await caller.art(A.gradient("  T H E   M A Z E  ",
                                    (A.HG, A.G, A.HC, A.C)))
        await caller.art(draw_maze(grid, player, exit_at))
        await caller.line("")
        await caller.line(f"  {A.HW}W A S D{A.RESET} or the arrow keys. "
                          f"{A.HW}Q{A.RESET} to give up.   moves: {moves}")
        key = await caller.key()
        step = {"w": (0, -1), "a": (-1, 0), "s": (0, 1), "d": (1, 0),
                "UP": (0, -1), "LEFT": (-1, 0), "DOWN": (0, 1),
                "RIGHT": (1, 0)}.get(key if len(key) > 1 else key.lower())
        if key.lower() == "q":
            return
        if step is None:
            continue
        nx, ny = player[0] + step[0], player[1] + step[1]
        if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]) and grid[ny][nx]:
            player = (nx, ny)
            moves += 1
        if player == exit_at:
            await caller.cls()
            await caller.art(draw_maze(grid, player, exit_at))
            await caller.line("")
            await caller.line(f"  {A.HG}{A.BOLD}OUT.{A.RESET} {moves} moves.")
            await caller.pause()
            return


# ---- THE ORACLE -------------------------------------------------------------

async def oracle(caller) -> None:
    """Ask it anything, and see exactly what kind of answer comes back.

    The point of the door is the labelling. Every reply is marked with where
    it came from: computed, known, read off its own state, or generated. A
    board that presented all four the same way would be the ordinary kind of
    lie a chat box tells.
    """
    await _banner(caller, "T H E   O R A C L E",
                  "It answers. The label tells you what the answer is worth.")
    await caller.line(f"  {A.GREY}blank line to leave.{A.RESET}")
    await caller.line("")

    while True:
        question = await caller.ask(f"  {A.HM}ask> {A.RESET}", limit=400)
        if not question.strip():
            return
        source, text = await caller.board.answer(question, caller)
        await caller.line("")
        await caller.line(f"  {_label(source)}")
        for line in A.wrap(text, caller.columns - 6):
            await caller.line(f"    {A.HW}{line}{A.RESET}")
        await caller.line("")


_LABELS = {
    "exact": f"{A.HG}[COMPUTED]{A.RESET} - arithmetic, done exactly, not guessed.",
    "known": f"{A.HG}[KNOWN]{A.RESET} - follows from something it was told.",
    "self": f"{A.HC}[FROM ITS OWN STATE]{A.RESET} - read off disk, true today.",
    "generated": f"{A.HY}[GENERATED]{A.RESET} - a continuation of your words, "
                 f"not an answer to them.",
}


def _label(source: str) -> str:
    return _LABELS.get(source, _LABELS["generated"])


# ---- TURING -----------------------------------------------------------------

async def turing(caller) -> None:
    """Two answers to your question. One is true. Which one?

    This is the door that earns its place. MotherBrain can answer some
    questions exactly - its size, its version, what it was told - and it can
    generate fluent prose about anything at all. The generated prose is the
    more convincing of the two, every time, and that is the whole problem
    with a language model this size presented as an assistant. Playing this
    is faster than being told.
    """
    await _banner(caller, "T U R I N G",
                  "Tell the honest answer from the fluent one.")
    await caller.line(f"  {A.GREY}Ask about MotherBrain itself - its size, "
                      f"its version, what it can see.{A.RESET}")
    await caller.line(f"  {A.GREY}blank line to leave.{A.RESET}")
    await caller.line("")

    right = wrong = 0
    rng = random.Random()

    while True:
        question = await caller.ask(f"  {A.HM}ask> {A.RESET}", limit=300)
        if not question.strip():
            break

        honest, made_up = await caller.board.two_answers(question, caller)
        if honest is None:
            await caller.line(f"  {A.HR}It has nothing true to say about "
                              f"that, so there is nothing to compare."
                              f"{A.RESET}")
            await caller.line(f"  {A.GREY}Try: how big are you / what version "
                              f"are you / can you see{A.RESET}\n")
            continue

        first_is_honest = rng.random() < 0.5
        pair = ((honest, made_up) if first_is_honest else (made_up, honest))
        for i, text in enumerate(pair, start=1):
            await caller.line("")
            await caller.line(f"  {A.HC}[{i}]{A.RESET}")
            for line in A.wrap(text, caller.columns - 8):
                await caller.line(f"      {A.HW}{line}{A.RESET}")
        await caller.line("")
        choice = await caller.ask(f"  which one is true? [1/2] ", limit=4)
        picked_first = choice.strip().startswith("1")
        if picked_first == first_is_honest:
            right += 1
            await caller.line(f"  {A.HG}Right.{A.RESET} The other one was "
                              f"generated - fluent, and about nothing.")
        else:
            wrong += 1
            await caller.line(f"  {A.HR}No.{A.RESET} The one you picked was "
                              f"generated. It reads better. That is the point.")
        await caller.line(f"  {A.GREY}score {right} right, {wrong} wrong"
                          f"{A.RESET}\n")

    if right or wrong:
        await caller.line(f"  final: {A.HG}{right}{A.RESET} right, "
                          f"{A.HR}{wrong}{A.RESET} wrong.")
        await caller.pause()
