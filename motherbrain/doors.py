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
    ("W", "WUMPUS", "Yob's dodecahedron, 1973. Five arrows."),
    ("L", "LUNAR", "Storer's lander, 1969. Under 5 km/h or a crater."),
    ("N", "ANIMAL", "1973. It learns the ones it gets wrong, and keeps them."),
    ("E", "ELIZA", "Weizenbaum, 1966. Compare it with the Oracle."),
    ("G", "GUESS", "The number is between 1 and 100. Eight tries."),
    ("Z", "THE MAZE", "Walk out of it. Drawn fresh every time."),
    ("R", "THE WYRM", "A daily-turn RPG. Forest, inn, armoury, dragon."),
    ("O", "THE ORACLE", "Ask MotherBrain anything. Watch what it does."),
    ("T", "TURING", "Two answers. One is honest. Which is which?"),
    ("P", "THE GALLERY", "Show it a picture and see what it makes of it."),
]

# Which doors need a model behind them. The rest run on nothing at all,
# which is what lets them ship as a release of their own.
NEEDS_MODEL = {"O", "T", "P"}


async def play(caller, key: str) -> bool:
    """Run the door named by `key`. Returns False if there is no such door."""
    door = {"H": hamurabi, "G": guess, "Z": maze, "W": wumpus, "L": lunar,
            "N": animal, "E": eliza, "R": wyrm,
            "O": oracle, "T": turing}.get(key.upper())
    if key.upper() == "P":
        from motherbrain.bbs import gallery

        await gallery(caller)
        return True
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


# ---- the classics -----------------------------------------------------------
#
# What a board actually ran. The BASIC canon - Creative Computing, `BASIC
# Computer Games`, the DECUS tapes - is where door games came from, and it
# is all public domain, so these are the real games rather than things in
# their shape. HUNT THE WUMPUS is Gregory Yob's, 1973. LUNAR LANDER is Jim
# Storer's, 1969. ANIMAL is from the DEC program library, 1973, and it
# learns from you and keeps what it learns, which on this board is worth
# more than the game is. ELIZA is Weizenbaum's, 1966, and it is here to be
# compared with the thing running the board.


async def wumpus(caller) -> None:
    """Hunt the Wumpus, Gregory Yob, 1973. The dodecahedron and all."""
    await _banner(caller, "H U N T   T H E   W U M P U S",
                  "Twenty rooms, three tunnels each, one wumpus.")

    rng = random.Random()
    # Yob's map: a dodecahedron, room by room, exactly as he numbered it.
    cave = [
        (1, 4, 7), (0, 2, 9), (1, 3, 11), (2, 4, 13), (0, 3, 5),
        (4, 6, 14), (5, 7, 16), (0, 6, 8), (7, 9, 17), (1, 8, 10),
        (9, 11, 18), (2, 10, 12), (11, 13, 19), (3, 12, 14), (5, 13, 15),
        (14, 16, 19), (6, 15, 17), (8, 16, 18), (10, 17, 19), (12, 15, 18),
    ]
    places = rng.sample(range(20), 6)
    you, wump, pit1, pit2, bat1, bat2 = places
    arrows = 5

    async def look() -> None:
        await caller.line("")
        await caller.line(f"  {A.HC}You are in room {you + 1}.{A.RESET}")
        for near in cave[you]:
            if near == wump:
                await caller.line(f"  {A.HR}You smell a wumpus.{A.RESET}")
            if near in (pit1, pit2):
                await caller.line(f"  {A.HY}You feel a draught.{A.RESET}")
            if near in (bat1, bat2):
                await caller.line(f"  {A.HM}You hear bats.{A.RESET}")
        tunnels = ", ".join(str(n + 1) for n in cave[you])
        await caller.line(f"  {A.GREY}Tunnels lead to {tunnels}."
                          f"  {arrows} arrow(s) left.{A.RESET}")

    while True:
        await look()
        move = (await caller.ask(f"  {A.HW}[S]hoot, [M]ove or [Q]uit? "
                                 f"{A.RESET}", limit=4)).strip().lower()
        if move.startswith("q"):
            return
        if move.startswith("s"):
            raw = await caller.ask(f"  {A.HW}Shoot into which room? {A.RESET}",
                                   limit=6)
            try:
                where = int(raw.strip()) - 1
            except ValueError:
                continue
            arrows -= 1
            if where == wump:
                await caller.line(f"\n  {A.HG}{A.BOLD}AHA! You got the "
                                  f"wumpus!{A.RESET}")
                await caller.pause()
                return
            await caller.line(f"  {A.GREY}Missed.{A.RESET}")
            # A missed arrow wakes it, and it moves. Yob's rule.
            wump = rng.choice(cave[wump] + (wump,))
            if wump == you:
                await caller.line(f"\n  {A.HR}The wumpus found you. "
                                  f"You lose.{A.RESET}")
                await caller.pause()
                return
            if arrows <= 0:
                await caller.line(f"\n  {A.HR}Out of arrows. You lose."
                                  f"{A.RESET}")
                await caller.pause()
                return
            continue

        raw = await caller.ask(f"  {A.HW}Move to which room? {A.RESET}",
                               limit=6)
        try:
            where = int(raw.strip()) - 1
        except ValueError:
            continue
        if where not in cave[you]:
            await caller.line(f"  {A.HR}No tunnel goes there.{A.RESET}")
            continue
        you = where
        if you == wump:
            await caller.line(f"\n  {A.HR}The wumpus eats you. You lose."
                              f"{A.RESET}")
            await caller.pause()
            return
        if you in (pit1, pit2):
            await caller.line(f"\n  {A.HR}You fall into a bottomless pit."
                              f"{A.RESET}")
            await caller.pause()
            return
        if you in (bat1, bat2):
            you = rng.randrange(20)
            await caller.line(f"  {A.HM}A giant bat carries you off to "
                              f"room {you + 1}.{A.RESET}")


async def lunar(caller) -> None:
    """Lunar Lander, Jim Storer, 1969. The physics is his, unchanged."""
    await _banner(caller, "L U N A R   L A N D E R",
                  "16,000 kg, 120 kg/s maximum burn. Land under 5 km/h.")

    altitude, velocity, fuel, mass = 120.0, 1.0, 1800.0, 16500.0
    gravity, exhaust = 0.001, 1.8
    clock = 0

    while altitude > 0:
        await caller.line("")
        bar = int(max(0, min(30, altitude / 4)))
        await caller.line(f"  {A.HB}{'│' * 1}{' ' * bar}{A.HY}▲{A.RESET}")
        await caller.line(
            f"  {A.HW}t={clock:>4}s   altitude {altitude:8.1f} m   "
            f"speed {velocity * 3600:7.1f} km/h   fuel {fuel:7.1f} kg"
            f"{A.RESET}")
        raw = await caller.ask(f"  {A.HY}burn, kg/s [0-120]: {A.RESET}",
                               limit=8)
        try:
            burn = max(0.0, min(120.0, float(raw.strip() or 0)))
        except ValueError:
            burn = 0.0
        if burn > fuel:
            burn = fuel
        fuel -= burn
        # Storer's step, ten seconds at a time.
        acceleration = gravity - burn * exhaust / mass
        altitude -= velocity * 10 + acceleration * 50
        velocity += acceleration * 10
        mass -= burn * 10 * 0.0
        clock += 10
        if fuel <= 0 and altitude > 0:
            await caller.line(f"  {A.HR}Out of fuel.{A.RESET}")

    speed = abs(velocity) * 3600
    await caller.line("")
    if speed < 5:
        await caller.line(f"  {A.HG}{A.BOLD}PERFECT LANDING at "
                          f"{speed:.1f} km/h.{A.RESET}")
    elif speed < 15:
        await caller.line(f"  {A.HY}Good landing - {speed:.1f} km/h. "
                          f"Some damage.{A.RESET}")
    else:
        await caller.line(f"  {A.HR}CRASH at {speed:.1f} km/h. A crater "
                          f"{speed / 8:.1f} m deep.{A.RESET}")
    await caller.pause()


async def animal(caller) -> None:
    """ANIMAL, 1973. It learns, it keeps what it learns, and it shows you.

    The only game here that genuinely gets better at something by being
    played, which on a board built around a model that cannot is worth
    saying out loud. Its whole knowledge is a binary tree of yes/no
    questions, and it is stored beside the board where you can read it.
    """
    import json

    await _banner(caller, "A N I M A L",
                  "Think of an animal. It learns the ones it gets wrong.")

    path = None
    tree = {"question": "Does it swim?",
            "yes": {"animal": "a fish"}, "no": {"animal": "a bird"}}
    board = getattr(caller, "board", None)
    if board is not None and getattr(board, "dir", None) is not None:
        path = board.dir / "animal.json"
        try:
            tree = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass

    def count(node) -> int:
        if "animal" in node:
            return 1
        return count(node["yes"]) + count(node["no"])

    await caller.line(f"  {A.GREY}It knows {count(tree)} animal(s) so far."
                      f"{A.RESET}\n")

    while True:
        node, parent, branch = tree, None, None
        while "animal" not in node:
            answer = (await caller.ask(
                f"  {A.HW}{node['question']} [y/n] {A.RESET}",
                limit=4)).strip().lower()
            parent, branch = node, "yes" if answer.startswith("y") else "no"
            node = node[branch]

        answer = (await caller.ask(
            f"  {A.HY}Is it {node['animal']}? [y/n] {A.RESET}",
            limit=4)).strip().lower()
        if answer.startswith("y"):
            await caller.line(f"  {A.HG}I knew it.{A.RESET}\n")
        else:
            what = (await caller.ask(
                f"  {A.HW}What was it? {A.RESET}", limit=40)).strip()
            if not what:
                return
            if not what.lower().startswith(("a ", "an ", "the ")):
                what = f"a {what}"
            question = (await caller.ask(
                f"  {A.HW}A question that is true of {what} but not of "
                f"{node['animal']}: {A.RESET}", limit=80)).strip()
            if not question:
                return
            if not question.endswith("?"):
                question += "?"
            replacement = {"question": question,
                           "yes": {"animal": what},
                           "no": {"animal": node["animal"]}}
            if parent is None:
                tree = replacement
            else:
                parent[branch] = replacement
            if path is not None:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(tree, indent=1),
                                    encoding="utf-8")
                except OSError:
                    pass
            await caller.line(f"  {A.HG}Learned. That is "
                              f"{count(tree)} animal(s), and it is written "
                              f"down.{A.RESET}\n")

        again = (await caller.ask(f"  {A.HW}Again? [Y/n] {A.RESET}",
                                  limit=4)).strip().lower()
        if again.startswith("n"):
            return


# ELIZA's script, from Weizenbaum's 1966 paper. The DOCTOR persona, cut to
# the rules that fire most often. It is here to stand next to the model:
# fifty lines of pattern matching against fifty-two million parameters, and
# people find this one more convincing.
_ELIZA_RULES = [
    (r"\bi need (.*)", ["Why do you need {0}?",
                        "Would it really help you to get {0}?"]),
    (r"\bwhy don'?t you (.*)", ["Do you really think I don't {0}?",
                                "Perhaps I will {0} in good time."]),
    (r"\bwhy can'?t i (.*)", ["Do you think you should be able to {0}?",
                              "What would it mean if you could {0}?"]),
    (r"\bi can'?t (.*)", ["How do you know you can't {0}?",
                          "Have you really tried?"]),
    (r"\bi am (.*)", ["Did you come to me because you are {0}?",
                      "How long have you been {0}?"]),
    (r"\bi'?m (.*)", ["How does being {0} make you feel?",
                      "Do you enjoy being {0}?"]),
    (r"\bare you (.*)", ["Why does it matter whether I am {0}?",
                         "Would you prefer it if I were not {0}?"]),
    (r"\bwhat (.*)", ["Why do you ask?", "What do you think?"]),
    (r"\bbecause (.*)", ["Is that the real reason?",
                         "What other reasons come to mind?"]),
    (r"\bi feel (.*)", ["Tell me more about feeling {0}.",
                        "Do you often feel {0}?"]),
    (r"\bi think (.*)", ["Do you doubt {0}?", "Do you really think so?"]),
    (r"\byes\b", ["You seem quite sure.", "I see."]),
    (r"\bno\b", ["Why not?", "Are you saying no just to be negative?"]),
    (r"\bmother|father|family\b",
     ["Tell me more about your family.",
      "How do you get along with your family?"]),
    (r"\bcomputer|model|machine\b",
     ["Do computers worry you?",
      "Why do you mention machines?",
      "What do you think machines have to do with your problem?"]),
    (r"\bsorry\b", ["There is no need to apologise.",
                    "Apologies are not necessary."]),
    (r"\bhello|hi\b", ["How do you do. Please tell me your problem."]),
]

_ELIZA_FALLBACK = [
    "Go on.", "Please tell me more.", "Can you elaborate on that?",
    "I see. And what does that suggest to you?", "How does that make you feel?",
    "Do you feel strongly about discussing such things?",
    "That is interesting. Please continue.",
]

_ELIZA_REFLECT = {
    "i": "you", "me": "you", "my": "your", "am": "are", "i'm": "you are",
    "mine": "yours", "myself": "yourself", "you": "I", "your": "my",
    "yours": "mine", "yourself": "myself", "are": "am",
}


def eliza_reply(said: str, rng: random.Random) -> str:
    """One DOCTOR response. Deterministic given the same rng, so it is testable."""
    import re

    lowered = said.lower().strip()
    for pattern, answers in _ELIZA_RULES:
        found = re.search(pattern, lowered)
        if not found:
            continue
        answer = rng.choice(answers)
        if found.groups():
            tail = _reflect(found.group(1)).rstrip(" .!?")
            return answer.format(tail)
        return answer
    return rng.choice(_ELIZA_FALLBACK)


def _reflect(text: str) -> str:
    """Swap the pronouns round, which is the entire trick."""
    return " ".join(_ELIZA_REFLECT.get(word, word) for word in text.split())


async def eliza(caller) -> None:
    """ELIZA, 1966. Fifty lines of patterns, next to fifty-two million weights."""
    await _banner(caller, "E L I Z A",
                  "Weizenbaum's DOCTOR, 1966. Compare it with the Oracle.")
    await caller.line(f"  {A.GREY}It matches patterns and turns your "
                      f"pronouns round. That is all it does.{A.RESET}")
    await caller.line(f"  {A.GREY}Weizenbaum was alarmed by how well it "
                      f"worked. Blank line to leave.{A.RESET}")
    await caller.line("")
    await caller.line(f"  {A.HC}ELIZA:{A.RESET} {A.HW}How do you do. "
                      f"Please tell me your problem.{A.RESET}")

    rng = random.Random()
    while True:
        said = (await caller.ask(f"\n  {A.HG}you:{A.RESET} ",
                                 limit=300)).strip()
        if not said:
            await caller.line(f"  {A.HC}ELIZA:{A.RESET} {A.HW}Goodbye. "
                              f"This was a good talk.{A.RESET}")
            await caller.pause()
            return
        await caller.line(f"  {A.HC}ELIZA:{A.RESET} "
                          f"{A.HW}{eliza_reply(said, rng)}{A.RESET}")


# ---- THE WYRM ---------------------------------------------------------------
#
# You asked for Legend of the Red Dragon. LORD is Seth Able Robinson's, it
# is still his, and copying its text, its characters or its structure would
# be taking somebody's work. What is not his is the genre he perfected: a
# daily-turn BBS role-playing game, where you get so many forest fights a
# day, spend the gold in a town, fight the other callers, and eventually go
# up the mountain after the dragon. That shape is what a board wanted, and
# this is an original game built in it - own name, own writing, own
# numbers, and the same daily rhythm.

WYRM_FOES = [
    (1, "a moss-backed toad", 6, 3, 8),
    (1, "a hedge-thief with a bad knife", 8, 4, 12),
    (2, "a starving wolf", 14, 6, 20),
    (2, "a tinker's cursed kettle", 16, 5, 26),
    (3, "a bog-lurker", 26, 9, 45),
    (3, "the miller's dead son", 30, 11, 60),
    (4, "a knight who does not know he lost", 48, 15, 110),
    (4, "something with too many arms", 55, 17, 140),
    (5, "the wyrm's herald", 80, 24, 240),
    (5, "a hollow king", 95, 27, 300),
]

WYRM_WEAPONS = [
    ("a sharpened fence post", 3, 0),
    ("a hand axe", 6, 200),
    ("a soldier's sword", 11, 900),
    ("a bill-hook, oiled", 18, 2600),
    ("the long knife of the drowned", 27, 7000),
    ("a blade with a name nobody says", 40, 18000),
]

WYRM_ARMOUR = [
    ("your own coat", 1, 0),
    ("a padded jerkin", 4, 150),
    ("boiled leather", 9, 750),
    ("a mail shirt", 15, 2200),
    ("plate, ill-fitting", 24, 6500),
    ("scale cut from something large", 36, 16000),
]

WYRM_TURNS = 15


def _wyrm_new(name: str) -> dict:
    return {"name": name, "level": 1, "hp": 30, "max_hp": 30,
            "weapon": 0, "armour": 0, "gold": 0, "bank": 0, "xp": 0,
            "turns": WYRM_TURNS, "day": "", "alive": True, "kills": 0,
            "slain_wyrm": False, "died": 0}


def _wyrm_store(board):
    import json

    if board is None or getattr(board, "dir", None) is None:
        return None, {}
    path = board.dir / "wyrm.json"
    try:
        return path, json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return path, {}


def _wyrm_save(path, players) -> None:
    import json

    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(players, indent=1), encoding="utf-8")
    except OSError:
        pass


async def wyrm(caller) -> None:
    """A daily-turn RPG in the shape every board had one in."""
    import time as _time

    board = getattr(caller, "board", None)
    path, players = _wyrm_store(board)
    who = caller.handle or "wanderer"
    me = players.get(who) or _wyrm_new(who)
    players[who] = me

    today = _time.strftime("%Y-%m-%d")
    if me["day"] != today:
        me["day"], me["turns"], me["alive"] = today, WYRM_TURNS, True
        me["hp"] = me["max_hp"]

    rng = random.Random()

    while True:
        await caller.cls()
        await caller.art(A.gradient("  T H E   W Y R M  ",
                                    (A.HR, A.R, A.HY, A.Y)))
        await caller.line(f"  {A.GREY}A daily-turn game. Come back "
                          f"tomorrow and the forest is full again."
                          f"{A.RESET}")
        await caller.line("")
        await _wyrm_status(caller, me)
        await caller.line("")
        await caller.art("\n".join(A.box("THE VILLAGE", [
            f"  {A.HY}[F]{A.RESET} {A.HW}the forest{A.RESET}"
            f"{A.GREY}      fight something. {me['turns']} turn(s) left"
            f"{A.RESET}",
            f"  {A.HY}[W]{A.RESET} {A.HW}the armoury{A.RESET}"
            f"{A.GREY}     buy a weapon or armour{A.RESET}",
            f"  {A.HY}[I]{A.RESET} {A.HW}the inn{A.RESET}"
            f"{A.GREY}         sleep it off, 20 gold{A.RESET}",
            f"  {A.HY}[B]{A.RESET} {A.HW}the strongbox{A.RESET}"
            f"{A.GREY}   gold you keep when you die{A.RESET}",
            f"  {A.HY}[C]{A.RESET} {A.HW}challenge{A.RESET}"
            f"{A.GREY}       fight another caller's record{A.RESET}",
            f"  {A.HY}[M]{A.RESET} {A.HW}the mountain{A.RESET}"
            f"{A.GREY}    go up after the wyrm{A.RESET}",
            f"  {A.HY}[S]{A.RESET} {A.HW}the board{A.RESET}"
            f"{A.GREY}       who has done what{A.RESET}",
            f"  {A.HY}[Q]{A.RESET} {A.HW}leave{A.RESET}",
        ], width=min(caller.columns, 74), frame=A.HR)))

        choice = (await caller.ask(f"  {A.HY}> {A.RESET}",
                                   limit=2)).strip().upper()
        if not choice or choice == "Q":
            _wyrm_save(path, players)
            return
        if choice == "F":
            await _wyrm_forest(caller, me, rng)
        elif choice == "W":
            await _wyrm_armoury(caller, me)
        elif choice == "I":
            await _wyrm_inn(caller, me)
        elif choice == "B":
            await _wyrm_bank(caller, me)
        elif choice == "C":
            await _wyrm_duel(caller, me, players, rng)
        elif choice == "M":
            await _wyrm_mountain(caller, me, rng)
        elif choice == "S":
            await _wyrm_board(caller, players)
        _wyrm_save(path, players)


async def _wyrm_status(caller, me: dict) -> None:
    weapon = WYRM_WEAPONS[me["weapon"]]
    armour = WYRM_ARMOUR[me["armour"]]
    bar = int(20 * me["hp"] / max(1, me["max_hp"]))
    await caller.line(
        f"  {A.HW}{me['name']}{A.RESET}  {A.GREY}level{A.RESET} "
        f"{A.HY}{me['level']}{A.RESET}   "
        f"{A.HR}{'█' * bar}{A.GREY}{'·' * (20 - bar)}{A.RESET} "
        f"{me['hp']}/{me['max_hp']}")
    await caller.line(
        f"  {A.GREY}carrying{A.RESET} {A.HW}{weapon[0]}{A.RESET} "
        f"{A.GREY}and wearing{A.RESET} {A.HW}{armour[0]}{A.RESET}")
    await caller.line(
        f"  {A.GREY}gold{A.RESET} {A.HY}{me['gold']:,}{A.RESET}   "
        f"{A.GREY}in the strongbox{A.RESET} {A.HY}{me['bank']:,}{A.RESET}   "
        f"{A.GREY}experience{A.RESET} {A.HY}{me['xp']:,}{A.RESET}   "
        f"{A.GREY}kills{A.RESET} {A.HY}{me['kills']}{A.RESET}")


def _wyrm_hit(me: dict, rng) -> int:
    return WYRM_WEAPONS[me["weapon"]][1] + rng.randint(1, 4) + me["level"]


async def _wyrm_fight(caller, me: dict, foe: tuple, rng, prize: int,
                      xp: int) -> bool:
    """One fight. True if you walked away from it."""
    _, name, hp, damage, _ = foe
    guard = WYRM_ARMOUR[me["armour"]][1]

    await caller.line("")
    await caller.line(f"  {A.HR}You meet {name}.{A.RESET}")
    while hp > 0 and me["hp"] > 0:
        action = (await caller.ask(
            f"  {A.HW}[A]ttack, [R]un? {A.RESET}", limit=2)).strip().lower()
        if action.startswith("r"):
            if rng.random() < 0.55:
                await caller.line(f"  {A.HY}You get away.{A.RESET}")
                await caller.pause()
                return True
            await caller.line(f"  {A.HR}It is faster than you.{A.RESET}")
        else:
            mine = _wyrm_hit(me, rng)
            hp -= mine
            await caller.line(f"  {A.HG}You hit for {mine}.{A.RESET}"
                              + (f"  {A.GREY}It has {max(0, hp)} left."
                                 f"{A.RESET}" if hp > 0 else ""))
            if hp <= 0:
                break
        theirs = max(1, damage + rng.randint(-2, 3) - guard // 2)
        me["hp"] -= theirs
        await caller.line(f"  {A.HR}It hits you for {theirs}.{A.RESET}"
                          f"  {A.GREY}{max(0, me['hp'])} left.{A.RESET}")

    if me["hp"] <= 0:
        me["hp"], me["alive"], me["died"] = 0, False, me["died"] + 1
        lost, me["gold"] = me["gold"], 0
        await caller.line("")
        await caller.line(f"  {A.HR}{A.BOLD}You are killed by {name}."
                          f"{A.RESET}")
        await caller.line(f"  {A.GREY}You lose {lost:,} gold. What is in "
                          f"the strongbox is safe.{A.RESET}")
        await caller.line(f"  {A.GREY}Come back tomorrow.{A.RESET}")
        await caller.pause()
        return False

    me["gold"] += prize
    me["xp"] += xp
    me["kills"] += 1
    await caller.line("")
    await caller.line(f"  {A.HG}{name} is dead.{A.RESET} "
                      f"{A.HY}+{prize} gold, +{xp} experience.{A.RESET}")
    threshold = me["level"] * 120
    if me["xp"] >= threshold:
        me["level"] += 1
        me["max_hp"] += 14
        me["hp"] = me["max_hp"]
        await caller.line(f"  {A.HC}{A.BOLD}You are level {me['level']}."
                          f"{A.RESET} {A.GREY}Wounds closed.{A.RESET}")
    await caller.pause()
    return True


async def _wyrm_forest(caller, me: dict, rng) -> None:
    if not me["alive"]:
        await caller.line(f"  {A.HR}You are dead until tomorrow.{A.RESET}")
        await caller.pause()
        return
    if me["turns"] <= 0:
        await caller.line(f"  {A.HY}No turns left today. The forest is "
                          f"quiet.{A.RESET}")
        await caller.pause()
        return
    me["turns"] -= 1
    band = [f for f in WYRM_FOES if abs(f[0] - me["level"]) <= 1] or WYRM_FOES
    foe = rng.choice(band)
    await _wyrm_fight(caller, me, foe, rng, foe[4] + rng.randint(0, 10),
                      foe[4] // 2 + 5)


async def _wyrm_armoury(caller, me: dict) -> None:
    while True:
        await caller.cls()
        rows = []
        for i, (name, power, price) in enumerate(WYRM_WEAPONS):
            mark = "*" if i == me["weapon"] else " "
            rows.append(f"  {A.HY}{mark}W{i}{A.RESET} {A.HW}{A.pad(name, 34)}"
                        f"{A.RESET}{A.GREY}hit {power:>3}   "
                        f"{price:>7,}g{A.RESET}")
        rows.append("")
        for i, (name, guard, price) in enumerate(WYRM_ARMOUR):
            mark = "*" if i == me["armour"] else " "
            rows.append(f"  {A.HY}{mark}A{i}{A.RESET} {A.HW}{A.pad(name, 34)}"
                        f"{A.RESET}{A.GREY}gd  {guard:>3}   "
                        f"{price:>7,}g{A.RESET}")
        await caller.art("\n".join(A.box(
            f"THE ARMOURY   you have {me['gold']:,} gold", rows,
            width=min(caller.columns, 74), frame=A.HY)))
        pick = (await caller.ask(f"  {A.HY}buy (e.g. W2), or [Q]: {A.RESET}",
                                 limit=4)).strip().upper()
        if not pick or pick == "Q":
            return
        table, index = (WYRM_WEAPONS, "weapon") if pick.startswith("W") \
            else (WYRM_ARMOUR, "armour")
        try:
            which = int(pick[1:])
            name, _, price = table[which]
        except (ValueError, IndexError):
            continue
        if which <= me[index]:
            await caller.line(f"  {A.GREY}You have better.{A.RESET}")
        elif me["gold"] < price:
            await caller.line(f"  {A.HR}You cannot afford {name}."
                              f"{A.RESET}")
        else:
            me["gold"] -= price
            me[index] = which
            await caller.line(f"  {A.HG}You take {name}.{A.RESET}")
        await caller.pause()


async def _wyrm_inn(caller, me: dict) -> None:
    if me["gold"] < 20:
        await caller.line(f"  {A.HR}Twenty gold. You have {me['gold']}."
                          f"{A.RESET}")
    elif me["hp"] >= me["max_hp"]:
        await caller.line(f"  {A.GREY}Nothing wrong with you.{A.RESET}")
    else:
        me["gold"] -= 20
        me["hp"] = me["max_hp"]
        await caller.line(f"  {A.HG}You sleep. The innkeeper says nothing "
                          f"about the blood.{A.RESET}")
    await caller.pause()


async def _wyrm_bank(caller, me: dict) -> None:
    await caller.line(f"  {A.GREY}On you {me['gold']:,}, in the box "
                      f"{me['bank']:,}.{A.RESET}")
    raw = (await caller.ask(f"  {A.HW}deposit how much? "
                            f"(negative to take out) {A.RESET}",
                            limit=12)).strip()
    try:
        amount = int(raw or 0)
    except ValueError:
        return
    if amount > 0:
        amount = min(amount, me["gold"])
        me["gold"] -= amount
        me["bank"] += amount
    else:
        amount = min(-amount, me["bank"])
        me["bank"] -= amount
        me["gold"] += amount
    await caller.line(f"  {A.HG}On you {me['gold']:,}, in the box "
                      f"{me['bank']:,}.{A.RESET}")
    await caller.pause()


async def _wyrm_duel(caller, me: dict, players: dict, rng) -> None:
    """Fight another caller's record, the way a board's RPG always did."""
    others = [p for n, p in players.items() if n != me["name"]]
    if not others:
        await caller.line(f"  {A.GREY}Nobody else has played yet."
                          f"{A.RESET}")
        await caller.pause()
        return
    for i, other in enumerate(others[:9], start=1):
        await caller.line(f"  {A.HY}{i}{A.RESET} {A.HW}"
                          f"{A.pad(other['name'], 20)}{A.RESET}"
                          f"{A.GREY}level {other['level']}, "
                          f"{other['kills']} kills{A.RESET}")
    pick = (await caller.ask(f"  {A.HY}challenge which? {A.RESET}",
                             limit=4)).strip()
    if not pick.isdigit() or not 1 <= int(pick) <= min(9, len(others)):
        return
    them = others[int(pick) - 1]
    foe = (them["level"], f"{them['name']}, who is not pleased to see you",
           them["max_hp"], WYRM_WEAPONS[them["weapon"]][1] + them["level"],
           them["gold"] // 2 + 20)
    if await _wyrm_fight(caller, me, foe, rng, them["gold"] // 2 + 20,
                         them["level"] * 40):
        stolen = them["gold"] // 2
        them["gold"] -= stolen
        await caller.line(f"  {A.HY}You take {stolen:,} gold off them."
                          f"{A.RESET}")
        await caller.pause()


async def _wyrm_mountain(caller, me: dict, rng) -> None:
    if me["level"] < 5:
        await caller.line(f"  {A.HR}You would not last the climb. Level 5."
                          f"{A.RESET}")
        await caller.pause()
        return
    if not me["alive"] or me["turns"] <= 0:
        await caller.line(f"  {A.HY}Not today.{A.RESET}")
        await caller.pause()
        return
    me["turns"] -= 1
    await caller.cls()
    await caller.line(f"  {A.HR}{A.BOLD}The wyrm is awake, and it was "
                      f"expecting somebody.{A.RESET}")
    foe = (6, "THE WYRM", 220, 34, 5000)
    if await _wyrm_fight(caller, me, foe, rng, 5000, 2000):
        me["slain_wyrm"] = True
        await caller.cls()
        await caller.art(A.gradient("  THE WYRM IS DEAD  ",
                                    (A.HY, A.HR, A.HY, A.HW)))
        await caller.line("")
        await caller.line(f"  {A.HW}You go back down with your arms full "
                          f"and nothing to say.{A.RESET}")
        await caller.line(f"  {A.GREY}The board remembers. Everyone will "
                          f"see it on the list.{A.RESET}")
        await caller.pause()


async def _wyrm_board(caller, players: dict) -> None:
    ranked = sorted(players.values(),
                    key=lambda p: (p["slain_wyrm"], p["level"], p["xp"]),
                    reverse=True)
    rows = []
    for i, player in enumerate(ranked[:15], start=1):
        crown = f"{A.HY}†" if player["slain_wyrm"] else " "
        rows.append(f"  {A.HY}{i:>2}{A.RESET} {crown} "
                    f"{A.HW}{A.pad(player['name'], 20)}{A.RESET}"
                    f"{A.GREY}level {player['level']:<4}"
                    f"{player['kills']:>4} kills   "
                    f"{player['bank'] + player['gold']:>8,}g{A.RESET}")
    await caller.art("\n".join(A.box("THE BOARD", rows or [
        f"  {A.GREY}nobody yet.{A.RESET}"],
        width=min(caller.columns, 74), frame=A.HR)))
    await caller.line(f"  {A.GREY}† has been up the mountain and come back."
                      f"{A.RESET}")
    await caller.pause()
