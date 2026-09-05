"""A windowed MotherBrain, offering the same four options as everything else.

Tkinter, and only Tkinter. It ships with Python on Windows and macOS and is one
apt package away on Linux, which means the desktop application needs no
install of its own - the point of a GUI is that somebody who will not open a
terminal can still use this, and telling them to pip install a toolkit first
would defeat it.

Everything slow happens on a worker thread: loading 47 million parameters takes
seconds, generation takes longer, and training a patch takes longer still. Tk
is not thread-safe, so workers never touch a widget. They put closures on a
queue and the main loop drains it, which is the only safe direction.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path

# The four options, in the order they are offered everywhere else.
OPTIONS = [
    ("1  Tell MotherBrain what kind of program to make",
     "Describe a program. It writes one, you save it, and you decide whether "
     "to run it."),
    ("2  Tell MotherBrain what to do",
     "An instruction: make, run, write, find, delete, list, or a shell "
     "command. Anything else, the model continues."),
    ("3  Teach MotherBrain something new",
     "Type or choose a file. It goes into the corpus, waiting to be learned."),
    ("4  Apply new knowledge as a patch (update)",
     "Learns everything fed since the last version, adds parameters, and "
     "ascends to the next version."),
]

MISSING_TK = """MotherBrain's window needs Tkinter, which is not installed here.

  Debian, Ubuntu, Kali   sudo apt install python3-tk
  Fedora                 sudo dnf install python3-tkinter
  Arch                   sudo pacman -S tk
  Windows, macOS         reinstall Python and tick "tcl/tk"

Nothing else is missing - the model is fine. Until then:

  mb console     the same four options, in this terminal
  mb serve       then open http://127.0.0.1:8000 in a browser
"""


class Bridge:
    """Passes results from worker threads back to the Tk main loop."""

    def __init__(self, root) -> None:
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.root.after(40, self._pump)

    def post(self, fn, *a) -> None:
        """Call fn(*a) on the main thread, soon."""
        self.q.put((fn, a))

    def _pump(self) -> None:
        while True:
            try:
                fn, a = self.q.get_nowait()
            except queue.Empty:
                break
            try:
                fn(*a)
            except Exception as exc:                      # noqa: BLE001
                # A failed update must not kill the pump, or the window
                # freezes with no explanation.
                print(f"gui: {exc}", file=sys.stderr)
        self.root.after(40, self._pump)


# A dark palette, defined once. Tk's defaults are a grey from 1994, and a
# window somebody is meant to sit in front of for an hour should not look like
# an error dialog.
PALETTE = {
    "bg":      "#0f1115",   # the window itself
    "panel":   "#181b22",   # cards, transcript, input
    "raised":  "#1f2430",   # a card under the pointer
    "line":    "#2a3040",   # borders
    "text":    "#e6e9ef",
    "muted":   "#8b93a5",
    "accent":  "#5eead4",   # the selected option, and MotherBrain's own voice
    "you":     "#7dd3fc",   # what you said
    "warn":    "#fbbf24",
    "bad":     "#f87171",
}


class Card:
    """One of the four options, drawn by hand.

    tk.Button ignores background colour on macOS, so a themed window built
    from real buttons is grey on one platform and dark on the others. A frame
    with labels inside behaves the same everywhere, which matters more here
    than the few lines it costs.
    """

    def __init__(self, parent, tk, index: int, title: str, hint: str,
                 command) -> None:
        self.tk = tk
        self.command = command
        self.enabled = False
        self.selected = False

        self.frame = tk.Frame(parent, bg=PALETTE["panel"],
                              highlightbackground=PALETTE["line"],
                              highlightthickness=1, cursor="hand2")
        self.frame.pack(fill="x", pady=3)

        inner = tk.Frame(self.frame, bg=PALETTE["panel"], padx=14, pady=10)
        inner.pack(fill="x")

        self.number = tk.Label(inner, text=str(index + 1), bg=PALETTE["panel"],
                               fg=PALETTE["muted"],
                               font=("TkDefaultFont", 15, "bold"), width=2)
        self.number.pack(side="left", padx=(0, 12))

        stack = tk.Frame(inner, bg=PALETTE["panel"])
        stack.pack(side="left", fill="x", expand=True)
        self.title = tk.Label(stack, text=title, bg=PALETTE["panel"],
                              fg=PALETTE["muted"], anchor="w",
                              font=("TkDefaultFont", 11, "bold"))
        self.title.pack(fill="x")
        self.hint = tk.Label(stack, text=hint, bg=PALETTE["panel"],
                             fg=PALETTE["muted"], anchor="w", justify="left",
                             wraplength=620, font=("TkDefaultFont", 9))
        self.hint.pack(fill="x", pady=(2, 0))

        for widget in (self.frame, inner, stack, self.number, self.title,
                       self.hint):
            widget.bind("<Button-1>", self._click)
            widget.bind("<Enter>", self._enter)
            widget.bind("<Leave>", self._leave)

    # The tests and the smoke script read a card the way they read a button.
    def cget(self, key: str):
        return self.title.cget("text") if key == "text" else ""

    def config(self, **kw) -> None:
        if "state" in kw:
            self.enabled = kw["state"] == "normal"
            self._paint()

    def _click(self, _event=None):
        if self.enabled:
            self.command()

    def _enter(self, _event=None):
        if self.enabled and not self.selected:
            self._fill(PALETTE["raised"])

    def _leave(self, _event=None):
        if not self.selected:
            self._fill(PALETTE["panel"])

    def _fill(self, colour: str) -> None:
        self.frame.config(bg=colour)
        for widget in self.frame.winfo_children():
            widget.config(bg=colour)
            for child in widget.winfo_children():
                child.config(bg=colour)
                for grandchild in child.winfo_children():
                    grandchild.config(bg=colour)

    def select(self, on: bool) -> None:
        self.selected = on
        self._paint()

    def _paint(self) -> None:
        self._fill(PALETTE["raised"] if self.selected else PALETTE["panel"])
        self.frame.config(highlightbackground=(
            PALETTE["accent"] if self.selected else PALETTE["line"]))
        if not self.enabled:
            self.title.config(fg=PALETTE["muted"])
            self.number.config(fg=PALETTE["line"])
            return
        self.title.config(fg=PALETTE["text"] if self.selected
                          else PALETTE["text"])
        self.number.config(fg=PALETTE["accent"] if self.selected
                           else PALETTE["muted"])


class Chip:
    """A small flat button - Send, Speak, Image - in the same style."""

    def __init__(self, parent, tk, text: str, command, accent: bool = False):
        self.tk = tk
        self.command = command
        self.enabled = False
        self.accent = accent
        self.label = tk.Label(parent, text=text, bg=PALETTE["panel"],
                              fg=PALETTE["muted"], padx=14, pady=7,
                              cursor="hand2", font=("TkDefaultFont", 10),
                              highlightbackground=PALETTE["line"],
                              highlightthickness=1)
        self.label.pack(fill="x", pady=2)
        self.label.bind("<Button-1>", lambda e: self.enabled and self.command())
        self.label.bind("<Enter>", lambda e: self.enabled and
                        self.label.config(bg=PALETTE["raised"]))
        self.label.bind("<Leave>", lambda e: self.label.config(bg=PALETTE["panel"]))

    def config(self, **kw) -> None:
        if "text" in kw:
            self.label.config(text=kw["text"])
        if "state" in kw:
            self.enabled = kw["state"] == "normal"
            self.label.config(fg=(PALETTE["accent"] if self.accent else
                                  PALETTE["text"]) if self.enabled
                              else PALETTE["muted"])

    def cget(self, key: str):
        return self.label.cget(key)


class App:
    def __init__(self, root, run_dir: str, corpus_dir: str, device: str,
                 max_tokens: int, steps: int, grow: int) -> None:
        import tkinter as tk
        from tkinter import scrolledtext

        self.tk = tk
        self.root = root
        self.run_dir, self.corpus_dir, self.device = run_dir, corpus_dir, device
        self.max_tokens, self.steps, self.grow = max_tokens, steps, grow
        self.bridge = Bridge(root)

        self.model = self.tok = self.dev = None
        self.version = 0
        self.mode = "do"          # which option is armed
        self.image_path: str | None = None
        self.last_code: str | None = None
        self.last_want: str = ""
        self.busy = False
        self.cap = None

        P = PALETTE
        root.title("MotherBrain")
        root.geometry("960x860")
        root.minsize(720, 560)
        root.configure(bg=P["bg"])

        head = tk.Frame(root, bg=P["bg"], padx=18, pady=14)
        head.pack(fill="x")
        tk.Label(head, text="MotherBrain", bg=P["bg"], fg=P["text"],
                 font=("TkDefaultFont", 17, "bold"), anchor="w").pack(anchor="w")
        self.header = tk.Label(head, text="loading …", bg=P["bg"],
                               fg=P["accent"], anchor="w",
                               font=("TkDefaultFont", 10))
        self.header.pack(anchor="w", pady=(2, 0))

        # The same stats block the terminal prints, in the same words. A number
        # that disagrees with itself between two windows is worse than none.
        #
        # It folds away, because ten lines of reference material at the top of
        # the window leaves the transcript - the part you actually read - as a
        # sliver on a laptop screen.
        # Folded by default: the header line above already carries version,
        # size, device and whether it can see, and the transcript is what the
        # window is for. Ctrl+B, or the fold itself, opens the detail.
        self.stats_open = False
        self.stats_wrap = tk.Frame(root, bg=P["panel"],
                                   highlightbackground=P["line"],
                                   highlightthickness=1)
        self.stats_wrap.pack(fill="x", padx=18, pady=(0, 10))
        self.stats_toggle = tk.Label(self.stats_wrap, text="▾  stats",
                                     bg=P["panel"], fg=P["muted"], anchor="w",
                                     padx=14, pady=6, cursor="hand2",
                                     font=("TkDefaultFont", 9))
        self.stats_toggle.pack(fill="x")
        self.stats_toggle.bind("<Button-1>", lambda e: self.toggle_stats())
        self.stats = tk.Label(self.stats_wrap, text="", font=("TkFixedFont", 9),
                              bg=P["panel"], fg=P["muted"], anchor="w",
                              justify="left", padx=14)
        self.stats_toggle.config(text="▸  stats")

        cards = tk.Frame(root, bg=P["bg"], padx=18)
        cards.pack(fill="x")
        self.buttons = []
        for i, (label, hint) in enumerate(OPTIONS):
            # The number is drawn separately, so strip the one in the text.
            title = label.split("  ", 1)[-1]
            card = Card(cards, tk, i, title, hint, lambda n=i: self.choose(n))
            self.buttons.append(card)

        # Each card already carries its own description, so this line says what
        # you can type instead of repeating it.
        self.hint = tk.Label(root, text="", bg=P["bg"], fg=P["muted"],
                             anchor="w", padx=18, pady=6, wraplength=880,
                             justify="left", font=("TkFixedFont", 9))
        self.hint.pack(fill="x")

        # Packed bottom-up and before the transcript: these reserve their space
        # first, so shrinking the window shrinks the transcript rather than
        # squeezing the input box down to a single line.
        self.status = tk.Label(root, text="", bg=P["panel"], fg=P["muted"],
                               anchor="w", padx=18, pady=6,
                               font=("TkDefaultFont", 9))
        self.status.pack(fill="x", side="bottom")

        self.extra = tk.Frame(root, bg=P["bg"], padx=18)
        self.extra.pack(fill="x", side="bottom", pady=(0, 8))

        row = tk.Frame(root, bg=P["bg"], padx=18)
        row.pack(fill="x", side="bottom", pady=(4, 6))

        self.view = scrolledtext.ScrolledText(
            root, wrap="word", height=15, font=("TkFixedFont", 10),
            state="disabled", padx=14, pady=12, relief="flat",
            bg=P["panel"], fg=P["text"], insertbackground=P["text"],
            selectbackground=P["line"], highlightthickness=1,
            highlightbackground=P["line"])
        self.view.pack(fill="both", expand=True, padx=18, pady=(0, 10))
        self.view.tag_configure("you", foreground=P["you"])
        self.view.tag_configure("mb", foreground=P["text"])
        self.view.tag_configure("note", foreground=P["muted"])
        self.view.tag_configure("bad", foreground=P["bad"])

        self.entry = tk.Text(row, height=3, wrap="word", relief="flat",
                             font=("TkDefaultFont", 11), bg=P["panel"],
                             fg=P["text"], insertbackground=P["accent"],
                             padx=12, pady=10, highlightthickness=1,
                             highlightbackground=P["line"],
                             highlightcolor=P["accent"])
        self.entry.pack(side="left", fill="both", expand=True)
        self.entry.bind("<Return>", self._on_return)
        self.entry.bind("<Shift-Return>", lambda e: None)

        side = tk.Frame(row, bg=P["bg"], padx=8)
        side.pack(side="right", fill="y")
        self.send_btn = Chip(side, tk, "Send", self.submit, accent=True)
        self.voice_btn = Chip(side, tk, "Speak", self.listen)
        self.image_btn = Chip(side, tk, "Image", self.pick_image)

        self.build_menus()
        self.choose(1)
        threading.Thread(target=self._load, daemon=True).start()

    # ---- menus ------------------------------------------------------------

    def build_menus(self) -> None:
        """A menu bar reaching everything the other two consoles can reach.

        The four buttons are the front door, but a window with no menus leaves
        `mb`'s other commands - checking out an old version, exporting, pricing
        a configuration - reachable only by typing them into option 2, which
        nobody discovers. Every entry here calls the same method the buttons
        and the typed commands do, so there is one implementation per action.
        """
        tk = self.tk
        # X11 draws menus from the system palette, which is a light grey that
        # sits badly above a dark window. macOS ignores these and uses its own
        # menu bar, which is correct there.
        style = dict(bg=PALETTE["panel"], fg=PALETTE["text"],
                     activebackground=PALETTE["accent"],
                     activeforeground=PALETTE["bg"],
                     borderwidth=0, relief="flat")
        bar = tk.Menu(self.root, **style)

        model = tk.Menu(bar, tearoff=0, **style)
        model.add_command(label="Status", accelerator="Ctrl+I",
                          command=lambda: self.dispatch("/status"))
        model.add_command(label="Versions", accelerator="Ctrl+L",
                          command=lambda: self.dispatch("/versions"))
        model.add_separator()
        model.add_command(label="Apply new knowledge as a patch…",
                          accelerator="Ctrl+U", command=self.apply_patch)
        model.add_command(label="Give it sight…", command=self.add_sight)
        model.add_command(label="Check out an earlier version…",
                          command=self.checkout_version)
        model.add_separator()
        model.add_command(label="Export the model…", accelerator="Ctrl+E",
                          command=self.export_model)
        model.add_command(label="Copy a workspace to a drive…",
                          command=self.copy_workspace)
        model.add_separator()
        model.add_command(label="Quit", accelerator="Ctrl+Q",
                          command=self.root.destroy)
        bar.add_cascade(label="Model", menu=model)

        do = tk.Menu(bar, tearoff=0, **style)
        for i, (label, _hint) in enumerate(OPTIONS):
            do.add_command(label=label, accelerator=f"Ctrl+{i + 1}",
                           command=lambda n=i: self.choose(n))
        bar.add_cascade(label="Do", menu=do)

        view = tk.Menu(bar, tearoff=0, **style)
        view.add_command(label="Show or hide stats", accelerator="Ctrl+B",
                         command=self.toggle_stats)
        view.add_command(label="Refresh stats", accelerator="Ctrl+R",
                         command=self.refresh_stats)
        view.add_command(label="Clear transcript", accelerator="Ctrl+K",
                         command=self._clear_view)
        view.add_separator()
        view.add_command(label="Look at an image…", command=self.pick_image)
        view.add_command(label="Forget the image",
                         command=self.forget_image)
        bar.add_cascade(label="View", menu=view)

        tools = tk.Menu(bar, tearoff=0, **style)
        tools.add_command(label="Price a configuration…",
                          command=self.scale_dialog)
        tools.add_command(label="Run a shell command…",
                          command=lambda: self.prompt_then("Shell command:",
                                                           "sh "))
        tools.add_command(label="Search these files…",
                          command=lambda: self.prompt_then("Search for:",
                                                           "find "))
        bar.add_cascade(label="Tools", menu=tools)

        helpmenu = tk.Menu(bar, tearoff=0, **style)
        helpmenu.add_command(label="What can I say?", accelerator="F1",
                             command=lambda: self.dispatch("/help"))
        helpmenu.add_command(label="About MotherBrain", command=self.about)
        bar.add_cascade(label="Help", menu=helpmenu)

        self.root.config(menu=bar)
        self.menubar = bar

        for key, fn in (
                ("<Control-i>", lambda e: self.dispatch("/status")),
                ("<Control-l>", lambda e: self.dispatch("/versions")),
                ("<Control-u>", lambda e: self.apply_patch()),
                ("<Control-e>", lambda e: self.export_model()),
                ("<Control-b>", lambda e: self.toggle_stats()),
                ("<Control-r>", lambda e: self.refresh_stats()),
                ("<Control-k>", lambda e: self._clear_view()),
                ("<Control-q>", lambda e: self.root.destroy()),
                ("<F1>", lambda e: self.dispatch("/help")),
                ("<Control-Key-1>", lambda e: self.choose(0)),
                ("<Control-Key-2>", lambda e: self.choose(1)),
                ("<Control-Key-3>", lambda e: self.choose(2)),
                ("<Control-Key-4>", lambda e: self.choose(3)),
        ):
            self.root.bind_all(key, fn)

    def dispatch(self, text: str) -> None:
        """Run a command exactly as if it had been typed into option 2.

        Menu entries go through the same parser as typing, so a menu can never
        drift from what the command does.
        """
        if self.busy or self.model is None:
            return
        self.write(f"\n> {text}\n", "you")
        self.run_worker(self._do, text)

    def prompt_then(self, question: str, prefix: str) -> None:
        """Ask for one line, then run it as a command."""
        from tkinter import simpledialog

        answer = simpledialog.askstring("MotherBrain", question, parent=self.root)
        if answer:
            self.dispatch(prefix + answer)

    def about(self) -> None:
        from tkinter import messagebox

        from motherbrain.cli import human

        if self.model is None:
            messagebox.showinfo("MotherBrain", "No model is loaded.")
            return
        sight = ("sees images" if getattr(self.model, "vision", None) is not None
                 else "text only")
        messagebox.showinfo(
            "About MotherBrain",
            f"MotherBrain v{self.version}\n"
            f"{self.model.n_params():,} parameters ({human(self.model.n_params())})\n"
            f"{sight}, running on {self.dev}\n\n"
            f"run   {self.run_dir}\n"
            f"corpus {self.corpus_dir}")

    # ---- loading ----------------------------------------------------------

    def _load(self) -> None:
        try:
            from motherbrain.cli import load_current
            from motherbrain.voice import detect

            model, tok, dev, version = load_current(self.run_dir, self.device)
            cap = detect()
        except Exception as exc:                          # noqa: BLE001
            self.bridge.post(self._load_failed, str(exc))
            return
        self.bridge.post(self._loaded, model, tok, dev, version, cap)

    def _loaded(self, model, tok, dev, version, cap) -> None:
        from motherbrain.cli import human

        self.model, self.tok, self.dev, self.version = model, tok, dev, version
        self.cap = cap
        sight = "sight" if getattr(model, "vision", None) is not None else "no sight"
        self.header.config(
            text=f"MotherBrain v{version} · {human(model.n_params())} parameters "
                 f"· {dev} · {sight}")
        for b in self.buttons:
            b.config(state="normal")
        self.send_btn.config(state="normal")
        if cap.listen:
            self.voice_btn.config(state="normal")
        else:
            self.voice_btn.config(text="🎤 (none)")
        if getattr(model, "vision", None) is not None:
            self.image_btn.config(state="normal")
        else:
            self.image_btn.config(text="🖼 (none)")
        self.refresh_stats()
        self.status.config(text="ready")
        self.entry.focus_set()

    def refresh_stats(self) -> None:
        """Redraw the stats block. Called on load and after every ascent."""
        from motherbrain.stats import gather, render

        if self.model is None:
            return
        try:
            summary = gather(self.run_dir, self.corpus_dir, model=self.model,
                             device=self.dev)
            self.stats.config(text=render(summary, width=78))
        except Exception as exc:                          # noqa: BLE001
            self.stats.config(text=f"  (stats unavailable: {exc})")

    def _load_failed(self, message: str) -> None:
        self.header.config(text="MotherBrain — no model loaded")
        self.write(f"Could not load a model from {self.run_dir}.\n{message}\n\n"
                   f"Try `mb bootstrap` in a terminal, or `mb status` to see "
                   f"what is missing.\n", "bad")
        self.status.config(text="not loaded")

    # ---- transcript -------------------------------------------------------

    def write(self, text: str, tag: str = "mb") -> None:
        self.view.config(state="normal")
        self.view.insert("end", text, tag)
        self.view.see("end")
        self.view.config(state="disabled")

    def say_note(self, text: str) -> None:
        self.write(text, "note")

    def clear_extra(self) -> None:
        for child in self.extra.winfo_children():
            child.destroy()

    def action(self, text: str, command, accent: bool = False):
        """A button in the row under the transcript, styled like the rest."""
        colour = PALETTE["accent"] if accent else PALETTE["text"]
        label = self.tk.Label(self.extra, text=text, bg=PALETTE["panel"],
                              fg=colour, padx=14, pady=8, cursor="hand2",
                              font=("TkDefaultFont", 10),
                              highlightbackground=PALETTE["line"],
                              highlightthickness=1)
        label.pack(side="left", padx=(0, 8))
        label.bind("<Button-1>", lambda e: command())
        label.bind("<Enter>", lambda e: label.config(bg=PALETTE["raised"]))
        label.bind("<Leave>", lambda e: label.config(bg=PALETTE["panel"]))
        return label

    # ---- options ----------------------------------------------------------

    EXAMPLES = {
        "make": "e.g.  a script that renames files    ·    a csv column average",
        "do": "e.g.  find TODO   ·   run script.py   ·   list files   ·   "
              "sh git status",
        "teach": "type or paste what it should learn, or choose a file",
        "apply": "no typing needed — press the button",
    }

    def toggle_stats(self) -> None:
        """Fold the stats panel away, and back."""
        self.stats_open = not self.stats_open
        if self.stats_open:
            self.stats.pack(fill="x", pady=(0, 10))
            self.stats_toggle.config(text="▾  stats")
        else:
            self.stats.pack_forget()
            self.stats_toggle.config(text="▸  stats")

    def choose(self, n: int) -> None:
        self.mode = ("make", "do", "teach", "apply")[n]
        self.hint.config(text=self.EXAMPLES[self.mode])
        for i, b in enumerate(self.buttons):
            b.select(i == n)
        self.clear_extra()

        if self.mode == "teach":
            self.action("Choose a file or folder to learn from…",
                        self.pick_corpus_path)
        if self.mode == "apply":
            self.action("Apply now — ascend to the next version",
                        self.apply_patch, accent=True)
            self.say_note("Option 4 needs no typing: press the button.\n")
        self.entry.focus_set()

    def _on_return(self, event):
        if event.state & 0x0001:       # shift held: newline, not send
            return None
        self.submit()
        return "break"

    def _take_input(self) -> str:
        text = self.entry.get("1.0", "end").strip()
        self.entry.delete("1.0", "end")
        return text

    def submit(self) -> None:
        if self.busy or self.model is None:
            return
        text = self._take_input()
        if not text:
            return
        self.write(f"\n> {text}\n", "you")
        if self.mode == "make":
            self.run_worker(self._make, text)
        elif self.mode == "teach":
            self.run_worker(self._teach, text)
        elif self.mode == "apply":
            self.apply_patch()
        else:
            self.run_worker(self._do, text)

    # ---- worker plumbing --------------------------------------------------

    def run_worker(self, fn, *a) -> None:
        self.busy = True
        self.send_btn.config(state="disabled")
        self.status.config(text="working …")

        def wrapper():
            try:
                fn(*a)
            except Exception as exc:                      # noqa: BLE001
                self.bridge.post(self.write, f"\n{type(exc).__name__}: {exc}\n", "bad")
            finally:
                self.bridge.post(self._done)

        threading.Thread(target=wrapper, daemon=True).start()

    def _done(self) -> None:
        self.busy = False
        self.send_btn.config(state="normal")
        self.status.config(text="ready")

    def _emit(self, piece: str, tag: str = "mb") -> None:
        self.bridge.post(self.write, piece, tag)

    # ---- option 1: write a program ---------------------------------------

    def _make(self, want: str) -> None:
        """Option 1, with the reasoning loop behind it.

        Generating once gives code that usually does not compile. Proposing
        several, checking each against the compiler, repairing what was merely
        interrupted, and keeping the one the model itself rates highest gives
        code that does - and the trace says how it got there rather than
        presenting the result as if it arrived whole.
        """
        from motherbrain.actions import CODE_CAVEAT
        from motherbrain.reasoning import reason_code

        self.last_want = want

        def on_step(step):
            mark = "✓" if step.ok else "✗"
            self._emit(f"  {mark} {step.name}\n", "note")
            if step.note:
                self._emit(f"      {step.note}\n", "note")

        self._emit("thinking …\n", "note")
        trace = reason_code(self.model, self.tok, self.dev, want,
                            attempts=4, max_tokens=self.max_tokens,
                            on_step=on_step)
        for step in trace.steps:
            if step.name.startswith(("closed", "scored", "chose", "kept",
                                     "nothing", "no candidate")):
                on_step(step)

        if not trace.succeeded:
            self._emit("\nnothing it wrote would compile, after "
                       f"{trace.goal and len(trace.steps)} steps.\n", "bad")
            self._speak("nothing compiled")
            return

        self.last_code = trace.answer
        self._emit("\n" + trace.answer + "\n")
        self._emit(f"\n{CODE_CAVEAT}\n", "note")
        self.bridge.post(self._offer_save)
        self._speak("program written")

    def _offer_save(self) -> None:
        self.clear_extra()
        self.action("Save as…", self.save_code, accent=True)
        self.action("Save and run", lambda: self.save_code(run=True))

    def save_code(self, run: bool = False) -> None:
        from tkinter import filedialog, messagebox

        from motherbrain.actions import default_filename

        if not self.last_code:
            return
        path = filedialog.asksaveasfilename(
            title="Save program", defaultextension=".py",
            initialfile=default_filename(self.last_want),
            filetypes=[("Python", "*.py"), ("All files", "*.*")])
        if not path:
            return
        Path(path).write_text(self.last_code, encoding="utf-8")
        self.say_note(f"written to {path}\n")
        if not run:
            return
        if not messagebox.askyesno(
                "Run it?",
                f"Run {Path(path).name}?\n\nIt was written by a small model "
                f"and nobody has reviewed it."):
            return
        self.run_worker(self._run_file, path)

    def _run_file(self, path: str) -> None:
        self._emit(f"\nrunning {path}\n", "note")
        try:
            proc = subprocess.run([sys.executable, path], capture_output=True,
                                  text=True, timeout=30)
        except subprocess.TimeoutExpired:
            self._emit("timed out after 30s\n", "bad")
            return
        if proc.stdout:
            self._emit(proc.stdout)
        if proc.stderr:
            self._emit(proc.stderr, "bad")
        self._emit(f"exit {proc.returncode}\n", "note")

    # ---- option 2: do what I say -----------------------------------------

    def _do(self, text: str) -> None:
        """Option 2: carry out an instruction, or continue it as a prompt.

        Parsing is the same deterministic table the terminal and browser
        consoles use — the model is never asked to interpret an instruction,
        only to continue text. Anything the table does not recognise is a
        prompt, which is the honest default for a base model.
        """
        from motherbrain.actions import stream
        from motherbrain.commands import parse

        cmd = parse(text)
        name, arg = cmd.name, cmd.text

        if name == "noop":
            return
        if name == "error":
            self._emit(cmd.args.get("message", "could not parse that") + "\n", "bad")
            return
        if name == "unknown":
            self._emit(f"no such command: /{cmd.args.get('command')}\n", "bad")
            return
        if name == "help":
            for label, hint in OPTIONS:
                self._emit(f"{label}\n    {hint}\n", "note")
            return
        if name == "clear":
            self.bridge.post(self._clear_view)
            return

        if name == "make":
            self._make(arg or text)
            return
        if name == "run":
            if arg:
                self._run_file(arg)
            else:
                self._emit("run needs a file\n", "bad")
            return
        if name == "ls":
            target = Path(arg or ".").expanduser()
            entries = sorted(c.name + ("/" if c.is_dir() else "")
                             for c in target.iterdir())
            self._emit("\n".join(entries) + "\n" if entries else "(empty)\n")
            return
        if name in ("cat", "see"):
            if not arg:
                self._emit(f"{name} needs a path\n", "bad")
                return
            target = Path(arg).expanduser()
            if name == "see" and getattr(self.model, "vision", None) is not None:
                self.image_path = str(target)
                self._emit(f"looking at {target.name}\n", "note")
                return
            body = target.read_text(encoding="utf-8", errors="replace")
            self._emit(body[:8000] + ("\n… truncated\n" if len(body) > 8000 else "\n"))
            return
        if name == "write":
            path, _, body = arg.partition(" ")
            if not path:
                self._emit("write needs a path\n", "bad")
                return
            Path(path).expanduser().write_text(body, encoding="utf-8")
            self._emit(f"written to {path}\n", "note")
            return
        if name == "delete":
            self.bridge.post(self._confirm_delete, arg)
            return
        if name == "find":
            if not arg:
                self._emit("find needs something to look for\n", "bad")
                return
            hits = 0
            for f in sorted(Path(".").rglob("*")):
                if not f.is_file() or f.stat().st_size > 1_000_000:
                    continue
                try:
                    for i, line in enumerate(f.read_text(encoding="utf-8",
                                                         errors="ignore").splitlines(), 1):
                        if arg in line:
                            self._emit(f"{f}:{i}: {line.strip()[:120]}\n")
                            hits += 1
                            if hits >= 100:
                                self._emit("… stopping at 100\n", "note")
                                return
                except OSError:
                    continue
            self._emit(f"{hits} match(es)\n", "note")
            return
        if name == "sh":
            if not arg:
                self._emit("sh needs a command\n", "bad")
                return
            try:
                proc = subprocess.run(arg, shell=True, capture_output=True,
                                      text=True, timeout=60)
            except subprocess.TimeoutExpired:
                self._emit("timed out after 60s\n", "bad")
                return
            self._emit((proc.stdout or "") + (proc.stderr or ""))
            self._emit(f"exit {proc.returncode}\n", "note")
            return
        if name == "learn":
            self._teach(arg)
            return
        if name in ("grow", "train"):
            self._apply()
            return
        if name == "checkout":
            from motherbrain.cli import load_current
            from motherbrain.patches import PatchStore

            wanted = cmd.args.get("version")
            store = PatchStore(self.run_dir, create=False)
            store.set_current(wanted)
            model, tok, dev, current = load_current(self.run_dir, self.device)
            self.bridge.post(self._adopt, model, tok, dev, current)
            self._emit(f"now at v{current} (v{store.head} is still the newest; "
                       f"nothing was lost)\n", "note")
            return
        if name == "export":
            from motherbrain.cli import export_model as write_export

            target = arg or f"models/motherbrain-v{self.version}.pt"
            written = write_export(self.run_dir, target, device=self.device,
                                   corpus_dir=self.corpus_dir)
            self._emit(f"wrote {target} ({written / 1e6:,.1f} MB)\n", "note")
            return
        if name == "scale":
            from motherbrain.cli import build_parser
            import contextlib
            import io

            preset = arg.strip() or "mother"
            buffer = io.StringIO()
            try:
                args = build_parser().parse_args(["scale", "--preset", preset])
                with contextlib.redirect_stdout(buffer):
                    args.func(args)
            except SystemExit:
                self._emit(f"no such preset: {preset}\n", "bad")
                return
            self._emit(buffer.getvalue())
            return
        if name in ("status", "version", "versions"):
            from motherbrain.cli import human
            from motherbrain.patches import PatchStore

            store = PatchStore(self.run_dir, create=False)
            self._emit(f"v{self.version} of v{store.head}, "
                       f"{human(self.model.n_params())} parameters\n", "note")
            for v in store.versions():
                self._emit(f"  v{v.version}  {v.n_documents} doc(s), "
                           f"loss {v.loss_before:.3f} -> {v.loss_after:.3f}\n", "note")
            return

        # Anything else: the model continues it.
        image = self._load_image()
        produced = []
        for piece in stream(self.model, self.tok, self.dev, text,
                            max_tokens=self.max_tokens, image=image):
            produced.append(piece)
            self._emit(piece)
        self._emit("\n")
        self._speak("".join(produced)[:300])

    def _clear_view(self) -> None:
        self.view.config(state="normal")
        self.view.delete("1.0", "end")
        self.view.config(state="disabled")

    def _confirm_delete(self, path: str) -> None:
        from tkinter import messagebox

        target = Path(path).expanduser() if path else None
        if target is None or not target.exists():
            self.write(f"no such path: {path}\n", "bad")
            return
        if not messagebox.askyesno("Delete?", f"Permanently delete {target}?"):
            return
        try:
            target.unlink()
            self.write(f"deleted {target}\n", "note")
        except OSError as exc:
            self.write(f"could not delete: {exc}\n", "bad")

    def _load_image(self):
        if not self.image_path or getattr(self.model, "vision", None) is None:
            return None
        from motherbrain.vision import load_image

        return load_image(self.image_path, self.model.cfg.image_size).to(self.dev)

    # ---- option 3: teach --------------------------------------------------

    def _teach(self, text: str) -> None:
        from motherbrain.data import Corpus

        corpus = Corpus(self.corpus_dir)
        corpus.add_text(text, "gui")
        corpus.write_meta()
        self.bridge.post(self._pending_note, corpus.n_documents)

    def pick_corpus_path(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(title="Learn from a file")
        if not path:
            return
        self.write(f"\n> learn from {path}\n", "you")
        self.run_worker(self._teach_path, path)

    def _teach_path(self, path: str) -> None:
        from motherbrain.data import Corpus

        corpus = Corpus(self.corpus_dir)
        files, chars = corpus.add_path(Path(path))
        corpus.write_meta()
        self._emit(f"fed {files} file(s), {chars:,} characters\n", "note")
        self.bridge.post(self._pending_note, corpus.n_documents)

    def _pending_note(self, n_documents: int) -> None:
        from motherbrain.patches import PatchStore

        pending = n_documents - PatchStore(self.run_dir, create=False).consumed_docs()
        self.say_note(f"{pending} document(s) fed and waiting. "
                      f"Option 4 turns them into the next version.\n")

    # ---- option 4: apply --------------------------------------------------

    def apply_patch(self) -> None:
        if self.busy or self.model is None:
            return
        self.write("\n> apply new knowledge as a patch\n", "you")
        self.run_worker(self._apply)

    def _apply(self) -> None:
        from motherbrain.cli import export_model, human, merged_model_path
        from motherbrain.patches import PatchConfig, create_patch

        self._emit("learning …\n", "note")

        def progress(info):
            step, total = info["step"], info["total"]
            if step % 10 == 0 or step == total:
                self._emit(f"  step {step}/{total}  loss {info['loss']:.4f}\n",
                           "note")

        version = create_patch(
            self.run_dir, self.corpus_dir, device=self.device,
            cfg=PatchConfig(mode="grow", grow_experts=self.grow, steps=self.steps),
            note="gui", progress_cb=progress)

        if version is None:
            self._emit("nothing new to learn - feed something first "
                       "(option 3).\n", "note")
            return

        self._emit(f"\nascended to v{version.version}: "
                   f"{human(version.params_before)} -> "
                   f"{human(version.params_after)} parameters, "
                   f"loss {version.loss_before:.3f} -> "
                   f"{version.loss_after:.3f}\n")

        from motherbrain.cli import load_current

        model, tok, dev, current = load_current(self.run_dir, self.device)
        self.bridge.post(self._adopt, model, tok, dev, current)

        target = merged_model_path(self.run_dir)
        try:
            written = export_model(self.run_dir, target, device=self.device)
            self._emit(f"exported {target} ({written / 1e6:,.1f} MB)\n", "note")
        except Exception as exc:                          # noqa: BLE001
            self._emit(f"could not export: {exc}\n", "bad")
        self._speak(f"version {version.version}")

    def _adopt(self, model, tok, dev, version) -> None:
        from motherbrain.cli import human

        self.model, self.tok, self.dev, self.version = model, tok, dev, version
        sight = "sight" if getattr(model, "vision", None) is not None else "no sight"
        self.header.config(
            text=f"MotherBrain v{version} · {human(model.n_params())} parameters "
                 f"· {dev} · {sight}")
        self.refresh_stats()

    # ---- the rest of the lineage -----------------------------------------

    def checkout_version(self) -> None:
        """Move the current version to an earlier one, without losing later ones."""
        from tkinter import simpledialog

        from motherbrain.patches import PatchStore

        store = PatchStore(self.run_dir, create=False)
        head = store.head
        if not head:
            self.say_note("there is only the base; nothing to check out.\n")
            return
        answer = simpledialog.askstring(
            "Check out a version",
            f"Which version? 0 is the base, {head} is the newest.\n"
            f"Later versions are kept either way.",
            initialvalue=str(store.current), parent=self.root)
        if answer is None:
            return
        self.dispatch(f"/checkout {answer.strip()}")

    def export_model(self) -> None:
        """Write the current version out as one self-contained file."""
        from tkinter import filedialog

        if self.model is None:
            return
        path = filedialog.asksaveasfilename(
            title="Export the model", defaultextension=".pt",
            initialfile=f"motherbrain-v{self.version}.pt",
            filetypes=[("MotherBrain model", "*.pt"), ("All files", "*.*")])
        if not path:
            return
        self.write(f"\n> export {path}\n", "you")
        self.run_worker(self._export_to, path)

    def _export_to(self, path: str) -> None:
        from motherbrain.cli import export_model as write_export

        written = write_export(self.run_dir, path, device=self.device,
                               corpus_dir=self.corpus_dir)
        self._emit(f"wrote {path} ({written / 1e6:,.1f} MB)\n", "note")
        self._emit("it carries the weights, config and tokenizer, and loads "
                   "with no code execution.\n", "note")

    def copy_workspace(self) -> None:
        """Copy a complete, runnable MotherBrain onto another disk."""
        from tkinter import filedialog

        target = filedialog.askdirectory(
            title="Copy MotherBrain to… (choose a folder)")
        if not target:
            return
        self.write(f"\n> copy a workspace to {target}\n", "you")
        self.run_worker(self._copy_workspace, target)

    def _copy_workspace(self, target: str) -> None:
        from motherbrain.cli import build_parser

        args = build_parser().parse_args(
            ["workspace", target, "--run", self.run_dir,
             "--corpus", self.corpus_dir])
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            args.func(args)
        self._emit(buffer.getvalue(), "note")

    def add_sight(self) -> None:
        """Attach and train a vision tower, as the next version."""
        from tkinter import messagebox

        if self.model is None or self.busy:
            return
        if getattr(self.model, "vision", None) is not None:
            self.say_note("this version can already see.\n")
            return
        if not messagebox.askyesno(
                "Give it sight?",
                "This trains a vision tower and ascends to the next version.\n\n"
                "It takes a while — tens of minutes on a CPU — and the window "
                "stays usable while it runs.\n\nStart now?"):
            return
        self.write("\n> give MotherBrain sight\n", "you")
        self.run_worker(self._add_sight)

    def _add_sight(self) -> None:
        from motherbrain.sight import create_sight_patch

        def progress(info):
            if info["step"] % 50 == 0 or info["step"] == info["total"]:
                self._emit(f"  step {info['step']}/{info['total']}  "
                           f"loss {info['loss']:.4f}\n", "note")

        self._emit("training a vision tower — this is the slow one.\n", "note")
        version, result = create_sight_patch(
            self.run_dir, device=self.device, progress_cb=progress)

        above = result["accuracy_after"] > result["chance"] * 2
        self._emit(f"\nv{version.parent} -> v{version.version}: "
                   f"{version.params_before:,} -> {version.params_after:,} "
                   f"parameters\n")
        self._emit(f"names {result['accuracy_after']:.1%} of held-out images "
                   f"(chance {result['chance']:.1%})\n",
                   "mb" if above else "bad")
        if not above:
            self._emit("that is not meaningfully above chance: the tower is "
                       "attached but has not learned to see.\n", "bad")

        from motherbrain.cli import load_current

        model, tok, dev, current = load_current(self.run_dir, self.device)
        self.bridge.post(self._adopt, model, tok, dev, current)

    def scale_dialog(self) -> None:
        """Price a configuration: what a given size would cost to run."""
        from tkinter import simpledialog

        from motherbrain.config import PRESETS

        answer = simpledialog.askstring(
            "Price a configuration",
            "Which preset?\n" + ", ".join(PRESETS),
            initialvalue="mother", parent=self.root)
        if answer:
            self.dispatch(f"/scale {answer.strip()}")

    def forget_image(self) -> None:
        self.image_path = None
        self.image_btn.config(text="🖼 Image")
        self.say_note("no longer looking at an image.\n")

    # ---- voice and images -------------------------------------------------

    def _speak(self, text: str) -> None:
        if self.cap is not None and self.cap.speak and text.strip():
            from motherbrain.voice import speak

            threading.Thread(target=speak, args=(text, self.cap),
                             daemon=True).start()

    def listen(self) -> None:
        if self.cap is None or not self.cap.listen:
            return
        self.status.config(text="listening …")
        self.voice_btn.config(state="disabled")

        def work():
            from motherbrain.voice import listen

            heard = listen(self.cap)
            self.bridge.post(self._heard, heard)

        threading.Thread(target=work, daemon=True).start()

    def _heard(self, heard: str | None) -> None:
        self.voice_btn.config(state="normal")
        self.status.config(text="ready")
        if not heard:
            self.say_note("did not catch that\n")
            return
        self.entry.delete("1.0", "end")
        self.entry.insert("1.0", heard)
        self.submit()

    def pick_image(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Look at an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.gif *.bmp *.webp"),
                       ("All files", "*.*")])
        if not path:
            return
        self.image_path = path
        self.image_btn.config(text="🖼 " + Path(path).name[:8])
        self.say_note(f"looking at {Path(path).name}. The next thing you send "
                      f"is conditioned on it.\n")


def run(run_dir: str, corpus_dir: str, device: str = "auto",
        max_tokens: int = 120, steps: int = 100, grow: int = 1) -> int:
    """Open the window. Returns a process exit code."""
    try:
        import tkinter as tk
    except ImportError:
        print(MISSING_TK, file=sys.stderr)
        return 1

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"no display to open a window on ({exc}).\n\n"
              f"On a headless machine or Android, serve it instead:\n"
              f"  mb serve --host 0.0.0.0\n"
              f"then open http://127.0.0.1:8000 in a browser.", file=sys.stderr)
        return 1

    App(root, run_dir, corpus_dir, device, max_tokens, steps, grow)
    root.mainloop()
    return 0
