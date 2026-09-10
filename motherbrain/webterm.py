"""The board in a browser, so a finger can drive it.

A phone has no telnet client worth the name and no mouse at all, and the
board is a telnet board driven by single keypresses. Bridging that could
have meant a second implementation of every screen. It does not, because
of one observation: the board already turns a *click* into the key a
caller would have typed, and a tap is a click.

So this is a relay and a renderer, and nothing else. A WebSocket carries
the same bytes the telnet socket carries; the page draws them; a tap on a
line sends the mouse report for that cell, which the board resolves
through the same hotspot table a desktop terminal's mouse goes through.
Every screen, every door, every menu, unchanged - and a phone keyboard for
the parts you type.

The renderer is deliberately line-oriented, because the board is: it
clears the screen, writes whole lines, returns the carriage, and erases a
line. It never addresses the cursor. Writing a full terminal emulator to
carry text that never needed one would be a lot of code to maintain for no
behaviour.
"""

from __future__ import annotations

import asyncio

# At module scope, not inside create_app. FastAPI resolves an endpoint's
# annotations against the module's globals: a `WebSocket` imported inside
# the factory cannot be found there, so the parameter is taken for a query
# argument and every connection is rejected with 403. This is the second
# time that has bitten this codebase and the reason is written down here so
# it is the last.
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,
      maximum-scale=1,user-scalable=no,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#05060f">
<title>MotherBrain BBS</title>
<style>
  :root { color-scheme: dark; --bg:#05060f; --fg:#c8c8c8; }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; height: 100%; background: var(--bg); color: var(--fg);
    font-family: "DejaVu Sans Mono", "Cascadia Mono", "Courier New", monospace;
    overscroll-behavior: none; -webkit-text-size-adjust: 100%;
  }
  body { display: flex; flex-direction: column; }
  #screen {
    flex: 1 1 auto; overflow: auto; padding: 6px 4px;
    white-space: pre; line-height: 1.15; letter-spacing: 0;
    -webkit-user-select: none; user-select: none;
  }
  #screen .row { display: block; min-height: 1.15em; }
  #screen .row:active { background: #1b2340; }
  #bar {
    flex: 0 0 auto; display: flex; flex-wrap: wrap; gap: 4px;
    padding: 6px 4px calc(6px + env(safe-area-inset-bottom));
    background: #0b1024; border-top: 1px solid #1d2748;
  }
  #bar button {
    flex: 1 1 auto; min-width: 44px; min-height: 44px;
    background: #16204a; color: #e6e6e6;
    border: 1px solid #2a3a72; border-radius: 6px;
    font: inherit; font-size: 15px; padding: 6px 8px;
  }
  #bar button:active { background: #2a3a72; }
  #bar button.wide { flex: 2 1 auto; }
  #typed {
    position: absolute; left: -9999px; width: 1px; height: 1px;
    opacity: 0;
  }
  #note {
    padding: 4px 8px; font-size: 12px; color: #7c8bb8; background: #0b1024;
  }
</style>
</head>
<body>
<div id="note">connecting…</div>
<div id="screen" aria-live="polite"></div>
<input id="typed" autocomplete="off" autocorrect="off"
       autocapitalize="off" spellcheck="false" aria-label="type">
<div id="bar"></div>
<script>
(function () {
  "use strict";

  var screenEl = document.getElementById("screen");
  var noteEl = document.getElementById("note");
  var barEl = document.getElementById("bar");
  var typedEl = document.getElementById("typed");

  // ---- the screen model -----------------------------------------------
  // Lines of runs. A run is {text, style}. The board writes whole lines,
  // returns the carriage and erases a line; it never moves the cursor
  // about, so this is the whole of what has to be modelled.
  var lines = [[]];
  var row = 0, col = 0;
  var style = {fg: 7, bg: null, bold: false, blink: false};
  var COLS = 80;

  var PALETTE = ["#101010", "#c04040", "#40b040", "#c0a040",
                 "#4060c0", "#b050b0", "#40b0b0", "#c8c8c8"];
  var BRIGHT = ["#606060", "#ff6060", "#60ff60", "#ffff60",
                "#6090ff", "#ff80ff", "#60ffff", "#ffffff"];

  function copyStyle(s) {
    return {fg: s.fg, bg: s.bg, bold: s.bold, blink: s.blink};
  }

  function reset() { style = {fg: 7, bg: null, bold: false, blink: false}; }

  function sgr(params) {
    if (params.length === 0) params = [0];
    for (var i = 0; i < params.length; i++) {
      var n = params[i];
      if (n === 0) reset();
      else if (n === 1) style.bold = true;
      else if (n === 5) style.blink = true;
      else if (n === 22) style.bold = false;
      else if (n === 25) style.blink = false;
      else if (n >= 30 && n <= 37) style.fg = n - 30;
      else if (n === 39) style.fg = 7;
      else if (n >= 40 && n <= 47) style.bg = n - 40;
      else if (n === 49) style.bg = null;
      else if (n >= 90 && n <= 97) { style.fg = n - 90; style.bold = true; }
      else if (n === 38 && params[i + 1] === 5) {
        style.fg = xterm(params[i + 2]); i += 2;
      } else if (n === 48 && params[i + 1] === 5) {
        style.bg = xterm(params[i + 2]); i += 2;
      }
    }
  }

  // 256-colour indices only appear in the picture gallery. Fold them onto
  // the sixteen the rest of the board uses rather than pretending.
  function xterm(n) {
    if (n < 16) return n % 8;
    if (n >= 232) return n < 244 ? 0 : 7;
    n -= 16;
    var r = Math.floor(n / 36), g = Math.floor((n % 36) / 6), b = n % 6;
    return (r > 2 ? 1 : 0) | (g > 2 ? 2 : 0) | (b > 2 ? 4 : 0);
  }

  function clearScreen() { lines = [[]]; row = 0; col = 0; }

  function put(text) {
    var line = lines[row] || (lines[row] = []);
    // Overwriting mid-line only happens after a carriage return, and the
    // board only does that to redraw a prompt - so replacing the tail is
    // exactly right and much simpler than a cell grid.
    if (col === 0) { lines[row] = [{t: text, s: copyStyle(style)}]; }
    else { line.push({t: text, s: copyStyle(style)}); }
    col += text.length;
  }

  function newline() { row += 1; col = 0; if (!lines[row]) lines[row] = []; }

  var pending = "";
  function feed(text) {
    text = pending + text; pending = "";
    var i = 0, buf = "";
    function flush() { if (buf) { put(buf); buf = ""; } }
    while (i < text.length) {
      var ch = text[i];
      if (ch === "\x1b") {
        if (text[i + 1] !== "[") { i += 1; continue; }
        var j = i + 2;
        while (j < text.length && !(text[j] >= "@" && text[j] <= "~")) j += 1;
        if (j >= text.length) { pending = text.slice(i); break; }
        flush();
        var body = text.slice(i + 2, j), final = text[j];
        var params = body.replace(/[^0-9;]/g, "").split(";")
                         .filter(function (p) { return p !== ""; })
                         .map(Number);
        if (final === "m") sgr(params);
        else if (final === "J") clearScreen();
        else if (final === "H") { row = 0; col = 0; }
        else if (final === "K") { lines[row] = []; col = 0; }
        i = j + 1;
        continue;
      }
      if (ch === "\r") { flush(); col = 0; i += 1; continue; }
      if (ch === "\n") { flush(); newline(); i += 1; continue; }
      if (ch === "\b") { flush(); col = Math.max(0, col - 1); i += 1; continue; }
      buf += ch; i += 1;
    }
    flush();
    draw();
  }

  function colourOf(n, bold) {
    if (n === null || n === undefined) return null;
    return (bold ? BRIGHT : PALETTE)[n % 8];
  }

  function draw() {
    var out = document.createDocumentFragment();
    for (var r = 0; r < lines.length; r++) {
      var div = document.createElement("span");
      div.className = "row";
      div.dataset.row = String(r + 1);
      var runs = lines[r];
      if (!runs.length) div.appendChild(document.createTextNode(" "));
      for (var k = 0; k < runs.length; k++) {
        var run = runs[k];
        var span = document.createElement("span");
        span.textContent = run.t;
        span.style.color = colourOf(run.s.fg, run.s.bold);
        if (run.s.bg !== null) span.style.background = colourOf(run.s.bg, false);
        if (run.s.blink) span.style.fontWeight = "bold";
        div.appendChild(span);
      }
      out.appendChild(div);
    }
    screenEl.replaceChildren(out);
    screenEl.scrollTop = screenEl.scrollHeight;
  }

  // ---- the wire --------------------------------------------------------
  var socket = null, ready = false;

  function connect() {
    var proto = location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(proto + "//" + location.host + "/ws");
    socket.binaryType = "arraybuffer";
    socket.onopen = function () {
      ready = true;
      noteEl.textContent = "connected - tap a menu line, or use the keys below";
      // After a frame: measuring the character cell before the browser has
      // laid the page out gives zero, and a board told it has zero columns
      // draws at its minimum and runs off the side of a phone.
      requestAnimationFrame(function () {
        requestAnimationFrame(sendSize);
      });
    };
    socket.onmessage = function (event) {
      var text = typeof event.data === "string"
        ? event.data
        : new TextDecoder("utf-8", {fatal: false}).decode(event.data);
      feed(text);
    };
    socket.onclose = function () {
      ready = false;
      noteEl.textContent = "disconnected - reload to call again";
      feed("\r\n\x1b[1;31mNO CARRIER\x1b[0m\r\n");
    };
  }

  function send(text) {
    if (ready) socket.send(text);
  }

  function sendSize() {
    // The page tells the board how wide it is, in the same units the
    // board draws in: characters. Measured, not guessed.
    var probe = document.createElement("span");
    probe.style.visibility = "hidden";
    probe.style.position = "absolute";
    probe.textContent = "0".repeat(80);
    screenEl.appendChild(probe);
    var per = probe.getBoundingClientRect().width / 80;
    screenEl.removeChild(probe);
    COLS = Math.max(38, Math.min(132,
      Math.floor((screenEl.clientWidth - 10) / (per || 8))));
    var rows = Math.max(12, Math.min(60,
      Math.floor(screenEl.clientHeight / 18)));
    send("\x00SIZE " + COLS + " " + rows + "\n");
  }

  // ---- a tap is a click ------------------------------------------------
  screenEl.addEventListener("click", function (event) {
    var line = event.target.closest(".row");
    if (!line) return;
    var box = line.getBoundingClientRect();
    var per = box.width / Math.max(1, line.textContent.length || COLS);
    var column = Math.max(1, Math.round((event.clientX - box.left) / per) + 1);
    var r = Number(line.dataset.row || 1);
    // The same SGR mouse report a terminal sends. The board resolves it
    // through the hotspot table every screen already builds.
    send("\x1b[<0;" + column + ";" + r + "M");
    send("\x1b[<0;" + column + ";" + r + "m");
    if (keyboardUp) typedEl.focus();
  });

  // ---- the keys a finger needs -----------------------------------------
  var KEYS = [
    ["ENTER", "\r", "wide"], ["←", "\x7f", ""],
    ["Q", "Q\r", ""], ["Y", "Y\r", ""], ["N", "N\r", ""],
    ["↑", "\x1b[A", ""], ["↓", "\x1b[B", ""],
    ["←←", "\x1b[D", ""], ["→", "\x1b[C", ""],
    ["SPACE", " ", "wide"], ["MENU", "?\r", ""], ["BYE", "G\r", ""],
    ["KEYBOARD", null, "wide"]
  ];

  var keyboardUp = false;

  KEYS.forEach(function (spec) {
    var button = document.createElement("button");
    button.textContent = spec[0];
    if (spec[2]) button.className = spec[2];
    button.addEventListener("click", function (event) {
      event.preventDefault();
      if (spec[1] === null) { keyboardUp = true; typedEl.focus(); return; }
      send(spec[1]);
      // Keep the soft keyboard where the caller put it: a button that
      // dismissed it every time would make typing a message impossible.
      if (keyboardUp) typedEl.focus();
    });
    barEl.appendChild(button);
  });

  // A physical keyboard is listened for on the document, so a desktop
  // browser needs no ceremony: open the page and type. Binding this to the
  // hidden input alone meant that clicking any button moved focus off it
  // and the next thing typed went nowhere.
  var SPECIAL = {
    Enter: "\r", Backspace: "\x7f", Tab: "\t", Escape: "\x1b",
    ArrowUp: "\x1b[A", ArrowDown: "\x1b[B",
    ArrowRight: "\x1b[C", ArrowLeft: "\x1b[D"
  };

  document.addEventListener("keydown", function (event) {
    if (event.metaKey || event.altKey) return;
    var out = SPECIAL[event.key];
    if (!out && event.ctrlKey && event.key.length === 1) {
      var code = event.key.toUpperCase().charCodeAt(0) - 64;
      if (code > 0 && code < 32) out = String.fromCharCode(code);
    }
    if (!out && !event.ctrlKey && event.key.length === 1) out = event.key;
    if (!out) return;
    send(out);
    event.preventDefault();
  });

  // Android and iOS soft keyboards fire `input` on a hidden field rather
  // than keydown, so that path is kept for them and cleared each time.
  typedEl.addEventListener("input", function () {
    var value = typedEl.value;
    if (value) { send(value); typedEl.value = ""; }
  });

  window.addEventListener("resize", function () {
    clearTimeout(window._mbResize);
    window._mbResize = setTimeout(sendSize, 200);
  });

  connect();
}());
</script>
</body>
</html>
"""


def create_app(board_host: str = "127.0.0.1", board_port: int = 23):
    """A page and a socket. Everything else is the board, unchanged."""
    app = FastAPI(title="MotherBrain BBS", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return PAGE

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "board": f"{board_host}:{board_port}"}

    @app.websocket("/ws")
    async def bridge(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            reader, writer = await asyncio.open_connection(board_host,
                                                           board_port)
        except OSError as exc:
            await websocket.send_text(
                f"\r\ncannot reach the board: {exc}\r\n")
            await websocket.close()
            return

        state = _Bridge(websocket, reader, writer)
        try:
            await asyncio.gather(state.to_browser(), state.to_board())
        except (WebSocketDisconnect, ConnectionError, asyncio.CancelledError):
            pass
        finally:
            writer.close()

    return app


class _Bridge:
    """One browser tab, holding one telnet connection to the board."""

    def __init__(self, socket, reader, writer) -> None:
        self.socket, self.reader, self.writer = socket, reader, writer
        self.telnet = None

    async def to_browser(self) -> None:
        """Board to page: answer the protocol, forward the text."""
        from motherbrain.client import Session

        self.telnet = Session(80, 24, "xterm-256color")
        while True:
            chunk = await self.reader.read(4096)
            if not chunk:
                await self.socket.close()
                return
            text, reply = self.telnet.feed(chunk)
            if reply:
                self.writer.write(reply)
                await self.writer.drain()
            if text:
                await self.socket.send_text(text)

    async def to_board(self) -> None:
        """Page to board. A size report is ours; everything else is keys."""
        from motherbrain.telnet import IAC

        while True:
            message = await self.socket.receive_text()
            if message.startswith("\x00SIZE "):
                await self._resize(message)
                continue
            data = message.encode("utf-8")
            self.writer.write(data.replace(bytes([IAC]), bytes([IAC, IAC])))
            await self.writer.drain()

    async def _resize(self, message: str) -> None:
        """The page measured itself. Pass it on as NAWS, as a terminal would."""
        try:
            _, columns, rows = message.strip().split()
            columns, rows = int(columns), int(rows)
        except ValueError:
            return
        if self.telnet is None:
            return
        self.telnet.columns, self.telnet.rows = columns, rows
        self.writer.write(self.telnet.size())
        await self.writer.drain()


async def serve(board_host: str, board_port: int, host: str,
                port: int) -> None:
    """Run the web front-end alongside the board, in the board's own loop."""
    import uvicorn

    config = uvicorn.Config(create_app(board_host, board_port), host=host,
                            port=port, log_level="warning")
    await uvicorn.Server(config).serve()
