"""XMODEM, because a bulletin board that cannot send you a file is a menu.

Ward Christensen wrote XMODEM in 1977 and it is still the only file
transfer every terminal program agrees on. The protocol is small enough to
state completely:

  * The receiver starts the conversation. It sends `C` if it wants CRC-16
    or `NAK` if it wants the original 8-bit checksum, once a second, until
    the sender answers.
  * The sender then sends blocks: SOH, the block number, the block number
    complemented, 128 bytes of data, and the check.
  * The receiver answers each block with ACK or NAK. NAK means send it
    again.
  * EOT ends the file, and is ACKed.

The 1K variant (STX and 1024-byte blocks) is here too, because sending a
50MB model file in 128-byte pieces is 400,000 round trips.

This is the sender half. The receiver half exists as well, and only because
a protocol you cannot test against is a protocol you are guessing at - the
tests drive one against the other.
"""

from __future__ import annotations

SOH, STX, EOT, ACK, NAK, CAN, SUB = 0x01, 0x02, 0x04, 0x06, 0x15, 0x18, 0x1A
CRC_REQUEST = ord("C")


def crc16(data: bytes) -> int:
    """CCITT CRC-16, the polynomial XMODEM-CRC uses (0x1021, no reflection)."""
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def checksum(data: bytes) -> int:
    return sum(data) & 0xFF


def frame(number: int, payload: bytes, crc: bool) -> bytes:
    """One block, ready for the wire. 128 or 1024 bytes, padded with SUB."""
    size = 1024 if len(payload) > 128 else 128
    body = payload.ljust(size, bytes([SUB]))
    head = bytes([STX if size == 1024 else SOH, number & 0xFF,
                  (~number) & 0xFF])
    tail = (crc16(body).to_bytes(2, "big") if crc
            else bytes([checksum(body)]))
    return head + body + tail


def blocks(data: bytes, size: int = 1024) -> list[bytes]:
    """Split a file into payloads. The last one is short and gets padded."""
    return [data[i:i + size] for i in range(0, len(data), size)] or [b""]


class Sender:
    """The sending half as a state machine, so any transport can drive it.

    Written this way rather than as a loop over a socket because the board
    is asyncio and the tests are not, and because a transfer that owns its
    own I/O cannot be tested without one.
    """

    def __init__(self, data: bytes, block_size: int = 1024,
                 max_retries: int = 10) -> None:
        self.payloads = blocks(data, block_size)
        self.index = 0
        self.crc = True
        self.started = False
        self.done = False
        self.cancelled = False
        self.retries = 0
        self.max_retries = max_retries

    def begin(self, byte: int) -> bytes | None:
        """The receiver's opening byte decides checksum or CRC."""
        if byte == CRC_REQUEST:
            self.crc = True
        elif byte == NAK:
            self.crc = False
        elif byte in (CAN, EOT):
            self.cancelled = True
            return None
        else:
            return None                       # noise before the handshake
        self.started = True
        return self.current()

    def current(self) -> bytes:
        return frame(self.index + 1, self.payloads[self.index], self.crc)

    def answer(self, byte: int) -> bytes | None:
        """React to one ACK/NAK/CAN. Returns what to send next, or None."""
        if not self.started:
            return self.begin(byte)
        if byte == CAN:
            self.cancelled = True
            return None
        if byte == ACK:
            self.retries = 0
            self.index += 1
            if self.index >= len(self.payloads):
                self.done = True
                return bytes([EOT])
            return self.current()
        if byte in (NAK, CRC_REQUEST):
            self.retries += 1
            if self.retries > self.max_retries:
                self.cancelled = True
                return bytes([CAN, CAN])
            return self.current()
        return None                            # anything else: keep waiting


class Receiver:
    """The receiving half, for tests and for anyone reading this to learn it."""

    def __init__(self, crc: bool = True) -> None:
        self.crc = crc
        self.data = bytearray()
        self.expect = 1
        self.done = False
        self._buf = bytearray()

    def start(self) -> bytes:
        return bytes([CRC_REQUEST if self.crc else NAK])

    def feed(self, chunk: bytes) -> bytes:
        """Consume bytes, return the acknowledgements to send back."""
        self._buf += chunk
        out = bytearray()
        while self._buf and not self.done:
            head = self._buf[0]
            if head == EOT:
                self._buf.pop(0)
                self.done = True
                out.append(ACK)
                continue
            if head == CAN:
                self._buf.pop(0)
                self.done = True
                continue
            size = 1024 if head == STX else 128
            total = 3 + size + (2 if self.crc else 1)
            if len(self._buf) < total:
                break
            block = bytes(self._buf[:total])
            del self._buf[:total]
            number, complement = block[1], block[2]
            body = block[3:3 + size]
            check = block[3 + size:]
            good = (number ^ complement) == 0xFF
            if self.crc:
                good = good and int.from_bytes(check, "big") == crc16(body)
            else:
                good = good and check[0] == checksum(body)
            if not good:
                out.append(NAK)
                continue
            if number == self.expect & 0xFF:
                self.data += body
                self.expect += 1
            out.append(ACK)                    # a repeat is ACKed, not stored
        return bytes(out)

    def file(self, length: int | None = None) -> bytes:
        """The received file. XMODEM pads, so the true length has to be told."""
        if length is not None:
            return bytes(self.data[:length])
        return bytes(self.data).rstrip(bytes([SUB]))
