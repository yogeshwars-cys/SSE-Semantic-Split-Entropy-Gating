"""Extract chosen members of a zip read from stdin, in one pass, without storing the archive (the UCI Kitsune zip
has no range support and its pcaps are not needed). Entries must carry their sizes in the local header.

usage: curl -sS <zip url> | python -I bench/zip_stream_extract.py <out dir> <substring> [<substring> ...]
(members whose name contains a substring and not "pcap")
"""
import os
import struct
import sys
import zlib

OUT, WANT = sys.argv[1], sys.argv[2:]
src = sys.stdin.buffer


def read(n):
    b = bytearray()
    while len(b) < n:
        c = src.read(min(n - len(b), 1 << 22))
        if not c:
            break
        b += c
    return bytes(b)


while True:
    h = read(30)
    if len(h) < 30 or h[:4] != b"PK\x03\x04":
        break
    _, flag, meth, _, _, _, cs, us, nl, el = struct.unpack("<HHHHHIIIHH", h[4:30])
    name = read(nl).decode("utf-8", "replace")
    extra = read(el)
    if cs == 0xFFFFFFFF:                                         # zip64 sizes in the extra field
        i = 0
        while i + 4 <= len(extra):
            tag, sz = struct.unpack("<HH", extra[i:i + 4])
            if tag == 1:
                us, cs = struct.unpack("<QQ", extra[i + 4:i + 20])
            i += 4 + sz
    if flag & 8:
        sys.exit(f"{name}: sizes in a data descriptor, cannot stream")
    keep = meth in (0, 8) and any(w in name for w in WANT) and "pcap" not in name and not name.endswith("/")
    print(f"{'KEEP' if keep else 'skip'} {name} {cs / 1e6:.1f} MB", flush=True)
    left = cs
    f = None
    if keep:
        path = os.path.join(OUT, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        f = open(path, "wb")
    z = zlib.decompressobj(-15) if keep and meth == 8 else None
    while left:
        c = src.read(min(left, 1 << 22))
        if not c:
            sys.exit("truncated")
        left -= len(c)
        if f:
            f.write(z.decompress(c) if z else c)
    if f:
        if z:
            f.write(z.flush())
        f.close()
print("done", flush=True)
