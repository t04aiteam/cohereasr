"""Eval: post real audio to the running service and check the transcript for hallucination.

Free (local CPU inference), slow (~100 s per song). Run before shipping a change to
models/guard.py or the decoding path:

    cohere_env/bin/python evals/eval_hallucination.py <audio> [--must "phrase"] [--forbid "phrase"]

Pass threshold: no loop by the guard's own definition (models.guard.find_loop),
every --must phrase present, no --forbid phrase present. Exit code 0 = pass.
Whole-transcript compression ratio is NOT used: a song that repeats its stanzas
scores 3.5 with a correct transcript.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.guard import find_loop  # noqa: E402


def post_file(url, path, language):
    """POST a file as multipart/form-data with the stdlib.

    Args:
        url: Endpoint URL.
        path: Audio file path.
        language: ISO 639-1 code.

    Returns:
        Parsed JSON response.
    """
    boundary = uuid.uuid4().hex
    with open(path, "rb") as fh:
        payload = fh.read()
    name = path.rsplit("/", 1)[-1]
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="language"\r\n\r\n{language}\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    body = head + payload + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.load(resp)


def main():
    """Run the eval and exit non-zero on failure."""
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--url", default="http://127.0.0.1:6221/transcribe/file")
    ap.add_argument("--language", default="vi")
    ap.add_argument("--must", action="append", default=[])
    ap.add_argument("--forbid", action="append", default=[])
    args = ap.parse_args()

    # self-check: a loop trips it, a song that repeats a stanza later does not
    assert find_loop("rõ ràng là một người nghiện " * 12)[0] == 12
    assert find_loop("la " * 7) is None and find_loop("xin chào các bạn") is None
    stanza = "nhà em ở lưng đồi nơi chim rừng thánh thót bầu trời xanh dịu ngọt "
    assert find_loop(stanza * 3) is None

    t0 = time.time()
    if args.audio.endswith(".txt"):  # score a saved transcript, e.g. the pre-fix output
        with open(args.audio) as fh:
            text = fh.read()
    else:
        text = post_file(args.url, args.audio, args.language)["transcription"]
    loop = find_loop(text, by_char=args.language in ("zh", "ja"))
    low = text.lower()
    fails = []
    if loop:
        fails.append(f"{loop[1]!r} repeated back-to-back x{loop[0]}")
    fails += [f"missing {p!r}" for p in args.must if p.lower() not in low]
    fails += [f"forbidden {p!r} present" for p in args.forbid if p.lower() in low]
    report = {
        "seconds": round(time.time() - t0, 1),
        "loop": loop,
        "fails": fails,
        "text": text,
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
