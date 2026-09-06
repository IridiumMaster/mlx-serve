#!/usr/bin/env bash
# Changed images must not lose the pre-media checkpoints when a one-entry
# cache evicts their donor. Run against a dedicated server: no other traffic.
# Example: --prefix-cache-entries 1 --prefix-cache-disk off --prefill-chunk 1024
#          --ssm-checkpoint-stride 1024 --ssm-checkpoint-max 8 --log-level debug
# Disk-only mode requires a freshly restarted server with a persisted TEXT
# request from the seed-disk mode. It must log a disk restore (check the log).
set -euo pipefail
PORT="${1:-11429}"
MODE="${2:-ram}"
python3 - "$PORT" "$MODE" <<'PY'
import base64
import json
import pathlib
import sys
import urllib.request

port, mode = sys.argv[1:]
assert mode in ("ram", "seed-disk", "disk"), mode
url = f"http://127.0.0.1:{int(port)}/v1/chat/completions"
fixtures = pathlib.Path("tests/fixtures")
images = []
for name in ("street-name-signs.jpg", "house.jpeg"):
    images.append("data:image/jpeg;base64," + base64.b64encode((fixtures / name).read_bytes()).decode())
prefix = "Reference material for cache retention testing. " * 500
question = "Describe the main subject visible in the image in one short sentence."

def ask(image=None, messages=None):
    if messages is None:
        content = [{"type": "text", "text": question}]
        if image is not None:
            content.insert(0, {"type": "image_url", "image_url": {"url": image}})
        messages = [{"role": "system", "content": prefix}, {"role": "user", "content": content}]
    payload = {"model": "mlx-serve", "messages": messages, "max_tokens": 48,
               "temperature": 0, "enable_thinking": False}
    request = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    usage = result["usage"]
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    text = result["choices"][0]["message"]["content"] or ""
    print(json.dumps({"mode": mode, "prompt": usage["prompt_tokens"], "cached": cached, "answer": text}), flush=True)
    return cached, text.lower()

if mode == "seed-disk":
    ask()
    print("Seed complete; restart the dedicated server before disk mode.")
else:
    # Identical token IDs and changing pixels: reuse must stop BEFORE image.
    count = 1 if mode == "disk" else 6
    for turn in range(count):
        which = turn % 2
        cached, answer = ask(images[which])
        if mode == "disk" or turn > 0:
            assert cached >= 2048, f"turn {turn}: lost pre-image state ({cached} cached)"
        words = ("sign", "street", "road", "intersection") if which == 0 else ("house", "home", "building")
        assert any(w in answer for w in words), f"turn {turn}: wrong-image answer: {answer}"
    print("PASS: pre-media prefix retained; image-specific answers remain correct.")
PY
