#!/usr/bin/env python3
"""photo-geolocator-01 entry: structured two-pass vision geolocator.

Pass 1 (every photo): one gpt-6-luna call with a clue-extraction
prompt. The model must first name the visible evidence (script/language
on signs, driving side, road markings, vegetation/climate, architecture,
vehicles/plates, sun/shadow direction) and only then commit to a
latitude/longitude plus a 0-1 confidence, as strict JSON.

Pass 2 (cheap second look, only where it can help): photos whose pass-1
call SUCCEEDED but with confidence < 0.35 get one more gpt-6-luna call
with the same image and a shorter prompt. The second answer replaces
the first only when its confidence is >= 0.60. At most 8 second looks.

Provider: the entry talks to whichever vendor the model belongs to.
A gpt-* model goes to the OpenAI gateway (POST /v1/chat/completions,
image inline as a data: URL) on OPENAI_API_KEY; a claude-* model goes
to the Anthropic gateway (POST /v1/messages) on ANTHROPIC_API_KEY.
Defaults are the GPT models because only an OpenAI key was pushed
for this account; set MAKEFUN_MODEL / MAKEFUN_SECOND_MODEL to a
claude-* model to use the Anthropic route instead (needs an
ANTHROPIC_API_KEY pushed). Each vendor is tracked dead separately,
so a 401 on one never kills calls on the other.

Fallback: any photo with no usable model answer (call refused, 401 with
no key pushed, unparseable answer, network error) gets the fixed prior
(50.0, 10.0) -- central Europe, the best single fixed point across the
three practice trials (mean score ~326k). Every photo is always
answered; the entry never crashes on a model failure.

Determinism: photos are processed in id order, model calls are
sequential, second-look candidates are ordered by (confidence, photo
id) with exact ties broken by a seeded shuffle from MAKEFUN_TRIAL_SEED.
No wall clock, no unseeded randomness, nothing time-dependent reaches
any request. A replay therefore sends the same model calls and posts
the same answers.

Budget (gpt-6-luna: input $0.10 / output $0.50 per MTok, per the
platform's pricing; a median ~45 KB photo is ~15k input tokens):
~$0.002/call, so 25 pass-1 calls + <=8 second looks cost ~$0.06-0.07
per trial actual, far under the $1.00/trial cap. (For comparison the
original Claude build was ~$0.16/trial: 25 Sonnet calls at $2.00/$10.00
per MTok + Haiku at $0.10/$0.50. The house baseline, 25 Sonnet calls,
measured $0.15/trial.)
Worst-case gateway reservation per call is ceil(body bytes / 3)
tokens at the dearest input class; sequential calls keep at most one
reservation in flight, so admission can never exhaust the $1.00
budget mid-trial.

Stdlib only. Env: MAKEFUN_ENV_URL, OPENAI_BASE_URL, OPENAI_API_KEY,
ANTHROPIC_BASE_URL, ANTHROPIC_API_KEY, MAKEFUN_TRIAL_SEED.
Optional overrides: MAKEFUN_MODEL, MAKEFUN_SECOND_MODEL,
MAKEFUN_MAX_SECOND_LOOKS.
"""

import base64
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

MODEL = os.environ.get("MAKEFUN_MODEL", "gpt-6-luna")
SECOND_MODEL = os.environ.get("MAKEFUN_SECOND_MODEL", "gpt-6-luna")
MAX_TOKENS = 512
SECOND_MAX_TOKENS = 384
MAX_SECOND_LOOKS = int(os.environ.get("MAKEFUN_MAX_SECOND_LOOKS", "8"))
SECOND_LOOK_BELOW = 0.35   # pass-1 confidence below this triggers a 2nd look
SECOND_LOOK_ACCEPT = 0.60  # 2nd answer replaces 1st only at/above this
HTTP_TIMEOUT_S = 120
RETRY_STATUSES = (408, 429, 500, 502, 503, 504, 529)
RETRIES = 2

# Best single fixed point across the three practice trials (mean score
# ~326k; the next-best fixed points score lower). A fallback, not a
# guess at any one photo: with no model it is the least-bad answer.
FALLBACK_LAT = 50.0
FALLBACK_LON = 10.0

PROMPT = (
    "You are a GeoGuessr expert. Look at this street-level photo and work out "
    "where on Earth it was taken. First list the visible clues, then commit.\n"
    "Clues to check, in order:\n"
    "1. Any text: language and script on signs, shops, plates, road paint.\n"
    "2. Driving side and road markings: yellow/white centre line, edge lines, "
    "sign shapes and colours.\n"
    "3. Vegetation, terrain and climate: trees, crops, desert, snow, coast.\n"
    "4. Architecture and infrastructure: roof shapes, power poles, fences, "
    "pavement, street furniture.\n"
    "5. Vehicles: models, steering side, plate colours and formats.\n"
    "6. Sun and shadows: which side the sun is on, for hemisphere hints.\n"
    "Then give your best estimate. If the photo is a featureless rural road "
    "with no readable clue, say so and give the centre of the most likely "
    "country or region, not a random precise point.\n"
    "Reply with ONLY one JSON object, no other text:\n"
    '{"lat": <degrees north, -90 to 90>, "lon": <degrees east, -180 to 180>, '
    '"confidence": <0 to 1>, "country": "<ISO code or null>", '
    '"reasoning": "<one or two sentences naming the decisive clues>"}'
)

SECOND_PROMPT = (
    "You are a GeoGuessr expert. Look at this street-level photo and say "
    "where it was taken. Check text/script on signs, driving side, road "
    "markings, vegetation, architecture, vehicles and plates. Reply with "
    "ONLY one JSON object, no other text:\n"
    '{"lat": <degrees north, -90 to 90>, "lon": <degrees east, -180 to 180>, '
    '"confidence": <0 to 1>, "country": "<ISO code or null>", '
    '"reasoning": "<one sentence naming the decisive clues>"}'
)


def log(msg):
    sys.stderr.write("photo-entry: %s\n" % msg)
    sys.stderr.flush()


def _opener():
    # No proxy: the sandbox reaches only the env and the gateway.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


_OPENER = _opener()


def http_json(method, url, body=None, headers=None, timeout=HTTP_TIMEOUT_S):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with _OPENER.open(req, timeout=timeout) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:  # connection error, timeout, DNS: no HTTP status
        raise IOError("HTTP %s %s failed: %s" % (method, url, e))


def http_bytes(url, timeout=HTTP_TIMEOUT_S):
    req = urllib.request.Request(url, method="GET")
    try:
        with _OPENER.open(req, timeout=timeout) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        raise IOError("GET %s failed: %s" % (url, e))


def is_lat(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(v) and -90.0 <= v <= 90.0


def is_lon(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(v) and -180.0 <= v <= 180.0


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    if isinstance(v, str):
        try:
            f = float(v.strip())
            return f if math.isfinite(f) else None
        except ValueError:
            return None
    return None


def parse_guess(text):
    """Extract {lat, lon, confidence} from a model reply, or None.

    Tries every {...} JSON object first (the prompt asks for exactly
    one), then falls back to the first in-range number pair, as the
    house starter does. Confidence defaults to 0.5 when the model
    gives coordinates but no confidence.
    """
    if not text:
        return None
    for m in _json_objects(text):
        try:
            o = json.loads(m)
        except (ValueError, TypeError):
            continue
        if not isinstance(o, dict):
            continue
        lat = _num(o.get("lat", o.get("latitude")))
        lon = _num(o.get("lon", o.get("lng", o.get("longitude"))))
        if lat is not None and lon is not None and is_lat(lat) and is_lon(lon):
            conf = _num(o.get("confidence"))
            conf = 0.5 if conf is None else max(0.0, min(1.0, conf))
            return {"lat": round(lat, 6), "lon": round(lon, 6),
                    "confidence": conf,
                    "country": o.get("country") if isinstance(
                        o.get("country"), str) else None}
    nums = []
    for tok in re.findall(r"-?\d+(?:\.\d+)?", text):
        try:
            nums.append(float(tok))
        except ValueError:
            pass
    for i in range(len(nums) - 1):
        if is_lat(nums[i]) and is_lon(nums[i + 1]):
            return {"lat": round(nums[i], 6), "lon": round(nums[i + 1], 6),
                    "confidence": 0.5, "country": None}
    return None


def _json_objects(text):
    out, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    out.append(text[start:i + 1])
                    start = None
    return out


class ModelDead(Exception):
    """The gateway refused with 401: no usable key, stop calling it."""


def ask_model(base_url, api_key, model, jpeg_b64, prompt, max_tokens):
    """One Anthropic Messages call via the gateway.

    Returns the reply text, or None on a final gateway refusal.
    Raises ModelDead on 401 (no key pushed) and IOError on transport
    failure. Retries transient 408/429/5xx/529 up to RETRIES times with
    fixed backoff; a gateway refusal (4xx, x-should-retry: false) is
    final and is never retried.
    """
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": jpeg_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    }).encode("utf-8")
    url = base_url.rstrip("/") + "/v1/messages"
    for attempt in range(RETRIES + 1):
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("x-api-key", api_key)
        req.add_header("anthropic-version", "2023-06-01")
        try:
            with _OPENER.open(req, timeout=HTTP_TIMEOUT_S) as res:
                payload = json.loads(res.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = e.read()
            if e.code == 401:
                raise ModelDead("gateway answered 401: no usable API key")
            should_retry = e.headers.get("x-should-retry") != "false"
            if e.code in RETRY_STATUSES and should_retry \
                    and attempt < RETRIES:
                time.sleep(2 * (attempt + 1))
                continue
            log("model call refused (HTTP %d): %s" % (
                e.code, raw[:200].decode("utf-8", "replace")))
            return None
        except Exception as e:
            if attempt < RETRIES:
                time.sleep(2 * (attempt + 1))
                continue
            log("model call transport error: %s" % e)
            return None
        text = "".join(
            b.get("text", "") for b in payload.get("content", [])
            if isinstance(b, dict) and b.get("type") == "text")
        return text
    return None


def ask_openai(base_url, api_key, model, jpeg_b64, prompt, max_tokens):
    """One OpenAI Chat Completions call via the gateway.

    Same contract as ask_model: reply text, None on a final gateway
    refusal, ModelDead on 401, IOError on transport failure. Retries
    transient 408/429/5xx/529 up to RETRIES times; a gateway refusal
    (4xx, x-should-retry: false) is final and never retried.
    max_completion_tokens is required by the gateway (a call without
    it is refused); stream is never requested.
    """
    body = json.dumps({
        "model": model,
        "max_completion_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + jpeg_b64}},
            ],
        }],
    }).encode("utf-8")
    url = base_url.rstrip("/") + "/chat/completions"
    for attempt in range(RETRIES + 1):
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", "Bearer " + api_key)
        try:
            with _OPENER.open(req, timeout=HTTP_TIMEOUT_S) as res:
                payload = json.loads(res.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = e.read()
            if e.code == 401:
                raise ModelDead("gateway answered 401: no usable API key")
            should_retry = e.headers.get("x-should-retry") != "false"
            if e.code in RETRY_STATUSES and should_retry \
                    and attempt < RETRIES:
                time.sleep(2 * (attempt + 1))
                continue
            log("model call refused (HTTP %d): %s" % (
                e.code, raw[:200].decode("utf-8", "replace")))
            return None
        except Exception as e:
            if attempt < RETRIES:
                time.sleep(2 * (attempt + 1))
                continue
            log("model call transport error: %s" % e)
            return None
        choices = payload.get("choices") or []
        if not choices:
            return None
        msg = (choices[0] or {}).get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):
            return "".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text")
        return content if isinstance(content, str) else None
    return None


class _Vendor:
    """One vendor's gateway endpoint plus its dead flag."""

    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.dead = not (base_url and api_key)


def _vendor_for(model, vendors):
    """The vendor a model belongs to: gpt-* -> OpenAI, else Anthropic."""
    return vendors["openai"] if model.startswith("gpt-") \
        else vendors["anthropic"]


def ask_any(model, jpeg_b64, prompt, max_tokens, vendors):
    """One model call on the right vendor's gateway (see ask_model).

    Raises ModelDead when that vendor has no key configured or
    answered 401; the vendor is then marked dead for the rest of
    the trial.
    """
    v = _vendor_for(model, vendors)
    if v.dead:
        raise ModelDead("no usable API key for %s" % model)
    try:
        if model.startswith("gpt-"):
            return ask_openai(v.base_url, v.api_key, model,
                              jpeg_b64, prompt, max_tokens)
        return ask_model(v.base_url, v.api_key, model,
                         jpeg_b64, prompt, max_tokens)
    except ModelDead:
        v.dead = True
        raise


def main():
    env_url = os.environ["MAKEFUN_ENV_URL"].rstrip("/")
    vendors = {
        "openai": _Vendor(
            os.environ.get("OPENAI_BASE_URL", "").rstrip("/"),
            os.environ.get("OPENAI_API_KEY", "")),
        "anthropic": _Vendor(
            os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/"),
            os.environ.get("ANTHROPIC_API_KEY", "")),
    }
    seed = os.environ.get("MAKEFUN_TRIAL_SEED", "photo-geolocator-01")
    rng = random.Random(hashlib.sha256(
        ("agent-fun/photo/v1:" + seed).encode("utf-8")).hexdigest())

    status, raw = http_json("GET", env_url + "/photos")
    if status != 200:
        log("GET /photos failed: HTTP %d" % status)
        return 1
    listed = json.loads(raw.decode("utf-8"))
    photos = [p for p in listed.get("photos", []) if not p.get("answered")]
    photos.sort(key=lambda p: p["id"])

    guesses = {}   # id -> {"lat","lon","confidence","country","model"}
    model_dead = _vendor_for(MODEL, vendors).dead
    if model_dead:
        log("no gateway configured for %s; model-free fallback" % MODEL)

    # ---- Pass 1: one model call per photo, in id order. ----
    for p in photos:
        pid = p["id"]
        try:
            _s, jpeg = http_bytes(env_url + p["url"])
        except IOError as e:
            log("%s: photo fetch failed: %s" % (pid, e))
            guesses[pid] = {"lat": FALLBACK_LAT, "lon": FALLBACK_LON,
                            "confidence": 0.0, "country": None, "model": None}
            continue
        if model_dead:
            guesses[pid] = {"lat": FALLBACK_LAT, "lon": FALLBACK_LON,
                            "confidence": 0.0, "country": None, "model": None}
            continue
        try:
            text = ask_any(MODEL,
                           base64.b64encode(jpeg).decode("ascii"),
                           PROMPT, MAX_TOKENS, vendors)
        except ModelDead as e:
            log(str(e))
            model_dead = True
            guesses[pid] = {"lat": FALLBACK_LAT, "lon": FALLBACK_LON,
                            "confidence": 0.0, "country": None, "model": None}
            continue
        g = parse_guess(text) if text else None
        if g is None:
            log("%s: no coordinates in pass-1 answer" % pid)
            guesses[pid] = {"lat": FALLBACK_LAT, "lon": FALLBACK_LON,
                            "confidence": 0.0, "country": None, "model": None}
        else:
            g["model"] = MODEL
            guesses[pid] = g

    # ---- Pass 2: cheap second look, low-confidence only. ----
    second_dead = _vendor_for(SECOND_MODEL, vendors).dead
    if not second_dead:
        cands = [(guesses[p["id"]]["confidence"], p["id"])
                 for p in photos
                 if guesses[p["id"]].get("model") == MODEL
                 and guesses[p["id"]]["confidence"] < SECOND_LOOK_BELOW]
        # Exact confidence ties break by seeded shuffle order (same
        # order on live and replay, since the seed is fixed per trial).
        shuffle = [pid for _, pid in cands]
        rng.shuffle(shuffle)
        tie = {pid: i for i, pid in enumerate(shuffle)}
        cands.sort(key=lambda c: (c[0], tie[c[1]], c[1]))
        for _conf, pid in cands[:MAX_SECOND_LOOKS]:
            try:
                _s, jpeg = http_bytes(env_url + "/photos/%s.jpg" % pid)
            except IOError as e:
                log("%s: second-look fetch failed: %s" % (pid, e))
                continue
            try:
                text = ask_any(SECOND_MODEL,
                               base64.b64encode(jpeg).decode("ascii"),
                               SECOND_PROMPT, SECOND_MAX_TOKENS, vendors)
            except ModelDead:
                log("second look stopped: gateway 401")
                break
            g2 = parse_guess(text) if text else None
            if g2 is not None and g2["confidence"] >= SECOND_LOOK_ACCEPT:
                log("%s: second look accepted (conf %.2f -> %.2f)" % (
                    pid, guesses[pid]["confidence"], g2["confidence"]))
                g2["model"] = SECOND_MODEL
                guesses[pid] = g2
            elif g2 is None:
                log("%s: second look unparseable, keeping pass 1" % pid)

    # ---- Post every answer (fallback included) in one batch. ----
    answers = [{"id": p["id"], "lat": guesses[p["id"]]["lat"],
                "lon": guesses[p["id"]]["lon"]} for p in photos]
    if not answers:
        log("nothing to post")
        return 0
    for attempt in range(2):
        try:
            status, raw = http_json("POST", env_url + "/answers",
                                    {"answers": answers})
        except IOError as e:
            log("POST /answers transport error: %s" % e)
            if attempt == 0:
                time.sleep(2)
                continue
            return 1
        if status == 200:
            log("posted %d answers: %s" % (
                len(answers), raw.decode("utf-8", "replace").strip()))
            return 0
        if status == 409:
            # Possibly a first POST landed but its reply was lost.
            try:
                _s, raw2 = http_json("GET", env_url + "/photos")
                n = json.loads(raw2.decode("utf-8")).get("answered", 0)
            except Exception:
                n = -1
            log("POST /answers 409; env reports %s answered" % n)
            return 0 if n == len(photos) else 1
        log("POST /answers refused: HTTP %d %s" % (
            status, raw[:200].decode("utf-8", "replace")))
        return 1
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyError as e:
        log("missing environment variable: %s" % e)
        sys.exit(1)
    except Exception as e:
        # Never die without trying to answer: if photos were listed
        # the loop above already fell back per photo, so reaching here
        # means the env itself failed; report and exit non-zero.
        log("fatal: %r" % (e,))
        sys.exit(1)
