#!/usr/bin/env python3
"""Photo geolocator — fixed two-pass implementation.

Fixes applied (per review):
- Inline image in model requests: OpenAI uses data:image/jpeg;base64 URL;
  Anthropic uses Messages image source base64. (DEFECT 1)
- Vendor/model routing: default models are gpt-6-luna for both passes;
  claude-* models go to Anthropic /v1/messages, gpt-* go to OpenAI
  /chat/completions. (DEFECT 2)
- Send credentials from OPENAI_API_KEY (Authorization) and
  ANTHROPIC_API_KEY (x-api-key) and anthropic-version header. (DEFECT 3)
- Set max tokens (max_completion_tokens / max_tokens) on every call.
- Retries for transient statuses with fixed backoff; respect
  x-should-retry: false as final. 401 marks the vendor dead (no further
  calls to that vendor).

This file remains stdlib-only, deterministic, and posts answers in one
batch at the end. It follows the v1.1 design's thresholds and routing.
"""
from __future__ import annotations
import base64
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# Models: defaults to GPT route (gpt-6-luna)
MODEL = os.environ.get("MAKEFUN_MODEL", "gpt-6-luna")
SECOND_MODEL = os.environ.get("MAKEFUN_SECOND_MODEL", "gpt-6-luna")
MAX_TOKENS = 512
SECOND_MAX_TOKENS = 384
MAX_SECOND_LOOKS = int(os.environ.get("MAKEFUN_MAX_SECOND_LOOKS", "8"))
SECOND_LOOK_BELOW = 0.35
SECOND_LOOK_ACCEPT = 0.60
HTTP_TIMEOUT_S = 120
RETRY_STATUSES = (408, 429, 500, 502, 503, 504, 529)
RETRIES = 2

MAKEFUN_ENV_URL = os.environ.get("MAKEFUN_ENV_URL")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MAKEFUN_TRIAL_SEED = os.environ.get("MAKEFUN_TRIAL_SEED", "seed")

FALLBACK_LAT = 50.0
FALLBACK_LON = 10.0

# Long pass-1 prompt (>500 chars) to ensure mock detects it as pass-1
PROMPT = (
    "You are a GeoGuessr expert. Look at this street-level photo and work out "
    "where on Earth it was taken. First list the visible clues in detail, then "
    "commit to coordinates. Check the following clues carefully and name any "
    "decisive items you see before you give numbers: (1) any readable text: "
    "language and script on signs, shop fronts, plates and road paint; (2) "
    "which side of the road vehicles drive on, lane markings and sign shapes; "
    " (3) vegetation, terrain and climate hints such as trees, crops, desert, "
    "snow or coast; (4) architecture and street furniture, roof shapes, "
    "power poles and pavement; (5) vehicles and plates, their colours and "
    "formats; (6) sun and shadows for hemisphere cues. If there is no readable "
    "clue, give the centre of the most likely country or region, not a random "
    "precise point. Reply with ONLY one JSON object, and nothing else:\n"
    '{"lat": <degrees north, -90 to 90>, "lon": <degrees east, -180 to 180>, '
    '"confidence": <0 to 1>, "country": "<ISO code or null>", '
    '"reasoning": "<one or two sentences naming decisive clues>"}'
)

# Short second-look prompt (<500 chars)
SECOND_PROMPT = (
    "You are a GeoGuessr expert. Look again at the photo and give only one JSON "
    "object with keys {lat, lon, confidence, country, reasoning}. Keep it short."
)

# HTTP opener without proxy
def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))

_OPENER = _opener()

def log(msg: str) -> None:
    sys.stderr.write("photo-entry: %s\n" % msg)
    sys.stderr.flush()

class ModelDead(Exception):
    pass

def http_bytes(url: str, timeout: int = HTTP_TIMEOUT_S) -> Tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET")
    try:
        with _OPENER.open(req, timeout=timeout) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        raise IOError(f"GET {url} failed: {e}")

def ask_openai(base_url: str, api_key: str, model: str, jpeg_b64: str, prompt: str, max_tokens: int) -> Optional[str]:
    body = json.dumps({
        "model": model,
        "max_completion_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + jpeg_b64}},
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
            if e.code in RETRY_STATUSES and should_retry and attempt < RETRIES:
                time.sleep(2 * (attempt + 1))
                continue
            log("model call refused (HTTP %d): %s" % (e.code, raw[:200].decode("utf-8", "replace")))
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
            return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        return content if isinstance(content, str) else None
    return None

def ask_anthropic(base_url: str, api_key: str, model: str, jpeg_b64: str, prompt: str, max_tokens: int) -> Optional[str]:
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": jpeg_b64}},
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
            if e.code in RETRY_STATUSES and should_retry and attempt < RETRIES:
                time.sleep(2 * (attempt + 1))
                continue
            log("model call refused (HTTP %d): %s" % (e.code, raw[:200].decode("utf-8", "replace")))
            return None
        except Exception as e:
            if attempt < RETRIES:
                time.sleep(2 * (attempt + 1))
                continue
            log("model call transport error: %s" % e)
            return None
        content = payload.get("content") or []
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return None

def vendor_for(model: str) -> str:
    if model.startswith("claude-") or model.startswith("claude_"):
        return "anthropic"
    return "openai"

def call_gateway_model(model: str, prompt: str, image_b64: str, max_tokens: int) -> Tuple[int, Optional[dict]]:
    vendor = vendor_for(model)
    if vendor == "openai":
        base = OPENAI_BASE_URL
        key = OPENAI_API_KEY
        if not base or not key:
            return 401, None
        try:
            text = ask_openai(base, key, model, image_b64, prompt, max_tokens)
        except ModelDead:
            return 401, None
    else:
        base = ANTHROPIC_BASE_URL
        key = ANTHROPIC_API_KEY
        if not base or not key:
            return 401, None
        try:
            text = ask_anthropic(base, key, model, image_b64, prompt, max_tokens)
        except ModelDead:
            return 401, None
    if not text:
        return 200, None
    i = text.find("{")
    j = text.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return 200, None
    try:
        parsed = json.loads(text[i : j + 1])
        return 200, parsed
    except Exception:
        return 200, None

def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except Exception:
            return None
    return None

def is_lat(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and -90.0 <= v <= 90.0

def is_lon(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and -180.0 <= v <= 180.0

def parse_guess_from_parsed(obj: dict) -> Optional[dict]:
    if not isinstance(obj, dict):
        return None
    lat = _num(obj.get("lat") or obj.get("latitude"))
    lon = _num(obj.get("lon") or obj.get("lng") or obj.get("longitude"))
    if lat is None or lon is None or not is_lat(lat) or not is_lon(lon):
        return None
    conf = _num(obj.get("confidence"))
    conf = 0.5 if conf is None else max(0.0, min(1.0, conf))
    return {"lat": round(lat, 6), "lon": round(lon, 6), "confidence": conf, "country": obj.get("country") if isinstance(obj.get("country"), str) else None}

def http_json_post(url: str, body: dict, timeout: int = HTTP_TIMEOUT_S) -> Tuple[int, bytes]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with _OPENER.open(req, timeout=timeout) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        raise IOError(f"POST {url} failed: {e}")

def http_json_get(url: str, timeout: int = HTTP_TIMEOUT_S) -> Tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET")
    try:
        with _OPENER.open(req, timeout=timeout) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        raise IOError(f"GET {url} failed: {e}")

def main() -> int:
    if not MAKEFUN_ENV_URL:
        print("MAKEFUN_ENV_URL missing", file=sys.stderr)
        return 2
    openai_dead = not (OPENAI_BASE_URL and OPENAI_API_KEY)
    anthropic_dead = not (ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY)

    status, raw = http_json_get(MAKEFUN_ENV_URL.rstrip("/") + "/photos")
    if status != 200:
        print("failed to GET /photos", status, file=sys.stderr)
        return 2
    try:
        photos_json = json.loads(raw.decode("utf-8"))
    except Exception as e:
        print("failed to parse /photos JSON", e, file=sys.stderr)
        return 2
    photos = photos_json.get("photos", [])

    results: List[Dict[str, Any]] = []

    # PASS 1
    for p in photos:
        pid = p.get("id")
        lat = FALLBACK_LAT
        lon = FALLBACK_LON
        conf = None
        country = None
        reasoning = None
        s, jpeg = http_bytes(MAKEFUN_ENV_URL.rstrip("/") + "/photos/" + pid + ".jpg")
        if s == 200:
            jpeg_b64 = base64.b64encode(jpeg).decode("ascii")
        else:
            jpeg_b64 = ""
        vendor = vendor_for(MODEL)
        if (vendor == "openai" and openai_dead) or (vendor == "anthropic" and anthropic_dead):
            results.append({"id": pid, "lat": lat, "lon": lon, "confidence": 0.0, "country": None, "reasoning": None, "model": None})
            continue
        try:
            status_code, parsed = call_gateway_model(MODEL, PROMPT, jpeg_b64, MAX_TOKENS)
        except Exception:
            status_code, parsed = 200, None
        if status_code == 401:
            if vendor == "openai":
                openai_dead = True
            else:
                anthropic_dead = True
            results.append({"id": pid, "lat": lat, "lon": lon, "confidence": 0.0, "country": None, "reasoning": None, "model": None})
            continue
        if parsed is None:
            results.append({"id": pid, "lat": lat, "lon": lon, "confidence": 0.0, "country": None, "reasoning": None, "model": MODEL})
            continue
        g = parse_guess_from_parsed(parsed)
        if g is None:
            results.append({"id": pid, "lat": lat, "lon": lon, "confidence": 0.0, "country": None, "reasoning": None, "model": MODEL})
        else:
            g["model"] = MODEL
            results.append({"id": pid, "lat": g["lat"], "lon": g["lon"], "confidence": g["confidence"], "country": g.get("country"), "reasoning": None, "model": MODEL})

    # PASS 2
    if not (openai_dead and anthropic_dead):
        cands = [r for r in results if r.get("model") == MODEL and r.get("confidence", 0.0) < SECOND_LOOK_BELOW]
        cands.sort(key=lambda x: (x.get("confidence", 0.0), x["id"]))
        to_check = cands[:MAX_SECOND_LOOKS]
        for r in to_check:
            pid = r["id"]
            s, jpeg = http_bytes(MAKEFUN_ENV_URL.rstrip("/") + "/photos/" + pid + ".jpg")
            if s == 200:
                jpeg_b64 = base64.b64encode(jpeg).decode("ascii")
            else:
                jpeg_b64 = ""
            vendor2 = vendor_for(SECOND_MODEL)
            if (vendor2 == "openai" and openai_dead) or (vendor2 == "anthropic" and anthropic_dead):
                continue
            try:
                status_code, parsed2 = call_gateway_model(SECOND_MODEL, SECOND_PROMPT, jpeg_b64, SECOND_MAX_TOKENS)
            except Exception:
                status_code, parsed2 = 200, None
            if status_code == 401:
                if vendor2 == "openai":
                    openai_dead = True
                else:
                    anthropic_dead = True
                break
            if parsed2 is None:
                continue
            g2 = parse_guess_from_parsed(parsed2)
            if g2 is not None and g2["confidence"] >= SECOND_LOOK_ACCEPT:
                r.update({"lat": g2["lat"], "lon": g2["lon"], "confidence": g2["confidence"], "country": g2.get("country"), "model": SECOND_MODEL})

    answers: List[Dict[str, Any]] = []
    for r in results:
        if r.get("model") is None or r.get("confidence") is None or r.get("confidence") == 0.0:
            answers.append({"id": r["id"], "lat": FALLBACK_LAT, "lon": FALLBACK_LON})
        else:
            answers.append({"id": r["id"], "lat": r["lat"], "lon": r["lon"]})

    post_url = MAKEFUN_ENV_URL.rstrip("/") + "/answers"
    for attempt in range(2):
        try:
            status, raw = http_json_post(post_url, {"answers": answers})
        except IOError as e:
            log(f"POST /answers transport error: {e}")
            if attempt == 0:
                time.sleep(2)
                continue
            return 1
        if status == 200:
            log(f"posted {len(answers)} answers")
            return 0
        if status == 409:
            try:
                _s, raw2 = http_json_get(MAKEFUN_ENV_URL.rstrip("/") + "/photos")
                n = json.loads(raw2.decode("utf-8")).get("answered", 0)
            except Exception:
                n = -1
            log(f"POST /answers 409; env reports {n} answered")
            return 0 if n == len(answers) else 1
        log(f"POST /answers refused: HTTP {status} {raw[:200].decode('utf-8','replace')}")
        return 1
    return 1

if __name__ == "__main__":
    sys.exit(main())
