# make.fun — Find where a photo was taken (photo-geolocator-01)

Entry for the [make.fun](https://make.fun) `photo-geolocator-01` bounty ($250).
25 street-level photos per trial from OpenStreetView-5M; the agent answers a
latitude/longitude for each. Score = 500,377 − total error in km, ranked over
6 sealed trials (150 photos). House baseline to beat: 460,000 (40,795 km).

## How it works (`main.py`, Python 3.13 stdlib only)

- **Pass 1:** one vision call per photo (default `gpt-6-luna` via the make.fun
  model gateway, OpenAI Chat Completions) with a structured clue-extraction
  prompt: signs/script, driving side, road markings, vegetation, architecture,
  vehicles/plates, sun/shadows — then a strict JSON
  `{lat, lon, confidence, country, reasoning}`.
- **Pass 2:** photos with pass-1 confidence < 0.35 get one second look with a
  shorter prompt; the new answer replaces the first only at confidence ≥ 0.60
  (max 8 second looks).
- **Fallback:** any photo with no usable model answer gets the fixed prior
  (50.0, 10.0) — the best single fixed point across the practice trials.
  Every photo is always answered; a model failure never crashes the run.
- Deterministic: photos in id order, sequential calls, seeded tie-breaks.
  A `claude-*` model can be selected with `MAKEFUN_MODEL` (Anthropic route)
  if an Anthropic key is pushed instead.

Cost is on the entrant's own API key (house pays none): ~$0.06–0.07 per trial
at gpt-6-luna pricing, far under the $1.00/trial cap.

## License

MIT — see `LICENSE`.
