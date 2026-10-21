"""Provenance stress test (manuscript 2.2, 3.2.3, research question 1): do C2PA credentials survive social-media processing?

Every test image is signed with a test certificate authority, AI images as
`trainedAlgorithmicMedia`, human images as `digitalCreation` (the best case:
every tool attaches credentials). Each signed file then goes through the same
conditions as the CNN and watermark stress tests and is checked three ways:

- stripped: the file exactly as re-encoded (what most platforms serve; the
  encoder writes no C2PA data),
- kept: the platform copies the original manifest into the re-encoded file
  (metadata-preserving upload); the manifest is there but its hash binding no
  longer matches the changed bytes,
- soft binding: the manifest is looked up in a repository of all signed
  images by perceptual hash (C2PA's way to recover stripped credentials). The
  match threshold is set on unsigned validation images, so at most
  `softbinding_target_fpr` of images that were never registered wrongly match
  one that was, under any condition.

Two forgeries are also checked on the clean files: moving a manifest from an
image of the other class onto an image (transplant), and signing an image with
the wrong claim using a certificate that is not on the trust list (self-signed).

Writes outputs/provenance/{results.csv, attacks.csv, softbinding.json, per_image.csv}.
Usage: python scripts/provenance_test.py [--max-images 500]
"""
import argparse
import csv
import io
import json
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from attribution.config import load_config, splits_dir
from attribution.data.splits import read_split
from attribution.signals import provenance as prov
from attribution.signals.provenance import ProvenanceStatus
from attribution.transforms.social_media import build_conditions, delivery_format, encode

MIME = {"JPEG": "image/jpeg", "WEBP": "image/webp"}
STATUSES = [s.value for s in ProvenanceStatus]
W: dict = {}  # per-worker state


def _init(cfg: dict, creds: prov.Credentials, forger: prov.Credentials) -> None:
    W.update(cfg=cfg, signer=prov.signer(creds), forger=prov.signer(forger), ctx=prov.context([creds.root_pem]))


def _decode(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as f:
        return f.convert("RGB")


def _source_type(label: int) -> str:
    p = W["cfg"]["provenance"]
    return p["ai_source_type"] if label == 1 else p["human_source_type"]


def _test_image(job: tuple[str, int]) -> dict:
    """Sign one image, then verify it under every condition, with the manifest stripped or kept."""
    path, label = job
    data = Path(path).read_bytes()
    signed, store = prov.sign(data, "image/jpeg", _source_type(label), W["signer"], W["ctx"])
    segments = prov.jpeg_manifest_segments(signed)
    img = _decode(signed)
    rows = []
    for name, t in build_conditions(W["cfg"]).items():
        if t is None:
            out, mime = signed, "image/jpeg"
        else:
            fmt, q = delivery_format(name, W["cfg"])
            out, mime = encode(t(img), fmt, q), MIME[fmt]
        stripped = prov.verify(out, mime, W["ctx"])
        kept = stripped if t is None else prov.verify(prov.carry_manifest(out, mime, segments, store), mime, W["ctx"])
        rows.append((stripped.status.value, stripped.claims_ai, kept.status.value, kept.claims_ai, prov.phash(_decode(out))))
    # forgery: same image signed with the opposite claim by a certificate outside the trust list
    forged, _ = prov.sign(data, "image/jpeg", _source_type(1 - label), W["forger"], W["ctx"])
    return {"rows": rows, "segments": segments, "self_signed": prov.verify(forged, "image/jpeg", W["ctx"]).status.value}


def _transplant(job: tuple[str, bytes]) -> str:
    """Another image's manifest copied onto this (unsigned) image."""
    path, segments = job
    return prov.verify(prov.carry_manifest(Path(path).read_bytes(), "image/jpeg", segments, b""), "image/jpeg", W["ctx"]).status.value


def _val_hashes(path: str) -> list[int]:
    with Image.open(path) as f:
        img = f.convert("RGB")
    out = []
    for name, t in build_conditions(W["cfg"]).items():
        if t is None:
            out.append(prov.phash(img))
        else:
            fmt, q = delivery_format(name, W["cfg"])
            out.append(prov.phash(_decode(encode(t(img), fmt, q))))
    return out


def _rates(statuses: list[str]) -> dict:
    return {f"{s}_rate": float(np.mean([x == s for x in statuses])) for s in STATUSES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-images", type=int, help="per class, for a quick run")
    ap.add_argument("--out", default="outputs/provenance")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    conditions = list(build_conditions(cfg))
    rng = random.Random(cfg["seed"])

    def subset(records):
        if args.max_images and len(records) > args.max_images:
            return rng.sample(records, args.max_images)
        return records

    test = read_split(splits_dir(cfg) / f"{args.split}.csv")
    images = subset([r for r in test if r.label == 1]) + subset([r for r in test if r.label == 0])
    val = subset(read_split(splits_dir(cfg) / "val.csv"))
    labels = np.array([r.label for r in images])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    creds, forger = prov.make_credentials(), prov.make_credentials("Untrusted forger")

    with Pool(args.workers, initializer=_init, initargs=(cfg, creds, forger)) as pool:
        print(f"signing and verifying {len(images)} test images under {len(conditions)} conditions", flush=True)
        res = pool.map(_test_image, [(r.path, r.label) for r in images], chunksize=16)
        # transplant: each image gets the manifest of an image of the other class
        ai_idx, human_idx = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
        donors = [human_idx[i % len(human_idx)] if l == 1 else ai_idx[i % len(ai_idx)] for i, l in enumerate(labels)]
        transplant = pool.map(_transplant, [(r.path, res[d]["segments"]) for r, d in zip(images, donors)], chunksize=16)
        print(f"hashing {len(val)} unsigned validation images for the soft-binding threshold", flush=True)
        val_hashes = np.array(pool.map(_val_hashes, [r.path for r in val], chunksize=16), dtype=np.uint64)

    # soft binding: repository = fingerprints of the signed originals
    repo = np.array([r["rows"][0][4] for r in res], dtype=np.uint64)
    target = cfg["provenance"]["softbinding_target_fpr"]
    val_min = np.stack([prov.hamming(val_hashes[:, c], repo).min(1) for c in range(len(conditions))], 1)
    threshold = max([t for t in range(65) if (val_min <= t).mean(0).max() <= target], default=-1)
    false_match = {name: float((val_min[:, c] <= threshold).mean()) for c, name in enumerate(conditions)}
    (out / "softbinding.json").write_text(json.dumps(
        {"hash": "64-bit DCT pHash", "threshold_bits": threshold, "target_fpr": target,
         "repository_size": len(repo), "false_match_rate": false_match}, indent=1))
    print(f"soft-binding threshold = {threshold} bits (unregistered images wrongly matched at most {target:.0%} under every condition)", flush=True)

    rows, per_image = [], []
    for c, name in enumerate(conditions):
        hashes = np.array([r["rows"][c][4] for r in res], dtype=np.uint64)
        d = prov.hamming(hashes, repo)
        own = d[np.arange(len(res)), np.arange(len(res))]
        dmin = d.min(1)
        recovered = (own <= threshold) & (own == dmin)
        wrong = (dmin <= threshold) & ~recovered
        for i, (r, img) in enumerate(zip(res, images)):
            s_status, s_ai, k_status, k_ai, _ = r["rows"][c]
            per_image.append((name, img.path, img.label, s_status, s_ai, k_status, k_ai, bool(recovered[i]), bool(wrong[i])))
        for label, cls in [(1, "ai"), (0, "human")]:
            m = labels == label
            sel = [r["rows"][c] for r, keep in zip(res, m) if keep]
            for policy, si in [("stripped", 0), ("kept", 2)]:
                statuses = [x[si] for x in sel]
                correct = [x[si] == "valid" and x[si + 1] == bool(label) for x in sel]
                rows.append({"policy": policy, "condition": name, "class": cls, "n": len(sel), **_rates(statuses),
                             "correct_rate": float(np.mean(correct)), "wrong_match_rate": None, "false_match_rate": None})
            rows.append({"policy": "soft_binding", "condition": name, "class": cls, "n": int(m.sum()),
                         **{f"{s}_rate": None for s in STATUSES}, "correct_rate": float(recovered[m].mean()),
                         "wrong_match_rate": float(wrong[m].mean()), "false_match_rate": false_match[name]})
        line = "  ".join(f"{p} {np.mean([x['valid_rate'] if p != 'soft_binding' else x['correct_rate'] for x in rows if x['policy'] == p and x['condition'] == name]):.1%}"
                         for p in ["stripped", "kept", "soft_binding"])
        print(f"  {name:>14}: credential usable  {line}", flush=True)

    attacks = [{"attack": "transplant", "n": len(transplant), **_rates(transplant)},
               {"attack": "self_signed", "n": len(res), **_rates([r["self_signed"] for r in res])}]
    for a in attacks:
        print(f"  attack {a['attack']}: accepted as valid {a['valid_rate']:.1%}, untrusted {a['untrusted_rate']:.1%}, invalid {a['invalid_rate']:.1%}")

    for fname, data in [("results.csv", rows), ("attacks.csv", attacks)]:
        with open(out / fname, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(data[0]))
            wr.writeheader()
            wr.writerows(data)
    with open(out / "per_image.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["condition", "path", "label", "stripped_status", "stripped_claims_ai", "kept_status", "kept_claims_ai",
                     "softbind_recovered", "softbind_wrong_match"])
        wr.writerows(per_image)
    print(f"results: {out / 'results.csv'}")


if __name__ == "__main__":
    main()
