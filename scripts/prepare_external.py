"""Build the external (unseen-generator) test set from WikiArt_VLM (manuscript section 3.8.2).

WikiArt_VLM (Fu et al., 2026; Zenodo 10.5281/zenodo.17880956, CC BY 4.0) pairs
39,530 WikiArt paintings with images generated from a caption of each painting
by FLUX.1-dev, F-Lite and Stable Diffusion 3.5-large. FLUX.1 and F-Lite are not
in the training data; SD 3.5 is a newer generation of the Stable Diffusion
family that AI-ArtBench contributes.

Reads the downloaded zips directly (no full extraction), samples
--per-generator images from each generator and the matching number of WikiArt
originals, and harmonizes them exactly like the training data (shorter side to
`preprocess.harmonize.size`, center crop, JPEG at `jpeg_quality`). When the
generated files can be matched to their originals by file name, sampling is
paired: every sampled painting appears once as a generated image (from one
generator) and once as the human original, so subject matter can't tell the
classes apart. Otherwise the two sides are sampled independently.

External images that are near-duplicates (dHash distance <= --max-distance) of
any image in the training pool are removed (AI-ArtBench's human images come from
ArtBench-10, which was drawn from WikiArt).

Writes <out>/<source>/<n>.jpg, <out>/external.csv (path,label,source,member),
<out>/removed_near_duplicates.csv and <out>/external_summary.json.
Usage:
  python scripts/prepare_external.py --zips /content/wikiart_vlm --inspect
  python scripts/prepare_external.py --zips /content/wikiart_vlm --per-generator 1500
"""
import argparse
import csv
import io
import json
import os
import random
import zipfile
from pathlib import Path

from PIL import Image

from attribution.config import load_config, splits_dir
from attribution.data.datasets import IMAGE_EXTS, Record
from attribution.data.eda import cross_split_near_duplicates
from attribution.data.splits import read_split
from attribution.transforms.harmonize import harmonize

GENERATORS = {"flux": "FLUX.zip", "f_lite": "F-Lite.zip", "sd35": "Stable-Diffusion.zip"}
HUMAN = ("wikiart", "Original.zip")


def image_members(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as z:
        return sorted(n for n in z.namelist()
                      if not n.endswith("/") and Path(n).suffix.lower() in IMAGE_EXTS and "__MACOSX" not in n)


def stem_key(member: str) -> str:
    """Matching key for a file: its name without folder or extension, lower case."""
    return Path(member).stem.lower()


def inspect(zips: dict[str, Path]) -> None:
    for source, path in zips.items():
        members = image_members(path)
        folders = sorted({str(Path(m).parent) for m in members})
        print(f"{source}: {path.name}: {len(members)} images in {len(folders)} folders")
        for f in folders[:5]:
            print(f"    folder: {f}")
        for m in members[:5]:
            print(f"    e.g. {m}")
    human = {stem_key(m) for m in image_members(zips[HUMAN[0]])}
    for source in GENERATORS:
        if source in zips:
            keys = {stem_key(m) for m in image_members(zips[source])}
            print(f"{source}: {len(keys & human)} of {len(keys)} file names also appear in {HUMAN[1]}")


def choose(zips: dict[str, Path], per_generator: int, seed: int, pairing: str) -> tuple[dict[str, list[str]], str]:
    """Members to use per source and the pairing mode actually used."""
    rng = random.Random(seed)
    members = {s: image_members(p) for s, p in zips.items()}
    by_key = {s: {stem_key(m): m for m in ms} for s, ms in members.items()}
    human_keys = set(by_key[HUMAN[0]])
    gens = [g for g in GENERATORS if g in zips]
    overlap = min(len(set(by_key[g]) & human_keys) / max(1, len(by_key[g])) for g in gens)
    mode = pairing if pairing != "auto" else ("paired" if overlap >= 0.5 else "random")
    chosen: dict[str, list[str]] = {}
    if mode == "paired":
        used: set[str] = set()
        human: list[str] = []
        for g in gens:
            candidates = sorted((set(by_key[g]) & human_keys) - used)
            rng.shuffle(candidates)
            take = candidates[:per_generator]
            used.update(take)
            chosen[g] = [by_key[g][k] for k in take]
            human += [by_key[HUMAN[0]][k] for k in take]
        chosen[HUMAN[0]] = human
    else:
        for g in gens:
            pool = members[g][:]
            rng.shuffle(pool)
            chosen[g] = pool[:per_generator]
        pool = members[HUMAN[0]][:]
        rng.shuffle(pool)
        chosen[HUMAN[0]] = pool[:per_generator * len(gens)]
    return chosen, mode


def extract(zips: dict[str, Path], chosen: dict[str, list[str]], out: Path, size: int, quality: int) -> list[dict]:
    """Harmonized copies (center crop, as in training) plus, under _squeezed/, the whole image resized to a square.

    The squeezed copy is only for the leakage check: another dataset may have resized
    the same painting without cropping, which gives a different perceptual hash.
    """
    rows = []
    for source, picked in chosen.items():
        label = 0 if source == HUMAN[0] else 1
        (out / source).mkdir(parents=True, exist_ok=True)
        (out / "_squeezed" / source).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zips[source]) as z:
            for i, member in enumerate(picked):
                dst = out / source / f"{i:06d}.jpg"
                try:
                    with Image.open(io.BytesIO(z.read(member))) as img:
                        harmonize(img, size).save(dst, "JPEG", quality=quality)
                        img.convert("RGB").resize((size, size), Image.BICUBIC).save(
                            out / "_squeezed" / source / f"{i:06d}.jpg", "JPEG", quality=quality)
                except (OSError, ValueError) as exc:
                    print(f"Warning: skipped {source}/{member}: {exc}")
                    continue
                rows.append({"path": str(dst), "label": label, "source": source, "member": member})
        print(f"  {source}: {sum(r['source'] == source for r in rows)} images", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--zips", required=True, help="folder with FLUX.zip, F-Lite.zip, Stable-Diffusion.zip, Original.zip")
    ap.add_argument("--out", default="data/external")
    ap.add_argument("--per-generator", type=int, default=1500)
    ap.add_argument("--pairing", choices=["auto", "paired", "random"], default="auto")
    ap.add_argument("--max-distance", type=int, default=5)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--inspect", action="store_true", help="only list what the zips contain")
    args = ap.parse_args()
    cfg = load_config(args.config)
    folder = Path(args.zips)
    zips = {s: folder / name for s, name in [*GENERATORS.items(), HUMAN] if (folder / name).is_file()}
    if HUMAN[0] not in zips or len(zips) < 2:
        raise FileNotFoundError(f"need {HUMAN[1]} and at least one generator zip in {folder}; found {sorted(zips)}")
    if args.inspect:
        inspect(zips)
        return

    out = Path(args.out)
    chosen, mode = choose(zips, args.per_generator, int(cfg["seed"]), args.pairing)
    print(f"sampling mode: {mode}; " + ", ".join(f"{s}: {len(m)}" for s, m in chosen.items()), flush=True)
    h = cfg["preprocess"]["harmonize"]
    rows = extract(zips, chosen, out, int(h["size"]), int(h["jpeg_quality"]))

    # leakage check against every image the models could have been trained on
    pool = [r for name in cfg["data"]["split"] for r in read_split(splits_dir(cfg) / f"{name}.csv")]
    external = [Record(r["path"], int(r["label"]), r["source"]) for r in rows]
    squeezed_of = {os.path.normpath(str(out / "_squeezed" / Path(r["path"]).relative_to(out))): os.path.normpath(r["path"])
                   for r in rows}
    squeezed = [Record(p, 0, "squeezed") for p in squeezed_of]
    print(f"checking {len(external)} external images (cropped and uncropped) against {len(pool)} training-pool images",
          flush=True)
    pairs = cross_split_near_duplicates({"external": external, "squeezed": squeezed, "pool": pool},
                                        args.max_distance, workers=args.workers)
    flagged = set()
    for p in pairs:
        if "pool" not in (p["split_a"], p["split_b"]):
            continue  # an external image vs its own squeezed copy
        side = "a" if p["split_a"] != "pool" else "b"
        path = os.path.normpath(p[f"path_{side}"])
        flagged.add(squeezed_of.get(path, path))
    import shutil

    shutil.rmtree(out / "_squeezed", ignore_errors=True)
    kept = [r for r in rows if os.path.normpath(r["path"]) not in flagged]
    removed = [r for r in rows if os.path.normpath(r["path"]) in flagged]

    for name, data in [("external.csv", kept), ("removed_near_duplicates.csv", removed)]:
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "label", "source", "member"])
            w.writeheader()
            w.writerows(data)
    summary = {
        "dataset": "WikiArt_VLM (Fu et al., 2026), Zenodo 10.5281/zenodo.17880956, CC BY 4.0",
        "sampling": mode,
        "per_generator_requested": args.per_generator,
        "harmonize": h,
        "max_distance": args.max_distance,
        "sources": {s: {"sampled": sum(r["source"] == s for r in rows),
                        "removed_near_duplicates": sum(r["source"] == s for r in removed),
                        "kept": sum(r["source"] == s for r in kept)} for s in chosen},
        "kept_total": len(kept),
    }
    (out / "external_summary.json").write_text(json.dumps(summary, indent=1))
    for s, c in summary["sources"].items():
        print(f"  {s}: {c['kept']} kept ({c['removed_near_duplicates']} removed as near-duplicates of training images)")
    print(f"wrote {out / 'external.csv'} ({len(kept)} images)")


if __name__ == "__main__":
    main()
