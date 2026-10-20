from attribution.config import load_config
from attribution.data.datasets import index_dataset, label_for
from attribution.data.eda import channel_mean_std, class_balance, cross_split_duplicates, image_stats
from attribution.data.splits import stratified_split


def test_default_rules_label_known_layouts():
    ds = load_config()["data"]["datasets"]
    k = ds["kannan_ai_human_art"]["classes"]
    assert label_for("train/NON_AI_GENERATED", k) == 0
    assert label_for("train/AI_GENERATED", k) == 1
    a = ds["ai_artbench"]["classes"]
    root = "Real_AI_SD_LD_Dataset"  # the real top folder, whose name contains "AI_"
    assert label_for(f"{root}/train/AI_SD_impressionism", a) == 1
    assert label_for(f"{root}/test/AI_LD_baroque", a) == 1
    assert label_for(f"{root}/train/impressionism", a) == 0
    assert label_for(f"{root}/test/ukiyo_e", a) == 0
    s = ds["superpotato9_ai_recognition"]["classes"]
    assert label_for("fakeV2/fake-v2", s) == 1
    assert label_for("real", s) == 0


def _make_dataset(root, art_image, n=6):
    for folder in ("real", "fake"):
        (root / folder).mkdir(parents=True)
        for i in range(n):
            art_image.rotate(i * 7 + (folder == "fake")).save(root / folder / f"{i}.png")
    (root / "notes").mkdir()
    art_image.save(root / "notes" / "x.png")


def test_index_split_and_eda(tmp_path, art_image):
    _make_dataset(tmp_path, art_image)
    spec = {"path": str(tmp_path), "classes": [{"match": "real", "label": 0}, {"match": "fake", "label": 1}]}
    records, unmatched = index_dataset("toy", spec)
    assert len(records) == 12 and unmatched == 1

    balance = class_balance(records)["overall"]
    assert balance["0"]["p"] == balance["1"]["p"] == 0.5

    splits = stratified_split(records, {"train": 0.5, "val": 0.25, "test": 0.25}, seed=0)
    assert sum(len(v) for v in splits.values()) == 12
    assert cross_split_duplicates(splits) == {}

    stats = [image_stats(r.path, 32) for r in splits["train"]]
    mean, std = channel_mean_std(stats)
    assert len(mean) == 3 and all(0 < m < 1 for m in mean) and all(s > 0 for s in std)


def test_duplicates_detected(tmp_path, art_image):
    _make_dataset(tmp_path, art_image, n=2)
    spec = {"path": str(tmp_path), "classes": [{"match": "*", "label": 0}]}
    records, _ = index_dataset("toy", spec)
    # notes/x.png is byte-identical to real/0.png (rotate(0) of the same image).
    by_name = {r.path.split("/")[-2] + "/" + r.path.split("/")[-1]: r for r in records}
    dup = cross_split_duplicates({"train": [by_name["real/0.png"]], "test": [by_name["notes/x.png"]]})
    assert dup == {"test+train": 1}


def test_sampling_balances_sources_and_redistributes_shortfall():
    from attribution.data.datasets import Record
    from attribution.data.splits import sample_per_class

    records = (
        [Record(f"a{i}", 0, "small") for i in range(10)]
        + [Record(f"b{i}", 0, "big1") for i in range(1000)]
        + [Record(f"c{i}", 0, "big2") for i in range(1000)]
    )
    out = sample_per_class(records, 100, seed=0)
    counts = {s: sum(r.source == s for r in out) for s in ("small", "big1", "big2")}
    assert counts == {"small": 10, "big1": 45, "big2": 45}
    pooled = sample_per_class(records, 100, seed=0, balance_sources=False)
    assert len(pooled) == 100


def test_deduplicate_keeps_one_copy_and_drops_label_conflicts():
    from attribution.data.datasets import Record
    from attribution.data.splits import deduplicate

    records = [Record("a", 0, "s"), Record("a2", 0, "s"), Record("b", 1, "s"), Record("c", 0, "s"), Record("c2", 1, "s")]
    hashes = ["h1", "h1", "h2", "h3", "h3"]
    kept, removed, conflicts = deduplicate(records, hashes)
    assert [r.path for r in kept] == ["a", "b"]
    assert removed == 3 and conflicts == 1
