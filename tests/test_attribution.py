"""
Offline tests for the two attribution methods and the attribution run.

Completeness and path variation are checked against scores whose right answer is
known by hand: a linear one, where integrated gradients is exact at any step
count, and an oscillating one, where no step count is enough. A tiny randomly
initialised ESM model stands in for ESM-2, and the UniProt request is replaced,
so nothing here needs a GPU or the network.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import make_record
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import EsmConfig, EsmModel

from src.config import load_config
from src.embeddings import pool
from src.interpretability.conservation import align_to_query, clean, conservation_scores
from src.interpretability.exact import split_margin
from src.interpretability.integrated_gradients import attribute, path_variation, quadrature
from src.interpretability.occlusion import occlude
from src.interpretability.scoring import classifier_margin, probe_direction, token_embedding
from src.model.esm2_lora import build_model

REPO = Path(__file__).resolve().parent.parent
N_CLASSES = 3
DEVICE = torch.device("cpu")


def load_module():
    path = REPO / "experiments" / "attribution" / "run_attribution.py"
    spec = importlib.util.spec_from_file_location("run_attribution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


attribution = load_module()


# ---------------------------------------------------------------------------
# Scores whose attribution is known by hand
# ---------------------------------------------------------------------------

@pytest.fixture
def linear_score():
    """A score linear in the token embeddings: integrated gradients is exact."""
    torch.manual_seed(0)
    table = torch.randn(30, 4)
    weights = torch.randn(4)

    def embed(input_ids):
        return table[input_ids]

    def score(tokens, mask):
        return (tokens[:, 1:-1] * weights).sum(dim=(1, 2))

    return embed, score, table, weights


def test_completeness_is_exact_for_a_linear_score(linear_score):
    embed, score, table, weights = linear_score
    ids = torch.tensor([[0, 5, 9, 14, 2]])
    baseline = table[7]
    result = attribute(embed, score, ids, baseline, n_steps=4, max_tokens=64)
    expected = ((table[ids[0, 1:-1]] - baseline) * weights).sum(dim=-1)
    assert np.allclose(result.scores, expected.numpy())
    # Not exactly zero: the sum is accumulated in float32
    assert result.relative_gap == pytest.approx(0.0, abs=1e-6)


def test_occlusion_is_exact_for_a_linear_score(linear_score):
    embed, score, table, weights = linear_score
    ids = torch.tensor([[0, 5, 9, 14, 2]])
    replacement = table[7]
    result = occlude(embed, score, ids, replacement, max_tokens=64)
    expected = ((table[ids[0, 1:-1]] - replacement) * weights).sum(dim=-1)
    assert np.allclose(result.scores, expected.numpy())


def test_path_variation_settles_for_a_linear_score(linear_score):
    embed, score, table, weights = linear_score
    ids = torch.tensor([[0, 5, 9, 14, 2]])
    coarse = path_variation(embed, score, ids, table[7], 20, max_tokens=64)
    fine = path_variation(embed, score, ids, table[7], 200, max_tokens=64)
    assert coarse["total_variation"] == pytest.approx(abs(coarse["net_change"]), rel=1e-6)
    assert fine["total_variation"] == pytest.approx(coarse["total_variation"], rel=1e-6)


def test_path_variation_grows_for_an_oscillating_score(linear_score):
    embed, _, table, weights = linear_score
    ids = torch.tensor([[0, 5, 9, 14, 2]])

    def rough(tokens, mask):
        return torch.sin(60 * (tokens[:, 1:-1] * weights).sum(dim=(1, 2)))

    coarse = path_variation(embed, rough, ids, table[7], 20, max_tokens=64)
    fine = path_variation(embed, rough, ids, table[7], 200, max_tokens=64)
    assert fine["total_variation"] > 3 * coarse["total_variation"]
    assert fine["total_variation"] > 10 * abs(fine["net_change"])


def test_quadrature_is_exact_for_polynomials():
    nodes, weights = quadrature(4)
    assert weights.sum().item() == pytest.approx(1.0)
    assert (weights * nodes ** 3).sum().item() == pytest.approx(0.25, abs=1e-5)


# ---------------------------------------------------------------------------
# Both methods on a stand-in for ESM-2
# ---------------------------------------------------------------------------

def save_tiny(tokenizer, path, token_dropout=True):
    config = EsmConfig(
        vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=1026,
        pad_token_id=tokenizer.pad_token_id, mask_token_id=tokenizer.mask_token_id,
        position_embedding_type="rotary", token_dropout=token_dropout,
    )
    torch.manual_seed(0)
    EsmModel(config, add_pooling_layer=False).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def tiny_model(path):
    torch.manual_seed(0)
    return build_model(path, N_CLASSES, 2, 4, 0.1, ["query", "value"], 0.1, DEVICE).eval()


@pytest.fixture(scope="module", params=[False, True], ids=["plain", "token_dropout"])
def model_and_ids(request, tokenizer, tmp_path_factory):
    path = save_tiny(tokenizer, tmp_path_factory.mktemp("tiny_esm"), request.param)
    model = tiny_model(path)
    ids = tokenizer("MKTAYIAKQRQISFVKSHFS", return_tensors="pt")["input_ids"]
    return model, ids, token_embedding(model, tokenizer.mask_token_id, DEVICE)


def test_embedding_route_matches_the_token_route(model_and_ids):
    model, ids, _ = model_and_ids
    mask = torch.ones_like(ids)
    with torch.no_grad():
        direct = model(input_ids=ids, attention_mask=mask)
        embedded = model.forward_embedded(model.embed_tokens(ids), mask)
    assert torch.allclose(direct, embedded, atol=1e-5)


def test_integrated_gradients_batching_does_not_change_the_scores(model_and_ids):
    model, ids, baseline = model_and_ids
    score = classifier_margin(model, 0, 1)
    one = attribute(model.embed_tokens, score, ids, baseline, n_steps=8, max_tokens=1)
    many = attribute(model.embed_tokens, score, ids, baseline, n_steps=8, max_tokens=10_000)
    assert np.allclose(one.scores, many.scores, atol=1e-5)


def test_swapping_the_classes_flips_the_sign(model_and_ids):
    model, ids, baseline = model_and_ids
    forward = attribute(
        model.embed_tokens, classifier_margin(model, 0, 1), ids, baseline, 8, 1024
    )
    backward = attribute(
        model.embed_tokens, classifier_margin(model, 1, 0), ids, baseline, 8, 1024
    )
    assert np.allclose(forward.scores, -backward.scores, atol=1e-5)


def test_a_baseline_equal_to_the_input_gives_no_credit(tokenizer, tmp_path):
    model = tiny_model(save_tiny(tokenizer, tmp_path))
    ids = tokenizer("AAAAAAAA", return_tensors="pt")["input_ids"]
    same = model.embed_tokens(ids)[0, 1].detach()
    result = attribute(model.embed_tokens, classifier_margin(model, 0, 1), ids, same, 4, 1024)
    assert np.allclose(result.scores, 0.0)
    assert result.gap == pytest.approx(0.0, abs=1e-6)


def test_occlusion_matches_masking_one_residue_at_a_time(model_and_ids):
    model, ids, baseline = model_and_ids
    score = classifier_margin(model, 0, 1)
    result = occlude(model.embed_tokens, score, ids, baseline, max_tokens=10_000)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        tokens = model.embed_tokens(ids)
        intact = score(tokens, mask).item()
        expected = []
        for position in range(1, ids.shape[1] - 1):
            one = tokens.clone()
            one[0, position] = baseline
            expected.append(intact - score(one, mask).item())
    assert np.allclose(result.scores, expected, atol=1e-5)
    assert result.score == pytest.approx(intact)


def test_occlusion_batching_does_not_change_the_scores(model_and_ids):
    model, ids, baseline = model_and_ids
    score = classifier_margin(model, 0, 1)
    one = occlude(model.embed_tokens, score, ids, baseline, max_tokens=1)
    many = occlude(model.embed_tokens, score, ids, baseline, max_tokens=10_000)
    assert np.allclose(one.scores, many.scores, atol=1e-6)


def test_occlusion_needs_a_residue(model_and_ids):
    model, _, baseline = model_and_ids
    with pytest.raises(ValueError, match="at least one residue"):
        occlude(model.embed_tokens, classifier_margin(model, 0, 1),
                torch.tensor([[0, 2]]), baseline, max_tokens=64)


def test_the_exact_split_adds_up_to_the_margin():
    torch.manual_seed(0)
    hidden = torch.randn(1, 7, 4)
    direction = torch.randn(4)
    result = split_margin(hidden, direction, 1.5)
    assert len(result.scores) == 5
    assert np.allclose(result.scores, (hidden[0, 1:-1] @ direction / 5).numpy())
    assert result.scores.sum() + result.offset == pytest.approx(result.score)
    assert result.gap < 1e-5


def test_the_exact_split_takes_one_sequence_with_residues():
    with pytest.raises(ValueError, match="One sequence at a time"):
        split_margin(torch.zeros(2, 5, 4), torch.zeros(4), 0.0)
    with pytest.raises(ValueError, match="at least one residue"):
        split_margin(torch.zeros(1, 2, 4), torch.zeros(4), 0.0)


def test_probe_direction_reproduces_the_fitted_margin():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 5))
    y = (x[:, 0] + rng.normal(0, 0.1, 60) > 0).astype(int) + (x[:, 1] > 1).astype(int)
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(x), y)
    direction, offset = probe_direction(
        clf.coef_, clf.intercept_, scaler.mean_, scaler.scale_, 2, 0
    )
    decision = clf.decision_function(scaler.transform(x))
    assert np.allclose(x @ direction + offset, decision[:, 2] - decision[:, 0])


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------

def test_alignment_keeps_query_positions():
    query = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    deleted = query[:10] + query[15:]
    inserted = query[:10] + "WWWWW" + query[10:]
    rows = align_to_query(query, [query, deleted, inserted])
    assert rows[0] == query
    assert all(len(row) == len(query) for row in rows)
    assert rows[1] == query[:10] + "-----" + query[15:]
    assert rows[2] == query


def test_alignment_accepts_rare_residues_and_unrelated_sequences():
    rows = align_to_query("MKTUYIAKQRQ", ["MKTOYIAKQRQ", "WWWW"])
    assert len(rows[0]) == 11
    assert "U" not in clean("MKTUY")


def test_conservation_scores():
    scores = conservation_scores(["AAAA", "AAAC", "AA-D", "AA-E"])
    assert scores[0] == pytest.approx(1.0)
    assert scores[2] == pytest.approx(0.5)
    assert 0 < scores[3] < 1
    assert conservation_scores(list("ACDEFGHIKLMNPQRSTVWY"))[0] == pytest.approx(0.0)
    assert conservation_scores(["-", "-"])[0] == 0.0


# ---------------------------------------------------------------------------
# Selection, annotations and statistics
# ---------------------------------------------------------------------------

def test_member_selection():
    ids = [f"p{i}" for i in range(50)]
    chosen = attribution.select_members(ids, ["p7"], 10, seed=1)
    assert chosen[0] == "p7"
    assert len(set(chosen)) == 10
    assert chosen == attribution.select_members(ids, ["p7"], 10, seed=1)
    assert len(attribution.select_members(ids[:3], ["p1"], 10, seed=1)) == 3
    with pytest.raises(ValueError, match="not in this family"):
        attribution.select_members(ids, ["missing"], 10, seed=1)


def test_competitor_is_the_true_class_when_wrong_else_the_runner_up():
    scores = np.array([0.0, 5.0, 3.0])
    assert attribution.choose_competitor(scores, 0) == (1, 0)
    assert attribution.choose_competitor(scores, 1) == (1, 2)


def test_annotations_are_zero_based_and_skip_uncertain_sites():
    def feature(kind, start, end, modifier="EXACT", code="ECO:0000255"):
        return {
            "type": kind,
            "location": {
                "start": {"value": start, "modifier": modifier},
                "end": {"value": end, "modifier": "EXACT"},
            },
            "evidences": [{"evidenceCode": code}],
        }

    entry = {
        "sequence": {"value": "MKTAY"},
        "features": [
            feature("Active site", 2, 2, code="ECO:0000269"),
            feature("Binding site", 3, 4),
            feature("Binding site", 1, 1, modifier="OUTSIDE"),
            feature("Signal", 1, 2),
        ],
    }
    assert attribution.parse_annotations(entry, "MKTAY") == {
        "active_site": [1], "binding_site": [2, 3], "active_site_experimental": [1],
    }
    assert attribution.parse_annotations(entry, "MKTAA") is None


def test_top_mask_and_hit_fraction():
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.0, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert attribution.top_mask(scores, 0.2).nonzero()[0].tolist() == [1, 3]
    assert attribution.top_mask(scores, 0.0).sum() == 1
    assert attribution.hit_fraction(scores, [1, 4], 0.2) == 0.5


def test_shift_test_separates_a_real_match_from_noise():
    rng = np.random.default_rng(0)
    signals = [rng.normal(size=200) for _ in range(5)]
    match = attribution.shift_test(
        signals, lambda s, i: attribution.rank_correlation(s, signals[i]), 200, 0
    )
    assert match["observed"] == pytest.approx(1.0)
    assert match["p_value"] < 0.01
    others = [rng.normal(size=200) for _ in range(5)]
    noise = attribution.shift_test(
        signals, lambda s, i: attribution.rank_correlation(s, others[i]), 200, 0
    )
    assert noise["p_value"] > 0.01
    assert abs(noise["null_mean"]) < 0.05


def test_constant_input_has_zero_correlation():
    assert attribution.rank_correlation(np.ones(5), np.arange(5.0)) == 0.0


# ---------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------

ALPHABETS = {0: "AILMV", 1: "DEKR", 2: "FWYH"}


def cache_embeddings(model, tokenizer, records, path):
    """The frozen model's mean embeddings, as src/embeddings.py would cache them."""
    means = []
    for record in records:
        ids = tokenizer(record["sequence"], return_tensors="pt")["input_ids"]
        mask = torch.ones_like(ids)
        with torch.no_grad(), model.esm.disable_adapter():
            hidden = model.esm(
                inputs_embeds=model.embed_tokens(ids), attention_mask=mask
            ).last_hidden_state
        means.append(pool(hidden, mask)[1][0].numpy())
    mean = np.array(means).astype(np.float16)
    np.savez(
        path,
        uniprot_ids=np.array([r["uniprot_id"] for r in records]),
        labels=np.array([r["label"] for r in records]),
        clusters=np.array([r["cluster_id"] for r in records]),
        mean=mean, cls=mean,
    )


def write_inputs(tmp_path, tokenizer, misaligned=False):
    rng = np.random.default_rng(0)
    processed = tmp_path / "processed"
    processed.mkdir()
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    model_path = save_tiny(tokenizer, tmp_path / "tiny")
    model = tiny_model(model_path)

    def sequence(label, length):
        return "".join(rng.choice(list(ALPHABETS[label]), size=length))

    parts = {}
    for split, per_class in [("train", 8), ("val", 4), ("test", 6)]:
        records = [
            make_record(
                f"{split}{label}-{i}" if split != "test" else f"{'abc'[label]}{i}",
                sequence=sequence(label, 30 + i), label=label,
                cluster_id=f"fam{'ABC'[label]}" if split == "test" else f"{split}{label}-{i}",
                ecs=(f"{label + 1}.1.1.1",),
            )
            for label in range(N_CLASSES)
            for i in range(per_class)
        ]
        parts[split] = records
        (processed / f"{split}.json").write_text(json.dumps(records))
        cache_embeddings(model, tokenizer, records, embeddings / f"{split}.npz")
    (processed / "label_map.json").write_text(json.dumps({"1.1": 0, "2.2": 1, "3.3": 2}))

    checkpoint = tmp_path / "best.pt"
    torch.save({"model": model.trainable_state()}, checkpoint)
    ids = [r["uniprot_id"] for r in parts["test"]]
    saved = {}
    for name in ["lora", "probe"]:
        saved[name] = tmp_path / f"{name}.npz"
        np.savez(
            saved[name], uniprot_ids=np.array(ids[::-1] if misaligned else ids),
            predictions=np.array([r["label"] for r in parts["test"]]),
        )

    cfg = load_config([
        f"data.output.processed_dir='{processed.as_posix()}'",
        f"embeddings.output_dir='{embeddings.as_posix()}'",
        f"lora.model_name='{model_path.as_posix()}'",
        "lora.r=2", "lora.alpha=4",
        "probe.C_grid=[0.1,1.0]",
        f"attribution.checkpoint='{checkpoint.as_posix()}'",
        f"attribution.lora_predictions_path='{saved['lora'].as_posix()}'",
        f"attribution.probe_predictions_path='{saved['probe'].as_posix()}'",
        "attribution.families=["
        "{cluster: famA, role: control, name: A, anchors: [a2]},"
        "{cluster: famB, role: failing, name: B, anchors: [b0]}]",
        "attribution.sequences_per_family=3",
        "attribution.robustness_sequences=1",
        "attribution.convergence_sequences=1",
        "attribution.variation_points=[10,20]",
        "attribution.convergence_steps=[2,4]",
        "attribution.n_shuffles=20",
        "attribution.max_homologs=4",
        f"attribution.work_dir='{(tmp_path / 'work').as_posix()}'",
        f"attribution.output_dir='{(tmp_path / 'results').as_posix()}'",
        f"tracking.dir='{(tmp_path / 'mlruns').as_posix()}'",
    ])
    return cfg, parts["test"]


def fake_fetch(records, calls):
    by_id = {r["uniprot_id"]: r for r in records}

    def fetch(base_url, accession):
        calls.append(accession)
        return {
            "sequence": {"value": by_id[accession]["sequence"]},
            "features": [{
                "type": "Active site",
                "location": {
                    "start": {"value": 3, "modifier": "EXACT"},
                    "end": {"value": 3, "modifier": "EXACT"},
                },
                "evidences": [],
            }],
        }, "TEST_RELEASE"

    return fetch


def test_run_end_to_end_and_reuses_annotations(tmp_path, tokenizer, monkeypatch):
    cfg, records = write_inputs(tmp_path, tokenizer)
    calls = []
    monkeypatch.setattr(attribution, "fetch_entry", fake_fetch(records, calls))
    attribution.run(cfg)
    assert len(calls) == 6
    attribution.run(cfg)
    assert len(calls) == 6

    results = json.loads((tmp_path / "results" / "results.json").read_text())
    assert results["uniprot_release"] == "TEST_RELEASE"
    assert [f["cluster"] for f in results["families"]] == ["famA", "famB"]
    family = results["families"][0]
    assert family["n_family"] == 6
    assert family["n_attributed"] == 3
    assert set(family["methods"]) == {"lora_masking", "probe_masking", "probe_exact"}
    assert list(family["anchors"]["lora_masking"]) == ["a2"]

    masking = family["methods"]["lora_masking"]
    assert "max_gap" not in masking
    assert masking["sites"]["active_site"]["scores"]["n_sequences"] == 3
    assert masking["sites"]["binding_site"]["scores"] is None
    assert "scores" not in masking["sequences"][0]
    assert masking["second_replacement_spearman"]["n_sequences"] == 1

    exact = family["methods"]["probe_exact"]
    assert exact["max_gap"] < 1e-4
    assert "second_replacement_spearman" not in exact
    for method in ["probe_masking", "probe_exact"]:
        assert np.isfinite(family["methods"][method]["max_difference_from_cached_margin"])

    assert set(family["method_agreement"]) == {
        "lora_masking_vs_probe_masking", "lora_masking_vs_probe_exact",
        "probe_masking_vs_probe_exact",
    }
    pair = family["method_agreement"]["probe_masking_vs_probe_exact"]
    assert -1 <= pair["mean_spearman"] <= 1
    assert 0 <= pair["mean_top_overlap"] <= 1

    assert len(family["integrated_gradients_convergence"]) == 1
    measured = family["integrated_gradients_convergence"][0]
    assert [row["n_steps"] for row in measured["lora"]["completeness_gap"]] == [2, 4]
    assert [row["n_points"] for row in measured["probe"]["variation"]] == [10, 20]
    assert set(measured["probe"]["variation"][0]) == {
        "n_points", "total_variation", "max_step", "net_change",
    }

    with np.load(tmp_path / "results" / "attributions.npz") as f:
        assert f["uniprot_ids"].tolist()[0] == "a2"
        assert f["lora_masking"].shape == (f["lengths"].sum(),)
        assert f["probe_exact"].shape == f["conservation"].shape


def test_misaligned_predictions_are_rejected(tmp_path, tokenizer):
    cfg, _ = write_inputs(tmp_path, tokenizer, misaligned=True)
    with pytest.raises(ValueError, match="not aligned"):
        attribution.run(cfg)


def test_a_changed_uniprot_sequence_drops_its_sites(tmp_path, tokenizer, monkeypatch):
    cfg, records = write_inputs(tmp_path, tokenizer)
    fetch = fake_fetch(records, [])

    def changed(base_url, accession):
        entry, release = fetch(base_url, accession)
        entry["sequence"]["value"] = "CHANGED"
        return entry, release

    monkeypatch.setattr(attribution, "fetch_entry", changed)
    attribution.run(cfg)
    family = json.loads((tmp_path / "results" / "results.json").read_text())["families"][0]
    assert family["n_sites_unavailable"] == 3
    assert family["methods"]["lora_masking"]["sites"]["active_site"]["scores"] is None
