"""
experiments/attribution/run_attribution.py

Which residues drive the predictions, on chosen families (T5.7, experiment E8).

Three attributions, and one method that was measured and rejected:
    lora_masking   the fine-tuned model, masking one residue at a time
    probe_masking  the probe, the same way
    probe_exact    the probe, split over positions exactly (src/.../exact.py)
    integrated gradients: not reported as scores. On this model the sum of the
        attributions misses the change in margin it should explain by a third to
        a half, and the error only halves when the step count quadruples, which
        puts a 5% gap at thousands of steps per sequence. The run measures that
        instead of hiding it: for a few sequences per family it records the
        completeness gap at two step counts and the total variation of the margin
        along the path at two resolutions, for both models. Frozen ESM-2 gives a
        path about three times smoother than the fine-tuned model, and still not
        smooth enough. That is the only evidence in the project about what
        fine-tuning did to the model's local geometry, so it is a result.

What this script does:
    1. Picks sequences from three failing families, the class one of them is
       mistaken for, and three families both models get right
    2. Attributes each one under all three methods, against the class the model
       would otherwise have chosen
    3. Compares the scores with conservation across the family, and with the
       active and binding sites annotated in UniProt
    4. Reports how far the methods agree with each other, because masking and the
       exact split answer different questions and mostly disagree
    5. Repeats a few sequences with a second replacement residue (D9), to show
       how much that choice matters
    6. Saves results.json and the per-residue arrays, and logs the run to MLflow

Why a circular shift as the null?
    Scores and conservation both vary smoothly along a sequence, which inflates
    any correlation. Rotating the scores keeps that smoothness and destroys only
    the match of positions. Sequences in a family are not independent, so these
    p-values describe this family's sample, with no claim about enzymes in general.

Why is using test allowed here?
    Nothing is fitted or chosen from these numbers; the saved models only explain
    themselves. UniProt annotations partly come from similarity to other family
    members, so a match to them supports a known family pattern, not a discovery.
"""

import json
import logging
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import requests
import torch
from omegaconf import DictConfig
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from transformers import EsmTokenizer

from src.config import cli_config, config_as_dict
from src.embeddings import embeddings_path
from src.interpretability.conservation import align_to_query, clean, conservation_scores
from src.interpretability.exact import split_margin
from src.interpretability.integrated_gradients import attribute, path_variation
from src.interpretability.occlusion import occlude
from src.interpretability.scoring import (
    AMINO_ACIDS,
    classifier_margin,
    frozen_hidden,
    mean_residue_embedding,
    probe_direction,
    probe_margin,
    token_embedding,
)
from src.model.esm2_lora import build_model
from src.probe import fit_and_select
from src.tracking import git_state, log_run
from src.training.trainer import get_device, guard_memory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

SITE_TYPES = {"Active site": "active_site", "Binding site": "binding_site"}
EXPERIMENTAL = "ECO:0000269"
SITE_NAMES = ["active_site", "active_site_experimental", "binding_site"]
METHODS = {
    "lora_masking": "fine-tuned model, masking one residue at a time",
    "probe_masking": "probe, masking one residue at a time",
    "probe_exact": "probe, split over positions exactly",
}
MAX_PATH_DIFFERENCE = 0.01      # logits through input_ids and through embeddings must agree
RETRY_SECONDS = (1, 5, 20)


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------

def select_members(ids: list[str], anchors: list[str], n: int, seed: int) -> list[str]:
    """The anchors, then a seeded random sample of the family, n in total."""
    missing = [a for a in anchors if a not in ids]
    if missing:
        raise ValueError(f"Anchors not in this family: {missing}")
    others = [i for i in ids if i not in anchors]
    rng = np.random.default_rng(seed)
    n_rest = max(0, min(n - len(anchors), len(others)))
    picked = rng.choice(len(others), size=n_rest, replace=False)
    return list(anchors) + [others[i] for i in sorted(picked)]


def choose_competitor(scores: np.ndarray, true_label: int) -> tuple[int, int]:
    """(predicted, competitor): the true class if the model is wrong, else the runner-up."""
    predicted = int(np.argmax(scores))
    if predicted != true_label:
        return predicted, true_label
    return predicted, int(np.argsort(-scores)[1])


# ---------------------------------------------------------------------------
# UniProt annotations
# ---------------------------------------------------------------------------

def parse_annotations(entry: dict, sequence: str) -> dict | None:
    """
    Zero-based site positions from a UniProt entry. None if UniProt's sequence
    has changed since the dataset was downloaded, as positions would not match.
    Sites with uncertain ends are skipped.
    """
    if entry.get("sequence", {}).get("value") != sequence:
        return None
    sites = {name: [] for name in SITE_NAMES}
    for feature in entry.get("features", []):
        name = SITE_TYPES.get(feature.get("type"))
        location = feature.get("location", {})
        start, end = location.get("start", {}), location.get("end", {})
        if name is None or start.get("modifier") != "EXACT" or end.get("modifier") != "EXACT":
            continue
        positions = list(range(start["value"] - 1, end["value"]))
        sites[name].extend(positions)
        codes = {e.get("evidenceCode") for e in feature.get("evidences", [])}
        if name == "active_site" and EXPERIMENTAL in codes:
            sites["active_site_experimental"].extend(positions)
    return {name: sorted(set(values)) for name, values in sites.items()}


def fetch_entry(base_url: str, accession: str) -> tuple[dict, str | None]:
    """One UniProt entry and the UniProt release that served it."""
    url = f"{base_url}/{accession}.json"
    params = {"fields": "sequence,ft_act_site,ft_binding"}
    for wait in (*RETRY_SECONDS, None):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code not in (429, 500, 502, 503, 504):
                response.raise_for_status()
                return response.json(), response.headers.get("X-UniProt-Release")
        except requests.ConnectionError:
            if wait is None:
                raise
        if wait is None:
            response.raise_for_status()
        time.sleep(wait)
    raise RuntimeError("Unreachable")


def load_annotations(cfg: DictConfig, records: list[dict], path: Path) -> dict:
    """Annotations by accession, cached, because they change with each UniProt release."""
    cache = {"release": None, "sites": {}}
    if path.exists():
        cache = json.loads(path.read_text())
    todo = [r for r in records if r["uniprot_id"] not in cache["sites"]]
    for i, record in enumerate(todo):
        entry, release = fetch_entry(cfg.attribution.uniprot_url, record["uniprot_id"])
        cache["sites"][record["uniprot_id"]] = parse_annotations(entry, record["sequence"])
        cache["release"] = release or cache["release"]
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            logger.info(f"UniProt annotations: {i + 1}/{len(todo)}")
    if todo:
        path.write_text(json.dumps(cache))
    return cache


# ---------------------------------------------------------------------------
# Comparison with conservation and annotated sites
# ---------------------------------------------------------------------------

def top_mask(scores: np.ndarray, fraction: float) -> np.ndarray:
    """True for the highest-scoring residues, at least one. Ties go to the earlier position."""
    k = max(1, int(np.ceil(len(scores) * fraction)))
    mask = np.zeros(len(scores), dtype=bool)
    mask[np.argsort(-scores, kind="stable")[:k]] = True
    return mask


def hit_fraction(scores: np.ndarray, positions: list[int], fraction: float) -> float:
    """The share of the given positions that are among the highest scores."""
    return float(top_mask(scores, fraction)[positions].mean())


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation; 0 if either input is constant."""
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def shift_test(scores: list[np.ndarray], statistic, n_shuffles: int, seed: int) -> dict:
    """
    The mean of statistic(scores, i) over sequences, and a one-sided p-value from
    rotating every sequence's scores by its own random offset.
    """
    rng = np.random.default_rng(seed)
    observed = float(np.mean([statistic(s, i) for i, s in enumerate(scores)]))
    null = np.array([
        np.mean([
            statistic(np.roll(s, rng.integers(1, len(s))), i) for i, s in enumerate(scores)
        ])
        for _ in range(n_shuffles)
    ])
    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "p_value": float((1 + (null >= observed).sum()) / (1 + n_shuffles)),
    }


def site_summary(
    scores: list[np.ndarray], sites: list[list[int]], fraction: float, n_shuffles: int, seed: int
) -> dict | None:
    """How often annotated sites are among the top scores, over sequences that have any."""
    keep = [i for i, positions in enumerate(sites) if positions]
    if not keep:
        return None
    kept_scores, kept_sites = [scores[i] for i in keep], [sites[i] for i in keep]
    test = shift_test(
        kept_scores, lambda s, i: hit_fraction(s, kept_sites[i], fraction), n_shuffles, seed
    )
    return {
        "n_sequences": len(keep),
        "mean_sites_per_sequence": float(np.mean([len(p) for p in kept_sites])),
        "share_in_top": test["observed"],
        "share_under_shift": test["null_mean"],
        "p_value": test["p_value"],
    }


def top_residues(row: dict, method: str, n: int) -> list[dict]:
    """The highest-scoring residues, numbered from 1 as in UniProt."""
    sites, scores = row["sites"] or {}, row["methods"][method]["scores"]
    result = []
    for position in np.argsort(-np.abs(scores), kind="stable")[:n]:
        position = int(position)
        result.append({
            "residue": f"{row['sequence'][position]}{position + 1}",
            "score": float(scores[position]),
            "conservation": float(row["conservation"][position]),
            "active_site": position in sites.get("active_site", []),
            "binding_site": position in sites.get("binding_site", []),
        })
    return result


def summarise_method(rows: list[dict], method: str, settings: DictConfig, seed: int) -> dict:
    """One method on one family: what it predicted, and what its scores line up with."""
    entries = [row["methods"][method] for row in rows]
    importance = [np.abs(entry["scores"]) for entry in entries]
    conservation = [row["conservation"] for row in rows]
    fraction, n_shuffles = settings.top_fraction, settings.n_shuffles
    correlation = shift_test(
        importance, lambda s, i: rank_correlation(s, conservation[i]), n_shuffles, seed
    )
    summary = {
        "method": METHODS[method],
        "accuracy": float(np.mean([e["predicted"] == e["true_label"] for e in entries])),
        "predicted": dict(Counter(e["predicted_subclass"] for e in entries)),
        "competitor": dict(Counter(e["competitor_subclass"] for e in entries)),
        "median_margin": float(np.median([e["margin"] for e in entries])),
        "score_vs_conservation": {
            "mean_spearman": correlation["observed"],
            "mean_spearman_under_shift": correlation["null_mean"],
            "p_value": correlation["p_value"],
        },
        "sites": {},
    }
    gaps = [e["gap"] for e in entries if e.get("gap") is not None]
    if gaps:
        summary["max_gap"] = float(np.max(gaps))
    differences = [
        e["difference_from_cached_margin"] for e in entries
        if e.get("difference_from_cached_margin") is not None
    ]
    if differences:
        summary["max_difference_from_cached_margin"] = float(np.max(differences))
    second = [e["baseline_spearman"] for e in entries if e.get("baseline_spearman") is not None]
    if second:
        summary["second_replacement_spearman"] = {
            "n_sequences": len(second),
            "mean": float(np.mean(second)),
            "min": float(np.min(second)),
        }
    for name in SITE_NAMES:
        positions = [(row["sites"] or {}).get(name, []) for row in rows]
        summary["sites"][name] = {
            "scores": site_summary(importance, positions, fraction, n_shuffles, seed),
            "conservation": site_summary(conservation, positions, fraction, n_shuffles, seed),
        }
    summary["sequences"] = [
        {
            **{k: v for k, v in row["methods"][method].items() if k != "scores"},
            "spearman_with_conservation": rank_correlation(
                np.abs(row["methods"][method]["scores"]), row["conservation"]
            ),
        }
        for row in rows
    ]
    return summary


def agreement(rows: list[dict], fraction: float) -> dict:
    """How far the methods pick out the same residues as each other."""
    result = {}
    for first, second in combinations(METHODS, 2):
        spearmans, overlaps = [], []
        for row in rows:
            a = np.abs(row["methods"][first]["scores"])
            b = np.abs(row["methods"][second]["scores"])
            spearmans.append(rank_correlation(a, b))
            top_a, top_b = top_mask(a, fraction), top_mask(b, fraction)
            overlaps.append(float((top_a & top_b).sum() / max(top_a.sum(), 1)))
        result[f"{first}_vs_{second}"] = {
            "mean_spearman": float(np.mean(spearmans)),
            "mean_top_overlap": float(np.mean(overlaps)),
        }
    return result


# ---------------------------------------------------------------------------
# The two models
# ---------------------------------------------------------------------------

def fit_probe(cfg: DictConfig, labels: list[int]) -> tuple:
    """Refit the probe as E1 did, and return it with the cached test embeddings."""
    data = {}
    for split in ["train", "val", "test"]:
        path = embeddings_path(cfg, split)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: uv run python -m src.embeddings")
        with np.load(path) as f:
            data[split] = {
                "ids": f["uniprot_ids"].tolist(),
                "labels": f["labels"],
                "mean": f["mean"].astype(np.float32),
            }
    scaler = StandardScaler().fit(data["train"]["mean"])
    clf, _ = fit_and_select(
        scaler.transform(data["train"]["mean"]), data["train"]["labels"],
        scaler.transform(data["val"]["mean"]), data["val"]["labels"], labels, cfg.probe,
    )
    if clf.classes_.tolist() != labels:
        # Otherwise a coefficient row would not belong to the class index it is read with
        raise ValueError("The probe did not see every class in training.")
    return clf, scaler, data["test"]["ids"], data["test"]["mean"]


def probe_scorer(context: dict, model, device: torch.device, true_label: int) -> tuple:
    """The probe's margin for one sequence: (scorer, direction, offset, predicted, competitor)."""
    scores = context["probe_scores"]
    predicted, competitor = choose_competitor(scores, true_label)
    direction, offset = probe_direction(
        context["coefficients"], context["intercepts"], context["centre"], context["scale"],
        predicted, competitor,
    )
    tensor = torch.tensor(direction, dtype=torch.float32, device=device)
    return probe_margin(model, tensor, offset), tensor, offset, predicted, competitor


def attribute_sequence(
    model, tokenizer, record, context, settings, index_to_subclass, device, robustness,
) -> dict:
    """All three attributions of one sequence, after checking the model agrees with itself."""
    true_label = int(record["label"])
    input_ids = tokenizer(record["sequence"], return_tensors="pt")["input_ids"].to(device)
    if input_ids.shape[1] != len(record["sequence"]) + 2:
        raise ValueError(f"{record['uniprot_id']}: tokens do not match residues one to one.")
    mask = torch.ones_like(input_ids)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=mask)
        tokens = model.embed_tokens(input_ids)
        embedded = model.forward_embedded(tokens, mask)
    if (logits - embedded).abs().max().item() > MAX_PATH_DIFFERENCE:
        raise RuntimeError("Logits through embeddings differ from logits through input_ids.")

    shared = {"uniprot_id": record["uniprot_id"], "true_label": true_label,
              "true_subclass": index_to_subclass[true_label]}

    def entry(predicted, competitor, margin, saved, scores, **extra):
        return {
            **shared, "predicted": predicted,
            "predicted_subclass": index_to_subclass[predicted],
            "competitor_subclass": index_to_subclass[competitor],
            "margin": margin, "agrees_with_saved_prediction": predicted == int(saved),
            "scores": scores, **extra,
        }

    def masking(score, replacement):
        return occlude(
            model.embed_tokens, score, input_ids, replacement, settings.occlusion_max_tokens
        )

    def robustness_spearman(score, scores):
        if not robustness:
            return None
        other = masking(score, context["second_replacement"])
        return rank_correlation(np.abs(scores), np.abs(other.scores))

    methods = {}

    # The fine-tuned model, by masking
    predicted, competitor = choose_competitor(logits[0].cpu().numpy(), true_label)
    score = classifier_margin(model, predicted, competitor)
    result = masking(score, context["mask_embedding"])
    methods["lora_masking"] = entry(
        predicted, competitor, result.score, context["lora_prediction"], result.scores,
        baseline_spearman=robustness_spearman(score, result.scores),
    )

    # The probe, by masking and by the exact split
    score, direction, offset, predicted, competitor = probe_scorer(
        context, model, device, true_label
    )
    cached = float(context["probe_scores"][predicted] - context["probe_scores"][competitor])
    result = masking(score, context["mask_embedding"])
    methods["probe_masking"] = entry(
        predicted, competitor, result.score, context["probe_prediction"], result.scores,
        difference_from_cached_margin=abs(result.score - cached),
        baseline_spearman=robustness_spearman(score, result.scores),
    )
    with torch.no_grad():
        hidden = frozen_hidden(model, tokens, mask)
    exact = split_margin(hidden, direction, offset)
    methods["probe_exact"] = entry(
        predicted, competitor, exact.score, context["probe_prediction"], exact.scores,
        gap=exact.gap, difference_from_cached_margin=abs(exact.score - cached),
    )

    guard_memory(device, settings.max_memory_fraction)
    return {**shared, "protein_name": record["protein_name"], "sequence": record["sequence"],
            "methods": methods}


def measure_convergence(model, tokenizer, record, context, settings, device) -> dict:
    """
    What integrated gradients would cost here, for both models.

    completeness_gap is the share of the change in margin the attributions fail to
    account for; it should fall towards zero as the step count rises. variation is
    the total up-and-down movement of the margin along the path, which settles with
    resolution if the path can be integrated at all.
    """
    input_ids = tokenizer(record["sequence"], return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    true_label = int(record["label"])
    predicted, competitor = choose_competitor(logits[0].cpu().numpy(), true_label)
    probe, _, _, _, _ = probe_scorer(context, model, device, true_label)
    scorers = {"lora": classifier_margin(model, predicted, competitor), "probe": probe}
    baseline = context["mask_embedding"]
    return {
        "uniprot_id": record["uniprot_id"],
        "n_residues": len(record["sequence"]),
        **{
            name: {
                "completeness_gap": [
                    {
                        "n_steps": int(n_steps),
                        "relative_gap": attribute(
                            model.embed_tokens, score, input_ids, baseline,
                            int(n_steps), settings.max_tokens,
                        ).relative_gap,
                    }
                    for n_steps in settings.convergence_steps
                ],
                "variation": [
                    path_variation(
                        model.embed_tokens, score, input_ids, baseline,
                        int(n_points), settings.occlusion_max_tokens,
                    )
                    for n_points in settings.variation_points
                ],
            }
            for name, score in scorers.items()
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: DictConfig) -> None:
    settings, lora = cfg.attribution, cfg.lora
    device = get_device()
    logger.info(f"Device: {device}")
    if device.type == "cpu" and settings.require_gpu:
        raise RuntimeError(
            "No GPU found, and masking needs a forward pass per residue: this would take days. "
            "Run with --set attribution.require_gpu=false to do it anyway."
        )
    processed_dir = Path(cfg.data.output.processed_dir)
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(settings.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    with open(processed_dir / "test.json") as f:
        test = json.load(f)
    with open(processed_dir / "label_map.json") as f:
        index_to_subclass = {int(v): k for k, v in json.load(f).items()}
    labels = sorted(index_to_subclass)
    ids = [r["uniprot_id"] for r in test]
    saved_predictions = {}
    for name, key in [("lora", "lora_predictions_path"), ("probe", "probe_predictions_path")]:
        with np.load(settings[key]) as f:
            if f["uniprot_ids"].tolist() != ids:
                raise ValueError(f"Test records and the saved {name} predictions are not aligned.")
            saved_predictions[name] = dict(zip(ids, f["predictions"].tolist()))
    by_id = {r["uniprot_id"]: r for r in test}

    families = []
    for offset, family in enumerate(settings.families):
        members = [r["uniprot_id"] for r in test if r["cluster_id"] == family.cluster]
        chosen = select_members(
            members, list(family.anchors), settings.sequences_per_family, settings.seed + offset
        )
        families.append({"family": family, "members": members, "chosen": chosen})

    annotations = load_annotations(
        cfg, [by_id[i] for item in families for i in item["chosen"]], work_dir / "annotations.json"
    )

    logger.info(
        "Refitting the probe on the cached embeddings. This is scikit-learn on the CPU and "
        "takes a few minutes, so the GPU stays idle until the first sequence."
    )
    clf, scaler, probe_ids, probe_mean = fit_probe(cfg, labels)
    if probe_ids != ids:
        raise ValueError("Test records and the cached embeddings are not aligned.")
    probe_row = dict(zip(ids, clf.decision_function(scaler.transform(probe_mean))))

    tokenizer = EsmTokenizer.from_pretrained(lora.model_name)
    model = build_model(
        lora.model_name, len(index_to_subclass), lora.r, lora.alpha, lora.dropout,
        lora.target_modules, lora.head_dropout, device,
    )
    best = torch.load(settings.checkpoint, map_location="cpu", weights_only=True)
    model.load_trainable_state(best["model"])
    model.eval()
    model.requires_grad_(False)
    shared_context = {
        "mask_embedding": token_embedding(model, tokenizer.mask_token_id, device),
        "second_replacement": mean_residue_embedding(
            model, tokenizer.convert_tokens_to_ids(list(AMINO_ACIDS)), device
        ),
        "coefficients": clf.coef_, "intercepts": clf.intercept_,
        "centre": scaler.mean_, "scale": scaler.scale_,
    }

    results_families = []
    arrays = {"ids": [], "lengths": [], "conservation": [], **{m: [] for m in METHODS}}
    for offset, item in enumerate(families):
        family = item["family"]
        logger.info(f"{family.name}: {len(item['chosen'])} of {len(item['members'])} sequences")
        rng = np.random.default_rng(settings.seed + offset)
        rows, convergence = [], []
        for position, uniprot_id in enumerate(item["chosen"]):
            record = by_id[uniprot_id]
            context = {
                **shared_context,
                "lora_prediction": saved_predictions["lora"][uniprot_id],
                "probe_prediction": saved_predictions["probe"][uniprot_id],
                "probe_scores": probe_row[uniprot_id],
            }
            started = time.perf_counter()
            row = attribute_sequence(
                model, tokenizer, record, context, settings, index_to_subclass, device,
                position < settings.robustness_sequences,
            )
            others = [i for i in item["members"] if i != uniprot_id]
            sampled = rng.choice(
                len(others), size=min(settings.max_homologs, len(others)), replace=False
            )
            alignment = align_to_query(
                record["sequence"], [by_id[others[i]]["sequence"] for i in sampled]
            )
            row["conservation"] = conservation_scores([clean(record["sequence"]), *alignment])
            row["sites"] = annotations["sites"].get(uniprot_id)
            rows.append(row)
            logger.info(
                f"  {uniprot_id}: {len(record['sequence'])} residues, {row['true_subclass']}; "
                f"fine-tuned said {row['methods']['lora_masking']['predicted_subclass']}, "
                f"probe said {row['methods']['probe_masking']['predicted_subclass']}, "
                f"{time.perf_counter() - started:.1f} s"
            )
            if position < settings.convergence_sequences:
                convergence.append(
                    measure_convergence(model, tokenizer, record, context, settings, device)
                )
            arrays["ids"].append(uniprot_id)
            arrays["lengths"].append(len(row["conservation"]))
            arrays["conservation"].append(row["conservation"])
            for method in METHODS:
                arrays[method].append(row["methods"][method]["scores"])

        results_families.append({
            "cluster": family.cluster, "role": family.role, "name": family.name,
            "n_family": len(item["members"]), "n_attributed": len(rows),
            "n_sites_unavailable": sum(row["sites"] is None for row in rows),
            "integrated_gradients_convergence": convergence,
            "method_agreement": agreement(rows, settings.top_fraction),
            "anchors": {
                method: {
                    row["uniprot_id"]: top_residues(row, method, settings.n_top_residues)
                    for row in rows if row["uniprot_id"] in family.anchors
                }
                for method in METHODS
            },
            "methods": {
                method: summarise_method(rows, method, settings, settings.seed)
                for method in METHODS
            },
        })

    commit, dirty = git_state()
    results = {
        "experiment": "E8 per-residue attributions on chosen families",
        "git_commit": commit,
        "dirty": dirty,
        "uniprot_release": annotations["release"],
        "probe_best_C": float(clf.C),
        "config": config_as_dict(settings),
        "families": results_families,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    np.savez(
        output_dir / "attributions.npz",
        uniprot_ids=np.array(arrays["ids"]),
        lengths=np.array(arrays["lengths"]),
        conservation=np.concatenate(arrays["conservation"]).astype(np.float32),
        **{m: np.concatenate(arrays[m]).astype(np.float32) for m in METHODS},
    )
    logger.info(f"Results saved to {results_path}")

    logged = {}
    for family in results_families:
        for method, summary in family["methods"].items():
            key = f"{family['cluster']}/{method}"
            logged[f"{key}/spearman"] = summary["score_vs_conservation"]["mean_spearman"]
    log_run(cfg, "decodec-interpretability", "attribution", metrics=logged,
            artifacts=[results_path])


if __name__ == "__main__":
    run(cli_config("Per-residue attributions on chosen families (E8)"))
