# DecodEC: Protein Language Model (PLM) fine-tuning for enzyme sub-class classification with homology-aware evaluation and calibrated uncertainty

**Author:** Iker Zapirain Gysling  
**Status:** In development. The data pipeline and a frozen-embedding baseline are built; fine-tuning, calibration and interpretability are not yet.  
**Domain:** Protein Engineering · Computational Biology · Production ML

---

## Current Result

| Model | Task | Split | Test macro-F1 | Test micro-F1 |
|---|---|---|---|---|
| Linear probe on frozen ESM-2 650M embeddings | 59 EC sub-classes | Homology-aware (MMseqs2, 30% identity, 80% coverage) | 0.494 | 0.679 |

This is a preliminary reference point, not a final baseline:

- It is a single run with no uncertainty estimate.
- The macro-F1 includes classes with too few test sequences to evaluate (EC 2.9 has 0, EC 1.9 has 1), which count as F1 = 0.
- The probe is untuned: CLS-token embedding, sequences truncated at 512 residues, no class weighting, regularisation not tuned on validation.

A corrected baseline, with an explicit rule for which classes are evaluable, comes before any fine-tuning results.

---

## The Problem

Enzyme classification from sequence is a foundational task in computational biology and drug discovery. Knowing which enzyme class a protein belongs to informs target identification, selectivity profiling, off-target risk assessment, and the design of selective inhibitors. Classical approaches rely on sequence similarity (BLAST, HMMs) and struggle with remote homologues — proteins that share a functional fold but have diverged beyond recognisable sequence identity.

Protein language models (PLMs) such as ESM-2 learn representations of protein sequences from evolutionary co-variation across very large sequence databases. These representations capture functional and structural information without explicit structural input, offering a route to classification that may generalise beyond shallow sequence similarity.

This project addresses four scientific questions:

- How well do ESM-2 embeddings encode enzyme sub-class identity, and how much does fine-tuning improve over a fair frozen-embedding baseline?
- How much do random splits inflate performance compared with homology-aware splits on this dataset?
- How calibrated are the model's confidence scores, and when should predictions be trusted?
- Which sequence positions drive classification decisions, and do they correspond to conserved residues?

---

## What Exists Today

| Component | Status |
|---|---|
| UniProtKB/Swiss-Prot download with EC filtering and checksums | Built |
| Preprocessing: length filter, deduplication, rare-class filter, label encoding | Built |
| MMseqs2 clustering and cluster-level train/val/test splits | Built |
| PyTorch Dataset with ESM-2 tokenisation | Built |
| Linear probe on frozen ESM-2 embeddings | Built (preliminary, see above) |
| ESM-2 + LoRA fine-tuning | Planned |
| Sequence-similarity nearest-neighbour baseline | Planned |
| Random vs homology-aware split comparison | Planned |
| Temperature-scaling calibration with ECE | Planned |
| Integrated gradients with MSA conservation cross-check | Planned |
| FastAPI inference endpoint, Docker image, CI, tests | Planned |

---

## Data Pipeline (as built)

Data comes from UniProtKB/Swiss-Prot, restricted to reviewed entries with at least one EC annotation (`reviewed:true AND ec:*`).

1. **Download.** Paginated UniProt REST download. Keeps sequences of 50–1024 residues whose EC numbers have at least two specific digits. The label is the EC sub-class (first two digits, e.g. 3.4) of the first listed EC number. The raw file's checksum is stored in `data/checksums.json`.
2. **Preprocess.** Removes exact duplicate sequences (40,056 removed from 257,018 records), then drops sub-classes with fewer than 100 sequences (16 dropped). Result: 216,962 sequences in **59 sub-classes**, covering all seven top-level EC classes (including EC 7, translocases).
3. **Cluster and split.** MMseqs2 (run in Docker) clusters sequences at 30% identity and 80% coverage, producing 15,935 clusters. Clusters, not sequences, are assigned to splits (70% / 15% / 15% of clusters per sub-class), so no cluster crosses splits. Result: 152,334 train / 32,629 val / 31,999 test sequences.

All settings are in `config/data_config.yaml`.

---

## Scientific Background

### Enzyme Classification

The Enzyme Commission (EC) system classifies enzymes hierarchically into seven top-level classes (oxidoreductases, transferases, hydrolases, lyases, isomerases, ligases and translocases), each subdivided into sub-classes and sub-subclasses. This project predicts the sub-class (two-digit EC number), which is harder than the top level while keeping enough data for most classes.

The challenge is generalisation to remote homologues: sequences that share a functional mechanism with training examples but have diverged beyond the reach of classical similarity tools.

### Why ESM-2

ESM-2 (Lin et al., 2023, *Science*) is a family of protein language models trained with a masked language modelling objective on UniRef sequences. The 650M-parameter model produces per-residue and sequence-level representations that capture local sequence patterns, structural constraints and evolutionary co-variation, which makes it a strong starting point for downstream classification.

### Why LoRA (planned)

Full fine-tuning of a 650M-parameter model needs substantial GPU memory and risks overwriting what the model learned in pre-training. Low-Rank Adaptation (LoRA, Hu et al., 2022) freezes the pre-trained weights and learns small low-rank updates to selected layers (here, the attention query and value projections), so only a small fraction of parameters is trained.

### Why Homology-Aware Splitting

With random splits, near-identical proteins can appear in both train and test, so test performance measures memorisation of close homologues rather than generalisation. Clustering at 30% identity and assigning whole clusters to one split removes this leakage. How large the inflation is on this dataset will be measured directly by comparing random and homology-aware splits on the same records.

### Calibration and Uncertainty (planned)

A model whose confidence scores do not match its empirical accuracy is unreliable in practice. Temperature scaling (Guo et al., 2017) will be fitted on the validation set, with Expected Calibration Error (ECE) and reliability diagrams reported before and after. Low-confidence predictions will be flagged rather than silently assigned to the top class.

---

## Key Design Decisions

**Why sub-classes rather than top-level EC classes?**  
Seven top-level classes make the task too easy to be informative. Sub-classes are harder while still leaving enough sequences per class for most of them.

**Why homology-aware splitting rather than random?**  
Random splits leak close homologues between train and test. The gap between random and homology-aware splits will be reported explicitly.

**Why temperature scaling rather than more complex calibration?**  
It has a single parameter, cannot change the predicted class, and is easy to interpret. More flexible methods (vector scaling, isotonic regression) will be tried only if per-class miscalibration remains.

**Why integrated gradients rather than attention weights?**  
Attention weights show how information is routed, not which inputs matter. Integrated gradients (Sundararajan et al., 2017) satisfy completeness and can be cross-checked against MSA conservation.

**Why config-driven runs?**  
Data settings live in YAML under `config/`, so a run can be reproduced from its config. Model and training configs will follow the same pattern.

---

## Evaluation Plan

### Metrics

| Metric | Description |
|---|---|
| Macro-F1 | Primary metric, over evaluable classes only; always reported with the number of classes it covers and an uncertainty estimate |
| Per-class F1 | Reported with support, to show which sub-classes are hardest |
| Top-3 accuracy | Relevant where a ranked shortlist is acceptable |
| ECE | Expected Calibration Error, with the binning stated |
| Reliability diagram | Visual calibration check |

### Baselines

- **Linear probe on frozen ESM-2 embeddings**: the reference for how much fine-tuning adds
- **Sequence-similarity nearest neighbour** (BLAST or MMseqs2 search against train): the classical baseline that matters most at a 30% split
- **Random forest on ESM-2 embeddings**: a non-linear, non-neural baseline
- **Random split vs homology-aware split**: same records and settings, only the split changes

### Known Failure Modes to Document

- **Sub-class boundary sequences** may be assigned to either of two classes with similar confidence.
- **Families absent from Swiss-Prot** cannot be classified correctly. Low maximum softmax probability is only a weak signal for out-of-distribution input.
- **Multi-functional enzymes** with EC numbers in more than one sub-class are labelled by their first EC number. Multi-label classification is a planned extension.
- **Classes too small to evaluate** on the test split are reported separately, with their counts.
- **Clustering sensitivity**: results may depend on the identity and coverage thresholds, which will be tested.

---

## Repository Structure

```
decodec/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
│
├── config/
│   └── data_config.yaml           # UniProt query, length and class filters, clustering settings
│
├── src/
│   ├── data/
│   │   ├── download.py            # UniProtKB/Swiss-Prot download with EC filtering
│   │   ├── preprocess.py          # Deduplication, rare-class filter, label encoding
│   │   ├── clustering.py          # MMseqs2 clustering and cluster-level splits
│   │   └── dataset.py             # PyTorch Dataset with ESM-2 tokenisation
│   │
│   ├── model/                     # planned: ESM-2 + LoRA
│   ├── training/                  # planned: training loop and metrics
│   ├── calibration/               # planned: temperature scaling and ECE
│   ├── interpretability/          # planned: integrated gradients
│   ├── inference/                 # planned: FastAPI endpoint
│   └── visualisation/             # planned: plots
│
├── data/                          # generated by the pipeline, not tracked (except checksums.json)
│   ├── raw/
│   ├── processed/
│   ├── clusters/
│   └── checksums.json
│
└── experiments/
    └── baseline_zeroshot/         # linear probe on frozen ESM-2 embeddings (preliminary)
        ├── run_baseline.py
        ├── results.json
        └── per_class_report.txt
```

---

## Quickstart

Requires Python 3.11, [uv](https://docs.astral.sh/uv/), Docker (for MMseqs2) and a CUDA GPU for the baseline.

### Installation

```bash
git clone https://github.com/izgys/decodec
cd decodec
uv sync --extra dev
```

### Data preparation

```bash
uv run python -m src.data.download --config config/data_config.yaml
uv run python -m src.data.preprocess --config config/data_config.yaml
uv run python -m src.data.clustering --config config/data_config.yaml
```

UniProt releases change over time, so a fresh download will usually not match the stored checksum.

### Baseline

```bash
uv run python experiments/baseline_zeroshot/run_baseline.py
```

---

## Roadmap

**v1 — Sequence-based classification**

- [x] Data pipeline (UniProtKB/Swiss-Prot download, filtering, deduplication, label encoding)
- [x] MMseqs2 homology-aware splitting
- [x] Preliminary linear probe on frozen ESM-2 embeddings
- [ ] Evaluable-class rule and corrected linear-probe baseline
- [ ] Sequence-similarity nearest-neighbour baseline
- [ ] ESM-2 + LoRA fine-tuning with MLflow tracking
- [ ] Random vs homology-aware split comparison
- [ ] Calibration (temperature scaling, ECE)
- [ ] Integrated gradients with MSA conservation cross-check
- [ ] FastAPI inference endpoint, Docker packaging, tests and CI

**v2 — Structure-aware extension**

- [ ] Attribution mapping to 3D structure (PDB retrieval, B-factor replacement)
- [ ] SaProt integration (structure-aware sequence tokens from Foldseek)
- [ ] Multi-label classification for multi-functional enzymes
- [ ] Newer protein language models (ESM-3 or later)

---

## Scientific References

- Lin Z et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 379:1123–1130. DOI: 10.1126/science.ade2574
- Hu EJ et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*. arXiv: 2106.09685
- Sundararajan M et al. (2017). Axiomatic Attribution for Deep Networks. *ICML*. arXiv: 1703.01365
- Guo C et al. (2017). On Calibration of Modern Neural Networks. *ICML*. arXiv: 1706.04599
- Steinegger M & Söding J (2017). MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. *Nat. Biotechnol.* 35:1026–1028. DOI: 10.1038/nbt.3988
- Steinegger M & Söding J (2018). Clustering huge protein sequence sets in linear time. *Nat. Commun.* 9:2542. DOI: 10.1038/s41467-018-04964-5
- The UniProt Consortium (2025). UniProt: the Universal Protein Knowledgebase in 2025. *Nucleic Acids Res.* 53:D609–D617. DOI: 10.1093/nar/gkae1010
- Rao R et al. (2021). MSA Transformer. *ICML*.
- Su J et al. (2024). SaProt: Protein Language Modeling with Structure-aware Vocabulary. *ICLR*. DOI (preprint): 10.1101/2023.10.01.560349

---

## Author

**Iker Zapirain Gysling**  
Computational Biochemist, PhD  
Barcelona, Spain  
[LinkedIn](https://linkedin.com/in/zgysling) · [GitHub](https://github.com/izgys)

---

## License

MIT License — see `LICENSE` for details.
