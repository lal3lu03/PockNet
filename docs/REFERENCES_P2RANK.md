# P2Rank Provenance Notes

This project contains original Python implementations that mirror the core
behaviour of the **P2Rank** binding pocket detection pipeline. No Groovy or
Java source code from the upstream project is used or redistributed here; the
logic was re-derived and rewritten from scratch based on public documentation
and observed behaviour of the official tool.

Key components inspired by P2Rank:

1. **Pocket aggregation heuristics**  
   The clustering radius, minimum cluster size, pocket growth radius, and score
   computation implemented in `post_processing/p2rank_like.py` follow the same
   algorithmic choices as P2Rank’s `PocketPredictor`. The Python code was
   authored specifically for this repository.

2. **Threshold defaults and adaptive sweep**  
   Default probability thresholds, scoring exponents, and the optional
   threshold sweep feature (**`--threshold-grid`**) are based on the values and
   evaluation practices documented by P2Rank so we can mirror its behaviour
   when selecting decision cut-offs.

3. **Tabular feature preparation**  
   The tabular feature set consumed by the models is reconstructed from the
   original `vectorsTrain_all_chainfix.csv` layout used by P2Rank (chemical
   descriptors, VolSite-derived values, protrusion, atom-table features, etc.),
   but the loading and preprocessing logic has been reimplemented in Python.

4. **Output schema**  
   The generated `pockets.csv` files match the CSV structure produced by
   P2Rank’s evaluation routines so downstream tooling (e.g., PyMOL scripts or
   benchmarking utilities) can be reused.

If you publish results that rely on these components, please cite the original
P2Rank work:

```
Krivák, R., & Hoksza, D. (2018).
P2Rank: Machine learning based tool for rapid and accurate prediction of ligand
binding sites from protein structure.
Journal of Cheminformatics, 10(1), 39.
https://doi.org/10.1186/s13321-018-0285-8
```

Repository: https://github.com/rdk/p2rank

Any future contributions that extend or modify the P2Rank-inspired logic should
update this document to keep provenance and attribution clear.
