# proc-posets

Reusable **poset / cospan / string-diagram** calculation, simulation,
estimation and comparison core for process-mining research.

The core is pure standard library (Python ≥3.10, no third-party packages);
numpy, networkx, pm4py and matplotlib sit behind optional extras and load
lazily, so a bare `import procposets` pulls nothing third-party.

```toml
dependencies = ["proc-posets"]        # dependency-free core
# "proc-posets[numeric]" numpy · "[graph]" networkx · "[pm4py]" model adapters
# · "[viz]" matplotlib/graphviz · "[all]" everything
```

It is not published to PyPI; consumers take it as a uv path dependency or from
an archived release.

## Third-party software, licences and attribution

Nothing third-party is vendored: every file in this repository is first-party
and MIT-licensed. The dependencies below are *declared*, never redistributed —
they are installed from PyPI by the user, and the base install pulls **none**
of them.

| extra | package | licence |
|---|---|---|
| `[numeric]` | numpy | BSD-3-Clause |
| `[graph]` | networkx | BSD-3-Clause |
| `[pm4py]` | pm4py | **AGPL-3.0** — see below |
| `[pm4py]` | pandas | BSD-3-Clause |
| `[viz]` | matplotlib | matplotlib licence (BSD-compatible) |
| `[viz]` | graphviz (Python bindings) | MIT; the `dot` binary itself is EPL-1.0 and is not bundled |

**pm4py is AGPL-3.0, and `[pm4py]` and `[all]` pull it in.** That does not
affect this package's own licence — no pm4py code is included here, and MIT is
compatible with the AGPL in the one direction that matters. It does affect
*you*: installing the extra creates a combined work on your machine, and if you
convey that combination, or offer it to users over a network, AGPL section 13
obligations attach to it. The base install and every other extra are free of
that, and nothing outside `procposets.discover`,
`procposets.adapters.outbound` and `procposets.viz.occn_vis` imports pm4py at
all.

### Algorithms implemented from published work

`procposets.occn` implements object-centric causal net discovery from **Liss,
Mensing and van der Aalst, "Object-Centric Causal Nets", CAiSE 2025**
(doi:10.1007/978-3-031-94571-7_6). It is written from the paper's published
pseudocode (Listing 1.1) and deliberately diverges from the authors' reference
implementation, which enumerates marker keys by set partition — so marker *key
labels* here differ from the reference even where marker *structure* agrees. No
code from that implementation is included or redistributed. Cite the paper for
the algorithm and this package for the implementation.

## Citing this software

Cite the archived release rather than the repository URL: a URL moves, a DOI
does not. `CITATION.cff` carries the machine-readable record.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22757993.svg)](https://doi.org/10.5281/zenodo.22757993)

| | DOI | resolves to |
|---|---|---|
| **concept** | [`10.5281/zenodo.22757993`](https://doi.org/10.5281/zenodo.22757993) | always the latest archived version |
| **version** | [`10.5281/zenodo.22757994`](https://doi.org/10.5281/zenodo.22757994) | release 0.1.0 |

Cite the **version** DOI when you name a specific release; the **concept** DOI
when you mean the software in general.

## Licence

MIT — see `LICENSE`.
