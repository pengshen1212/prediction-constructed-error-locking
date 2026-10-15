# Input Data

Raw and processed benchmark datasets are not included in this repository. This avoids redistributing third-party data without a verified redistribution license. The experiment scripts expect the files to be obtained from their original dataset sources and placed in the path layout below.

| Dataset | Expected local path | Files used | Notes |
|---|---|---|---|
| Cora hypergraph | `data/cora-cc/` | `features.pickle`, `hypergraph.pickle`, `labels.pickle` | The experiment uses a hypergraph-formatted Cora variant; verify source and license for the exact preprocessing before redistribution. |
| DBLP hypergraph | `data/dblp-ca/` | `features.pickle`, `hypergraph.pickle`, `labels.pickle` | The experiment uses a preprocessed DBLP co-author hypergraph; verify source and license for the exact preprocessing before redistribution. |
| Walmart-trips | `data/web-hypergraphs/walmart-trips/` | `hyperedges-walmart-trips.txt`, `node-labels-walmart-trips.txt`, `label-names-walmart-trips.txt` | Node-labeled hypergraph files. Consult the original dataset documentation and license before use or redistribution. |

Some scripts still embed absolute paths from the original development workstation. Update those paths to this layout before rerunning. The saved outputs in `results/` can be inspected without downloading the inputs.
