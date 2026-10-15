# Official Relation Audit for Paper2

Status: completed descriptive audit (one prespecified seed per method). This directory records an audit of the learned relations emitted by two author-maintained reference implementations. It is separate from the existing NumPy LGS reimplementation.

## Methods selected

1. **IDGL**: authors' NeurIPS 2020 implementation, `hugochan/IDGL`, frozen commit `08129c596a574bef6d34b72ab745dd19c683a6d4`. Use the repository's Cora `config/cora/idgl.yml`, retaining its architecture, optimizer, graph-learning settings and split. It learns a graph from node embeddings and jointly optimizes it for classification.
2. **Pro-GNN**: authors' KDD 2020 implementation, `ChandlerBang/Pro-GNN`, frozen commit `90e11956bf8145dc2ff441fc9923bc95b535f289`, using its cited `DSE-MSU/DeepRobust` implementation, frozen separately in `provenance.json`. Use the official Cora meta-attack 15% settings from `scripts/meta/cora_meta.sh`; evaluate the learned normalized graph selected by validation performance.

The source check did not find a public official LGS implementation. Existing LGS scripts in the manuscript are explicitly reimplementations, not official code. CEST does not propagate labels over its pseudo-label similarity graph. IDGL and Pro-GNN therefore test a broader learned-relation boundary; neither should be described as an exact hard-pseudo-label-graph reproduction.

## Audit definition

For each trained checkpoint, save the exact nonnegative relation supplied to its graph aggregation, the full-node logits, labels and train/validation/test indices. Convert the saved relation `W` to `K[u,v] = W[u,v] / sum_j W[u,j]`. This row normalization preserves the class argmax of a single nonnegative aggregation `W @ onehot(prediction)` because it only scales each target row by a positive scalar. For Pro-GNN, the saved graph is the selected symmetric-normalized propagation matrix; for IDGL, it is the final relation after graph-skip and update-adjacency blends.

Freeze each method's original hard predictions as the relation-audit input. Report the standard method test accuracy separately from this frozen-relation diagnostic. For one, two, five and ten row-stochastic steps, report on test nodes:

- certificate coverage: `K^t[u, predicted_class[u]] > 0.5`;
- corrected errors: initially wrong, propagated class equals truth;
- harmful changes: initially correct, propagated class becomes wrong;
- wrong-to-wrong changes and all prediction changes.

These are relation diagnostics, not the methods' full nonlinear test-time forward pass. The one-step argmax is equivalent to the corresponding nonnegative linear aggregation; later powers are a frozen-kernel stress diagnostic. Preserve per-node records so every aggregate can be reconstructed.

## Performance and observed results

IDGL seed 42 reaches 84.4% on the official 2,708-node Cora split (paper reference: 84.5 +/- 0.3%). Pro-GNN seed 10 reaches 76.06% on the official 2,485-node Cora largest connected component under 15% Meta-Self perturbation (paper reference: 76.40 +/- 1.27%). Both single runs are within the corresponding reported ranges. The datasets/splits differ, so these accuracies are not compared against each other.

At one frozen-relation step, IDGL has 95.5% certificate coverage; among 1,000 test nodes, 6 errors correct, 12 initially correct nodes become wrong, 3 change to another wrong class, and 21 classes change in total. Pro-GNN has 100% certificate coverage; among 1,988 test nodes, there are no one-step class changes, corrections, or harmful changes. These are descriptive single-checkpoint results, not estimates over training randomness. Later powers t=2,5,10 are stress diagnostics only.

Do not tune on test labels or select checkpoints/seeds by audit outcomes. The reference-performance check passed for these individual runs, but one seed does not establish seed-averaged performance or variability.

IDGL's current PyYAML API needs a runtime compatibility shim for the repository's old `yaml.load(stream)` call. The shim supplies `yaml.FullLoader` and does not change model code. Pro-GNN's graph is captured by an observer around the official `test` method; the training source is not patched. Its official training and test completed, followed by a failed optional input-graph export (SciPy `.toarray()` was called on a Tensor). The captured relation/logits and completed test are intact; a converter was fixed and regression-tested without rerunning the expensive training.

## Scope and reporting

Both selected methods learn propagation graphs from embeddings or supervised graph objectives; their learned relations are not constructed solely by thresholding the same hard pseudo-labels they later propagate. A low lock coverage is an applicability-boundary result, not a failure of either method. High correction or harmful-change counts are conditional on the frozen one-hot propagation probe, not evidence that the published method's normal forward pass gains or loses that accuracy.

Results, runtime logs, exact official commits, compatibility notes and input hashes belong in this directory. If either method cannot be run near its normal performance under the official configuration, report the audit as incomplete for that method; do not substitute a local reimplementation and call it official.
