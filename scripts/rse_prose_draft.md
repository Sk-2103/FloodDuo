# RSE upgrade — new + deepened prose (draft for assembly)

## NEW §3.6 — Architecture design and sensitivity  (place after §3.5 Ablation)

Heading: "3.6 Architecture Design and Sensitivity"

PARA A (design search):
The FloodDuo configuration was selected by a controlled design search over the three
components that carry the trainable capacity: the in-block adapter, the local post-hoc
adapter, and the decoder. Holding the frozen dual-encoder backbone fixed, we compared a
low-rank (LoRA) in-block adapter against the frequency-domain Earth-Adapter, three local
adapters (atrous depth-wise convolution, strip convolution, and a wavelet variant), and
three decoders (UPerNet, LinkNet, and DPT), selecting on validation performance to avoid
tuning on held-out regions. The frequency-domain Earth-Adapter consistently outperformed
its rank-limited LoRA counterpart, indicating that decomposing tokens into low- and
high-frequency bands is better matched to the mixture of smooth open-water interiors and
sharp boundary structure that characterises inundation than an unstructured low-rank
perturbation. Among local adapters the atrous depth-wise convolution was strongest,
consistent with its explicit multi-scale receptive field (dilations 1, 2 and 3)
re-injecting precisely the neighbourhood structure that patch-16 attention discards.
Among decoders the lightweight LinkNet, which merges the multi-depth pyramid additively
and coarse-to-fine, matched or exceeded the heavier UPerNet and DPT while training far
fewer parameters, so we adopt it as the FloodDuo decoder.

PARA B (ablation ladder — refer new ladder figure + Table 2):
Figure L quantifies how each design decision translates into cross-region accuracy under
the leave-one-region-out protocol, building the full FloodDuo model one component at a
time. A single frozen spectral encoder with adapters (DOFA-L) already transfers better
than the from-scratch U-Net, confirming that a pretrained representation is the larger
part of the cross-region advantage. Adding the DINOv3 spatial branch (the dual design)
leaves FloodPlanet almost unchanged but lifts UFO markedly, showing that the spatial
branch supplies the fine urban detail that the spectral branch alone cannot resolve,
whereas on FloodPlanet the spectral branch already absorbs most of the radiometric
variability. Substituting Clay for DOFA-L as the spectral encoder adds a small,
consistent gain, attributable to Clay's native patch-8, GSD-aware pretraining. The final
and largest single increment on the harder FloodPlanet benchmark comes from replacing the
scalar-gate concatenation with the disagreement-gated fusion: because this step changes
only the fusion (encoders, adapters and decoder are held fixed), it isolates the
contribution of the disagreement mechanism itself. Every component contributes, and the
disagreement-gated fusion is the dominant driver of the FloodPlanet cross-region gain.

PARA C (LR sensitivity + robustness — refer LR figure):
FloodDuo is insensitive to its principal training hyperparameter. A validation sweep of
the learning rate over [1e-4, 1e-3] varies region-mean validation IoU by less than 1
point on FloodPlanet and is essentially flat on UFO (Figure S), so we fix the learning
rate at 5e-4 for both benchmarks without per-dataset tuning. This robustness, together
with the frozen backbones and the fixed 80-epoch, seed-42, no-checkpoint-selection
protocol, means the reported cross-region numbers reflect the architecture rather than
hyperparameter search on the evaluation regions. We also observed during protocol design
that the dual-encoder model is robust to the input normalisation choice, whereas the
convolutional baseline is sensitive to it (U-Net gains several points from per-image
normalisation on the radiometrically varied urban scenes); we therefore fix
dataset-statistics normalisation for FloodPlanet and per-image normalisation for UFO for
all models, a choice that only helps the baselines.

## DEEPEN §3.1 (append interpretation after the boxplot/PR discussion)

ADD SENTENCES:
The shape of the gap is as informative as its size. Across held-out regions the
single-encoder baselines, and the U-Net in particular, fail asymmetrically: their
precision remains high while recall collapses, i.e., they miss water in unfamiliar
radiometric environments rather than hallucinating it. FloodDuo instead stays near the
precision-equals-recall diagonal, so its advantage is concentrated exactly where
generalization is hardest, the recall of genuine but atypically-coloured water. This is
the behaviour an operational system needs: errors of omission during a flood are far
costlier than a modest loss of precision.

## DEEPEN §3.4 fine-scale (append)

ADD:
The boundary-resolved error curve makes the division of labour between the two encoders
explicit. Near water edges, where the spatial branch dominates, FloodDuo's error falls
fastest; deep in the interior of water bodies, where the spectral branch dominates, its
margin over the convolutional baseline widens, because the spectral encoder recognises
open water that the CNN, lacking scene context, leaves as false negatives. The two
branches are therefore complementary not only in principle but in the spatial location of
the errors they each prevent.

## DEEPEN §4.2 disagreement (append)

ADD:
Practically, this turns FloodDuo into a self-diagnosing mapper. Because the disagreement
map is bounded, comparable across scenes, and available at inference without labels, an
operator can rank or threshold predictions by confidence, route the highest-disagreement
tiles to human review, and quantify how much of a flood map is trustworthy before any
validation data exist, a capability that conventional single-model segmentation does not
provide.
