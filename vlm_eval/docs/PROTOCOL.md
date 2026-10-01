# SR/SQ/SC semantic evaluation with JPEG--identity normalization

This document specifies the VLM evaluation protocol implemented in this
directory. It uses Qwen3.5-9B with greedy decoding, thinking disabled,
schema-constrained JSON, and no method, filename, bitrate, or conventional
metric metadata in the prompt.

## Independent judgments

For each source/reconstruction pair, the evaluator makes three calls.

1. **Source inventory (source image only).** Extract a global interpretation and
   at most 12 nonredundant semantic units with importance weights
   \(w_r\in\{1,\ldots,5\}\).
2. **Reconstruction inventory (reconstruction only).** Extract visible semantic
   units. Each unit has importance \(w_c\), identity confidence
   \(p_c\in[0,100]\), and four integer quality ratings in \([0,10]\): clarity,
   structural integrity, naturalness, and artifact non-dominance.
3. **Consistency matching (text inventories only).** Match source and
   reconstruction units bidirectionally and judge global scene similarity.
   Images and reconstruction confidence/quality ratings are withheld.

The source inventory is cacheable and must be reused for every reconstruction
of the same source.

## Raw scores

For reconstruction unit \(c\), intrinsic quality is

```text
q_c = 10 * (0.30 * clarity + 0.30 * structure
            + 0.20 * naturalness + 0.20 * artifact_non_dominance)
```

Raw semantic recognizability and quality are

```text
SR_raw = sum(w_c * p_c) / sum(w_c)
SQ_raw = sum(w_c * q_c) / sum(w_c)
```

The text matcher supplies importance-weighted source recall \(rho\), candidate
precision \(pi\), their harmonic mean \(F\), and global similarity \(g\):

```text
F  = 2 * rho * pi / (rho + pi)
SC = 0.40 * g + 0.60 * F
```

SC is already source matched and is not normalized.

## JPEG--identity normalization (JI10)

Evaluate two anonymous reconstruction controls for every source with the same
reconstruction-only prompt:

- **identity:** the clean source itself;
- **JPEG:** the source encoded as baseline JPEG, quality 1, 4:2:0 subsampling.

For \(M\in\{SR,SQ\}\), let \(I_i^M\) and \(J_i^M\) denote the identity and JPEG
raw scores. The per-image denominator is

```text
D_i^M = I_i^M - J_i^M,              if I_i^M - J_i^M > 10
        calibrated_fallback^M,       otherwise
```

The frozen fallback headrooms are 30.434 for SR and 37.015 for SQ. The final
score is

```text
M_i = 100 * clip((M_i_raw - J_i^M) / D_i^M, 0, 1)
```

Normalize each image before dataset averaging. Report clipping rates alongside
means when diagnosing saturation.

## Frozen runtime settings

- Model: `Qwen/Qwen3.5-9B`
- Revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Temperature: 0
- Seed: 0
- Thinking: disabled
- Structured output: JSON schema
- Default image budget: 262,144 pixels

See [`../README.md`](../README.md) for executable commands and manifest examples.
