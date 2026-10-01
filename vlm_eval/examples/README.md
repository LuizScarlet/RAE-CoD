# Example manifest

`manifest.jsonl` demonstrates the source/reconstruction pair schema. The
included images provide identity, two degradation levels, and source-mismatch
controls for a small smoke test. From the `vlm_eval/` directory, after installation:

```bash
CUDA_VISIBLE_DEVICES=0 vlm-sem-dist examples/manifest.jsonl \
  --model /path/to/Qwen3.5-9B \
  --output outputs/example_results.jsonl \
  --reference-cache outputs/example_reference_cache.json \
  --max-pairs 4
```
