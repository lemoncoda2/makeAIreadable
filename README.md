# makeAIreadable

Decoupled collaboration training: improve coding via RL while restoring human-readable collaboration text via regenerate + DPO.

Implementation: [`decoupled_collab/`](decoupled_collab/). Spec: [`GOAL_decoupled_collaboration.md`](GOAL_decoupled_collaboration.md).

```bash
cd decoupled_collab
bash scripts/smoke_test.sh          # no GPU
# on 4×V100 server after setup:
python src/run_pipeline.py --config configs/pipeline_config.yaml
```
