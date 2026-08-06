# Trained weights

- `best.pt` - best checkpoint by validation mAP (used for all reported results and Grad-CAM figures)
- `last.pt` - final-epoch checkpoint

Evaluate with:

```bash
bash scripts/validate.sh weights/best.pt
```
