# Training Logs

This directory stores training logs and TensorBoard event files.

## Viewing Logs

### TensorBoard

```bash
pip install tensorboard
tensorboard --logdir ./logs/
```

Then open http://localhost:6006 in your browser.

### Log Files

Training logs are organized by experiment name:

```
logs/
├── log_ft_iam/                # IAM fine-tuning logs
│   └── events.out.tfevents.*  # TensorBoard events
├── log_ft_SROIE/              # SROIE fine-tuning logs
│   └── events.out.tfevents.*
└── log_ft_str_benchmarks/     # STR fine-tuning logs
    └── events.out.tfevents.*
```

## Tracked Metrics

- **Training loss** — Cross-entropy loss per batch
- **Validation loss** — Loss on the validation set
- **Learning rate** — Current learning rate (with warmup schedule)
- **Gradient norm** — For monitoring training stability
