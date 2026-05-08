# Results

This directory stores evaluation results and model predictions.

## Output Structure

After running evaluation, results are saved here:

```
results/
├── eval_iam/
│   └── generate-test.txt     # Predicted text for IAM test set
├── eval_sroie/
│   └── generate-test.txt     # Predicted text for SROIE test set
├── eval_str/
│   └── generate-test.txt     # Predicted text for STR benchmarks
├── ft_iam/
│   └── checkpoint_best.pt    # Best fine-tuned model on IAM
├── ft_SROIE/
│   └── checkpoint_best.pt    # Best fine-tuned model on SROIE
└── ft_str_benchmarks/
    └── checkpoint_best.pt    # Best fine-tuned model on STR
```

## Output Format

The `generate-test.txt` file contains lines like:
```
H-0    -0.1234    predicted text here
T-0    ground truth text here
```

Where:
- `H-<id>` = hypothesis (model prediction) with log-probability score
- `T-<id>` = target (ground truth)

## SROIE Submission

For SROIE, convert the output to the required zip format:
```bash
python src/convert_to_SROIE_format.py
```
Then submit at: https://rrc.cvc.uab.es/?ch=13&com=evaluation&task=2
