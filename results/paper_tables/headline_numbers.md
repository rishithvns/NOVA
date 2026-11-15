# Headline numbers

## Base (88M)
- Consequential (primary): raw margin AUROC 0.912 [0.878, 0.941] -> margins + first-order + curvature 0.924 [0.896, 0.950]
- Harmful vs harmless (quant.): raw margin AUROC 0.504 [0.463, 0.545] -> margins + first-order + curvature 0.589 [0.530, 0.640]
- Fine-tuning flip: raw margin AUROC 0.773 [0.761, 0.786] -> margins + first-order + curvature 0.782 [0.769, 0.794]
- primary test (y_consequential): delta AUROC -0.005 [-0.020, +0.007], p = 0.784, Holm p = 1.000
- primary test (y_noise): delta AUROC +0.002 [+0.001, +0.003], p = 0.002, Holm p = 0.006
- primary test (y_hidden): delta AUROC -0.000 [-0.001, -0.000], p = 0.975, Holm p = 1.000
- primary test (y_ft): delta AUROC -0.001 [-0.002, +0.001], p = 0.772, Holm p = 1.000
- primary test (y_ft_input): delta AUROC -0.001 [-0.003, +0.001], p = 0.809, Holm p = 1.000
- primary test (y_harmful): delta AUROC +0.007 [-0.002, +0.016], p = 0.062, Holm p = 0.186
- primary test (y_quant): delta AUROC -0.001 [-0.001, +0.000], p = 0.961, Holm p = 1.000
- primary test (y_quant_harmful): delta AUROC +0.002 [-0.001, +0.005], p = 0.109, Holm p = 0.436
- primary test (y_quant_damage_given_flip): delta AUROC +0.064 [+0.023, +0.107], p = 0.001, Holm p = 0.005
- repair at 10%: learned: raw margin: 45.5% of damage removed
- repair at 10%: learned: margins: 45.5% of damage removed
- repair at 10%: learned: margins + first-order: 42.8% of damage removed
- repair at 10%: learned: margins + first-order + curvature: 52.1% of damage removed
- repair at 10%: learned: all features: 49.8% of damage removed
- repair at 10%: heuristic: smallest raw margin: 40.6% of damage removed
- repair at 10%: heuristic: smallest first-order flip distance: 40.1% of damage removed
- repair at 10%: heuristic: smallest second-order flip distance: 40.1% of damage removed
- repair at 10%: oracle (true damage): 100.0% of damage removed
- repair at 10%: random (expected): 10.0% of damage removed
- closed-form flip probability, noise 0.05 x RMS: Brier 1st order 0.0009 -> 2nd order 0.0009 (delta +0.0000 [-0.0000, +0.0000], p = 0.227); Spearman 0.437 -> 0.437
- closed-form flip probability, noise 0.1 x RMS: Brier 1st order 0.0017 -> 2nd order 0.0016 (delta +0.0001 [+0.0000, +0.0002], p = 0.006); Spearman 0.571 -> 0.571
- closed-form flip probability, noise 0.2 x RMS: Brier 1st order 0.0040 -> 2nd order 0.0031 (delta +0.0009 [+0.0007, +0.0012], p = 0.001); Spearman 0.715 -> 0.718
- closed-form flip probability, noise 0.4 x RMS: Brier 1st order 0.0144 -> 2nd order 0.0082 (delta +0.0063 [+0.0053, +0.0074], p = 0.001); Spearman 0.829 -> 0.851
- tokens whose routing flips under 4-bit quantization: 7.3% (mean over MoE layers)

## Granite (1.3B)
- Consequential (primary): raw margin AUROC 0.784 [0.744, 0.846] -> margins + first-order + curvature 0.834 [0.804, 0.870]
- Harmful vs harmless (quant.): raw margin AUROC 0.480 [0.457, 0.516] -> margins + first-order + curvature 0.557 [0.534, 0.582]
- Fine-tuning flip: raw margin AUROC 0.735 [0.718, 0.753] -> margins + first-order + curvature 0.729 [0.707, 0.754]
- primary test (y_consequential): delta AUROC -0.018 [-0.034, -0.004], p = 0.996, Holm p = 1.000
- primary test (y_noise): delta AUROC +0.003 [-0.003, +0.008], p = 0.145, Holm p = 0.290
- primary test (y_hidden): delta AUROC -0.003 [-0.006, +0.001], p = 0.908, Holm p = 1.000
- primary test (y_ft): delta AUROC -0.002 [-0.009, +0.004], p = 0.681, Holm p = 1.000
- primary test (y_ft_input): delta AUROC +0.007 [-0.004, +0.015], p = 0.125, Holm p = 0.465
- primary test (y_harmful): delta AUROC -0.008 [-0.022, +0.002], p = 0.919, Holm p = 1.000
- primary test (y_quant): delta AUROC -0.001 [-0.004, +0.002], p = 0.807, Holm p = 1.000
- primary test (y_quant_harmful): delta AUROC -0.004 [-0.019, +0.008], p = 0.688, Holm p = 1.000
- primary test (y_quant_damage_given_flip): delta AUROC +0.031 [-0.011, +0.068], p = 0.077, Holm p = 0.308
- repair at 10%: learned: raw margin: 12.0% of damage removed
- repair at 10%: learned: margins: 13.7% of damage removed
- repair at 10%: learned: margins + first-order: 13.5% of damage removed
- repair at 10%: learned: margins + first-order + curvature: 12.7% of damage removed
- repair at 10%: learned: all features: 13.2% of damage removed
- repair at 10%: heuristic: smallest raw margin: 16.7% of damage removed
- repair at 10%: heuristic: smallest first-order flip distance: 14.6% of damage removed
- repair at 10%: heuristic: smallest second-order flip distance: 14.5% of damage removed
- repair at 10%: oracle (true damage): 91.7% of damage removed
- repair at 10%: random (expected): 10.0% of damage removed
- closed-form flip probability, noise 0.05 x RMS: Brier 1st order 0.0127 -> 2nd order 0.0113 (delta +0.0015 [+0.0003, +0.0027], p = 0.010); Spearman 0.730 -> 0.717
- closed-form flip probability, noise 0.1 x RMS: Brier 1st order 0.0373 -> 2nd order 0.0305 (delta +0.0068 [+0.0029, +0.0109], p = 0.001); Spearman 0.792 -> 0.757
- closed-form flip probability, noise 0.2 x RMS: Brier 1st order 0.1573 -> 2nd order 0.1022 (delta +0.0551 [+0.0457, +0.0651], p = 0.001); Spearman 0.680 -> 0.618
- closed-form flip probability, noise 0.4 x RMS: Brier 1st order 0.5000 -> 2nd order 0.3186 (delta +0.1814 [+0.1633, +0.1992], p = 0.001); Spearman 0.294 -> 0.199
- tokens whose routing flips under 4-bit quantization: 50.2% (mean over MoE layers)

## Benchmarks (acc_norm)
task          arc_challenge  arc_easy  hellaswag  openbookqa   piqa   sciq
model                                                                     
dense                 0.204     0.386      0.284       0.256  0.598  0.560
dense_sft             0.214     0.413      0.273       0.240  0.579  0.606
moe                   0.218     0.403      0.298       0.250  0.588  0.564
moe_sft               0.214     0.393      0.275       0.252  0.561  0.602
random guess          0.250     0.250      0.250       0.250  0.500  0.250