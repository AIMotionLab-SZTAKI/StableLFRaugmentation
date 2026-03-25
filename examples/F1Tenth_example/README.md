# F1Tenth Simulation example

## Usage
To reproduce the experiments, navigate to the directory containing the F1Tenth example scripts and display the available command-line options:
```bash
cd examples/F1Tenth_example/
python3 f1tenth_example.py -h
```

This command returns the following description of the script and its arguments:
```text

usage: f1tenth_example.py [-h] [--seed SEED] [--SNR {0,40,30,20}] [--state_augm_type {static,dynamic}]
        [--LFR_struct {zero,lower-triang,WP,contr}]

Example script for testing LFR-based model augmentation on the F1Tenth identification example.

options:
  -h, --help            show this help message and exit
  --seed SEED           Random seed.
  --SNR {0,40,30,20}    Added Gaussian noise to training with specified sensor-to-noise ratio (SNR).
                        If 0, no noise is added.
  --state_augm_type {static,dynamic}
                        Type of state augmentation. Options: [static, dynamic]
  --LFR_struct {zero,lower-triang,WP,contr}
                        LFR matrix parametrization. Options: [zero, lower-triang, WP, contr].
                        'zero' for Dzw=0,
                        'lower-triang' for Dzw strictly lower triangular,
                        'WP' for well-posed parametrization,
                        'contr' for contracting parametrization.
```

The file `hyperparam_config.yaml` contains the hyperparameter configuration used to generate the results reported in the paper.
Modifying this file allows users to explore alternative configurations; however, such changes naturally can lead to different model performance.

All results reported were obtained using **Python 3.10** on **Ubuntu 24.04.3 LTS**.

> [!WARNING]
> Note that the reported results were obtained using different random seeds, all selected from 0, ..., 9.
> Furthermore, the results provided in the paper have been achieved by using a slightly different implementation. Due to these changes, the final results may differ from those reported.

## Example
To run the dynamic well-posed parametrization with added noise, resulting in 40 dB SNR, from the initial seed 1, run the following command:
```bash
python3 f1tenth_example.py --seed 1 --SNR 40 --state_augm_type dynamic --LFR_struct WP
```
