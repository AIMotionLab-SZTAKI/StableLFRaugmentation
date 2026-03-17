# Cascaded Tanks Benchmark Example

## Usage
To reproduce the experiments, navigate to the directory containing the benchmark scripts and display the available command-line options:
```bash
cd StableLFRaugmentation/examples/CascadedTanks_benchmark/
python3 cascaded_tanks_example.py -h 
```

This command returns the following description of the script and its arguments:
```bash
Example script for testing LFR-based model augmentation on the Cascaded Tanks benchmark problem.

options:
  -h, --help                show this help message and exit
  --seed SEED               Random seed.
  --LFR_struct {WP,contr}   LFR matrix parametrization. Options: [WP, contr].
                            'WP' for well-posed parametrization,
                            'contr' for contracting parametrization.
```

The file `hyperparams.yaml` contains the hyperparameter configuration used to generate the results reported in the paper. Modifying this file allows users to explore alternative configurations; however, such changes naturally can lead to different model performance.

All results reported were obtained using **Python 3.10** on **Ubuntu 24.04.3 LTS**.

### Well-posed LFR Model Augmentation
The well-posed (WP) LFR model augmentation can be evaluated using:
```bash
python3 cascaded_tanks_example.py --seed 3 --LFR_struct WP
```
Note that the reported benchmark results were obtained using this specific random seed. Due to potential differences in numerical behavior across operating systems and Python versions, exact reproducibility of the results may require using the same software environment as reported.

### Contracting LFR model augmentation
The well-posed (WP) LFR model augmentation structure can be evaluated by the following command:
```bash
python3 cascaded_tanks_example.py --seed 3 --LFR_struct contr
```
