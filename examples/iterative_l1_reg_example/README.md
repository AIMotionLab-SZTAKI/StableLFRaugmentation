# Model Augmentation Structure Discovery Example
To run the code for testing out the iterative L1 regularization applied to automatically discover the underlying model augmentation structure, first,
navigate to the appropriate dictionary and then run the experiment code, as:
```bash
cd StableLFRaugmentation/examples/iterative_l1_reg_example/
python3 l1_reg_example.py
```

To try the example with different hyperparameters, edit the `hyperparams.yaml` file.

> [!WARNING]
> Note that the results reported in the [paper](https://arxiv.org/abs/2604.11421) were obtained using a slightly different implementation.
> Consequently, the results produced by the provided script may differ slightly; however, the overall conclusions remain unchanged.
