# StableLFRaugmentation
A unified LFR-based model augmentation approach to combine physics-based models and learning components with well-posedness and stability guarantees. Contains the code implementation and example scripts for the paper titled *Data-driven augmentation of first-principles models under constraint-free well-posedness and stability guarantees*. The paper is available on arXiv (coming soon).

## Installation
**1. Clone the repository**
```bash
git clone https://github.com/AIMotionLab-SZTAKI/StableLFRaugmentation.git
cd StableLFRaugmentation
```
**2. Create virtual environment (recommended)**
```bash
python3 -m venv venv
source venv/bin/activate
```
**3. Install package and dependencies**
```bash
pip install -e .
```

## Example usage
The following script illustrates the building blocks of the toolbox with a minimal working example. The [examples](examples/) folder contains several more advanced applications.

```python
import numpy as np
import jax
from matplotlib import pyplot as plt
from model_augmentation_jax import LinearTimeInvariantSystem, StaticLFRAugmentation
from model_augmentation_jax.utils import NRMSE_loss, compute_normalization_constants


jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

# Generate or load data
np.random.seed(0)
U = np.random.normal(size=10_000) # Input sequence
x = [0, 0] # Initial state
ylist = [] # Output sequence
for uk in U:
    ylist.append(x[0] + np.random.normal(loc=0., scale=0.01))  # Compute output
    x = 0.9 * x[0] + 0.1 * x[1] + 0.1 * uk + 0.02 * x[0] * x[1], \
       -0.2 * x[0] + 0.95 * x[1] + 0.05 * uk - 0.1 * x[0]**3 # Advance state

# Split dataset
Y = np.array(ylist)
Y_train = Y[:9000]
Y_test = Y[9000:]
U_train = U[:9000]
U_test = U[9000:]

# create LTI baseline model (with approximate params)
A_mx = np.array([[0.88, 0.11], [-0.2, 0.94]])
B_mx = np.array([[0.1], [0.]])
C_mx = np.array([[1., 0.]])
fp_model = LinearTimeInvariantSystem(A=A_mx, B=B_mx, C=C_mx)

# simulate baseline model to approximate constants for normalization
Yhat_train_base, Xhat_train_base = fp_model.simulate(U_train)  # starts from x0 = 0
norm = compute_normalization_constants(U_train, Y_train, Xhat_train_base)

# create augmented model
model = StaticLFRAugmentation(known_sys=fp_model, hidden_layers=2, nodes_per_layer=8, activation="tanh", nz=2, nw=2,
                              norm_dict=norm)

# set training options
model.set_optimization_parameters(adam_epochs=100, lbfgs_epochs=500, train_x0=True, verbosity=50)

# train the model
model.fit(Y_train, U_train)

# estimate initial state based on first 10 samples of the test data
x0_test = model.learn_x0(U_test[:10], Y_test[:10])

# simulate model
Yhat_test, _ = model.simulate(U_test, X0=x0_test)

nrmse = NRMSE_loss(Yhat_test[10:], Y_test[10:])  # only consider the part that was not used in state estimation

# visualize model output
sim_idx = np.arange(U_test.shape[0])
plt.figure(figsize=(7,3), layout="tight")
plt.plot(sim_idx, Y_test, label="True data")
plt.plot(sim_idx[10:], Yhat_test[10:], '--', label=f"Model sim. (NRMS = {nrmse:.2}%)")
plt.legend()
plt.grid()
plt.xlabel("Sim. index")
plt.ylabel("Output")
plt.show()
```
<img width="700" height="300" alt="example0_fig" src="https://github.com/user-attachments/assets/447b685c-3c42-426e-aa19-f8ec1ff4dcd4" />

## License
This project is licensed under the BSD 3-Clause, see the [LICENSE](LICENSE) file in this repository.

## Citation
If you use this repository in your research, please cite:
```bibtex
coming soon
```

## Acknowledgments
Coming soon

## Contact
For questions or collaboration:
- Open an issue
- Contact the maintainers via GitHub
- Contact the corresponding author: [gyorokbende@sztaki.hu](mailto:gyorokbende@sztaki.hu)

## TODO:
- [ ] Upload to arXiv
- [ ] Add paper reference to every Readme in the project
- [ ] Add citation
- [ ] Add ack
