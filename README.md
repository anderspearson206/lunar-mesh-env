# lunar-mesh-env

A custom [PettingZoo](https://pettingzoo.farama.org/)-compatible environment for Multi-Agent Reinforcement Learning (MARL) on the lunar surface.

The simulation focuses on a team of agents (rovers) in a lunar-like terrain, where they must establish communication routes using multiple frequencies while managing energy consumption.

Support for dynamic radio map prediction via [RadioLunaDiff](https://github.com/anderspearson206/RadioLunaDiff) is integrated directly into the environment.

For older Gymnasium support, see the [legacy-gym branch](https://github.com/anderspearson206/lunar-mesh-env/tree/legacy-gym)

## Features

- **Multi-Agent Architecture:** Built on the PettingZoo API for cooperative multi-agent tasks.
- **Radio Propagation:** Simulates complex signal propagation and interference.
- **Neural Network Integration:** Includes support for deep learning-based radio map prediction (RadioLunaDiff).

## Installation

**Prerequisite:** Python 3.10 is recommended for the best compatibility with PyTorch and PettingZoo.

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/anderspearson206/lunar-mesh-env.git](https://github.com/anderspearson206/lunar-mesh-env.git)
    cd lunar-mesh-env
    ```

2.  **Set up Conda environment (recommended):**

    ```bash
    conda create -n lunar_mesh python=3.10
    conda activate lunar_mesh
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Example

The repository includes a script to run the multi-agent simulation with the default configuration.

To run the MARL environment:

```bash
python examples/run_marl_simulation.py
```
