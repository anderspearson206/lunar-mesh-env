# lunar-mesh-env

A custom [Gymnasium](https://gymnasium.farama.org/)-compatible environment for simulating agent based lunar mesh networks. This environment is built on [mobile-env](https://github.com/j-schuer/mobile-env) but is for mesh netowkrs

The simulation focuses on a small number of agents (rovers) in a lunar-like terrain, where they must establish communication routes using multiple frequencies and manage their energy consumption.

Support for dynamic radio map prediction via [RadioLunaDiff](https://github.com/anderspearson206/RadioLunaDiff) was added.

## Installation

Note, due to dependency issues with mobile-env, this does not work on python>3.10

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/anderspearson206/lunar-mesh-env.git
    cd lunar-mesh-env
    ```

2.  **Install Dependencies:**
    For use without the RadioLunaDiff model:
    `bash
pip install -r requirements.txt
`

For use with the RLD model:
`bash
    pip install -r nn_requirements.txt
    `

## Example

The `examples/run_simulation.py` script provides a simple way to run the environment and see the visualization. You will need radio map data for this

The `examples/run_simulation_nn.py` script will do the exact same thing as `examples/run_simulation.py`, but with use of the RLD neural network. In the future this will be adjusted to take advantage of the RLD NN's ability to dynamically produce radio maps, but for now the side by side comparison is helpful.
