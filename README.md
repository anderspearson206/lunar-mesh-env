# lunar-mesh-env

A custom [Gymnasium](https://gymnasium.farama.org/)-compatible environment for simulating agent based lunar mesh networks. This environment is built on [mobile-env](https://github.com/j-schuer/mobile-env) but is for mesh netowkrs

The simulation focuses on a small number of agents (rovers) in a lunar-like terrain, where they must establish communication routes using multiple frequencies and manage their energy consumption.


## Installation

Note, due to dependency issues with mobile-env, this does not work on python>3.10

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/anderspearson206/lunar-mesh-env.git
    cd lunar-mesh-env
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Example

The `examples/run_simulation.py` script provides a simple way to run the environment and see the visualization. You will need radio map data for now
