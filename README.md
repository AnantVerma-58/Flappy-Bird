# Reinforcement Learning in Flappy Bird Environment

## Project Overview

This project utilizes the Proximal Policy Optimization (PPO) algorithm to train a Reinforcement Learning agent in a custom environment simulating the Flappy Bird game. The primary objective is to enable the agent to navigate the environment, avoiding obstacles, and collecting fuel to maximize its survival duration.

## Prerequisites

Ensure you have the following dependencies installed:

- NumPy
- Matplotlib
- OpenCV
- PyTorch
- TorchVision
- OpenAI Gym

## Custom Environment

The dynamics of the custom environment are defined in the `ChopperEnv.py` script, emulating the Flappy Bird game with specific characteristics.

## Neural Network Architecture

The agent's policy is implemented using a convolutional neural network defined in the `Policy` class. The architecture comprises convolutional layers followed by fully connected layers.
We used default weighted VGG16 model through transfer learning to train a different model with output as action space.

## Training

The PPO algorithm is employed to train the agent in the environment. The `reinforce` function executes the training loop, updating the policy based on calculated returns and log probabilities.

## Evaluation

Evaluate the trained agent's performance using the `evaluate_agent` function. This function runs the agent in the environment for a specified number of episodes.


## Saving and Loading Model

The trained model can be saved and loaded using the PyTorch `torch.save` and `torch.load` functions. Example code for saving and loading is provided in the script.

## Acknowledgments

This implementation draws inspiration from OpenAI's Gym and Proximal Policy Optimization.
