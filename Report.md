# Continuous Control DDPG Project - Report

Justin Hess

# Introduction

This project uses the DDPG policy-based method algorithm that is used to learn the optimal policy in a model-free Reinforcement Learning setting. A double-jointed arm can move to target locations in this environment. A reward of +0.1 is provided for each step that the agent's arm is in the goal location. The goal of the agent is to maintain its position at the target location for as many time steps as possible within a given episode.

The observation space consists of 33 variables corresponding to the position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector is a number between -1 and 1.

# Deep Deterministic Policy Gradients Algorithm

Deep Deterministic Policy Gradient (DDPG) is an Actor-Critic method with a twist. The actor produces a deterministic policy by mapping states to actions instead of a standard stochastic policy, and the critic evaluates actor's policy output. The critic is updated using the temporal difference (TD) error and the actor is trained using the deterministic policy gradient algorithm. The critic neural network in DDPG is used to approximate the maximizer over the Q-values of the next state, and not as a learned baseline, as have seen so far. One of the limitations of the DQN agent is that it is not straightforward to use in continuous action spaces. For example, how do you get the value of continuous action with DQN architecture? This is the problem DDPG solves.

## Replay Buffer

The Replay Buffer uses a queue for a replay buffer. It randomly selects past experience tuples to avoid correlation among actions and next state. It randomly samples data from the queue for more efficient sample reuse. This avoids temporal correlations during training.

## Multi Agent Learning using DDPG

The agents learn by using the actor network to choose continuous actions from evaluating states, and then inputting those actions to the critic network which outputs a singular Q value (like in DQN learning's target network) for the action from the actor.

Q_target = r + γ * Q_target(s', μ_target(s'))

μ_target(s') is the target actor's deterministic action that it chooses for continuous actions.

The target network updates its gradient based on deterministic actions that the critic network perceive are apt for a state, action pair:
∇_θμJ ≈ E[∇_aQ(s,a∣θ^Q)∇_θμμ(s∣θ^μ)]

Both DQN and DDPG use target networks (slowly updated versions of main networks) to stabilize training:

𝜃′← τθ + (1−τ)θ′ with small τ

The actor is trained using the deterministic policy gradient theorem, where gradients flow from the critic's evaluation of the actor's chosen actions back through the actor network. The actor's objective is to maximize the critic's Q-value estimate: it learns to select actions that the critic believes will yield the highest returns.

The critic network learns a single Q-value for each given state-action pair for discrete actions and drives both learning and action selection. It's trained using temporal difference learning with target networks for stability. In this way it is somewhat similar to the network in the DQN algorithm, but that learns the maximal Q-values for all possible discrete state, action pairs.

## Actor-Critic Network architecture

The neural network architectures of both the actor and critic are comprised of two fully connected hidden layers of 256 and 128 hidden units, respectively, each with ReLU activations. The hyperbolic tanh activation is used on the output layer for the actor network so that the entries in the action vector is a number between -1 and 1. The Adam optimizer is used for both actor and critic networks.

# Results

The scenario where multiple agents are used (20) was able to achieve consistent rewards of over +30 once it got to 110 episodes, and continued until 125 epsiodes where they finished training.

# Future Improvements

It is recommended to investigate using more batch normalization in the actor and critic networks to speed up training, as well as make the agents converge faster. The training is noticably slow even when using GPUs, so speeding up training as well would prove to be fruitful in helping multi agents learn more quickly.