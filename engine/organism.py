from dataclasses import dataclass

import numpy as np
from scipy.special import softmax

from engine.nn import NeuralNetworkLayer, MSELoss

@dataclass
class Action:
    value: int
    logits: np.ndarray
    odds: np.ndarray

@dataclass
class StateTransition:
    state: np.ndarray
    action: Action
    reward: float
    next_state: np.ndarray
    terminated: bool = False
    truncated: bool = False


class Organism(object):
    def __init__(self, d_in, d_latent, num_actions, learning_rate: float, memory_size: int = 0):
        self.init_learning_rate = learning_rate
        self.sensor = NeuralNetworkLayer(d_in, d_latent, alpha=learning_rate)
        self.actor = NeuralNetworkLayer(d_latent, num_actions, alpha=learning_rate, prev=self.sensor)

        self.memory_size = memory_size
        self.memory = []

    def observe(self, state: np.ndarray):
        # the latent space represents how the organism perceives the environment
        latent_state = self.sensor.forward(state, inference=True)
        return latent_state

    def act(self, latent_state: np.ndarray) -> Action:
        logits = self.actor.forward(latent_state, inference=True)
        value = np.argmax(logits, axis=-1) if logits.ndim > 1 else np.argmax(logits)
        odds = softmax(logits, axis=-1) if logits.ndim > 1 else softmax(logits)
        return Action(value=value, logits=logits, odds=odds)

    def store_transition(self, transition: StateTransition):
        self.memory.append(transition)
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)

    def mutate(self):
        """in-place mutation of the organism's neural network weights and biases."""
        # for layer in self.nn:
        #     layer.mutate(scale=0.05)
        ...

    def learn(self, gamma: float = 0.99):
        """
        Simple Q-learning update using numpy only.
        Assumes discrete actions and one-step transitions.
        """
        if not self.memory:
            return

        self.actor.inference = False
        self.actor.trainable = True
        self.sensor.inference = False
        self.sensor.trainable = True

        batch = self.memory if len(self.memory) < self.memory_size else self.memory[-self.memory_size:]
        for transition in batch:
            # Forward pass for current state
            state_latent = self.sensor.forward(transition.state)
            q_values = self.actor.forward(state_latent)
            q_values = np.squeeze(q_values)

            # Forward pass for next state
            next_state_latent = self.sensor.forward(transition.next_state)
            next_q_values = self.actor.forward(next_state_latent)
            next_q_values = np.squeeze(next_q_values)

            target = q_values.copy()
            # if done:
            #     target[action] = reward
            # else:
            target[transition.action.value] = transition.reward + gamma * np.max(next_q_values)

            # Compute loss and gradients (MSE loss)
            loss_grad = MSELoss.backward(q_values.reshape(1, -1), target.reshape(1, -1))
            # Backprop through actor
            loss_grad = self.actor.backward(loss_grad.reshape(1, -1))
            self.actor.update_params()
            loss_grad = self.sensor.backward(loss_grad.reshape(1, -1))
            self.sensor.update_params()

    def compatible_with(self, other: "Organism") -> bool:
        """
        Compare the architecture and connections of the sensor neural network
        between this organism and another. Returns True if identical, else False.
        """
        def layer_signature(layer, visited):
            # Recursively build a tuple signature for the layer and its connections
            if id(layer) in visited:
                return None  # Prevent cycles
            visited.add(id(layer))
            # Signature: (class, d_in, d_out, number of prev, number of next, sorted signatures of prev/next)
            prev_sigs = []
            for p in layer.get_prev():
                sig = layer_signature(p, visited)
                if sig is not None:
                    prev_sigs.append(sig)
            next_sigs = []
            for n in layer.get_next():
                sig = layer_signature(n, visited)
                if sig is not None:
                    next_sigs.append(sig)
            return (
                layer.__class__.__name__,
                getattr(layer, "d_in", None),
                getattr(layer, "d_out", None),
                len(layer.get_prev()),
                len(layer.get_next()),
                tuple(sorted(prev_sigs)),
                tuple(sorted(next_sigs)),
            )

        sig_self = layer_signature(self.sensor, set())
        sig_other = layer_signature(other.sensor, set())
        return sig_self == sig_other


from copy import deepcopy

def traverse_layers(root_layer):
    """Yield all layers in the graph starting from root_layer (DFS, avoids duplicates)."""
    visited = set()
    stack = [root_layer]
    while stack:
        layer = stack.pop()
        if id(layer) in visited:  # prevent cycles. TODO: allow, for more complexity?
            continue
        visited.add(id(layer))
        yield layer
        # Traverse next layers (children)
        nexts = layer.get_next()
        if isinstance(nexts, list):
            stack.extend(nexts)
        elif nexts is not None:
            stack.append(nexts)

def mate(a: Organism, b: Organism, p: float = 0.05) -> "Organism":
    if not a.compatible_with(b):
        print("Organisms are not compatible for mating.")
        return None

    child = deepcopy(a)

    # Traverse both graphs in the same order (assumes identical structure)
    layers_a = list(traverse_layers(a.sensor))
    layers_b = list(traverse_layers(b.sensor))
    layers_child = list(traverse_layers(child.sensor))
    assert len(layers_a) == len(layers_b) == len(layers_child)

    for layer_a, layer_b, layer_child in zip(layers_a, layers_b, layers_child):
        # Mix weights
        mask_w = (np.random.rand(*layer_a.w.shape) < p)
        layer_child.w = np.where(mask_w, layer_a.w, layer_b.w)
        # Mix biases
        mask_b = (np.random.rand(*layer_a.b.shape) < p)
        layer_child.b = np.where(mask_b, layer_a.b, layer_b.b)

    child.mutate()
    return child
