import numpy as np
from engine.organism import Organism, mate, traverse_layers, Action, StateTransition

def test_organism_forward_and_act():
    org = Organism(d_in=4, d_latent=8, num_actions=3, learning_rate=0.01)
    state = np.random.randn(1, 4)
    latent = org.observe(state)
    assert latent.shape == (1, 8)
    action = org.act(latent)
    assert isinstance(action, Action)
    assert action.logits.shape == (1, 3)
    assert action.odds.shape == (1, 3)

def test_organism_learn():
    org = Organism(d_in=2, d_latent=4, num_actions=2, learning_rate=0.01, memory_size=10)
    # Fill memory with dummy transitions
    for _ in range(5):
        state = np.random.randn(1, 2)
        latent = org.observe(state)
        action = org.act(latent)
        reward = np.random.rand()
        next_state = np.random.randn(1, 2)
        done = np.random.choice([True, False])
        transition = StateTransition(state, action, reward, next_state, done, False)
        org.store_transition(transition)
    org.learn()
    # No assertion, just check that it runs without error

def test_organism_compatible_with():
    org1 = Organism(2, 4, 2, 0.01)
    org2 = Organism(2, 4, 2, 0.01)
    assert org1.compatible_with(org2)
    # Change org2's architecture
    org2.sensor = org2.actor  # break the structure
    assert not org1.compatible_with(org2)
    assert not org1.compatible_with(Organism(2, 5, 2, 0.01))

def test_traverse_layers_no_cycles():
    org = Organism(2, 4, 2, 0.01)
    layers = list(traverse_layers(org.sensor))
    assert len(layers) == 2  # sensor and actor
    # Should not revisit any layer
    assert len(layers) == len(set(id(l) for l in layers))

def test_mate_produces_child():
    org1 = Organism(2, 4, 2, 0.01)
    org2 = Organism(2, 4, 2, 0.01)
    child = mate(org1, org2, p=0.05)
    assert child is not None
    # Child should be compatible with parents
    assert org1.compatible_with(child)
    assert org2.compatible_with(child)
    # Child weights should be a mix of parents
    layers1 = list(traverse_layers(org1.sensor))
    layers2 = list(traverse_layers(org2.sensor))
    layers_child = list(traverse_layers(child.sensor))
    for l1, l2, lc in zip(layers1, layers2, layers_child):
        assert lc.w.shape == l1.w.shape == l2.w.shape
        assert lc.b.shape == l1.b.shape == l2.b.shape
