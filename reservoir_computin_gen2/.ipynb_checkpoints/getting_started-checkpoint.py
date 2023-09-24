import reservoirpy as rpy

rpy.verbosity(0)  # no need to be too verbose here
rpy.set_seed(42)  # make everyhting reproducible !

from reservoirpy.nodes import Reservoir

reservoir = Reservoir(100, lr=0.5, sr=0.9)

import numpy as np
import matplotlib.pyplot as plt

X = np.sin(np.linspace(0, 6*np.pi, 100)).reshape(-1, 1)

plt.figure(figsize=(10, 3))
plt.title("A sine wave.")
plt.ylabel("$sin(t)$")
plt.xlabel("$t$")
plt.plot(X)
plt.show()

s = reservoir(X[0].reshape(1, -1))

print("New state vector shape: ", s.shape)

s = reservoir.state()

states = np.empty((len(X), reservoir.output_dim))
for i in range(len(X)):
    states[i] = reservoir(X[i].reshape(1, -1))

plt.figure(figsize=(10, 3))
plt.title("Activation of 20 reservoir neurons.")
plt.ylabel("$reservoir(sin(t))$")
plt.xlabel("$t$")
plt.plot(states[:, :20])
plt.show()

states = reservoir.run(X)
from reservoirpy.nodes import Reservoir, Ridge, FORCE, ESN

reservoir = reservoir.reset()

states_from_null = reservoir.run(X, reset=True)

a_state_vector = np.random.uniform(-1, 1, size=(1, reservoir.output_dim))

states_from_a_starting_state = reservoir.run(X, from_state=a_state_vector)

previous_states = reservoir.run(X)

with reservoir.with_state(reset=True):
    states_from_null = reservoir.run(X)
    
# as if the with_state never happened !
states_from_previous = reservoir.run(X) 

from reservoirpy.nodes import Ridge

readout = Ridge(ridge=1e-7)

X_train = X[:50]
Y_train = X[1:51]

plt.figure(figsize=(10, 3))
plt.title("A sine wave and its future.")
plt.xlabel("$t$")
plt.plot(X_train, label="sin(t)", color="blue")
plt.plot(Y_train, label="sin(t+1)", color="red")
plt.legend()
plt.show()

train_states = reservoir.run(X_train, reset=True)

readout = readout.fit(train_states, Y_train, warmup=10)

test_states = reservoir.run(X[50:])
Y_pred = readout.run(test_states)

plt.figure(figsize=(10, 3))
plt.title("A sine wave and its future.")
plt.xlabel("$t$")
plt.plot(Y_pred, label="Predicted sin(t)", color="blue")
plt.plot(X[51:], label="Real sin(t+1)", color="red")
plt.legend()
plt.show()

from reservoirpy.nodes import Reservoir, Ridge

reservoir = Reservoir(100, lr=0.5, sr=0.9)
ridge = Ridge(ridge=1e-7)

esn_model = reservoir >> ridge

esn_model = esn_model.fit(X_train, Y_train, warmup=10)

print(reservoir.is_initialized, readout.is_initialized, readout.fitted)

Y_pred = esn_model.run(X[50:])

plt.figure(figsize=(10, 3))
plt.title("A sine wave and its future.")
plt.xlabel("$t$")
plt.plot(Y_pred, label="Predicted sin(t+1)", color="blue")
plt.plot(X[51:], label="Real sin(t+1)", color="red")
plt.legend()
plt.show()