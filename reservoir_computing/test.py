from reservoirpy.datasets import mackey_glass

X = mackey_glass(n_timesteps=2000)

from reservoirpy.nodes import Reservoir, Ridge

reservoir = Reservoir(units=100, lr=0.3, sr=1.25)
readout = Ridge(output_dim=1, ridge=1e-5)

esn = reservoir >> readout

predictions = esn.fit(X[:500], X[1:501]).run(X[501:-1])

from reservoirpy.observables import rmse, rsquare

print("RMSE:", rmse(X[502:], predictions), "R^2 score:", rsquare(X[502:], predictions))

