# stan_predict

A simple example of how one might use Stan to predict on new data.

Example usage:

```python
import stan_predict as sp

# simulate some train and test data
X_train, y_train = sp.simulate_data(0., 1.)
X_test, y_test = sp.simulate_data(0., 1., seed=43)

# make a stan model and fit it
model = sp.StanPredictor()
model.fit(X_train, y_train)

# predict on the test data (produces samples for each X_test)
y_pred = model.predict(X_test)

# the model is faster at predicting after the first time (stan code is compiled already)
X_new, y_new = sp.simulate_data(0., 1., seed=44)
y_fast = model.predict(X_new)
```
