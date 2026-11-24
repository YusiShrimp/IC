Note that the `scenario.npz` file in this directory contains *subsampled* output from
the model answer. The file contains arrays for:

* velocity
* altitude
* mass
* radius
* angle
* distance

Some of the initial conditions can be found in the first row of the file,
and are also given here along with other key input parameters:

* velocity = 19 km/s
* radius = 35 m
* angle = 45 degrees
* pancake_factor = 6.5
* strength = 10 MPa
* density = 3000 kg/m^3

These are provided for you to visually check your code results against. Remember your
code outputs will be at a different temporal resolution to these subsampled results.
These arrays only cover the time until shortly after the airburst.

Load this array using:

```python
data = np.load('tests/scenario.npz')
v = data['velocity']
```