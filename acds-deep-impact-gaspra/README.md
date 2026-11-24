# deepimpact

*Team Gaspra* · ACDS Deep Impact project, Imperial College London

deepimpact is a Python tool that predicts what happens when a small asteroid
enters the atmosphere above the UK: where it bursts, how far the blast damage
reaches on the ground, and which postcodes and how many people are affected.

It has two parts:

- *The airburst solver* follows the asteroid from 100 km altitude down to the
  burst or the ground.
- *The damage mapper* turns the burst into damage zones on a map, a list of
  postcodes and a head count, and can estimate the risk when the asteroid's
  properties are uncertain.

This README has two sections. [For users](#for-users) explains how to install
and run the tool. [For developers](#for-developers) explains how it is built
and how to work on it.

---

## For users

### Installation

Python 3.14 or later is required. From the base directory:


conda env create -f env.yml
conda activate deepimpact


or, in an existing environment:


pip install -e .


### Download the data

The postcode and population files are too large for the repository. Download
them once (about 115 MB, saved in resources/):


python download_data.py


### Quick start

python
import deepimpact

# 1. Simulate the fall of the asteroid
earth = deepimpact.Planet()
result = earth.solve_atmospheric_entry(radius=35, angle=45, strength=1e7,
                                       density=3000, velocity=19e3)
result = earth.calculate_energy(result)
outcome = earth.analyse_outcome(result)

# 2. Find where the blast is centred and how far each damage level reaches
pressures = [1e3, 3.5e3, 27e3, 43e3]
blat, blon, damrad = deepimpact.damage_zones(outcome, lat=55.2, lon=-2.5,
                                             bearing=217., pressures=pressures)

# 3. Find who is inside each zone
locator = deepimpact.GeospatialLocator()
postcodes = locator.get_postcodes_by_radius((blat, blon), damrad)
population = locator.get_population_by_radius((blat, blon), damrad)

# 4. Draw the largest zone on a map and save it as a web page
fmap = deepimpact.plot_circle(blat, blon, damrad[0])
fmap.save('damage_map.html')


The complete script, including the extension features, is in
examples/example.py:


python examples/example.py


### What you give it and what you get back

| Step | Call | Main inputs | Output |
|---|---|---|---|
| Fall of the asteroid | Planet.solve_atmospheric_entry | radius (m), velocity (m/s), density (kg/m³), strength (Pa), angle (degrees) | table of velocity, mass, angle, altitude, distance and radius against time |
| Energy released | Planet.calculate_energy | the table above | the same table with dedz, the energy lost per km of altitude (kt TNT/km) |
| Summary of the event | Planet.analyse_outcome | the table with dedz | dictionary: airburst or cratering, burst altitude, distance, energy and angle |
| Damage zones | damage_zones | the outcome, entry latitude and longitude, bearing, pressures (Pa) | surface zero (latitude, longitude) and one radius (m) per pressure |
| Angle-dependent zones | damage_ellipses | the outcome, the radii | three semi-axes (m) per radius |
| Postcodes and population | GeospatialLocator | surface zero and the radii or semi-axes | list of postcodes and number of people per zone |
| Risk under uncertainty | impact_risk | a Planet, a CSV file of scenarios, one pressure | probability per postcode; mean and standard deviation of people affected |
| Maps | plot_circle, plot_damage_zones | surface zero and the radii | interactive folium map |

Units throughout: metres, seconds, kilograms, pascals, degrees. Energies are in
kilotons of TNT. Bearings are measured clockwise from north.

The four pressure levels used in the project are 1 kPa (about 10% of windows
break), 3.5 kPa (about 90% of windows break), 27 kPa (wood buildings
flattened) and 43 kPa (brick buildings collapse).

### Guides and examples

| File | What it shows |
|---|---|
| User guide/user_guide_solver.ipynb | step-by-step guide to the airburst solver |
| UserGuide_Part_b.ipynb | step-by-step guide to the damage mapper, with figures and maps |
| examples/example.py | the whole chain in one script |
| examples/solver_plots.ipynb | plots of the fall and the accuracy of the solver |
| examples/chelyabinsk_fit.ipynb | fitting the model to the 2013 Chelyabinsk event |

### Things to keep in mind

- The Earth is treated as a sphere of radius 6371 km.
- Population is counted on a 1 km grid of 2011 census residents: a grid square
  counts in full if its centre is inside the zone.
- A blast radius of 0 means that pressure level is not reached on the ground.
- The model is intended for asteroids of roughly 10 to 100 m in diameter.

---

## For developers

### Module layout

```
my_module/
│
├── gitignore
├── __init__.py
├── damage.py
|   ├── _shared_locator
|   ├── surface_zero
|   ├── blast_radii
|   ├── damage_zones
|   ├── damage_ellipses
|   └── impact_risk
├── locator.py
|   ├── great_circle_distance
|   ├── _initial_bearing
|   ├── _in_bounding_box
|   ├── GeospatialLocator.__init__
|   ├── GeospatialLocator._candidates
|   ├── GeospatialLocator.get_postcodes_by_radius
|   ├── GeospatialLocator.get_population_by_radius
|   ├── GeospatialLocator._in_ellipse_zones
|   ├── GeospatialLocator.get_postcodes_by_ellipse
|   ├── GeospatialLocator.get_population_by_ellipse
|   └── GeospatialLocator.
├── mapping.py
|   └── plot_circle
└── solver.py
    ├── Planet.__init__
    ├── Planet._derivatives
    ├── Planet._rk4_step
    ├── Planet.solve_atmospheric_entry
    ├── Planet.calculate_energy
    ├── Planet.analyse_outcome
    └── Planet.analytical_solution
```

### How it works

*Solver (solver.py).* The state of the asteroid is seven numbers: speed,
mass, trajectory angle, altitude, downrange distance, radius and spreading
rate. They follow the ordinary differential equations given in
AirburstSolver.ipynb (drag, gravity, ablation, lift, and spreading after
break-up).

- *Integration:* classical fourth-order Runge-Kutta with a fixed internal
  step of at most 0.05 s. Output is returned at the time step the user asks
  for.
- *Events:* break-up (ram pressure exceeds the strength), the pancake limit
  and ground contact are located inside a step by bisection or Newton's method
  and the step is shortened to land on them. Without this, each event would be
  detected up to one step late.
- *Atmosphere:* exponential (default), constant, or tabular. The tabular
  option interpolates the logarithm of the density, because density falls
  roughly exponentially with height.
- *Energy and outcome:* calculate_energy differentiates kinetic energy
  with respect to altitude. analyse_outcome takes the altitude of the peak as
  the burst point and classifies the event as an airburst or a cratering event.

*Damage (damage.py).*

- *Surface zero* is found with the great-circle destination formulas, from
  the entry point, the bearing and the distance travelled.
- *Blast radius:* the pressure formula cannot be inverted by hand, so it is
  solved with Newton's method (SciPy solvers were not permitted). The function
  is decreasing and convex, and the starting guess is chosen on the side from
  which Newton's method cannot overshoot.
- *Angle-dependent zones:* two half-ellipses with the same area as the
  circle, squashed along the trajectory according to the burst angle.
- *Impact risk:* one pass over the scenarios in the file. The probability of
  a postcode is the number of scenarios in which it is inside the zone divided
  by the total number of scenarios. Scenarios that do no damage on the ground
  count as zero.

*Locator (locator.py).*

- *Distances* use the haversine formula, vectorised with numpy broadcasting.
- *Data* is read once when the locator is created. Postcodes without a real
  location in the file are dropped.
- *Look-ups* first discard points outside a bounding box around the zone,
  then compute the exact distance for the rest.

*Mapping (mapping.py).* Thin wrappers around folium that draw the zones,
surface zero and a legend.

### Design choices

- *The public interface is fixed by the project brief* and is used by the
  automated scorer. Function names, arguments and return types must not change.
- *Helper functions are private or module-level* (for example surface_zero
  and blast_radii in damage.py) so that each public function stays short
  and each step can be tested on its own.
- *No hidden state:* every function works from its arguments. The one
  exception is the locator kept between calls of impact_risk, because loading
  its data is the slowest step and the object is read-only.

### Tests


pytest tests/


The suite has more than 400 tests. Most do not need the downloaded data; the
ones that do are in test_locator.py, test_risk.py and
test_deepimpact.py.

| File | Covers |
|---|---|
| test_solver.py | integration, events, atmospheres, comparison with the analytical solution |
| test_outcome.py | energy calculation and outcome classification |
| test_damage.py | surface zero, blast radii, ellipses |
| test_locator.py | distances, postcode and population look-ups |
| test_risk.py | impact risk, with a fake solver and locator and with the real ones |
| test_deepimpact.py | the tests provided with the project |

Expected values in the tests come from hand calculations and from the formulas
in the project notebooks, not from the output of the code.

### Code style and continuous integration


flake8 .


Two GitHub Actions workflows run on every push and pull request: one runs
flake8, the other installs the package, downloads the data, runs the tests and
builds the documentation with warnings treated as errors.

### Documentation


python -m sphinx docs html


The HTML pages are written to html/. The function reference is generated
from the docstrings, which follow the numpy style.

### Contributing

1. Open an issue describing the task.
2. Create a short-lived branch for it.
3. Make small commits, each with one change and a message that says what
   changed.
4. Open a pull request that references the issue. It needs passing checks and
   one review before it is merged.
5. Record any substantial use of AI tools in references.md.

### Known limitations and possible extensions

- Only the naive population method is implemented.
- impact_risk runs the scenarios one after another; they are independent and
  could be run in parallel.
- The internal time step is fixed; an adaptive step would be faster for long,
  slow descents.

---

## More information

The project specification is in ProjectDescription.ipynb,
AirburstSolver.ipynb and DamageMapper.ipynb. Sources and the record of AI
use are in references.md.

## Licence

MIT. See LICENSE.