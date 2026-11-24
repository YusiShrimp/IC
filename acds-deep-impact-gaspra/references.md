# References

<!--
# AI-usage

Tool:
Asked:
Kept:
Changed:
Checked:
Reviewed by: @
-->
# AI-usage

## A1
Tool: Claude (Anthropic)
Asked: Help understanding RK4 and how to apply it to the atmospheric entry ODEs; feedback on my `solve_atmospheric_entry` code; how to load and interpolate the tabular atmosphere; how to split out `_derivatives` and `_rk4_step` to match the team's tests; and fixes for the ground-stop row and for the stalled-asteroid case.
Kept: The ground-stop interpolation idea, the stalled-asteroid block at the end of the loop (marked "Claude contribution" in the code), and the `_derivatives` and `_rk4_step` methods (pulled out of my own loop by Claude).
Changed: Wrote the loop, break-up logic, pancake clamp and output spacing myself and adapted the suggestions to it (e.g. kept `h = dt/5` and the `abs(dt_h - dt)` test instead of Claude's alternatives; `np.loadtxt` for the table, `right=0.0` after testing).
Checked: Compared against `scenario.npz` (errors about 1e-3 with strength 1e7), checked output spacing for several `dt`, and ran `tests/test_solver.py` and `tests/test_deepimpact.py`.
Reviewed by: @Gabriel Chidiac

## A2

#### Git conflicts
Tool: Copilot
Asked: For help using Github on instances where rebasing and merge conflicts where present (mainly on first day making edits in branches).
Kept: Suggestions that would lead to no conflicts and issues with working branches.
Changed: A few suggestions that would make the tree messy, and made my own edits that would lead to groups working in a single branch.
Checked: Viewed changes frequently on GitHub to ensure the PRs created were free of conflicts.
Reviewed by: @ada-ssn26 (Sebastian Navarrete)

#### Testing for calculate_energy and analyse_outcome
Tool: Google Gemini
Asked: Methods for pandas DataFrame and numpy to conduct checks of DataFrame objects without needing to involve loops. In addition, I brainstormed with the AI tool on a cases for analyse_outcome for how cratering should be handled for scenrios where in the data frame there is an instance where altitude is less than current radius.
Kept: Methods such as np.allclose. pd.isna(), pd.all(), and pd.eq().all() that were used in some assert statements.
Changed: The parameters of the methods to accomodate for the parameters and outputs of the tests.
Checked: I verified the tests would perform based on what I wanted them to do by occupying dummy inputs and outputs and using print statements to observe the assertions would correctly verify what is required. For the brainstormed test case, I had a clue that it should fail based on a message from the scorer run, from the AI tool I obtained some assurance that cratering should be handled differently and is where I suggested it being through looking at altitude and radius which is what I fixed.
Note: Also made use of Lecture notes to identify equations needed for manual verifications (particularly made use of forward/backward/central difference equations from Computational Mathematics module). Also used Numpy and Pandas documentation.
Reviewed by: @ada-ssn26 (Sebastian Navarrete)

#### Plotting slope of RK4 error
Tool: Copilot:
Asked: For help using matplotlib to give a nice visual of a line that could be shown in the convergence plot of errors to indicate that the error matches the expected RK4 error.
Kept: Used the suggestions of the AI tool to encorporate the new line.
Changed: Some visual paramters.
Checked: Verfied with the plot to ensure the new line matched with what was expected.
Reviewed by: @ada-ssn26 (Sebastian Navarrete)

#### solver plots and Chelyabinsk fit (examples/solver_plots.ipynb; examples/chelyabinsk_fit.ipynb)
Tool: OpenAI Codex
Asked: Help implement and explain solver plots; check them against the project requirements; fit the Chelyabinsk energy-deposition data using the known entry conditions and the specified parameter ranges.
Kept: The plotting workflow using the solver output and calculate_energy; the bounded parameter search using differential_evolution followed by least_squares; plots comparing the model with the supplied observations.
Changed: No further team edits reported yet.
Checked: Executed all code cells in both notebooks using the Python (acds) kernel. Confirmed that the solver plots and fit outputs were generated without cell errors. The fit was compared with the baseline parameters. The full pytest suite was not run for this notebook work.
Reviewed by: @ada-yj1226

#### calculate_energy and analyse_outcome (deepimpact/solver.py)
Tool: OpenAI Codex
Asked: Help implement and explain the energy-loss calculation and outcome analysis functions, following the requirement that calculate_energy use values from the solver result dataframe.
Kept: calculate_energy derives kinetic energy from mass and velocity, estimates its change with altitude using numpy.gradient, converts the result to kt TNT/km, and returns a copy of the dataframe with a dedz column. analyse_outcome finds the peak energy-deposition row and returns the outcome type, burst properties, and energy.
Changed: The final implementation handles empty and degenerate inputs. It classifies an event as cratering when the peak-deposition altitude is no higher than the object's radius; cratering energy is the larger of the energy lost and the remaining kinetic energy at the final row.
Checked: The A2 tests cover output format, input preservation, expected dedz values, degenerate altitude data, missing required columns, and Airburst/Cratering outcomes. The tests passed during A2 development.
Reviewed by: @ada-yj1226


## B2

#### tests/test_locator.py: use of pytest and great_circle_distance

Tool: Claude (Anthropic)
Asked: Explain the use of mark in pytest; Review tests.
Kept: Format of using @mark; Add missing .T in test symmetry; Expand test function explanation in test_performance
Checked: Ran pytest and hand-calculated values
Reviewed by: @ada-sz2626

#### tests/test_locator.py: GeospatialLocator.__init__

Tool: Claude (Anthropic)
Asked: Review tests for the data loading in GeospatialLocator.__init__; explain how the pandas checks work.
Kept: Add the test of no -9999 cells left, and default norm.
Changed: Checked each constant and set PC_COL, LAT_COL, LON_COL to match.
Checked: Ran pytest and hand-calculated values
Reviewed by: @ada-sz2626

#### tests/test_locator.py: _radius

Tool: Claude (Anthropic)
Asked: Explain the meaning of function get_postcodes_by_radius; Hints for the brute-force postcode lookup; Review tests.
Kept: The test skeleton.
Changed: Chose IMPERIAL from the coordinates of 'SW7 2AZ' in the postcode file; set brute_force_postcodes function; fixed between() calls.
Checked: Ran the tests on the real data: test_postcodes_inside_uk found 12,687 postcodes with placeholder coordinates (latitude 99.999999). Checked NORTH_SEA has no postcodes or populated cells nearby.
Reviewed by: @ada-sz2626

#### tests/test_locator.py: _population

Tool: Claude (Anthropic)
Asked: Review Test; List tests could be reused in Extensions _population from _radius; Explain how the known-value tests use real numbers from the data(single cell, whole UK).
Kept: Kept and Checked the real numbers of data given. The format of test unknown methods.
Checked: Ran the tests on the real data. single_cell and whole_uk take their expected values straight from loc.pop.
Reviewed by: @ada-sz2626

#### tests/test_locator.py: _ellipse

Tool: Claude (Anthropic)
Asked: List tests could be reused in Extensions _ellipse from _radius; Explain why three semi-axis in ellipse and how bearing works.
Kept: Tests in float input and reverse bearing.
Checked: Ran pytest and hand-calculated values. Use previous output in _radius functions.
Reviewed by: @ada-sz2626

#### tests/test_locator.py: _in_bounding_box

Tool: Claude (Anthropic)
Asked: Hints in testing once JY added a bounding-box pre-filter to speed up the lookups; write those tests; write section headers and a file docstring in the same style as the solver tests.
Kept: Skeleton of tests of _in_bounding_box on its own, Add _initial_bearing, _in_ellipse_zones in ellipse section
Changed: Added SHETLAND and CORNWALL to the constants.
Checked: pytest tests against JY's _in_bounding_box: all pass.
Reviewed by: @ada-sz2626

#### tests/test_mapping.py: plot_circle and plot_damage_zones

Tool: Claude (Anthropic)
Asked: Explain the use of folium; Number of tests should be ran on mapping.
Kept: The skeleton of tests.
Changed: Printed the folium objects first to find where the centre and radius are stored (location, options['radius'])
Checked: Ran pytest tests/test_mapping.py: all pass. Opened the map in a browser through python -m http.server to see the zones, after the OpenStreetMap tiles were blocked for a local file.
Reviewed by: @ada-sz2626

#### tests/test_mapping.py: _ellipse_outline and plot_damage_ellipses

Tool: Claude (Anthropic)
Asked: The difference between circle and ellipse mapping; List tests could be reused
Kept: Bearing 0 puts a_front to the north and a_behind to the south; reversed bearing with front and behind swapped gives the same zone.
Checked: Ran pytest tests/test_mapping.py: all passed.
Reviewed by: @ada-sz2626

#### tests/test_mapping.py: plot_risk and plot_trajectory

Tool: Claude (Anthropic)
Asked: Explain _postcode_sector, plot_risk and plot_trajectory; Explain the impact_risk create the postcodes table
Kept: For plot_risk, use the small hand-made table of five postcodes so every expected value (dot location, highest and mean probability, postcodes affected, sectors with no dot) can be worked out by hand
Checked: Ran pytest tests/test_mapping.py, all passed
Reviewed by: @ada-sz2626

#### great_circle_distance (deepimpact/locator.py; issues #4, #20)

Tool: Claude (Anthropic)
Asked: Explain the haversine formula and numpy broadcasting; provide a vectorised implementation; later add input range checking (ValueError) and the 1-D single-point output shape
Kept: Haversine implementation with broadcasting ((n, 1) and (1, m) arrays) and the np.clip guard; ValueError for latitude outside [-90, 90] or longitude outside [-180, 180]; removing the axis of a single 1-D point
Changed: Comment wording; the (n,) output shape for a 1-D point was chosen in the pair to match the docstring example
Checked: Teacher's reference values in tests/test_deepimpact.py (XPASS); all tests in tests/test_locator.py; hand-calculated distances (pi*R/180 for 1 degree, pi*R for antipodes)
Reviewed by: @ada-jl3326

#### GeospatialLocator.__init__ (deepimpact/locator.py; issue #5)

Tool: Claude (Anthropic)
Asked: Explain pandas read_csv, the .asc file layout, numpy reshape and boolean masks
Kept: Default file paths relative to locator.py (os.path); reading ncols, nrows and NODATA_value from the .asc header; np.loadtxt + reshape(3, nrows, ncols); mask pop != nodata to drop -9999 cells
Checked: 1,771,908 postcodes; 127,675 populated cells; total population 63,184,827 (matches the 2011 UK census); works when run from another directory; full test suite 28 passed, 6 xpassed
Reviewed by: @ada-jl3326

#### get_postcodes_by_radius (deepimpact/locator.py; issue #27)

Tool: Claude (Anthropic)
Asked: Explain how to look up postcodes within each radius with numpy (to_numpy, boolean masks, tolist); found and handled 12,687 postcodes with the placeholder location (99.999999, 0.0)
Kept: Valid-location postcode arrays built once in __init__ (_pc_latlon, _pc_codes); distances with self.norm; one list per radius with dist < r and tolist()
Checked: Example locations (Imperial College: SW7 2AZ and SW7 2BU inside 1.5 km; Cambridge: 19 postcodes within 200 m, 2 within 100 m); removed postcodes are IM/GY/JE/BT; full test suite passes
Reviewed by: @ada-jl3326

#### get_population_by_radius (deepimpact/locator.py)

Tool: Claude (Anthropic)
Asked: Explain the naive population method and np.column_stack
Kept: Cell centres as an (N, 2) array; distances with self.norm; for each radius the sum of pop where dist < r, converted to int; NotImplementedError for methods other than 'naive'
Checked: Radius 0 gives 0 and a 2000 km radius gives the UK total 63,184,827; example locations (Imperial College, Cambridge); full test suite passes
Reviewed by: @ada-jl3326

#### get_postcodes_by_ellipse and get_population_by_ellipse (deepimpact/locator.py; issue #44)

Tool: Claude (Anthropic)
Asked: Explain how to test whether a point is inside the two half-ellipses; design the bearing helper, the zone-mask helper and the two query functions
Kept: Distance from self.norm plus the initial bearing formula from DamageMapper.ipynb, rotated into along-track and crossrange coordinates; one private helper shared by both queries; the ellipse inequality multiplied out so that a zero semi-axis gives an empty zone
Checked: Hand-worked points (a_perp 5000, a_front 2000, a_behind 3000 m, bearing 90: 2.5 km west inside, 2.5 km east outside, 3 km north inside); equal semi-axes give exactly the radius results for postcodes and population (Imperial, Cambridge); bearing + 180 with a_front and a_behind swapped gives the same zones; teacher's ellipse tests pass
Reviewed by: @ada-jl3326

#### Bounding-box pre-filter: _in_bounding_box, _candidates, _pop_latlon (deepimpact/locator.py; issue #52)

Tool: Claude (Anthropic)
Asked: Explain how a latitude-longitude bounding box can skip far-away points; design the box helper, the candidate check and suggest the changes to the four queries
Kept: Latitude limit r / R and longitude limit arcsin(sin(r / R) / cos(lat0)); no longitude limit when the circle covers a pole; short way round near longitude 180; box skipped when self.norm is not great_circle_distance; box of the largest semi-axis for the ellipse zones; _pop_latlon built once in __init__
Checked: All four queries give identical results to the previous version (30 random centres with random radii and ellipse parameters); no point inside the circle falls outside the box (random points worldwide, including near the poles and longitude 180); full test suite 206 passed; 10 km query: postcodes 0.044 -> 0.015 s, population 2.8 -> 0.6 ms
Reviewed by: @ada-jl3326

#### plot_damage_zones and its helpers (deepimpact/mapping.py)

Tool: Claude (Anthropic)
Asked: Explain folium (Circle vs CircleMarker, tooltip and popup, FeatureGroup and LayerControl) and branca legends; design plot_damage_zones and the popup, legend, layer-control and zoom helpers
Kept: Circles drawn largest first; colour set by the damage level of each pressure (DamageMapper.ipynb table), so the same pressure always has the same colour; legend built from the same lists as the colours; exactly one layer control, kept last; zoom using the locator's bounding-box formula
Checked: Colours, drawing order and popup text in the rendered HTML; maps viewed in a browser; partner's tests in tests/test_mapping.py
Reviewed by: @ada-jl3326

#### plot_damage_ellipses and _ellipse_outline (deepimpact/mapping.py)

Tool: Claude (Anthropic)
Asked: Explain how to draw the two half-ellipses on a map
Kept: Edge points from the ellipse equation, turned into a distance and bearing from surface zero and moved there with surface_zero (the locator's ellipse test in reverse); same colours, popups and layers as the circles
Checked: An edge shrunk by 0.1% is inside and one enlarged by 0.1% is outside according to GeospatialLocator._in_ellipse_zones; equal semi-axes give a circle of that radius; partner's tests
Reviewed by: @ada-jl3326

#### plot_risk and _postcode_sector (deepimpact/mapping.py)

Tool: Claude (Anthropic)
Asked: Compare ways of showing the impact_risk output on a map
Kept: One dot per postcode sector, coloured on a fixed 0-1 scale (branca LinearColormap) by the highest probability in the sector; popups with the mean probability and the number of affected postcodes
Changed: Chose sector aggregation and the highest probability (conservative) after comparing the options
Checked: Hand calculation for sector SW7 2 against its popup; run on the output of impact_risk for the default scenarios; partner's tests
Reviewed by: @ada-jl3326

#### plot_trajectory (deepimpact/mapping.py)

Tool: Claude (Anthropic)
Asked: Guide to draw the ground track from the entry point to surface zero with direction arrows
Kept: 50 points along the great circle from surface_zero; arrows with folium.plugins.PolyLineTextPath; a marker at the entry point
Checked: The track ends exactly at surface zero and its bearing from the entry point equals the trajectory bearing at every point; partner's tests
Reviewed by: @ada-jl3326

#### plot_damage_zones as one call for a whole scenario (deepimpact/mapping.py)

Tool: Claude (Anthropic)
Asked: Combine a suggestion from the review of #69 (one plot_damage_zones for circles, ellipses and the trajectory; leave out zones with a zero radius) with the existing design
Kept: plot_damage_zones decides what to draw and calls _plot_circles, plot_damage_ellipses (semi_axes=) and plot_trajectory (entry=); new arguments added at the end, so existing calls still work; the zoom covers the zones and the entry point; levels with a zero radius are left out. Suggested tests for a whole scenario
Changed: Kept plot_damage_ellipses and plot_trajectory public, so they can still be used on their own
Checked: All of the partner's existing tests pass; circles, ellipses, and ellipses with the trajectory drawn from the damage_zones docstring example; the track ends at surface zero; ValueError when semi_axes or entry is given without a bearing; pytest all pass.
Reviewed by: @ada-jl3326, @ada-sz2626

## B1: tests for damage_zones, damage_ellipses and impact_risk; impact_risk speed-up

Tool: Claude (Anthropic)
Asked: Help with writing tests for damage_zones, damage_ellipses and impact_risk, and with speeding up repeated impact_risk calls.
Kept: Parts of the test structure, and the shared locator in impact_risk, after review.
Changed: Adapted the tests to the project brief and to our code, and combined them with my B1 partner's tests.
Checked: Expected values were checked against the formulas in DamageMapper.ipynb; the full test suite and flake8 pass.
Reviewed by: @ada-gc126



## B1: deepimpact/damage.py

#### surface_zero (deepimpact/damage.py)

Tool: Claude (Anthropic)
Asked: Review my implementation of the surface zero calculation.
Kept: Small fixes from the review (np.clip, %360 to wrap longitude into [-180,180]).
Changed: The implementation is my own. Formulas taken from the notebook.
Checked: Ran tests/test_damage.py and tests/test_deepimpact.py; flake8 passes.
Reviewed by: @ada-uu26


#### blast_radii (deepimpact/damage.py)

Tool: Claude (Anthropic)
Asked: Review my Newton solver for the blast radii; help me with the stopping condition.
Kept: Review comments on the docstring, and solving for all pressures at once with numpy arrays instead of one pressure at a time.
Changed: The method is my own choice: solving p(s) = pressure for s with Newton's method, the starting guess taken from each term of the formula on its own, and returning 0 where the damage level does not reach the ground.
Checked: Ran tests/test_damage.py; substituted the radii back into the overpressure formula from DamageMapper.ipynb; flake8 passes.
Reviewed by: @ada-uu26


#### damage_zones (deepimpact/damage.py)

Tool: Claude (Anthropic)
Asked: Review my implementation combining surface_zero and blast_radii.
Kept: The flake8 errors pointed out in the review, which I fixed.
Changed: The implementation is my own: ValueError for pressures that are not positive; a float for a single pressure and a list for several.
Checked: Ran tests/test_damage.py and tests/test_deepimpact.py; flake8 passes.
Reviewed by: @ada-uu26


#### damage_ellipses (deepimpact/damage.py)

Tool: Claude (Anthropic)
Asked: Review my implementation of the angle-dependent blast zone from DamageMapper.ipynb.
Kept: Review comments on the docstring and comments.
Changed: Fixed flake8 issue and added ValueError when a semi-axis would not be positive.
Checked: Ran tests/test_damage.py; hand-checked that the ellipse area equals the circle area; flake8 passes.
Reviewed by: @ada-uu26


#### impact_risk

Tool: Claude (Anthropic)
Asked: Explain the uncertainty analysis; provide an implementation of the scenario loop and postcode probabilities.
Kept: Probability = scenarios that hit the postcode / total scenarios; misses counted as zero; mean and standard deviation of population.
Changed: Fixed a missing import and flake8 errors; shared locator added afterwards by my B1 partner.
Checked: Results on the provided impact file match an independent calculation; tests/test_risk.py and flake8 pass.
Reviewed by: @ada-uu26
