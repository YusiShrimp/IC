# References

## Generative AI Usage Log

The entries below summarise how I used OpenAI Codex to think through selected numerical methods and interpret results while preparing the coursework. The prompts are concise records of those queries.

### Question 1 — checking the hand-derived and SciPy interpolants

- **Tool:** OpenAI Codex
- **Context:** Question 1B. I wanted to compare my linear polynomial with SciPy in a way that checked more than just whether the curves looked similar.
- **Prompt:** “I have a hand-derived linear interpolant and a SciPy interpolant through the same two points. What numerical check would be useful alongside plotting both curves, and what should I conclude if their maximum difference is near floating-point precision?”
- **How I used the response:** I used the suggestion to compare the polynomials over a dense set of points and report their maximum absolute difference. I ran the calculation and checked the plotted curves in my notebook.

### Question 1 — choosing a genuinely quadratic interpolant and extending it

- **Tool:** OpenAI Codex
- **Context:** Questions 1C–D. I needed a third point that made the interpolant quadratic, then examples of additional data that preserved that polynomial.
- **Prompt:** “The first two points define a straight line. How can I choose a third point at x=0.3 so the interpolant is genuinely quadratic, and then generate extra distinct points that the same quadratic will still interpolate?”
- **How I used the response:** I used the reasoning to check my choice of third point and to construct added points from the resulting polynomial. I verified in code that the expanded datasets still lie on that curve.

### Question 1 — switching from interpolation to least squares

- **Tool:** OpenAI Codex
- **Context:** Question 1E. The added observations do not all lie on the Q1C quadratic, so that polynomial is no longer an interpolant of the full dataset.
- **Prompt:** “For these five points, the original Q1C quadratic misses two observations. How should I set up a new quadratic least-squares fit, and how can I use the plot and residuals to show what has changed from interpolation?”
- **How I used the response:** I used the explanation to organise the matrix formulation and comparison. I implemented the fit with NumPy and based my discussion on the resulting coefficients, plot, and residuals.

### Question 1 — comparing MSE and MAE with an outlier

- **Tool:** OpenAI Codex
- **Context:** Question 1F. I was comparing two five-point datasets, one with a visually good fit and one with a single large outlier.
- **Prompt:** “In my outlier dataset, the least-squares curve is pulled toward one point. How can I explain why its MSE grows much more sharply than its MAE, while still being clear that both metrics indicate a worse fit?”
- **How I used the response:** I used the response to frame the comparison of the two metrics. I calculated MSE and MAE for both datasets and used those values in the final explanation.

### Question 1 — demonstrating over-fitting and extrapolation

- **Tool:** OpenAI Codex
- **Context:** Question 1G. I wanted to contrast a high-degree interpolant with a lower-degree fit on noisy data and show what happens outside the observed interval.
- **Prompt:** “I have eight perturbed samples from sin(pi*x). Would comparing a degree-seven interpolant with a degree-three least-squares fit show over-fitting clearly? What should I compare inside and outside the data interval, and what limitation should I mention?”
- **How I used the response:** I used the suggestions to plan the comparisons and explain the synthetic example's limits. I generated the fits, plots, and error values in my notebook and wrote the conclusions from those results.

### Question 2 — deriving the finite-difference error orders

- **Tool:** OpenAI Codex
- **Context:** Question 2A. I was checking which Taylor-series terms cancel in the two central-difference formulas and what leading terms remain.
- **Prompt:** “When the Taylor expansions for f(x+h) and f(x-h) are substituted into the second-order central formula, which terms cancel? For the fourth-order stencil, how do the weights cancel the lower-order terms and leave an O(h^4) derivative error?”
- **How I used the response:** I used the explanation to review the cancellation argument and expected leading error orders. I checked it against my handwritten derivation.

### Question 2 — reading the convergence rates

- **Tool:** OpenAI Codex
- **Context:** Questions 2B–C. The exact derivative of exp(x) at x=1 is known, so I used it to check the finite-difference implementations and their convergence.
- **Prompt:** “My error table approaches observed orders 1, 2, and 4 as h is reduced. How should I connect those slopes to the Taylor analysis, and what does this test establish about the code without claiming it proves the implementation is correct for every case?”
- **How I used the response:** I used the response to explain how agreement in both values and convergence rates supports confidence in the implementations, while noting the limits of this single test. I checked the orders using the results from my code.

### Question 2 — understanding very large and very small step sizes

- **Tool:** OpenAI Codex
- **Context:** Question 2D. I wanted to explain the different error behaviour at the two extremes of the step-size range.
- **Prompt:** “In my exp(x) test, the errors are irregular for tiny h and become enormous for very large h. How do cancellation and round-off explain the small-step behaviour, and why do the Taylor-based approximations break down for large steps?”
- **How I used the response:** I used the explanation to interpret the error curve and selected values in my table, distinguishing round-off effects from truncation error.

### Question 3 — separating code verification from solution verification

- **Tool:** OpenAI Codex
- **Context:** Questions 3B–C. I compared RK4 with the analytical logistic solution, then compared trajectories at different time steps.
- **Prompt:** “For my logistic ODE, I compare RK4 against the analytical solution and also refine the time step against a fine-step reference. How should I distinguish code verification from solution verification, and what would count as evidence that the time step is adequate?”
- **How I used the response:** I used the distinction to structure the two checks. I calculated the errors and trajectory differences in my notebook and described what the refinement results indicate.

### Question 3 — interpreting synthetic validation and failure cases

- **Tool:** OpenAI Codex
- **Context:** Questions 3D–F. I fitted the logistic model to part of a synthetic dataset, checked held-out observations, demonstrated two failures, and compared Euler with RK4.
- **Prompt:** “I generated noisy observations from a logistic curve, fitted r and K using some observation times, and held the others out. How should I explain what that validation exercise can show and why it does not validate the model for a real population? Also, how can I connect a coarse-step Euler failure and a wrong-sign growth term to stability, accuracy, and efficiency?”
- **How I used the response:** I used the suggestions to organise the limitations of synthetic validation and the discussion of numerical failures and solver comparison. I ran the fitting and numerical examples in my notebook and based the conclusions on the resulting plots and error values.
