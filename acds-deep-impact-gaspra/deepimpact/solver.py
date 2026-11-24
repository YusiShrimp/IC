"""
This module contains the atmospheric entry solver class
for the Deep Impact project
"""
import os
import numpy as np
import pandas as pd

__all__ = ['Planet']


class Planet():
    """
    The class called Planet is initialised with constants appropriate
    for the given target planet, including the atmospheric density profile
    and other constants
    """

    def __init__(self, atmos_func='exponential',
                 atmos_filename=os.sep.join((os.path.dirname(__file__), '..',
                                             'resources',
                                             'AltitudeDensityTable.csv')),
                 Cd=1., Ch=0.1, Q=1e7, Cl=1e-3, Rp=6371e3, g=9.81,
                 H=8000., rho0=1.2, pancake_factor=6.5):
        """
        Set up the initial parameters and constants for the target planet

        Parameters
        ----------
        atmos_func : string, optional
            Function which computes atmospheric density, rho, at altitude, z.
            Default is the exponential function rho = rho0 exp(-z/H).
            Options are 'exponential', 'tabular' and 'constant'

        atmos_filename : string, optional
            Name of the filename to use with the tabular atmos_func option

        Cd : float, optional
            The drag coefficient

        Ch : float, optional
            The heat transfer coefficient

        Q : float, optional
            The heat of ablation (J/kg)

        Cl : float, optional
            Lift coefficient

        Rp : float, optional
            Planet radius (m)

        rho0 : float, optional
            Air density at zero altitude (kg/m^3)

        g : float, optional
            Surface gravity (m/s^2)

        H : float, optional
            Atmospheric scale height (m)

        pancake_factor : float, optional
            The factor of the initial radius at which
            fragmentation will stop. Must be finite and at least 1.
            Default is 6.5.
        """

        if atmos_func not in ('exponential', 'tabular', 'constant'):
            raise ValueError(
                "atmos_func must be 'exponential', 'tabular' or 'constant'"
            )

        # ------------------
        # Input value checks
        # ------------------

        # Drag (Cd), heat transfer (Ch) and lift (Cl) coefficients and
        # surface gravity (g) must be finite and non-negative
        for name, value in (
                ('Cd', Cd), ('Ch', Ch), ('Cl', Cl), ('g', g)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

        # Heat of ablation (Q) divides the mass-loss rate (dm/dt),
        # so it must be finite and strictly positive
        if not np.isfinite(Q) or Q <= 0:
            raise ValueError("Q must be finite and positive")

        # Planet radius must be strictly positive. Infinity is allowed
        # (in a flat planet the curvature terms vanish), but NaN is not
        if np.isnan(Rp) or Rp <= 0:
            raise ValueError("Rp must be positive (or positive infinity)")

        # Exponential atmosphere, rho = rho0 * exp(-z / H): the scale
        # height H is a length and must be finite and positive
        if atmos_func == 'exponential' and (
                not np.isfinite(H) or H <= 0):
            raise ValueError("H must be finite and positive")

        # rho0 is the ground density (exponential) or the constant density
        # (constant). It must be finite and non-negative; zero is allowed
        # and gives a vacuum
        if atmos_func in ('exponential', 'constant') and (
                not np.isfinite(rho0) or rho0 < 0):
            raise ValueError("rho0 must be finite and non-negative")

        # The pancake factor is the limit on r / r_0. The radius only stays
        # constant or grows after break-up, so the limit must be >= 1
        if not np.isfinite(pancake_factor) or pancake_factor < 1:
            raise ValueError("pancake_factor must be finite and at least 1")

        # Input constants
        self.Cd = Cd
        self.Ch = Ch
        self.Q = Q
        self.Cl = Cl
        self.Rp = Rp
        self.g = g
        self.H = H
        self.rho0 = rho0
        self.pancake_factor = pancake_factor
        self.atmos_filename = atmos_filename

        if atmos_func == 'exponential':
            # rho_a = rho0 * np.exp(-z / H)
            self.rhoa = lambda z: rho0 * np.exp(-z / H)

        elif atmos_func == 'tabular':
            # Density read from external file, with a column
            # for altitude and one for the corresponding
            # density of the asteroid at that altitude

            # Column 0 is altitude (m, increasing),
            # column 1 is atmospheric density (kg/m^3)
            alt_rho_data = np.loadtxt(atmos_filename)

            altitude = alt_rho_data[:, 0]
            rho = alt_rho_data[:, 1]

            # logarithm of rho computed once, rather than on every lookup
            log_rho = np.log(rho)

            # The solver asks for the density at arbitrary altitudes (e.g.
            # the intermediate RK4 stages), which fall between table
            # entries, so interpolation is needed. Density falls roughly
            # exponentially (seen from plotting), so we interpolate log(rho)
            # linearly and exponentiate. This is more accurate than
            # interpolating rho directly.
            # right=-inf gives zero density above the table (space);
            # below the table np.interp keeps the ground value.
            self.rhoa = lambda z: np.exp(
                np.interp(z, altitude, log_rho, right=-np.inf)
            )

        elif atmos_func == 'constant':
            # Density set to the constant value provided by the
            # initialisation of the class object.
            self.rhoa = lambda x: rho0

    def _derivatives(self, t, state, density, phase):
        """
        Right-hand side of the ODE system for the asteroid, f in
        dy/dt = f(t, y).

        Parameters
        ----------
        t : float
            Time in s. Unused, because the equations do not depend on t
            explicitly, but kept so the signature matches what the RK4
            step calls.

        state : array-like, shape (7,)
            Current state [v, m, theta, z, x, r, u]:
            speed (m/s), mass (kg), trajectory angle to the horizontal
            (rad), altitude (m), downrange distance (m), radius (m), and
            u = dr/dt, the spreading rate (m/s).

        density : float
            Density of the asteroid material in kg/m^3 (not the air
            density). Only used in the spreading acceleration.

        phase : {'intact', 'spreading', 'capped'}
            Stage of the asteroid:
            'intact'    - before break-up: the radius is constant, so
                          dr/dt and du/dt are zero.
            'spreading' - after break-up, while r < pancake_factor * r_0:
                          dr/dt = u and du/dt = Cd * rho_a * v^2 /
                          (density * r).
            'capped'    - the pancake limit has been reached: spreading
                          has stopped, so dr/dt and du/dt are zero.

        Returns
        -------
        numpy.ndarray, shape (7,)
            Time derivatives [dv/dt, dm/dt, dtheta/dt, dz/dt, dx/dt,
            dr/dt, du/dt], in the same order and units as state / s.

        Raises
        ------
        ValueError
            If phase is not one of the three names above.
        """
        if phase not in ("intact", "capped", "spreading"):
            raise ValueError(f"Unknown phase: {phase!r}")

        # Get relevant variables from state
        v, m, theta, z, _, r, u = state

        # Precompute some values that area used repeatedly
        rho_a = self.rhoa(z)
        area = np.pi * r**2
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        rho_area = rho_a * area

        # Determine whether there is spreading
        spreading = phase == "spreading"

        # Returns the set of differential equations which define
        # motion for a rigid sphere
        return np.array([
            # dv/dt
            (-self.Cd * rho_area * v**2) / (2 * m)
            + self.g * sin_theta,
            # dm/dt
            (-self.Ch * rho_area * v**3) / (2 * self.Q),
            # dtheta/dt
            (self.g * cos_theta) / v
            - (self.Cl * rho_area * v) / (2 * m)
            - (v * cos_theta) / (self.Rp + z),
            # dz/dt
            -v * sin_theta,
            # dx/dt
            (v * cos_theta) / (1 + z / self.Rp),
            # dr/dt
            u if spreading else 0.0,
            # du/dt
            (self.Cd * rho_a * v**2) / (density * r) if spreading else 0.0,
        ], dtype=float)

    def _rk4_step(self, rhs, t, state, h):
        """
        Advance the state by one classical fourth-order Runge-Kutta step.

        Parameters
        ----------
        rhs : callable
            Function taking time and state and returning their time
            derivatives. The derivative vector must have the same shape as
            state, and rhs must not modify state.
        t : float
            Time at the start of the step, in seconds. rhs is evaluated
            at the RK4 stage times t, t + h/2, and t + h.
        state : numpy.ndarray, shape (n,)
            State at time t. Any size works; the solver uses n = 7:
            velocity (m/s), mass (kg), trajectory angle to the horizontal
            (rad), altitude (m), downrange distance (m), radius (m), and
            spreading rate dr/dt (m/s).
        h : float
            Step duration in seconds (h > 0).

        Returns
        -------
        numpy.ndarray, shape (n,)
            Estimated state at time t + h. This is a new array; the input
            state is not modified.
        """
        k1 = rhs(t, state)
        k2 = rhs(t + h / 2, state + (h / 2) * k1)
        k3 = rhs(t + h / 2, state + (h / 2) * k2)
        k4 = rhs(t + h, state + h * k3)
        return state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _input_checks(self, radius, velocity, density, strength,
                      angle, init_altitude, dt, radians):
        # Radius, velocity, density, angle, initial altitude and
        # output time step must all be finite.
        for name, value in (
                ('radius', radius),
                ('velocity', velocity),
                ('density', density),
                ('angle', angle),
                ('init_altitude', init_altitude),
                ('dt', dt)):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")

        # Radius must be positive.
        if radius <= 0:
            raise ValueError("radius must be positive")

        # Speed is not directional, so must be positive.
        if velocity <= 0:
            raise ValueError("velocity must be positive")

        # Density is a positive quantity.
        if density <= 0:
            raise ValueError("density must be positive")

        # Angle should be between 0 and 90 degress (pi/2 rad).
        angle_limit = np.pi / 2 if radians else 90.0
        if not 0 <= angle <= angle_limit:
            raise ValueError(
                f"angle must be between 0 and {angle_limit}"
            )

        # The asteroid must start at or above the ground.
        if init_altitude < 0:
            raise ValueError("init_altitude must be non-negative")

        # The output time step must be positive, to move forward in time.
        if dt <= 0:
            raise ValueError("dt must be positive")

        # Strength must be non-negative, but can be infinite (an asteroid
        # that never breaks up).
        if np.isnan(strength) or strength < 0:
            raise ValueError(
                "strength must be non-negative or positive infinity"
            )

    def _first_state_init(self, radius, velocity, density, angle,
                          init_altitude, radians):
        # Initialise variables defining the first state
        # Initial velocity given parameter
        v_0 = velocity
        # Mass of a sphere: (4/3)*np.pi * radius**3 * density
        m_0 = (4/3)*np.pi * radius**3 * density
        # Consistently operate on angle in radians
        theta_0 = angle if radians else np.deg2rad(angle)
        # Initial altitude
        z_0 = init_altitude
        # Initial downrange distance is 0
        x_0 = 0
        # Initial radius of the asteroid
        r_0 = radius
        # sr = spreading rate = dr / dt
        sr_0 = 0
        # Return initial state
        return np.array([v_0, m_0, theta_0, z_0, x_0, r_0, sr_0])

    def _aux_vars_init(self, radius, density, init_altitude, dt):
        # Initialise other variables for the RK4 solver
        # Start from a non-broken up state
        breakup = False
        # Set the start time at 0
        t = 0
        # Maximum internal time step, in s: dt/2, limited to the range
        # 0.01 to 0.05 for accuracy and speed, and never larger than dt
        # (so at most one RK4 step per output row). Individual steps are
        # shortened further to land exactly on events and output times.
        h = min(max(0.01, min(dt / 2, 0.05)), dt)
        # dt_h tracks the time elapsed since the last stored output row
        dt_h = 0
        # Safety maximum number of steps
        max_steps = 1e6
        # n_step counts the steps taken
        n_step = 0
        # Start from intact status
        phase = "intact"
        # Maximum radius of the asteroid
        r_0 = radius
        r_max = self.pancake_factor * r_0
        # Fraction of the original mass from which point we terminate
        ablated_fraction = 1e-6
        # Initial mass
        m_0 = (4/3)*np.pi * radius**3 * density
        z_0 = init_altitude
        return (
            breakup, t, h, dt_h, max_steps, n_step,
            phase, r_max, ablated_fraction, m_0, z_0
        )

    def _base_rk4_step(self, h, dt_h, t, dt, rhs, prev):
        # Limit the internal RK4 step so it does not pass the next
        # requested output time.
        h_step = min(h, dt - dt_h)

        # A trial step that is too large can produce invalid
        # intermediate values. Ignore arithmetic warnings while checking
        # it; rejected trials are recomputed with a smaller step.
        with np.errstate(all="ignore"):
            state = self._rk4_step(rhs, t, prev, h_step)
            # Reject implausible results and halve the step, retrying
            # from the same previous state, up to 80 times.
            for _ in range(80):
                dm = prev[1] - state[1]
                # The following are heuristic step-rejection tolerances,
                # not physical limits.
                bad = (
                    not np.all(np.isfinite(state))
                    or state[0] <= 0
                    # Mass must not increase or drop by more than 5%.
                    or dm < 0 or dm > 0.05 * prev[1]
                    # Limit the velocity change to 10% per step.
                    or abs(state[0] - prev[0]) > 0.10 * prev[0]
                    # Do not allow velocity gains beyond the
                    # gravity-only bound.
                    or state[0] > prev[0] + 2 * self.g * h_step + 1e-9
                )
                # Accept the step once it is already very small, so a hard
                # stretch cannot stall the run.
                if not bad or h_step < 1e-6:
                    break
                h_step *= 0.5
                state = self._rk4_step(rhs, t, prev, h_step)

        return h_step, state

    def _edge_event_1(self, state, h_step, rhs, prev, t_prev):
        # EVENT 1: mass reached zero during this step.
        # The overshoot state is meaningless (negative mass, and the
        # equations divide by m), so it is discarded.
        # We bisect the step duration, recomputing from prev each time,
        # then set the mass at the last positive-mass state to exactly zero.
        if state[1] > 0:
            return None

        lo, hi = 0.0, h_step
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if self._rk4_step(rhs, t_prev, prev, mid)[1] > 0:
                lo = mid
            else:
                hi = mid

        state = self._rk4_step(rhs, t_prev, prev, lo)
        state[1] = 0.0
        return state, t_prev + lo

    def _edge_event_2(self, state, h_step, rhs, ram, prev, t_prev, strength):
        # EVENT 2: break-up happened during this step.
        # We find (by bisection) the shortest sub-step after which
        # rho_a * v^2 exceeds the strength, redo the step with that
        # length, and switch break-up on. Without this, break-up is
        # detected up to one step late, causing inaccuracy issues
        # due to an error proportional to h.
        if ram(state) <= strength:
            return None

        lo, hi = 0.0, h_step
        # Bisection loop
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if ram(self._rk4_step(rhs, t_prev, prev, mid)) > strength:
                hi = mid
            else:
                lo = mid

        h_step = hi
        state = self._rk4_step(rhs, t_prev, prev, h_step)
        return h_step, state

    def _edge_event_3(self, state, r_max, h_step, rhs, t_prev, prev):
        # EVENT 3: the radius reached the pancake limit during this step.
        # Find (by bisection) the sub-step at which r reaches the limit
        # and redo the step with that length.
        # Again to avoid accuracy issues.
        if state[5] < r_max:
            return None

        lo, hi = 0.0, h_step
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if self._rk4_step(rhs, t_prev, prev, mid)[5] >= r_max:
                hi = mid
            else:
                lo = mid
        h_step = hi
        state = self._rk4_step(rhs, t_prev, prev, h_step)
        return h_step, state

    def _edge_event_4(self, state, h_step, rhs, prev, t_prev, r_max):
        # EVENT 4: ground. Find the partial step h_g that lands on
        # z = 0 (linear first guess, then Newton refinement)

        if state[3] >= 0:
            return None

        # Linear guess of appropriate step
        h_g = h_step * prev[3] / (prev[3] - state[3])

        # 3 iterations of RK4 for Newton refinement
        for _ in range(3):
            ground = self._rk4_step(rhs, t_prev, prev, h_g)
            dzdt = -ground[0] * np.sin(ground[2])
            # Adjust h_g using Newton's method
            h_g -= ground[3] / dzdt

        # Final estimation of ground contact state
        ground = self._rk4_step(rhs, t_prev, prev, h_g)

        # Remove rounding residue
        ground[3] = 0.0
        if ground[5] >= r_max:
            ground[5] = r_max
            ground[6] = 0.0

        return ground, t_prev + h_g

    def _edge_event_5(self, state, ablated_fraction, m_0):
        # EVENT 5: ablated. Mass only decays asymptotically, so treat
        # a negligible remainder as fully ablated: zero it,
        # record it, stop.
        if state[1] > ablated_fraction * m_0:
            return None

        ablated_state = state.copy()
        ablated_state[1] = 0.0
        return ablated_state

    def solve_atmospheric_entry(
            self, radius, velocity, density, strength, angle,
            init_altitude=100e3, dt=0.05, radians=False):
        """
        Solve the system of differential equations for a given impact scenario

        Parameters
        ----------
        radius : float
            The radius of the asteroid in meters

        velocity : float
            The entry speed of the asteroid in meters/second

        density : float
            The density of the asteroid in kg/m^3

        strength : float
            The strength of the asteroid (i.e. the maximum pressure it can
            take before fragmenting) in N/m^2

        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            Must be between 0 and 90 degrees. If 'radians' is set to True,
            the input must be between 0 and pi/2 radians.

        init_altitude : float, optional
            Initial altitude in m; must be finite and non-negative.

        dt : float, optional
            The positive, finite output timestep, in s.

        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the dataframe will have the same units as the
            input

        Raises
        ------
        ValueError
            If a physical input is non-finite or outside its valid range.

        Returns
        -------
        Result : DataFrame
            A pandas dataframe containing the solution to the system.
            Includes the following columns:
            'velocity', 'mass', 'angle', 'altitude',
            'distance', 'radius', 'spreading_rate', 'time'
        """

        # ------------------
        # Input value checks
        # ------------------

        self._input_checks(radius, velocity, density, strength,
                           angle, init_altitude, dt, radians)

        state = self._first_state_init(radius, velocity, density, angle,
                                       init_altitude, radians)

        (
            breakup, t, h, dt_h, max_steps, n_step, phase,
            r_max, ablated_fraction, m_0, z_0
        ) = self._aux_vars_init(radius, density, init_altitude, dt)

        # Stop once the asteroid keeps less than 1e-8 of its initial
        # kinetic energy: it has deposited essentially all of it, so the
        # energy deposition, burst values and damage are unaffected, but a
        # slow fall to the ground could otherwise take hours of model time.
        v_0 = velocity
        ke_stop_fraction = 1e-8

        # Define the RHS of the ODE for a neater call
        def rhs(tt, y):
            return self._derivatives(tt, y, density, phase)

        # Ram pressure rho_a * v^2 of a state (the break-up criterion).
        def ram(y):
            return self.rhoa(y[3]) * y[0]**2

        # Initialise list of states and output time steps
        all_states = [state]
        all_ts = [t]

        # The solver keeps running as long as all of these hold:
        #   - z >= 0: the asteroid has not yet hit the ground
        #   - v > 1: the asteroid is still moving (sufficiently). Below
        #     1 m/s only a slow drift is left, which can last for hours
        #     of simulated time without changing the outcome
        #   - m > min mass: the asteroid has not been mostly ablated
        #   - no rebound: the asteroid has not climbed back above its
        #     starting altitude
        #   - KE > 1e-8 KE_0: the asteroid still has non-negligible energy
        #   - n_step < max_steps: the step limit has not been reached,
        #     in case the other conditions fail to end the loop
        while (
            state[3] >= 0
            and state[0] > 1
            and state[1] > ablated_fraction * m_0
            and not (state[2] < 0 and state[3] > z_0)  # rebound
            and state[1] * state[0]**2 > ke_stop_fraction * m_0 * v_0**2
            and n_step < max_steps
        ):

            # Save in case needed to back up
            prev = state
            t_prev = t

            # Break-up check at the start of the step. This still matters
            # when the asteroid *starts* above the strength limit.
            breakup = breakup or (ram(state) > strength)

            # Determine the phase:
            #   - intact: has not broken up yet
            #   - spreading: has broken up and is spreading
            #   - capped: has broken up and spread to the pancake limit,
            #     so it is no longer spreading
            if not breakup:
                phase = "intact"
            elif state[5] < r_max:
                phase = "spreading"
            else:
                phase = "capped"

            # Once the asteroid is slow (below 20 m/s) it is only drifting
            # down and the solution changes slowly, so one RK4 step per
            # output row (at most 0.05 s) is enough. This halves the work
            # in the long final descent of small asteroids.
            h_now = min(dt, 0.05) if state[0] < 20. else h

            h_step, state = self._base_rk4_step(
                h_now, dt_h, t, dt, rhs, prev
            )

            # EVENT 1: mass reached zero during this step.
            mass_event = self._edge_event_1(state, h_step, rhs, prev, t_prev)
            if mass_event is not None:
                state, t = mass_event
                all_states.append(state)
                all_ts.append(t)
                break

            # EVENT 2: break-up happened during this step.
            if not breakup:
                breakup_event = self._edge_event_2(
                    state, h_step, rhs, ram, prev, t_prev, strength
                )
                if breakup_event is not None:
                    h_step, state = breakup_event
                    # The next step starts in the spreading phase.
                    breakup = True

            # EVENT 3: the radius reached the pancake limit during this step.
            if phase == "spreading":
                pancake_event = self._edge_event_3(
                    state, r_max, h_step, rhs, t_prev, prev
                )
                if pancake_event is not None:
                    h_step, state = pancake_event

            # Clamp: we remove the tiny overshoot of r and stop the spreading
            # rate, so spreading is exactly off from here on.
            if state[5] >= r_max:
                state[5] = r_max
                state[6] = 0.0

            # Advance the time and the time since the last output row
            t = t + h_step
            dt_h = dt_h + h_step

            # We purposefully add the entry for z = 0 in the
            # results, even if it is not dt seconds away from the last
            # step, because we do want to have all the info at z = 0
            if state[3] < 0:
                # EVENT 4: ground. Find the partial step h_g that lands on
                # Always record ground contact, even if it is between
                # output times.
                ground_event = self._edge_event_4(
                    state, h_step, rhs, prev, t_prev, r_max
                )
                if ground_event is not None:
                    ground, ground_t = ground_event
                    all_states.append(ground)
                    all_ts.append(ground_t)

            else:
                # EVENT 5: ablated
                ablated_state = self._edge_event_5(
                    state, ablated_fraction, m_0
                )
                if ablated_state is not None:
                    state = ablated_state
                    all_states.append(state.copy())
                    all_ts.append(t)
                    break

                if dt_h >= dt - 1e-9:       # An output time has been reached.
                    # Save the time and the state and restart the counter
                    dt_h = 0
                    all_states.append(state.copy())
                    all_ts.append(t)

            # Update state counter
            n_step += 1

        # Edge case: the asteroid stopped in the air (v <= 1 or negligible
        # kinetic energy), left the atmosphere, or the step cap was reached.
        # We keep its final state so the table ends where the run ended.
        if state[3] >= 0 and all_ts[-1] != t:
            all_states.append(state.copy())
            all_ts.append(t)

        # Convert collected data into pandas df
        data = np.array(all_states)
        data = np.column_stack((data, np.array(all_ts)))

        # Retransform to radians if necessary
        if not radians:
            data[:, 2] = np.rad2deg(data[:, 2])

        # Define column names for df
        cols = [
            'velocity', 'mass', 'angle', 'altitude',
            'distance', 'radius', 'spreading_rate', 'time'
        ]
        df = pd.DataFrame(data, columns=cols)

        return df

    def calculate_energy(self, result):
        """
        Function to calculate the kinetic energy lost per unit altitude in
        kilotons TNT per km, for a given solution. The calculation should
        use only the columns of result (velocity, mass, angle, altitude,
        distance, radius, spreading_rate and time), as returned by
        solve_atmospheric_entry.
        It should not rely on any other values computed during the solve.

        Parameters
        ----------
        result : DataFrame
            A pandas dataframe with columns for the velocity, mass, angle,
            altitude, horizontal distance, radius and spreading rate as a
            function of time

        Returns : DataFrame
            Returns a copy of the dataframe with additional column ``dedz``
            which is the kinetic energy lost per unit altitude

        """

        required = {'mass', 'velocity', 'altitude'}
        missing = required.difference(result.columns)
        if missing:
            raise ValueError(
                'result is missing required columns: '
                + ', '.join(sorted(missing))
            )

        output = result.copy()
        if output.empty:
            output['dedz'] = pd.Series(index=output.index, dtype=float)
            return output

        kinetic_energy = 0.5 * output['mass'].to_numpy() * np.square(
            output['velocity'].to_numpy()
        )
        altitude = output['altitude'].to_numpy()

        # d(kinetic energy)/dz is positive while the object descends and
        # loses energy. Convert J/m to kt TNT/km.
        if len(output) == 1:
            # A single altitude sample is not enough to estimate a gradient.
            energy_loss_per_metre = np.full(1, np.nan, dtype=float)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                energy_loss_per_metre = np.gradient(
                    kinetic_energy, altitude
                )
            # Repeated altitudes make dK/dz undefined at the affected rows.
            energy_loss_per_metre[~np.isfinite(energy_loss_per_metre)] = np.nan

        output['dedz'] = (
            energy_loss_per_metre * 1000.0 / 4.184e12
        )
        return output

    def analyse_outcome(self, result):
        """
        Inspect a pre-found solution to calculate the impact and airburst stats

        Parameters
        ----------
        result : DataFrame
            pandas dataframe with velocity, mass, angle, altitude, horizontal
            distance, radius, spreading rate and dedz as a function of time

        Returns
        -------
        outcome : Dict
            dictionary with details of the impact event, which should contain
            the following 6 keys:

            * ``outcome`` (which should contain one of the
              following strings: ``Airburst`` or ``Cratering``)
            * ``burst_peak_dedz``
            * ``burst_altitude``
            * ``burst_distance``
            * ``burst_energy``
            * ``burst_angle`` (in the same units as the input angle:
              degrees, unless ``radians=True`` was used)
        """

        outcome = {'outcome': 'Unknown',
                   'burst_peak_dedz': 0.,
                   'burst_altitude': 0.,
                   'burst_distance': 0.,
                   'burst_energy': 0.,
                   'burst_angle': 0.}

        required = {
            'dedz',
            'altitude',
            'distance',
            'angle',
            'mass',
            'velocity',
            'radius',
        }
        missing = required.difference(result.columns)
        if missing:
            raise ValueError(
                'result is missing required columns: '
                + ', '.join(sorted(missing))
            )
        if result.empty:
            return outcome

        dedz = result['dedz'].to_numpy()
        if not np.any(np.isfinite(dedz)):
            return outcome
        peak_position = int(np.nanargmax(dedz))
        peak = result.iloc[peak_position]

        initial_energy = 0.5 * result['mass'].iloc[0] * (
            result['velocity'].iloc[0] ** 2
        )
        peak_energy = 0.5 * peak['mass'] * peak['velocity'] ** 2
        final = result.iloc[-1]
        final_energy = 0.5 * final['mass'] * final['velocity'] ** 2

        is_cratering = peak['altitude'] <= peak['radius']
        if is_cratering:
            energy_lost = initial_energy - final_energy
            burst_energy = max(energy_lost, final_energy)
            outcome.update({
                'outcome': 'Cratering',
                'burst_peak_dedz': final['dedz'],
                'burst_altitude': 0.,
                'burst_distance': final['distance'],
                'burst_energy': burst_energy / 4.184e12,
                'burst_angle': final['angle'],
            })
        else:
            energy_lost = initial_energy - peak_energy
            outcome.update({
                'outcome': 'Airburst',
                'burst_peak_dedz': peak['dedz'],
                'burst_altitude': peak['altitude'],
                'burst_distance': peak['distance'],
                'burst_energy': energy_lost / 4.184e12,
                'burst_angle': peak['angle'],
            })

        return outcome

    def analytical_solution(
            self, z, v0, C_D, R, m0, theta, rho_0, H, z0=100000
    ):
        """
        Calculate the analytical velocity of an object during atmospheric entry
        using an exponential density model and aerodynamic drag.

        Parameters
        ----------
        z : float or ndarray
            Current altitude(s) above sea level (meters).
        v0 : float
            Initial velocity at the starting altitude (m/s).
        z0 : float
            Initial reference altitude (meters).
        R : float
            Radius of the object, assuming a circular cross-section (meters).
        C_D : float
            Drag coefficient of the object.
        rho_0 : float
            Reference air density at sea level (kg/m^3).
        H : float
            Atmospheric scale height (meters).
        m0 : float
            Mass of the object (kg).
        theta : float
            Flight path angle relative to the horizontal (degrees).

        Returns
        -------
        float or ndarray
            Calculated velocity of the object at the given
            altitude(s) `z` (m/s).
        """
        A = np.pi * R**2
        theta = np.radians(theta)
        const = (C_D*A*rho_0*H)/(2*m0*np.sin(theta))
        exponential = np.exp(
            const * (np.exp(-z0/H) - np.exp(-z/H))
            )
        return v0 * exponential
