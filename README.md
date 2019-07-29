# LIGO Controls Problems
A few examples of control system challenges in the LIGO project, so that clever interested researchers can help to solve them.

### Table of Contents
* [Damping of Mirror Suspension](#sus_damping)
   - [Background](#background_sus)
* [Angular Control of Optical Cavity Mirrors](#asc_cav)
* [Hierarchical Control of Mirror Suspension for Optical Cavity Control](#darm_hierarchy)
* [Optimal ID of Linear Systems](#opt_TF)

<a name="sus_damping"></a>
## Damping of Mirror Suspension with local sensors

1. Minimize RMS velocity or displacement of the test mass (mirror). The RMS of test mass longitudinal motion (x), and lateral motion (y) divided by 1000, and vertical motion (z) divided by 1000, must be < 10^-19 m/rHz at 10 Hz, falling faster than 1/f^4 (except for bounce mode peak at ~12 Hz). Pitch (rotation about y) noise, and Yaw (rotation about z) noise, must each be < 10^-17 rad/rHz at 10 Hz, falling faster than 1/f^2. Roll motion should be damped, but specific limit is required.
1. Filter sensor noise so the resulting contribution to test mass motion satisfies the criteria given above.
1. Avoid saturating the actuators. (MORE DEFINITION NEEDED HERE)
1. Locally damped plant as seen by the global alignment controls should be 'simple' and easy to put in a feedback loop.

<a name="asc_cav"></a>
## Angular control of a Fabry-Perot cavity with a quad suspension on each end
1. Need to have enough gain below 3 Hz
1. Need to have low feedthrough of angular noise into DARM
1. Does it have to be MIMO?
1. The DRMI loops are all low bandwidth. Only cHard is a challenging loop.
1. Shape opto-mechanical response to all for 'simple' global alignment controls
1. Minimize relative motion of interferometric DoFs
1. Avoid saturating the actuators

<a name="darm_hierarchy"></a>
## Hierarchical control of the differential arm cavity lengths, actuating on all 4 stages.
1. Want to have a ~100 Hz bandwidth loop for the longitudinal controls
1. How stable should the crossovers be? Variation in the crossovers can distort the calibration.
1. Given the existing seismic vibration and sensor noise, try to have a factor of 10-30 between the peak range of the actuator and the RMS of the control signal.
1. Minimize relative motion of interferometric DoFs
1. Avoid saturating the actuators


<a name="opt_TF"></a>
## Optimal Identification of Linear, Time-Invariant Systems
1. It takes us too long to make transfer function measurements.
1. > 98% of our measurements are LTI, so no need for thinking of time-dependent, nonlinear systems.
1. > 90% of our systems can be considered SISO for basic design purposes.
1. For the ones where MIMO is necessary, we can always considered a reduced order MIMO system rather than the full state space with all possible off-diagonal elements.
1. [Description in LIGO Control Systems Wiki](https://wiki.ligo.org/CSWG/OptTF)
1. [README in OptimalSysID dir](OptimalSysID/README.rst)


------
