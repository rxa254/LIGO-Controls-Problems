# LIGO-Controls-Problems
A few examples of control system challenges in the LIGO project

### Table of Contents
* [Damping of Mirror Suspension](#sus_damping)
   - [Background](#background_sus)
* [Angular Control of Optical Cavity Mirrors](#asc_cav)
* [Hierarchical Control of Mirror Suspension for Optical Cavity Control](#darm_hierarchy)


<a name="sus_damping"></a>
### Damping of Mirror Suspension with local sensors

1. Minimize RMS velocity or displacement of the test mass (mirror). The RMS of test mass longitudinal motion (x), and lateral motion (y) divided by 1000, and vertical motion (z) divided by 1000, must be < 10^-19 m/rHz at 10 Hz, falling faster than 1/f^4 (except for bounce mode peak at ~12 Hz). Pitch (rotation about y) noise, and Yaw (rotation about z) noise, must each be < 10^-17 rad/rHz at 10 Hz, falling faster than 1/f^2. Roll motion should be damped, but specific limit is required.
1. Filter sensor noise so the resulting contribution to test mass motion satisfies the criteria given above.
1. Avoid saturating the actuators. (MORE DEFINITION NEEDED HERE)
1. Locally damped plant as seen by the global alignment controls should be 'simple' and easy to put in a feedback loop.

<a name="asc_cav"></a>
### Angular control of a Fabry-Perot cavity with a quad suspension on each end
1. Need to have enough gain below 3 Hz
1. Need to have low feedthrough of angular noise into DARM
1. Does it have to be MIMO?
1. The DRMI loops are all low bandwidth. Only cHard is a challenging loop.
1. Shape opto-mechanical response to all for 'simple' global alignment controls
1. Minimize relative motion of interferometric DoFs
1. Avoid saturating the actuators

<a name="darm_hierarchy"></a>
### Hierarchical control of the differential arm cavity lengths, actuating on all 4 stages.
1. Want to have a ~100 Hz bandwidth loop for the longitudinal controls
1. How stable should the crossovers be? Variation in the crossovers can distort the calibration.
1. Given the existing seismic vibration and sensor noise, try to have a factor of 10-30 between the peak range of the actuator and the RMS of the control signal.
1. Minimize relative motion of interferometric DoFs
1. Avoid saturating the actuators

------
<a name="background_sus"></a>
#### Background on the Quadruple Pendulum Suspension
![Alt](/quad_sketch.jpg "LIGO Suspension")
The LIGO test mass optics are supported by a suspension system, or assembly, which provides passive isolation (from motion of the seismically isolated optics tables) in all degrees of freedom. The test mass suspensions are quadruple, ie. they have four suspended masses. The assembly is depicted in the quad_sketch.jpg. Each of the test mass suspensions is comprised of two adjacent chains, each chain having four masses suspended from one another. The main chain includes the test mass optic as the lowest mass. The adjacent, reaction chain provides an isolated set of masses for force reaction. The bottom mass in the reaction chain is the Compensaton Plate (CP). A structure surrounds and cages the suspended masses and mounts to the seismically isolated optics table. 

Vibration isolation for the test mass is accomplished with a 4-stage pendulum and 3 stages of cantilevered blade springs, providing isolation in all 6 degrees-of-freedom above approximately 1 Hz. The suspension is designed to couple 22 of the 24 quasi-rigid body modes (all but the 2 highest frequency) of each isolation chain so that they are observable and controllable at the top mass (4 wires between masses to couple pitch and roll modes; non-vertical wires to couple pendulum modes).

For each chain, all the quadruple suspension rigid body modes below 9 Hz can be actively damped from the top stage. Sensing for this local damping is accomplished with integral optical shadow sensors (or with independent optical lever sensors). The shadow sensors are collocated with the suspension actuators (see eDrawing d060218_bosem_assembly.easm) and have a noise level of 3x10^-10 m/√Hz at 1 Hz. Force actuation on the upper three masses is accomplished with coil/magnet actuators. Six degree-of-freedom actuation is provided at the top mass of each chain, by reacting against the suspension structure. These actuators are used for the local damping of 22 modes (each chain). The next two masses can be actuated in the pitch, yaw and piston directions, by applying forces between adjacent suspended masses. These stages are used for global interferometer control. Low noise current drive electronics, combined with the passive filtering of the suspension, limit the effect of actuation noise at the test mass.
![Alt](/Quad_Sensors_Actuators.png "Suspension Acuators and Sensors")
