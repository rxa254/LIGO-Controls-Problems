# LIGO-Controls-Problems
A few examples of control system challenges in the LIGO project


### Local damping of quadruple suspension using shadow sensors on the TOP stage

1. Minimize RMS velocity or displacement of single mirror
2. Filter sensor noise so the resulting contribution to test mass motion is < 1e-19 m/rHz @ 10 Hz along the longitudinal DOF
3. Avoid saturating the actuators.

### Angular control of a Fabry-Perot cavity with a quad suspension on each end
1. Need to have enough gain below 3 Hz
2. Need to have low feedthrough of angular noise into DARM
3. Does it have to be MIMO?
4. The DRMI loops are all low bandwidth. Only cHard is a challenging loop.
5. Shape opto-mechanical response to all for 'simple' global alignment controls
6. Minimize relative motion of interferometric DoFs

### Hierarchical control of the differential arm cavity lengths, actuating on all 4 stages.
1. Want to have a 100 Hz bandwidth loop for the longitudinal controls
2. How stable should the crossovers be? Variation in the crossovers can distort the calibration.
3. Given the existing seismic vibration and sensor noise, try to have a factor of 10-30 between the peak range of the actuator and the RMS of the control signal.
4. Minimize relative motion of interferometric DoFs

### Background on the Quadruple Pendulum Suspension
The LIGO test mass optics are supported by a suspension system, or assembly, which provides passive isolation (from motion of the seismically isolated optics tables) in all degrees of freedom. The test mass suspensions are quadruple, ie. they have four suspended masses. The assembly is depicted in the quad_sketch.jpg. Each of the test mass suspensions is comprised of two adjacent chains, each chain having four masses suspended from one another. The main chain includes the test mass optic as the lowest mass. The adjacent, reaction chain provides an isolated set of masses for force reaction. The bottom mass in the reaction chain is the EndReaction Mass (ERM) (in the case of an ETM suspension). A structure surrounds and cages the suspended masses and mounts to the seismically isolated optics table. 

Vibration isolation for the test mass is accomplished with a 4-stage pendulum and 3 stages of cantilevered blade springs, providing isolation in all 6 degrees-of-freedom above approximately 1 Hz. The suspension is designed to couple 22 of the 24 quasi-rigid body modes (all but the 2 highest frequency) of each isolation chain so that they are observable and controllable at the top mass (4 wires between masses to couple pitch and roll modes; non-vertical wires to couple pendulum modes).

For each chain, all the quadruple suspension rigid body modes below 9 Hz can be actively damped from the top stage. Sensing for this local damping is accomplished with integral optical shadow sensors (or with independent optical lever sensors). The shadow sensors are collocated with the suspension actuators (see eDrawing d060218_bosem_assembly.easm) and have a noise level of 3x10^-10 m/√Hz at 1 Hz. Force actuation on the upper three masses is accomplished with coil/magnet actuators. Six degree-of-freedom actuation is provided at the top mass of each chain, by reacting against the suspension structure. These actuators are used for the local damping of 22 modes (each chain). The next two masses can be actuated in the pitch, yaw and piston directions, by applying forces between adjacent suspended masses. These stages are used for global interferometer control. Low noise current drive electronics, combined with the passive filtering of the suspension, limit the effect of actuation noise at the test mass.
