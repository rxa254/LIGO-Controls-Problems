# Quadruple Suspension Damping

simple_quad_pendulum_damping.m sets up an example of damping feedback for the quadruple pendulum. The pendulum is simplified to be only a single degree of freedom (DOF) at each stage, which is the DOF parallel to the laser beam axis. The damping feedback is applied from the top mass to the top mass, however, we actually care about the damping of the unsensed test mass (bottom stage). The code plots the input disturbance and noise sources, the damped test mass amplitude spectrum, the impulse response, and the loop gain transfer function.

All other files in this directory support simple_quad_pendulum_damping.m

-----
<a name="background_sus"></a>
### Background on the Quadruple Pendulum Suspension
![Alt](/quad_sketch.jpg "LIGO Suspension")
The LIGO test mass optics are supported by a suspension system, or assembly, which provides passive isolation (from motion of the seismically isolated optics tables) in all degrees of freedom. The test mass suspensions are quadruple, ie. they have four suspended masses. The assembly is depicted in the quad_sketch.jpg. Each of the test mass suspensions is comprised of two adjacent chains, each chain having four masses suspended from one another. The main chain includes the test mass optic as the lowest mass. The adjacent, reaction chain provides an isolated set of masses for force reaction. The bottom mass in the reaction chain is the Compensaton Plate (CP). A structure surrounds and cages the suspended masses and mounts to the seismically isolated optics table. 

Vibration isolation for the test mass is accomplished with a 4-stage pendulum and 3 stages of cantilevered blade springs, providing isolation in all 6 degrees-of-freedom above approximately 1 Hz. The suspension is designed to couple 22 of the 24 quasi-rigid body modes (all but the 2 highest frequency) of each isolation chain so that they are observable and controllable at the top mass (4 wires between masses to couple pitch and roll modes; non-vertical wires to couple pendulum modes).

For each chain, all the quadruple suspension rigid body modes below 9 Hz can be actively damped from the top stage. Sensing for this local damping is accomplished with integral optical shadow sensors (or with independent optical lever sensors). The shadow sensors are collocated with the suspension actuators (see eDrawing d060218_bosem_assembly.easm) and have a noise level of 3x10^-10 m/√Hz at 1 Hz. Force actuation on the upper three masses is accomplished with coil/magnet actuators. Six degree-of-freedom actuation is provided at the top mass of each chain, by reacting against the suspension structure. These actuators are used for the local damping of 22 modes (each chain). The next two masses can be actuated in the pitch, yaw and piston directions, by applying forces between adjacent suspended masses. These stages are used for global interferometer control. Low noise current drive electronics, combined with the passive filtering of the suspension, limit the effect of actuation noise at the test mass.
![Alt](/Quad_Sensors_Actuators.png "Suspension Acuators and Sensors")
