# Quadruple Suspension Damping

simple_quad_pendulum_damping.m sets up an example of damping feedback for the quadruple pendulum. The pendulum is simplified to be only a single degree of freedom (DOF) at each stage, which is the DOF parallel to the laser beam axis. The damping feedback is applied from the top mass to the top mass, however, we actually care about the damping of the unsensed test mass (bottom stage). The code plots the input disturbance and noise sources, the damped test mass amplitude spectrum, the impulse response, and the loop gain transfer function.

All other files in this directory support simple_quad_pendulum_damping.m
