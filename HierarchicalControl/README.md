# Hierarchical control of the differential arm cavity lengths, actuating on all 4 stages.

In this problem, we want to have a ~100 Hz UGF for the Differential Arm (DARM)
length. This is the difference in the length of the two Fabry-Perot arm cavities.

We must have enough gain so as to suppress the ~micron scale fluctuations < 1 Hz
and yet the actuators must be quiet enough so as not to introduce more than
10^-20 m/rHz of noise at 10 Hz and 10^-21 m/rHz at 100 Hz.

The quadruple suspension has 4 stages, each with different range and noise
specifications (listed below). The hierarchical control loops used for the
DARM feedback can drive all 4 stages. The crossovers of these loops must be
stable enough so as to not saturate the actuators, and also to be robust such
that the overall actuator TF changes very little. In the GW band, we need the
overall DARM actuator TF to remain stable at the level of 1% and 3 deg. so that
the interferometer calibration remains stable.



1. Want to have a 100 Hz bandwidth loop for the longitudinal controls.
2. How stable should the crossovers be? Variation in the crossovers can distort the calibration.
3. Given the existing seismic vibration and sensor noise, try to have a factor of 10-30 between the peak range of the actuator and the RMS of the control signal.
4. Minimize relative motion of interferometric DoFs
5. Avoid saturating the actuators. Less than 1 saturation per hour.
6. DARM RMS must < 1e-14 m RMS in the 0-100 Hz band.
