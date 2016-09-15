# LIGO-Controls-Problems
A few examples of control system challenges in the LIGO project


### Local damping of quadruple suspension using shadow sensors on the TOP stage

1. Minimize RMS velocity or displacement of single mirror
2. Filter sensor noise to be < 1e-19 m/rHz @ 10 Hz
3. Minimize relative motion of interferometric DoFs
4. Shape opto-mechanical response to all for 'simple' global alignment controls


### Angular control of a Fabry-Perot cavity with a quad suspension on each end
1. Need to have enough gain below 3 Hz
2. Need to have low feedthrough of angular noise into DARM
3. Does it have to be MIMO?
4. The DRMI loops are all low bandwidth. Only cHard is a challenging loop.

### Hierarchical control of the differential arm cavity lengths, actuating on all 4 stages.
1. Want to have a 100 Hz bandwidth loop for the longitudinal controls
2. How stable should the crossovers be? Variation in the crossovers can distort the calibration.
3. Given the existing seismic vibration and sensor noise, try to have a factor of 10-30 between the peak range of the actuator and the RMS of the control signal.
