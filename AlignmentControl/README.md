# Angular control of a LIGO suspended optical cavity

One of the limits to the noise performance of the LIGO interferometers is the feedthrough of the angular feedback noise into the longitudinal position of the mirrors (in the direction of the laser beam axis).

There are several reasons for this:
1. The angular sensors have a noise floor of ~1e-13-1e-14 rad/rHz, whereas the displacement noise goal for the mirrors is 1e-20 m/rHz @ 10 Hz.
1. The angular loops must suppress the large seismic motions at frequencies of 0-3 Hz.
1. The plant varies in time due to thermally induced curvature changes in the mirrors and optical radiation pressure instabilities.
1. The feedback loops need to be reasonably stable and also have sharp low pass filtering so as to cut the noise off at f > 10 Hz.


### some requirements
1. Need to have enough gain below 3 Hz
2. Need to have low feedthrough of angular noise into DARM
3. Does it have to be MIMO?
4. The DRMI loops are all low bandwidth. Only cHard is a challenging loop.
5. Shape opto-mechanical response to all for 'simple' global alignment controls
6. Minimize relative motion of interferometric DoFs
7. Avoid saturating the actuators
