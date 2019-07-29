Optimal Transfer Functions !
=============

.. image:: http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/WIENERHAMMERSTEINPROCESS/WienerHammProcessNoise.png

.. |Build Status| |Doc Status| |License| |Discord|

There is always the question of "what is optimum?", but for this problem we're defining it in the manner of
`Pintelon and Shoukens <https://books.google.com/books?id=up5UX7KuJDcC&dq=pintelon+schoukens&source=gbs_navlinks_s>`_:

for a linear, time-invariant system, where there is already some rough knowledge of the plant, they describe a multi-sine approach which maximizes the weighted Fisher Information (i.e. minimizes the determinant of the weighted covariance matrix).

Or, in other words, this minimizes the uncertainty of the poles and zeros of the system. So in this case, given a fixed measurement time and injection energy, the P&S algorithm gives the best estimation of the plant parameters.

Some examples of places where we ought to SysID better:

#. The main DARM TF used for doing the h(t) calibration.
#. LSC loop shapes; we often find surprises in these loops, especially at low frequencies.
#. length <--> angle cross-coupling in the mirror suspensions. In addition to the ID of the open-loop mechanical plant, we must consider the modifications to the plant due to the interferometer radiation pressure forces/torques, the local damping dynamics, as well as the impacts of the global angle and length control loops.
#. Seismic Isolation platforms have a lot of features below 1 Hz so the measurements can be very time-consuming.
#. After EQs or mechanical work on the mirrors/suspensions in vacuum, we often want to check that the suspension is working ideally. Would be great to do this quickly and programmatically rather than use generic DTT templates.

FAQ:
1. Isn't the Schroeder phase algorithm the best way to do broadband excitations?* No, this is detailed in the first chapter of their book.
 #. *What's wrong with just blasting everything with lots of noise all the time?* Takes too long, breaks lock, induces nonlinearity in the measurements.
 #. *Can we just take really careful swept sine measurements?* Yes, but that will <u>always be somewhat sub-optimal</u>. i.e. taking a swept sine measurement with lots of integration and thousands of points takes much more time and gives not much more information
 #. *If we just inject sine waves at the pole/zero frequencies, won't we be missing any unexpected features in the plant?* <u>Not necessarily</u>. P&S describe a 'nullstream' approach to find surprises when doing SysID on an otherwise well-known LTI system.

Developing an Optimal multi-sine algorithm:
====
We want to make some reasonable plant models with realistic noise and saturations. Then we run our measurement code on it to verify.
   1 (optional) Drive with broadband noise using a rough guess at the plant parameters. Check to see if this consistent with initial guess. If so, move on to step #2.
   1 Load zpk model of the LTI system.
   1 Define parameters for the sensing noise and force noise sources.
   1 zpk model of the whitening / dewhitening filters used to condition signals for the ADCs and DACs.
   1 Include saturation in the electronics + ADC/DAC.
   1 Calculate multi-sine waveform to optimize weighted Fisher Information.
   1 <span style='background-color: transparent;'>Check simulation for saturations. What is an acceptable saturation rate for SysID?</span>
   1 Run measurement. Use Tukey window to ramp excitation on/off.
   1 Ramp off excitations and return all settings to nominal in case of interferometer lock loss, or other error state.
   1 Use inference to determine plant parameters and uncertainties.
   1 If necessary (e.g. uncertainties too large or initial parameters too far from actual), refine plant and noise parameters in steps 2-3 and repeat steps 4-9.
---
---+++++ useful references:
   * [[https://books.google.com/books?id=up5UX7KuJDcC&dq=pintelon+schoukens&source=gbs_navlinks_s][System ID: A Freq Domain Approach]]
   * Larry Price [[https://dcc.ligo.org/LIGO-G1400084][presentation]] on the concept in DCC
   * [[https://git.ligo.org/dennis-coyne/SysID][Dennis Coyne's GitLab ]]on SysID
   * Larry's Git repo?
   * Public GitHub repo on [[https://github.com/rxa254/LIGO-Controls-Problems][LIGO Controls Problems]]



.. |Build Status| image:: https://travis-ci.com/adafruit/circuitpython.svg?branch=master
   :target: https://travis-ci.org/adafruit/circuitpython
.. |Doc Status| image:: https://readthedocs.org/projects/circuitpython/badge/?version=latest
   :target: http://circuitpython.readthedocs.io/
.. |Discord| image:: https://img.shields.io/discord/327254708534116352.svg
   :target: https://adafru.it/discord
.. |License| image:: https://img.shields.io/badge/License-MIT-brightgreen.svg
   :target: https://choosealicense.com/licenses/mit/
