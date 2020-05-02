# Simulation of the ASC
This project aims to create a time-domain simulation of the ASC. It currently has the following components:

1) Displacement of the ISI suspension platform along the arm and OSEM noise from top-stage damping driving displacement (along arm) and pitch are transferred to the test mass by using Fourier-domain methods. This means that noise time series are generated as fixed size batch, and in lack of a full state-space model that includes the input spectra of noises, this is the best one can do. However, maybe there are clever ways to turn input noise spectra into effective state-space models.

2) A time-domain model is constructed for the pitch dynamics between penultimate mass (PUM) and test mass (TST). These dynamics include a power-dependent Sigg-Sidles torque. They are defined as [b,a] transfer-function model, and subsequently converted into a state-space model, which is convenient for continuous sampling of the system coordinate (test-mass pitch).

3) The readout of the test-mass pitch motion is given a readout noise (so far, only for the hard-mode readout). 

4) A linear ASC feedback filter is implemented (as used in LIGO in the past).

Eventually, the system will be extended to describe all soft and hard modes and include the nonlinearity of A2L coupling.

## Project ideas:
* Test nonlinear feedback control such as reinforcement learning

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a [Creative Commons Attribution 4.0 International
License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
