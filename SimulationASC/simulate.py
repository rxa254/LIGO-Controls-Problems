"""Main simulator runner for ASCam."""

import json
import time

from absl import app
from absl import flags
import control
from ASCam import *

flags.DEFINE_integer('fs', 256, 'Sampling frequency [Hz]')
flags.DEFINE_integer('T_batch', 1024, 'Duration of time-series of input noises [s]')
flags.DEFINE_integer('T_fft', 64, 'Duration of FFT segment for plots [s]')
# Plant parameters.
flags.DEFINE_float('P', 56700, 'Light power inside arm cavities [W]')
flags.DEFINE_float('dydth_soft', -2.1e3, 'Beam offset to angle coefficient [m/rad]')
flags.DEFINE_float('dydth_hard', 4.5e4, 'Beam offset to angle coefficient [m/rad]')

flags.DEFINE_float('n_hard', 3e-14, 'Spectral density of hard-mode readout noise [rad/rtHz]')
flags.DEFINE_string('plot_dir', 'plots', 'Directory to save plots in.')

FLAGS = flags.FLAGS

"""
TO-DO
1) Implement angular to length coupling and simulate DARM noise to get a performance evaluation of ASC control. For
   this, the function bilinear.py can be used, which was developed by the Caltech group (R Adhikari et al).
"""

def main(argv):
    del argv  # Unused.

    data = {'sampling_frequency': FLAGS.fs, 'duration_batch': FLAGS.T_batch, 'duration_fft': FLAGS.T_fft}
    physics = {'P': FLAGS.P, 'dydth_soft': FLAGS.dydth_soft, 'dydth_hard': FLAGS.dydth_hard}
    sensing = {'noise_hard_mode': FLAGS.n_hard}
    print(json.dumps(FLAGS.flag_values_dict(), indent=2))

    noise_files=['noise_inputs/n_ISI_L.txt', 'noise_inputs/n_osem_L.txt', 'noise_inputs/n_osem_P.txt']
    transfer_files=['transfer_functions/tf_topL_2_tstP.txt', 'transfer_functions/tf_topNL_2_tstP.txt', 'transfer_functions/tf_topNP_2_tstP.txt']
    reference_data_file='noise_inputs/aicReferenceData_Aplus.txt'
    asc_plant = Plant(physics, data, plot_dir=None, noise_files=noise_files, transfer_files=transfer_files)
    asc_sensors = Sensors(sensing, data)
    asc_controller = Controller(data, plot_dir=None)

    transfer_functions = True
    sim_open_loop = True
    sim_closed_loop = True
    plot_dir = FLAGS.plot_dir

    if transfer_functions:
        # bode plots of state-space models
        open_loop_sys = control.series(asc_plant.get_pumP_2_tstP_SS_sys(), asc_controller.get_feedback_filter_sys())
        bode_plot(open_loop_sys, filename=os.path.join(plot_dir, 'bode_open_loop.png'))
        analyze_sys(open_loop_sys, filename=os.path.join(plot_dir, 'pzmap_open_loop.png'))

        # measure transfer functions (with white noise input)
        asc_plant.reset_counters()
        transfer_function(open_loop_sys, data['duration_batch'], data['sampling_frequency'],
                          filename=os.path.join(plot_dir, 'bode_open_loop_measured.png'))

        transfer_function(asc_controller.get_feedback_filter_sys(), data['duration_batch'], data['sampling_frequency'],
                          filename=os.path.join(plot_dir, 'bode_feedback_measured.png'))

    # open-loop simulation
    if sim_open_loop:
        open_loop_run(asc_plant, asc_sensors, asc_controller, data, plot_dir)

    # closed-loop simulation
    if sim_closed_loop:
        closed_loop_run(asc_plant, asc_sensors, asc_controller, data, plot_dir, reference_data_file)


if __name__ == '__main__':
    start_time = time.time()
    app.run(main)
    print("--- %s seconds ---" % np.round(time.time() - start_time))
