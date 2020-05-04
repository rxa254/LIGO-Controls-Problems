import numpy as np
import scipy.signal as signal
import control
import matplotlib.pyplot as plt

import pandas as pd
import argparse
import pprint as pp
import sys
import time

"""
ASCam is an ASC time-domain simulation to test novel feedback-filter designs. 

Produced by Jan Harms

version 1.0 (04/26/2020)
ASCam implements pitch dynamics with noise inputs from ISI-L and TOP NL/NP from damping OSEMs. The dynamics
include a power-dependent Sigg-Sidles torque feedback. ASCam simulates the test-mass pitch hard-mode readout. 
In lack of a state-space model for the ISI/TOP input noises, they are produced by Fourier methods in fixed-size 
batches.

"""

def plot_psd(timeseries, T_fft, fs, ylabel='Spectrum [Hz$^{-1/2}$]', filename='Spectrum.png'):
    n_fft = T_fft*fs
    window = signal.kaiser(n_fft, beta=35)  # note that beta>35 does not give you more sidelobe suppression
    ff, psd = signal.welch(timeseries, fs=fs, window=window, nperseg=n_fft, noverlap=n_fft//2)

    rms = np.sqrt(1./T_fft*np.sum(psd))

    plt.figure()
    plt.loglog(ff, np.sqrt(psd), label='rms = {:5.2e}'.format(rms))
    plt.xlim(0.1, 100)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def bode_plot(sys_ss, filename='plot'):
    ff = np.logspace(-1, 2, 500)
    mag, phase, w = control.bode(sys_ss, 2*np.pi*ff)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.semilogx(ff, 20*np.log10(mag))  # Bode magnitude plot
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Transfer function, mag [dB]')
    plt.xlim(0.1, 100)
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.semilogx(ff, phase*180./np.pi)  # Bode phase plot
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Transfer function, phase [deg]')
    plt.xlim(0.1, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./plots/bode_' + filename + '.png', dpi=300)
    plt.close()

def analyze_sys(sys_ss, filename='plot'):
    plt.figure()
    control.pzmap(sys_ss, Plot=True, grid=True, title='Pole Zero Map')
    plt.xlim(-5,5)
    plt.ylim(-5, 5)
    plt.tight_layout()
    plt.savefig('./plots/pzmap_' + filename + '.png', dpi=300)
    plt.close()

    gm, pm, sm, wg, wp, ws = control.stability_margins(sys_ss)
    print('------', filename, '------')
    print('Gain margin ({:5.2f}'.format(wg/(2*np.pi)), 'Hz): {:5.2f}'.format(gm))
    print('Phase margin ({:5.2f}'.format(wp/(2*np.pi)), 'Hz): {:5.2f}'.format(pm))
    print('Stability margin ({:5.2f}'.format(ws/(2*np.pi)), 'Hz): {:5.2f}'.format(sm))

def transfer_function(sys_ss, T, fs, T_fft=64, ylabel='Transfer function', filename='./plots/transfer_function.png'):

    # Fourier amplitudes of white noise
    re = np.random.normal(0, 1, T*fs//2+1)
    im = np.random.normal(0, 1, T*fs//2+1)
    wtilde = re + 1j*im
    wtilde[0] = 0

    input_signal = np.fft.irfft(wtilde)*fs
    tt = np.linspace(0, T, len(input_signal)+1)
    tt = tt[0:-1]

    t, output, x = control.forced_response(sys_ss, U=input_signal, T=tt)

    n_fft = T_fft * fs
    window = signal.hann(n_fft)  # note that beta>35 does not give you more sidelobe suppression
    ff, pxy = signal.csd(input_signal, output, fs=fs, window=window, nperseg=n_fft, noverlap=n_fft//2)
    ff, pxx = signal.welch(input_signal, fs=fs, window=window, nperseg=n_fft, noverlap=n_fft//2)

    tf = pxy/pxx

    fi = np.logical_and(ff>0.1, ff<100)     # constrain plotted values since this leads to better automatic y-range in the plot
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.semilogx(ff[fi], 20*np.log10(np.abs(tf[fi])))  # Bode magnitude plot
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(ylabel +', mag [dB]')
    plt.xlim(0.1, 100)
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.semilogx(ff[fi], np.unwrap(np.angle(tf[fi])*180/np.pi, discont=180))  # Bode phase plot
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(ylabel + ', phase [deg]')
    plt.xlim(0.1, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


class Plant:

    def __init__(self, physics, data, plotit=False):

        self.fs = data['sampling_frequency']
        self.T_batch = data['duration_batch']
        self.T_fft = data['duration_fft']

        self.plotit = plotit

        self.ns = []                                # noise models read from files as PSDs
        self.tfs = []                               # transfer functions read from files as complex amplitudes
        self.tst_noise_t = []                       # time series of test-mass noise from ISI stage 2, and damping OSEMs at top-mass

        self.SiggSidles_fb_sys = []     # state-space model of Sigg-Sidles feedback
        self.pumP_2_tstP_SS_sys = []    # state-space model of pumP (torque) to tstP (angle) with Sigg-Sidles feedback
        self.pumP_2_tstP_SS_state = np.zeros((12, 1))
        self.last_pumP_2_tstP_SS_input = 0.

        self.ti = 0                                 # index running through input noise batch

        self.P = physics['P']
        self.dydth_soft = physics['dydth_soft']
        self.dydth_hard = physics['dydth_hard']

        self.set_models(plotit=plotit)              # definition of state-space models

        self.read_noise_from_top(plotit=plotit)             # read models for test-mass pitch noise from ISI/TOP OSEMs
        self.read_sus_transfer_functions(plotit=plotit)     # read transfer functions ISI/TOP -> TST
        self.create_tst_noise_from_top(plotit=plotit)       # create batch of test-mass pitch noise from ISI/TOP OSEMs

    def reset_counters(self):
        self.ti = 0

    def read_noise_from_top(self, plotit=False):
        files = ['n_ISI_L.txt', 'n_osem_L.txt', 'n_osem_P.txt']
        units = ['m', 'm', 'rad']

        self.ns = []
        for k in range(len(files)):
            dn = pd.read_csv('./noise_inputs/'+files[k], names=['ff', 'rPSD'], delimiter=' ', skipinitialspace=True)
            ff = np.array(dn[['ff']].values.flatten())
            rpsd = np.array(dn[['rPSD']].values.flatten())

            fi = files[k].rfind('.')
            name = files[k][:fi]

            self.ns.append({'name': name, 'ff': ff, 'rPSD': rpsd, 'unit': units[k]})

            if plotit:
                plt.figure()
                plt.loglog(ff, rpsd)
                plt.xlim(0.1, 100)
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('Model spectrum, {0} [{1}]'.format(name, units[k]+'/$\sqrt{\\rm Hz}$'))
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('./plots/' + name + '.png', dpi=300)
                plt.close()

    def read_sus_transfer_functions(self, plotit=False):
        files = ['tf_topL_2_tstP.txt', 'tf_topNL_2_tstP.txt', 'tf_topNP_2_tstP.txt']
        units = [['rad', 'm'], ['rad', 'm'], ['rad', 'rad']]         # unit ['A','B'] means A/B

        self.tfs = []
        for k in range(len(files)):
            dtf = pd.read_csv('./transfer_functions/'+files[k], names=['ff', 'transfer'], delimiter=' ', skipinitialspace=True)
            ff = np.array(dtf[['ff']].values.flatten())
            tf = np.array(list(map(complex, dtf[['transfer']].values.flatten())))

            fi = files[k].rfind('.')
            name = files[k][:fi]

            self.tfs.append({'name': name, 'ff': ff, 'tf': tf, 'unit': units[k]})

            if plotit:
                plt.figure()
                plt.loglog(ff, np.abs(tf))
                plt.xlim(0.1, 100)
                plt.xlabel('Frequency [Hz]')
                plt.ylabel(name+' ['+units[k][0]+'/'+units[k][1]+']')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('./plots/' + name + '.png', dpi=300)
                plt.close()

    def create_tst_noise_from_top(self, plotit=False):

        frequencies = np.linspace(0, self.fs//2, self.T_batch*self.fs//2+1)

        # for Sigg-Sidles correction to noise spectra
        mag_SS, phase, w = control.freqresp(self.SiggSidles_fb_sys, 2*np.pi*frequencies)

        delta_freq = 1./self.T_batch
        norm = 0.5 * (1. / delta_freq)**0.5

        noises_t = []
        for k in range(len(self.ns)):
            # Fourier amplitudes of white noise
            re = np.random.normal(0, norm, len(frequencies))
            im = np.random.normal(0, norm, len(frequencies))
            wtilde = re + 1j * im

            # convolve with noise root PSD (note that ss or [b,a] models lead to divergence)
            rpsd = np.interp(frequencies, self.ns[k]['ff'], self.ns[k]['rPSD'], left=0, right=0)
            tf = np.interp(frequencies, self.tfs[k]['ff'], self.tfs[k]['tf'], left=0, right=0)
            ctilde = wtilde * rpsd * tf * mag_SS.flatten()

            # set DC = 0
            ctilde[0] = 0

            n_t = np.fft.irfft(ctilde) * self.fs
            noises_t.append(n_t)

            if plotit:
                name = self.tfs[k]['name']+'x'+self.ns[k]['name']
                unit = self.tfs[k]['unit'][0]+'/$\sqrt{\\rm Hz}$'

                plot_psd(n_t, self.T_fft, self.fs,
                         ylabel='Spectrum, {0} [{1}]'.format(name,unit), filename='./plots/'+name+'_S.png')

        self.tst_noise_t = np.sum(noises_t, 0)

        if plotit:
            plot_psd(self.tst_noise_t, self.T_fft, self.fs,
                     ylabel='Test-mass pitch noise from ISI and TOP [rad/$\sqrt{\\rm Hz}$]',
                     filename='./plots/n_tstP_from_isi_top_S.png')

    def set_models(self, plotit=False):
        """
        The following model is based on the zpk models from
        https://alog.ligo-la.caltech.edu/aLOG/index.php?callRep=41815

        The system defined here has its input at TST P (angle), which makes it possible to easily inject the
        ISI / TOP OSEM noise in the feedback model.
        """

        zz = np.array([-2.107342e-01 + 2.871199e+00j, -2.107342e-01 - 2.871199e+00j])
        pp = np.array([-1.543716e-01 + 2.727201e+00j, -1.543716e-01 - 2.727201e+00j, -8.732026e-02 + 3.492316e+00j,
                       -8.732026e-02 - 3.492316e+00j, -3.149511e-01 + 9.411627e+00j, -3.149511e-01 - 9.411627e+00j])
        k = 9.352955e+01
        [num, den] = signal.zpk2tf(zz, pp, k)
        pumP_2_tstP_sys = control.tf2ss(control.TransferFunction(num, den))

        # TST P to P transfer function
        zz = np.array([-1.772565e-01 + 2.866176e+00j, -1.772565e-01 - 2.866176e+00j, -1.755293e-01 + 7.064508e+00j,
                       -1.755293e-01 - 7.064508e+00j])
        pp = np.array([-1.393094e-01 + 2.737083e+00j, -1.393094e-01 - 2.737083e+00j, -8.749749e-02 + 3.493148e+00j,
                       -8.749749e-02 - 3.493148e+00j, -3.185553e-01 + 9.347665e+00j, -3.185553e-01 - 9.347665e+00j])
        r = 2 * self.P / 299792458. * self.dydth_hard
        k = 2.567652*r
        [num, den] = signal.zpk2tf(zz, pp, k)
        SiggSidles_sys = control.tf2ss(control.TransferFunction(num, den))

        # for Sigg-Sidles correction to ISI / TOP OSEM noise spectra
        self.SiggSidles_fb_sys = control.feedback(1, SiggSidles_sys, sign=1)

        # the Sigg-Sidles feedback sign needs to be checked
        self.pumP_2_tstP_SS_sys = control.feedback(pumP_2_tstP_sys, SiggSidles_sys, sign=1)

        if plotit:
            bode_plot(pumP_2_tstP_sys, 'pumP_2_tstP')
            bode_plot(SiggSidles_sys, 'SiggSidles')

    def get_pumP_2_tstP_SS_sys(self):
        return self.pumP_2_tstP_SS_sys

    def sample_tstP(self, pum_input_signal=None):

        # ISI L / TOP OSEM noise
        output = self.tst_noise_t[self.ti]
        self.ti += 1

        if pum_input_signal is not None:
            t, pumP, x = control.forced_response(self.pumP_2_tstP_SS_sys, U=[self.last_pumP_2_tstP_SS_input, pum_input_signal],
                                                   T=[0., 1./self.fs], X0=self.pumP_2_tstP_SS_state)
            self.last_pumP_2_tstP_SS_input = pum_input_signal
            self.pumP_2_tstP_SS_state = x[:, 1]

            # add signal from PUM P input torque
            output += pumP[1]

        return output

class Controller:

    def __init__(self, controls, data, plotit=False):

        self.fs = data['sampling_frequency']
        self.T_fft = data['duration_fft']

        self.n_hard = controls['noise_hard_mode']

        self.feedback_sys = []

        self.controller_state = np.zeros((19, 1))
        self.last_controller_input = 0.

        self.set_feedback_filter(plotit)

    def reset_counters(self):
        pass

    def set_feedback_filter(self, plotit=False):
        # Example: ASC feedback filter used 2019(?) at LIGO for hard mode

        ## dc gain 30 for low noise; 50 for high bandwidth
        dc_gain = 30.0

        ## optical response in [ct/rad]
        K_opt = 4.44e10

        l2_ct2tau = 7.629e-5 * 0.268e-3 * 0.0309

        factor = dc_gain * K_opt * l2_ct2tau

        ## ctrl
        zz = np.array([-3.5 + 1.5j, -3.5 - 1.5j, -1 + 4j, -1 - 4j,
                       -0.3436+4.11j, -0.3436-4.11j, -0.7854+9.392j, -0.7854-9.392j])
        pp = np.array([-3.5+1.5j, -3.5-1.5j, -1+4j, -1-4j,
                       -78.77+171.25j, -78.77-171.25j, -0.062832, -628.32])

        zz = np.array([-0.3436+4.11j, -0.3436-4.11j, -0.7854+9.392j, -0.7854-9.392j])
        pp = np.array([-78.77+171.25j, -78.77-171.25j, -0.062832, -628.32])
        k = 5797.86
        [num, den] = signal.zpk2tf(zz, pp, k)
        self.feedback_sys = control.tf2ss(control.TransferFunction(num, den))

        ## top mass
        [num, den] = signal.zpk2tf([-2. * np.pi * 0.1], [0], 1)
        self.feedback_sys = control.series(self.feedback_sys, control.tf2ss(control.TransferFunction(num, den)))

        ## low-pass
        zz, pp, k = signal.ellip(2, 1., 40., 2. * np.pi * 10., analog=True, output='zpk')
        [num, den] = signal.zpk2tf(zz, pp, k)
        self.feedback_sys = control.series(self.feedback_sys, control.tf2ss(control.TransferFunction(num, den)))
        zz, pp, k = signal.ellip(4, 1., 10., 2. * np.pi * 20., analog=True, output='zpk')
        [num, den] = signal.zpk2tf(zz, pp, k)
        self.feedback_sys = control.series(self.feedback_sys, control.tf2ss(control.TransferFunction(num, den)))

        ## boost
        zz = np.array([-0.322 + 0.299j, -0.322 - 0.299j, -0.786 + 0.981j, -0.786 - 0.981j,
                       -1.068 + 2.753j, -1.068 - 2.753j, -1.53 + 4.13j, -1.53 - 4.13j])
        pp = np.array([-0.161 + 0.409j, -0.161 - 0.409j, -0.313 + 1.217j, -0.313 - 1.217j,
                       -0.268 + 2.941j, -0.268 - 2.941j, -0.24 + 4.39j, -0.24 - 4.39j])
        k = factor
        [num, den] = signal.zpk2tf(zz, pp, k)
        self.feedback_sys = control.series(self.feedback_sys, control.tf2ss(control.TransferFunction(num, den)))

        if plotit:
            bode_plot(self.feedback_sys, 'feedback')

    def sample_feedback(self, input_signal=0.):

        input_signal += np.random.normal(0, (self.fs/2.)**0.5 * self.n_hard)

        t, output, x = control.forced_response(self.feedback_sys, U=[self.last_controller_input, input_signal],
                                               T=[0, 1./self.fs], X0=self.controller_state)

        self.last_controller_input = input_signal
        self.controller_state = x[:, 1]

        return output[1]

    def get_last_controller_input(self):
        return self.last_controller_input

    def get_feedback_filter_sys(self):
        return self.feedback_sys

def open_loop_run(asc_plant, asc_controller, data):
    asc_plant.reset_counters()

    N = data['duration_batch']*data['sampling_frequency']

    tstP_t = np.zeros((N,))
    readout_t = np.zeros((N,))
    control_t = np.zeros((N,))
    for k in range(N-1):
        tstP_t[k+1] = asc_plant.sample_tstP()
        control_t[k+1] = asc_controller.sample_feedback(input_signal=tstP_t[k+1])
        readout_t[k+1] = asc_controller.get_last_controller_input()
        if np.abs(tstP_t[k+1])>1:
            print('Diverging time series at', np.round(100.*k/N),'%')
            sys.exit(0)
        if np.mod(k, np.round(N/10)) == 0:
            print(np.round(100.*k/N), '% done of open-loop simulation')

    plot_psd(tstP_t, data['duration_fft'], data['sampling_frequency'],
             ylabel='TST P [rad/$\sqrt{\\rm Hz}$]', filename='./plots/tstP_open_loop.png')
    plot_psd(control_t, data['duration_fft'], data['sampling_frequency'],
             ylabel='Control output [Nm/$\sqrt{\\rm Hz}$]', filename='./plots/control_output_open_loop.png')
    plot_psd(readout_t, data['duration_fft'], data['sampling_frequency'],
             ylabel='Control input [rad/$\sqrt{\\rm Hz}$]', filename='./plots/control_input_open_loop.png')

def closed_loop_run(asc_plant, asc_controller, data):
    asc_plant.reset_counters()

    N = data['duration_batch']*data['sampling_frequency']

    tstP_t = np.zeros((N,))
    readout_t = np.zeros((N,))
    control_t = np.zeros((N,))
    for k in range(N-1):
        tstP_t[k+1] = asc_plant.sample_tstP(pum_input_signal=-control_t[k])
        control_t[k+1] = asc_controller.sample_feedback(input_signal=tstP_t[k+1])
        readout_t[k+1] = asc_controller.get_last_controller_input()
        if np.abs(tstP_t[k+1]) > 1:
            print('Diverging time series at', np.round(100.*k/N),'%')
            sys.exit(0)
        if np.mod(k, np.round(N/10)) == 0:
            print(np.round(100.*k/N), '% done of closed-loop simulation')

    plot_psd(tstP_t, data['duration_fft'], data['sampling_frequency'],
             ylabel='TST P [rad/$\sqrt{\\rm Hz}$]', filename='./plots/tstP_closed_loop.png')
    plot_psd(control_t, data['duration_fft'], data['sampling_frequency'],
             ylabel='Control output [Nm/$\sqrt{\\rm Hz}$]', filename='./plots/control_output_closed_loop.png')
    plot_psd(readout_t, data['duration_fft'], data['sampling_frequency'],
             ylabel='Control input [rad/$\sqrt{\\rm Hz}$]', filename='./plots/control_input_closed_loop.png')


"""
TO-DO
1) Implement angular to length coupling and simulate DARM noise to get a performance evaluation of ASC control. For
   this, the function bilinear.py can be used, which was developed by the Caltech group (R Adhikari et al).
"""

def main(args):
    # later, use args this way to turn them into values: float(args['critic_lr'])
    data = {'sampling_frequency': int(args['fs']), 'duration_batch': int(args['T_batch']), 'duration_fft': int(args['T_fft'])}
    physics = {'P': float(args['P']), 'dydth_soft': float(args['dydth_soft']), 'dydth_hard': float(args['dydth_hard'])}
    controls = {'noise_hard_mode': float(args['n_hard'])}

    asc_plant = Plant(physics, data, plotit=False)
    asc_controller = Controller(controls, data, plotit=False)

    transfer_functions = True
    sim_open_loop = True
    sim_closed_loop = True

    if transfer_functions:
        # bode plots of state-space models
        open_loop_sys = control.series(asc_plant.get_pumP_2_tstP_SS_sys(), asc_controller.get_feedback_filter_sys())
        bode_plot(open_loop_sys, filename='open_loop')
        analyze_sys(open_loop_sys, filename='open_loop')

        # measure transfer functions (with white noise input)
        asc_plant.reset_counters()
        transfer_function(open_loop_sys, data['duration_batch'], data['sampling_frequency'],
                          filename='./plots/bode_open_loop_measured.png')

        transfer_function(asc_controller.get_feedback_filter_sys(), data['duration_batch'], data['sampling_frequency'],
                          filename='./plots/bode_feedback_measured.png')

    # open-loop simulation
    if sim_open_loop:
        open_loop_run(asc_plant, asc_controller, data)

    # closed-loop simulation
    if sim_closed_loop:
        closed_loop_run(asc_plant, asc_controller, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for ASC simulation')

    # simulation settings
    parser.add_argument('--fs', help='sampling frequency [Hz]', default=256)
    parser.add_argument('--T_batch', help='duration of time-series of input noises [s]', default=1024)
    parser.add_argument('--T_fft', help='duration of FFT segment for plots [s]', default=64)

    # plant parameters
    parser.add_argument('--P', help='light power inside arm cavities [W]', default=56700)
    parser.add_argument('--dydth_soft', help='beam offset to angle coefficient [m/rad]', default=-2.1e3)
    parser.add_argument('--dydth_hard', help='beam offset to angle coefficient [m/rad]', default=4.5e4)

    parser.add_argument('--n_hard', help='Spectral density of hard-mode readout noise [rad/rtHz]', default=1e-13)

    args = vars(parser.parse_args())

    pp.pprint(args)

    start_time = time.time()
    main(args)
    print("--- %s seconds ---" % np.round(time.time() - start_time))
