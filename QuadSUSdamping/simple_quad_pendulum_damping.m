% This script sets up a simple example of damping feedback for the
% quadruple pendulum. The pendulum is simplified to be only a single degree
% of freedom (DOF) at each stage, which is the DOF parallel to the laser beam axis.
% 
% The damping feedback is applied from the top mass to the top mass,
% however, we actually care about the damping of the test mass (bottom
% stage).
% 
% The code plots the input disturbance and noise sources, the damped test
% mass amplitude spectrum, the impulse response, and the loop gain transfer
% function.

%% Initializations

% frequency vector over which the plots are made
freq = logspace(-2,2,1e5);


% bode plot options
bodeopts = bodeoptions;
bodeopts.Grid = 'On';
bodeopts.FreqUnits = 'Hz';
bodeopts.MagUnits = 'abs';
bodeopts.MagScale = 'log';
bodeopts.Title.FontSize = 25;
bodeopts.YLabel.FontSize = 25;
bodeopts.XLabel.FontSize = 25;
bodeopts.TickLabel.FontSize = 25;
bodeopts.XLim = {[freq(1) freq(end)]};

%% Load the quad pendulum

% Load the quad pendulum model. Has units of meters / Newton
load simple_long_quadmodel % loads quad pendulum model as a state space variable called simple_long_quadmodel. For simplicity, the model only includes 1 degree of freedom (DOF) per stage, the DOF parallel to the laser beam

% Input and output indices of the quad model. These are valid for both the
% undamped and damped cases. The field L, refers to the
% 'Longitudinal or Length' DOF, which is the DOF parallel to the laser
% beam. All other DOFs are not included in this model.
out.top.disp.L = 1;
out.uim.disp.L = 2;
out.pum.disp.L = 3;
out.tst.disp.L = 4;
 in.gnd.disp.L = 1;
in.top.drive.L = 2;
in.uim.drive.L = 3;
in.pum.drive.L = 4;
in.tst.drive.L = 5;

% pull out the top to top TF (this should speed things up later)
TopL2TopL  = simple_long_quadmodel(out.top.disp.L,in.top.drive.L);

% quad model number of ins and outs
[quad_output_num,quad_input_num] = size(simple_long_quadmodel);

% group plant parameters into a convenient struct format
plant_params.act.longitdinal = 1; % damping actuator gain
plant_params.undamped_ss = simple_long_quadmodel; % full state space system
plant_params.undamped_input_num = quad_input_num; % number of quad pendulum inputs
plant_params.undamped_output_num = quad_output_num;  % number of quad pendulum outputs
plant_params.undamped_out = out; % output indices
plant_params.undamped_in = in; % input indices

%% load model noise sources

% OSEM (the sensors used in the damping) sensor noise
OSEMnoise_rawasd = [sqrt(10/freq(1)) 1 1] * (1e-10 / sqrt(2)); % basic model of OSEM noise spectrum [m/rHz]
OSEMnoise_rawfreq = [freq(1) 10 10000]; % frequency vector for the basic OSEM noise model [Hz]
OSEMnoise = transpose( 10.^interp1(log10(OSEMnoise_rawfreq),log10(OSEMnoise_rawasd),log10(freq)) ); % interpolate the basic OSEM noise model over the chosen frequency vector. This interpolation works best in logspace.
% figure,loglog(freq,OSEMnoise),xlabel('Frequency (Hz)'),ylabel('Amplitude (m/\surdHz)'),title('OSEM noise model for the longitudinal DOF'),grid on

% Seismic noise
load suspoint_L % loads a data struct called suspoint_L. The units are m/sqrt(Hz)
% The seismic data in suspoint_L was posted to the DCC at T1500318, and is from the GS13s of LHO ETMX during ER7. 
% The controls configuration according to T1500318 is "Aggressive level 3 isolation filters, LLO 90mhz blends on St1 X and Z, 45mhz blend on Y, 250mhz blends on RX/RY, 750mhz blend on RZ, St2 250mhz lowpassed blends on X,Y,Z,RZ, rdr .43hz notch sensor correction on St1 X&Y, Mittelman broadband sensor correction on HEPI Z and VL FF on X,Y&Z."
seismicnoise = transpose( interp1(suspoint_L.freq,suspoint_L.asd,freq) );


%% Create damping filter and close the loop

% simple damping loop design
damping_filter = zpk(-2*pi*0,-2*pi*20*[1;1],1e6);

damping_loop_gain = damping_filter*TopL2TopL; % loop gain TF

[damped_quad_model,damping_filter_input_index,damping_filter_output_index] = make_closed_loop_DampQuad(plant_params,damping_filter);

%% Calculate the total test mass displacement

% Transfer function from seismic noise to the damped test mass motion
damped_seismic_to_test_mass_TF = damped_quad_model(out.tst.disp.L,in.gnd.disp.L);

% Transfer function from damping sensor noise to the damped test mass motion
damped_sensor_to_test_mass_TF = damped_quad_model(out.tst.disp.L,damping_filter_input_index);

% For reference, the transfer function from seismic noise to the undamped test mass motion
undamped_seismic_to_test_mass_TF = simple_long_quadmodel(out.tst.disp.L,in.gnd.disp.L);

% test mass displacement contributions
test_mass.seismicnoise_contribution = abs(squeeze(freqresp(damped_seismic_to_test_mass_TF,2*pi*freq))) .* seismicnoise;
test_mass.sensornoise_contribution = abs(squeeze(freqresp(damped_sensor_to_test_mass_TF,2*pi*freq))) .* OSEMnoise;
test_mass.total_displacement = sqrt(test_mass.seismicnoise_contribution.^2 + test_mass.sensornoise_contribution.^2); % total is the incoherent sum of all noise contributions

% For reference, the undamped test mass displacement
test_mass.undamped = abs(squeeze(freqresp(undamped_seismic_to_test_mass_TF,2*pi*freq))) .* seismicnoise;

%% Make some plots

% noise inputs
figure,loglog(freq,seismicnoise,freq,OSEMnoise,'LineWidth',3)
set(gca,'FontSize',25)
xlabel('Frequency (Hz)')
ylabel('Amplitude (m/\surdHz)')
title('Suspension noise inputs along the longitudinal DOF (parallel to cavity axis)'),grid on
legend('Suspension point motion','Top mass damping sensor noise')

% test mass displacement
figure,loglog(freq,test_mass.undamped,'k',...
              freq,test_mass.seismicnoise_contribution,...
              freq,test_mass.sensornoise_contribution,...
              freq,test_mass.total_displacement,...
              'LineWidth',3)
set(gca,'FontSize',25)
xlabel('Frequency (Hz)')
ylabel('Amplitude (m/\surdHz)')
title('Test mass displacement along the cavity axis'),grid on
legend('Undamped','damped seismic contribution','damped sensor noise contribution','total damped')

% Impulse response
figure
impulse(damped_seismic_to_test_mass_TF,0:0.001:100),grid on
title('Impulse response from suspension point displacement to the test mass displacement')

% Loop gain transfer function
figure
bodeplot(damping_loop_gain,bodeopts)
title('Damping loop gain transfer function')