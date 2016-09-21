function [noise_ASD, response_FR] = SEI_sensor_noise(sensor_name, freq)
% SEI_sensor_noise  Noise estimates for HAM-ISI, BSC-ISI, and HEPI SEI sensors
%  draft version
%
%  noise = SEI_sensor_noise('sensor_name', freq)
%  noise is the ASD of the self noise of the sensor, in m/rtHz
%    this includes the noise of the default LIGO preamp, if any
%    see T0900XX for more details.
%  freq is the frequency vector (Hz)
%  sensor_name is a string defining the amplifier in question
%   so far 'ADE_1mm', 'ADE_p25mm', 'Kaman_1mm',
%       'L4C', 'GS13', 'STS2' 'T240meas', and 'T240spec' have been included.
%       (the code is not case sensitive)
%
% This can also be called with an optional second output argument
% which returns the frequency response of the sensitivity of the sensor
% in volts/ meter, e.g.
%
% [noise, response_FR] = SEI_sensor_noise('GS13',freq)
%
% Brian Lantz, Sept 24, 2009
%
% https://dcc.ligo.org/LIGO-T0900450
%
% Rana, 2015 - adding some high noise, cheap sensors for comparison
% might be useful for many sensor arrays for NN estimation / subtraction
%

if strncmpi(sensor_name,'ADE_1mm',5)
    % see Andy Stein SEI log enty 1311, nov 4 overnight plot
    freq_data   = [.001,  .002,  .01,    .1,    .7,    100];
    noise_data  = [ 5e-8, 9e-9, 2.5e-9, 5e-10, 2e-10, 2e-10];
    
    lognoise    = interp1(log10(freq_data),log10(noise_data),log10(freq));
    noise_ASD   = 10.^lognoise;
    response_FR = 10/1e-3 * ones(size(freq)); % 10 V per mm
    
elseif strncmpi(sensor_name,'ADE_p25mm',5)
    freq_data   =   [.001,   .002,    .01,      .1,    .7,    100];
    noise_data  = [1.5e-08, 2.7e-09, 7.5e-10, 1.5e-10, 6e-11, 6e-11];
    
    lognoise    = interp1(log10(freq_data),log10(noise_data),log10(freq));
    noise_ASD   = 10.^lognoise;
    response_FR = 10/0.25e-3 * ones(size(freq)); % 10V/ 0.25 mm
    
elseif strncmpi(sensor_name,'L4C',3)
    freq_data   =   [.04,   .52,     .8,       1.4,     4,       10,      100];
    noise_data  = [1.0e-06, 1.0e-10, 2.3e-11, 7.0e-12, 2.3e-12, 8.5e-13, 8.0e-14];
    
    lognoise    = interp1(log10(freq_data),log10(noise_data),log10(freq));
    noise_ASD   = 10.^lognoise;
    response_FR = 10/0.25e-3 * ones(size(freq)); % 10V/ 0.25 mm
    
elseif strncmpi(sensor_name,'T240spec',5)
    % this is taken directly from the T240_noise_spec_ASD.m code
    data = [...
        0.001	-171.688; ...
        0.003	-179.481; ...
        0.007	-183.636; ...
        0.018	-187.532; ...
        0.044	-189.610; ...
        0.082	-190.130; ...
        0.226	-190.130; ...
        0.530	-189.091; ...
        1.000	-187.273; ...
        2.257	-183.377; ...
        3.634	-180.000; ...
        6.335	-174.286; ...
        9.803	-168.312];
    
    
    freq_spec = data(:,1);
    w_spec    = 2*pi*freq_spec;
    
    dB_accel_power_spec = data(:,2);
    accel_amp_spec      = 10.^(dB_accel_power_spec/20);
    disp_amp_spec       = accel_amp_spec ./(w_spec.^2);
    lognoise  = interp1(log10(freq_spec), log10(disp_amp_spec), log10(freq));
    noise_ASD = 10.^lognoise;
    response_FR = 1196 .* 2*pi*freq;  % V/m 
    disp('warning, the T240 response is only valid between about 20 mHz and 40 Hz')

elseif strncmpi(sensor_name,'T240meas',5)
    %eyeballed from the 090310 data sets
    disp('The 0.01 to 0.03 Hz data from T240meas is just 2*T240spec, and not really measured')
    disp('    (the 0.03 Hz point is measured)')
    freq_data   = [.01, .03,    .1,    0.3,    1,     3,     10    ];
    noise_data  = [2*1.4e-7, 2e-8, 1.5e-9, 2e-10, 4e-11, 5e-12, 1.5e-12];
    
    lognoise    = interp1(log10(freq_data),log10(noise_data),log10(freq));
    noise_ASD   = 10.^lognoise;

    response_FR = 1196 .* 2*pi*freq;  % V/m 
    disp('warning, the T240 response is only valid between about 20 mHz and 40 Hz')

elseif strncmpi(sensor_name,'GS13meas',5)
    % taken directly from GS13_noise_measured_March2007 
    % the noise floor of the GS13 measured on the Tech Demo
    %  this is based on the data from the Tech Demo 3/15/2007
    %  using 2 witnesses with good ADCs, the new readouts, and a Q of ~ 5
    %  it is a little bigger than the expected noise based on the
    %  spec sheet for the LT1012 readout.
    %
    %  BTL, Oct 5, 2008
    %
    % see log entry http://ligo.phys.lsu.edu:8080/SEI/1288

    % data from graphclicks
    
    % at low freq, it looks to scale as x10 for every x2 in freq
    % is a power law of -3.32 (expect -3.5 from 1/f and sens scaling)
    % we will use the 3.5 to be conservative
    % 2e-8 * 10^3.5 = 6.3e-5
    % also assume a 1/f falloff at high freq
    full_data = [...
        0.01   2e-8 * 10^3.5
        0.101	2.138e-8;...
        0.201	1.958e-9;...
        0.400	1.892e-10;...
        0.792	1.732e-11;...
        0.994	7.262e-12;...
        1.258	3.887e-12;...
        1.655	2.449e-12;...
        2.588	1.462e-12;...
        3.954	9.725e-13;...
        8.202	4.423e-13;...
        10.710	3.280e-13;...
        24.796	1.137e-13;...
        46.087	5.610e-14;...
        91.208	2.053e-14
        1e3      2e-15;];
    
    data_freq   = full_data(:,1);
    data_noise  = full_data(:,2);
    lognoise   = interp1(log10(data_freq),log10(data_noise),log10(freq));
    noise_ASD  = 10.^lognoise;
    response_FR = NaN * ones(size(freq));
    disp(' warning, the freq resp for the GS13 is not yet defined')
    
elseif strncmpi(sensor_name,'GS13calc',6)
    freq_data   =   [.01    .1       .5      .8        1.2     3       10      100];
    noise_data  = [1.9e-5, 6.1e-9, 1.8e-11, 3.0e-12, 1.3e-12, 5.3e-13, 1.4e-13, 1.3e-14];
    
    lognoise    = interp1(log10(freq_data),log10(noise_data),log10(freq));
    noise_ASD   = 10.^lognoise;
    response_FR = NaN*ones(size(freq));
    disp(' warning, the freq resp for the GS13 is not yet defined')
    
elseif strncmpi(sensor_name,'CMG40T',6)
    % Guralp CMG-40T noise spec from the Manual
    % in units of dB((m^2/s^4)/Hz)
    % Rana - 12/12/2009
    freq_data   =   [0.01    0.1    0.3   1.0    5       10      30];
    acc_data  =   [-165,  -175,  -174, -172,  -166,   -155,  -150];
    acc_data = 10.^(acc_data/20);
    
    lognoise    = interp1(log10(freq_data), log10(acc_data) ,log10(freq),...
        'spline');
    acc_ASD   = 10.^lognoise;
    noise_ASD = acc_ASD ./ (2*pi*freq).^2;
    response_FR = NaN*ones(size(freq));
    disp(' warning, the freq resp for the CMG40 is not yet defined')    

elseif strncmpi(sensor_name, 'Wilcoxon731A', 12)
    % Wilcoxon 731A data from the datasheet
    % http://www.wilcoxon.com/vi_index.cfm?PD_ID=33
    % in units of ug/rHz
    % Rana - June 2015
    % f < 2 Hz noise is just a guess
    freq_data =  [0.02  0.2     2     10     100];
    acc_data  =  [0.05 0.05  0.03   0.01  0.004];
    acc_data = acc_data / 1e6 * 9.81;
    
    lognoise    = interp1(log10(freq_data), log10(acc_data) ,log10(freq),...
                          'spline');
    acc_ASD   = 10.^lognoise;
    noise_ASD = acc_ASD ./ (2*pi*freq).^2;
    response_FR = NaN*ones(size(freq));
    %disp(' warning, the freq resp for the 731A is not yet defined')  

elseif strncmpi(sensor_name, 'Wilcoxon731_207', 13)
    % Wilcoxon 731-207 data from the datasheet
    % http://www.wilcoxon.com/vi_index.cfm?PD_ID=33
    % in units of ug/rHz
    % Rana - June 2015
    % f < 2 Hz noise is just a guess
    freq_data =  [0.02  0.2     2     10     100];
    acc_data  =  [0.5   0.5  0.28   0.09    0.03];
    acc_data = acc_data / 1e6 * 9.81;
    
    lognoise    = interp1(log10(freq_data), log10(acc_data) ,log10(freq),...
                          'spline');
    acc_ASD   = 10.^lognoise;
    noise_ASD = acc_ASD ./ (2*pi*freq).^2;
    response_FR = NaN*ones(size(freq));
    %disp(' warning, the freq resp for the 731-207 is not yet defined')  
    
    
else
    disp('  error in SEI_sensor_noise  ');
    disp('defined sensors are: ''ADE_1mm'', ''ADE_p25mm'', ''GS13meas'', ''GS13calc''');
    disp('                     ''L4C'',  ''T240meas'',and ''T240spec''')
    disp('still need to define ''STS2'' and ''Kaman_1mm'' ')
    noise_ASD   = NaN * ones(size(freq));
    response_FR = NaN * ones(size(freq));
    
end


