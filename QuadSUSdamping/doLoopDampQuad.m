% Global loop optimization
%
% This particular example makes a damping loop for the quad Longitudinal DOF (parallel to the laser beam). 
% 
% The damping loop is found uses the particle swarm optimziation algorithm
% on the damping filter parameters, which include the poles, zeros, and
% the unity gain frequency. For the cost function it considers the damping sensor noise injected
% inot the optical cavity, the amount the modes are damped, and how stable
% the loop is.

% clear
% clc
% close all

%% Initializations

debug = 0;
set_logspace = 1; % 1 means the code searches the parameters in log space, e.g. searched_parameter_value = log10(actual_parameter_value)
                  % 0 means the code searches the parameters in linear space

% cost function shape: the cost function switches from a quadratic shape for small cost values to a linear shape for larger cost values, to keep it from blowing up 
quadratic_linear_crossover_cost = 1000; % this is the cost value where the function switches from quadratic to linear
linear_crossover_factor = sqrt(quadratic_linear_crossover_cost); % sqrt is taken here, so it doesn't need to be done at every single evalution of the cost function
plant_params.linear_crossover_factor = linear_crossover_factor;

% number of particles in swarm
Nswarms = 5; % 10000

% setting the bounds for the variables to be searched over
fmin    = 2*pi*2e-3; % minumum frequency bound for damping filter poles and zeros
fmax    = 2*pi*300;  % maximum frequency bound for damping filter poles and zeros
Qmin = 1e-2; % minimum Quality factor bound value for complex ploes and zeros in the damping filter
Qmax = 1e2;  % maximum Quality factor bound value for complex ploes and zeros in the damping filter
% gain_bounds = [1e-10 1e10];
ugf_bounds = 2*pi*[3 9];

if set_logspace % if the parameter search is to be done in logspace
    fmin = log10(fmin);
    fmax = log10(fmax);
    Qmin = log10(Qmin);
    Qmax = log10(Qmax);
%     gain_bounds = log10([1e-10 1e10]);
    ugf_bounds = log10(ugf_bounds);
end

% estimated run time of code on 2012 MacBook Pro, 2.3 GHz Intel Core i7, 8 GB RAM, running Matlab R2015b
estimated_run_time = 25*Nswarms;
disp(['Estimated run time = ',...
      num2str(estimated_run_time),...
      ' sec = ' num2str(estimated_run_time/60),' min = ',...
       num2str(estimated_run_time/3600),' hrs'])
   
% set initial damping filter poles & zeros for the first particle in the swarm
polesz_real = 2*pi*[20 20 100]; % frequency [rad/s]
zerosz_real = 2*pi*[3e-3 250];  % frequency [rad/s]
polesz_complex = 2*pi*[1 2 2 8 20 1]; % frequency [rad/s], Q, etc
zerosz_complex = 2*pi*[1 5 2 3 10 1]; % frequency [rad/s], Q, etc

% list of parameters
Longz = [polesz_real zerosz_real polesz_complex zerosz_complex];
    plant_params.numbers.real_poles = length(polesz_real);
    plant_params.numbers.real_zeros = length(zerosz_real);
    plant_params.numbers.complex_poles = length(polesz_complex);
    plant_params.numbers.complex_zeros = length(zerosz_complex);
pz = [Longz];
% pz = [pz 1];  % last element is the initial guess for overall gain
pz = [pz 2*pi*3.4];  % last element is the initial guess ugf in rad/s
pzz = [pz];

% randomize the initial starting positions of the other particles in the swarm
for k = 2:Nswarms
%     pzz = [pzz; pz.*(0.5 + rand(size(pz)))];
    pzz = [pzz; pz.*(0.75 + 0.5*rand(size(pz)))];
end

% propogate the parameter bounds to all parameters in the list
LB=[]; UB=[];
for k = 1:(plant_params.numbers.real_poles+plant_params.numbers.real_zeros) % bounds for real poles and zeros
    LB = [LB fmin];
    UB = [UB fmax];
end
for k=1:0.5*(plant_params.numbers.complex_poles+plant_params.numbers.complex_zeros)  % bounds for complex poles and zeros
    LB = [LB fmin]; LB = [LB Qmin];   % Lower and
    UB = [UB fmax]; UB = [UB Qmax];   % Upper bounds for the swarm values
end

if set_logspace % if the parameter search was chosen to be logspace
    pzz = log10(pzz);
end
LB    = [LB ugf_bounds(1)];  % add ugf bounds
UB    = [UB  ugf_bounds(2)];

nvars = length(UB); % number of variables

% check initial positions are not out of bounds, if so adjust them to be in bounds
for kk = 2:Nswarms
    for ii = 1:nvars
        
        if pzz(kk,ii) < LB(ii)
            pzz(kk,ii) = 1.1*LB(ii);
        elseif pzz(kk,ii) > UB(ii)
            pzz(kk,ii) = 0.9*UB(ii);
        end
    
    end  
end

% Frequency vector
ff = logspace(-1, 2, 633);
minf = min(ff); maxf = max(ff);
plant_params.ff = ff;
ww = 2*pi*ff;
plant_params.ww = ww;

plant_params.debug = debug;


%% Load the quad pendulum model and sensor noise

% Load the quad pendulum model. Has units of meters / Newton
load simple_long_quadmodel % loads quad pendulum model as a state space variable called simple_long_quadmodel. For simplicity, the model only includes 1 degree of freedom (DOF) per stage, the DOF parallel to the laser beam

% OSEM (the sensors used in the damping) sensor noise
plant_params.OSEMnoise = 1e-10 / sqrt(2); % BOSEM noise above 10 Hz (m/rHz)

% quad model number of ins and outs
[quad_output_num,quad_input_num] = size(simple_long_quadmodel);

% Input and output indices of the quad model. The field L, refers to the
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
plant_params.plant = TopL2TopL; 

% quad technical noise requirement = 1/10 of longitudinal suspension thermal noise: T010007-v5.
sus_tech_noise_freq = linspace(10,20,20); % 10 to 20 Hz, the frequencies over which we will test the noise performance of the damping loop
sus_tech_noise_req = (1e-20)*(10^2)./(sus_tech_noise_freq.^2); % the magnitude of the noise performance requirement of the frequency band above.
plant_params.Noise_req_freq = (sus_tech_noise_freq); % frequency data for the noise requirement
plant_params.Noise_req_df = sus_tech_noise_freq(2) - sus_tech_noise_freq(1); % differential frequency, needed to integrate over the noise over the frequency band
plant_params.Noise_req_asd = transpose(sus_tech_noise_req); % ASD data for the noise requirement

                    
% plant parameters needed for the cost function calculation
plant_params.act.longitdinal = 1; % damping actuator gain
plant_params.undamped_ss = simple_long_quadmodel; % full state space system
plant_params.undamped_input_num = quad_input_num; % number of quad pendulum inputs
plant_params.undamped_output_num = quad_output_num;  % number of quad pendulum outputs
plant_params.undamped_out = out; % output indices
plant_params.undamped_in = in; % input indices


%% Particle SWarm Optimization

% set particle swarm optimization options
hybridopts = optimoptions('fmincon',...
                  'Display','iter',...
                  'MaxIter', 150,...
                  'TolFun', 1e-2,...
                  'MaxFunEvals', 1911);

options = optimoptions('particleswarm',...
               'SwarmSize', nvars*Nswarms,...
               'UseParallel', 1,...
               'InitialSwarm', pzz,...
               'MaxIter', 300,...
               'SelfAdjustment',   1.49,...
               'SocialAdjustment', 1.49,...
               'TolFun', 3e-2,...
               'Display', 'iter',...
               'HybridFcn',{@fmincon, hybridopts});


% RUNS the Particle Swarm ========
if debug
    xout = pz;  % just use the initial guess
else
    tic
    [xout, fval, exitflag] =...
        particleswarm(@(x) mkLoopCostDampQuad(x, plant_params, debug, set_logspace),...
                                        nvars, LB, UB, options);
    toc
end

% make the closed loop plant
Hlong = make_filter_from_params_DampQuad(xout, plant_params, debug, set_logspace); % damping filter
[damped_quad_model,Hlong_input_index] = make_closed_loop_DampQuad(plant_params, Hlong); % closed loop system

% save these temporary results: this allows running on a remote, Dropbox
% synced machine and then using the code block below to make the plots
save Data/SwarmResults.mat xout damped_quad_model plant_params Hlong ff minf maxf

%% list final costs
load Data/SwarmResults.mat

pc = pole(damped_quad_model);
real_pc = real(pc);

stability_cost = sum(find(real_pc > 0));

% high Q or settling time cost
damp_time = 1 ./ abs(real(pc));
max_damp_time = max(damp_time);
disp(['max damp time = ',num2str(max_damp_time)])
pole_angle = angle(pc);
damp_ratio = sin(pole_angle-pi/2);
Q = 0.5./damp_ratio;
[max_Q,max_Q_ind] = max(Q);
disp(['max Q = ',num2str(max_Q),'at ',num2str(abs(pc(max_Q_ind))/(2*pi)),' Hz'])
Qscale = 3;
Qcost = min((max_Q/Qscale)^2,plant_params.linear_crossover_factor*max_Q/Qscale);

% Gain peaking cost 
sensitivity_TF     = prescale(1/(1+plant_params.plant*Hlong),{2*pi*minf, 2*pi*maxf}); % Sensitivity TF
sensitivity_dB     = 20*log10(abs(squeeze(freqresp(sensitivity_TF, plant_params.ww))));
[max_gain_peaking,maxgainpeaking_freqind]   = max(sensitivity_dB);
disp(['max gain peaking = ',num2str(max_gain_peaking)])
gain_peaking_scale = 2;
gain_peaking_cost  = min((max_gain_peaking/gain_peaking_scale)^2,...
                          plant_params.linear_crossover_factor*max_gain_peaking/gain_peaking_scale);

% sensor noise to test mass TF
prescaled_model = prescale(damped_quad_model, {2*pi*plant_params.Noise_req_freq(1),...
    2*pi*plant_params.Noise_req_freq(end)});
sensor_noise_to_testmass_TF = prescaled_model(plant_params.undamped_out.tst.disp.L,...
    Hlong_input_index);
                      
% noise cost
test_mass_damping_noise_asd = plant_params.OSEMnoise *...
    abs(squeeze(freqresp(sensor_noise_to_testmass_TF,...
    2*pi*plant_params.Noise_req_freq)));
% noise_integral_over_band = plant_params.Noise_req_df * sum(test_mass_damping_noise_asd ./...
%     plant_params.Noise_req_asd)
% noise_cost = min(noise_integral_over_band^2,plant_params.linear_crossover_factor *...
%     noise_integral_over_band);
[max_noise_ratio,maxfreqind] = max(test_mass_damping_noise_asd ./ plant_params.Noise_req_asd);
disp(['max noise ratio = ',num2str(max_noise_ratio)])
noise_cost = min(max_noise_ratio^2,plant_params.linear_crossover_factor*max_noise_ratio);
disp(['Noise cost = ',num2str(noise_cost)])

% total cost
sss = mkLoopCostDampQuad(xout, plant_params, 0, 1);
disp(['Total cost = ',num2str(sss)])
      
[z,p,k] = zpkdata(Hlong,'v');
z = sort(-z/2/pi)
p = sort(-p/2/pi)

%% Make some plots

tnowstr = datestr(now, 'yymmdd_HHMM');

% noise plot
freq = logspace(log10(minf), log10(maxf), 1000);
% calculate test mass displacement noise from damping loop
test_mass_damping_noise_asd_hi_res = plant_params.OSEMnoise *...
    abs(squeeze(freqresp(sensor_noise_to_testmass_TF,2*pi*freq)));

figure(4042)
    loglog(freq, test_mass_damping_noise_asd_hi_res,...
           plant_params.Noise_req_freq, plant_params.Noise_req_asd,...
           plant_params.Noise_req_freq, test_mass_damping_noise_asd,'*b',...
           'LineWidth',3)
    grid on
    set(gca,'FontSize',22)
    set(gca,'YTick',logspace(-21,-9,13))
    axis([1 30 1e-21 .21e-9])
    title(['BOSEM noise contribution to test mass displacement; noise ',num2str(round(max_noise_ratio,1)),' x requirement at ',num2str(round(plant_params.Noise_req_freq(maxfreqind),1)),' Hz'],'FontSize',17)
    ylabel('Displacement (m/\surdHz)')
    xlabel('Frequency (Hz)')
    legend('Test mass motion','Technical noise requirement - T010007-v5','Sampled frequencies for noise cost')
    FillPage('w')
    saveas(gcf,['../Data/Noise_',tnowstr,'.pdf'])

% CLtopLtoTopL = plant_params.plant/(1 + plant_params.plant*Hlong);
figure(33003)
[impulse_data,impulse_time] = ...
    impulse(damped_quad_model(out.tst.disp.L, in.top.drive.L), 100);
impulse_data = impulse_data / max(abs(impulse_data));
plot(impulse_time,impulse_data,...
    [impulse_time(1) impulse_time(end)], exp(-1)*[1 1],'--k',...
    [impulse_time(1) impulse_time(end)],-exp(-1)*[1 1],'--k'...
    )
legend('Top L to Test L','1/e decay level')
grid on
    set(gca,'FontSize',22,'XTick',0:10:1000)
    title(['Impulse response; longest 1/e settling time of all poles = ',num2str(round(max_damp_time,1)),' s; max Q = ',num2str(round(max_Q,1)),' at ',num2str(round(abs(pc(max_Q_ind))/(2*pi),1)),' Hz'],'FontSize',13)
    ylabel('Displacement (arbitrary units)')
    xlabel('Time (seconds)')
    % set the line thickness
    ll = findobj(gcf,'type','line');
    set(ll, 'linewidth', 2);
    FillPage('w')
    saveas(gcf,['../Data/Ringdown_',tnowstr,'.pdf'])
    
bopp = bodeoptions;
bopp.FreqUnits = 'Hz';
bopp.Grid = 'On';
bopp.PhaseWrapping = 'On';

figure(111)
bodemag(sensitivity_TF,...
     prescale(plant_params.plant,{2*pi*0.1, 2*pi*11})*Hlong,...
     Hlong,...
     ff, bopp)
  title(['Quad Sus Damping Design; max gain peaking = ',num2str(round(max_gain_peaking,1)),' dB at ',num2str(round(plant_params.ff(maxgainpeaking_freqind),1)),' Hz'],'FontSize',17)
  xlabel('Frequency','FontSize',22)
  legend('Sensitivity', 'Open Loop Gain', 'Damping Filter')
     % set the line thickness
  ll = findobj(gcf,'type','line');
  ylim([-90 90])
  set(ll,'linewidth',3);
  set(gca,'FontSize',22)
  FillPage('w')
  saveas(gcf,['../Data/Sensitivity_',tnowstr,'.pdf'])

% figure,bode(prescale(damped_quad_model(out.top.disp.L,in.gnd.disp.L),{2*pi*0.01,2*pi*1000}))
%     title('Bode plot - Closed Loop Gnd L to test L')
% % set the line thickness
%     ll = findobj(gcf,'type','line');
%     set(ll,'linewidth',3);
%     FillPage('w')

figure(222),bode(prescale(plant_params.plant*Hlong,{2*pi*0.01,2*pi*1000}))
    title('Bode plot - Damping Loop Gain Loop')
% set the line thickness
    ll = findobj(gcf,'type','line');
    set(ll,'linewidth',3);
    FillPage('w')
    saveas(gcf,['../Data/LoopGain_',tnowstr,'.pdf'])



%% save data
save(['Data/QuadDampResults_' tnowstr],'Hlong', 'plant_params', 'xout', 'damped_quad_model', 'Hlong_input_index', 'out', 'ff', 'minf', 'maxf', 'ww', 'Nswarms', 'quadratic_linear_crossover_cost')

