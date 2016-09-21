function varargout = mkLoopCostDampQuad(fparams, plant_params, flag, set_logspace)
% given a vector of complex numbers, use them to compute the cost function
%
% flag can be 0 or 1 for plotting

debug = plant_params.debug;

% make the ZPK obj
Hlong = make_filter_from_params_DampQuad(fparams,plant_params,debug,set_logspace);

% make the closed loop plant
[damped_quad_model, Hlong_input_index] = make_closed_loop_DampQuad(plant_params,Hlong);

% sensor noise to test mass TF
prescaled_model = prescale(damped_quad_model, {2*pi*plant_params.Noise_req_freq(1),...
    2*pi*plant_params.Noise_req_freq(end)});
sensor_noise_to_testmass_TF = prescaled_model(plant_params.undamped_out.tst.disp.L,...
    Hlong_input_index);

% calculate test mass displacement noise from damping loop
test_mass_damping_noise_asd = plant_params.OSEMnoise *...
    abs(squeeze(freqresp(sensor_noise_to_testmass_TF,...
    2*pi*plant_params.Noise_req_freq)));

%% Loop Al-Gebra
% plant = prescale(ss(plant_params.plant),{2*pi*0.20, 2*pi*100});
% % Hdoop = ss(Hpum) +...
% %         ss(Hlong);
% Hdoop = ss(Hlong);
% G1 = Hlong  * plant_params.act.longitdinal;
% %G = zpk(G);
% 
% G1 = prescale(G1 * plant, {2*pi*0.2, 2*pi*100});
% % G = G1 + G2;
% G = G1;
% 
% ff      = plant_params.ff;
% d.plant = squeeze(freqresp(plant, 2*pi*ff));
% d.Hlong   = squeeze(freqresp(Hlong,   2*pi*ff));
% d.G1    = squeeze(freqresp(G1,     2*pi*ff));
% % d.G = d.G1 + d.G2;
% d.G = d.G1;
%% Noise?
% erms   = 0;
% conrms = 0;

% get the poles of the closed loop gain
% pc = pole(1 / (1 - G));
pc = pole(damped_quad_model);
real_pc = real(pc);

stability_cost = sum(find(real_pc > 0));

% iff closed loop poles are in left half place
%   compute the rest of the costs
if stability_cost == 0
    
    % high Q cost
%     pole_freq = mag(pc);
    pole_angle = angle(pc);
    damp_ratio = sin(pole_angle-pi/2);
    Q = 1./damp_ratio;
    max_Q = max(Q);
    Qscale = 4;     % try to keep the closed loop Q below this number
    Qcost = min((max_Q/Qscale)^2,...
                 plant_params.linear_crossover_factor*max_Q/Qscale);
    
    % Gain peaking cost 
    sensitivity_TF = prescale(1/(1+plant_params.plant*Hlong),{2*pi*0.1, 2*pi*10}); % Sensitivity TF
    sensitivity_dB = 20*log10(abs(freqresp(sensitivity_TF, plant_params.ff)));
    max_gain_peaking = max(sensitivity_dB);
    gain_peaking_scale = 2;
    gain_peaking_cost = min((max_gain_peaking/gain_peaking_scale)^2,...
        plant_params.linear_crossover_factor*max_gain_peaking/gain_peaking_scale);
    
    % noise cost
    % / (plant_params.Noise_req_Delta_Freq);
    noise_integral_over_band = plant_params.Noise_req_df *...
        sum(test_mass_damping_noise_asd ./ plant_params.Noise_req_asd);
    noise_cost = min(noise_integral_over_band^2,...
        plant_params.linear_crossover_factor*noise_integral_over_band);
    
    % total cost
%     sss = min(Qscale*max_Q,  (Qscale*max_Q)^2) +...    
%           min(noise_integral_over_band, noise_integral_over_band^2) + ...
%           gain_peaking_cost^2;
      
      sss = 10*Qcost + gain_peaking_cost + noise_cost;

else
    % if closed loop response is unstable just give a high cost and exit
    sss = 10000*sum(logsig(real_pc));
end

varargout{1} = sss;

%% plot if flag = 1
if flag
    
    sys.G = G;
    sys.G1 = G1;
    sys.G2 = G2;
    sys.Hdoop = Hdoop;
    sys.plant = plant;
    sys.d = d;
    varargout{2} = sys;
    
    disp(' ')
    str1 = ['  Error Signal RMS = ' num2str(erms,2) ' pm'];
    str2 = ['Control Signal RMS = ' num2str(conrms,2) ' mN'];
    if debug
        display(str1)
        display(str2)
    end
    % error signal spectrum
    figure(113)
    loglog(ff, errs/plant_params.FPgain*1e6,  'b',...
           ff, ermss/plant_params.FPgain*1e6, 'b--',...
           ff, con_pum, 'r',...
           ff, conrmss_p, 'r--',...
           ff, con_tm, 'k',...
           ff, conrmss_t, 'k--')
    grid on
    grid minor
    xlabel('Freq [Hz]')
    ylabel('um/\surdHz or N/\surdHz')
    %title('In-Loop Error Signal')
    legend('Error Signal', 'Error RMS',...
           'Control PUM', 'PUM RMS',...
           'Control TM', 'TM RMS',...
           'Location','SouthWest')
    ylim([1e-14 1e-4])
    text(11, 3.0e-6, str1)
    text(11, 0.5e-6, str2)


    orient landscape
    set(gcf,'Position', [600 0 800 500])
    set(gcf,'PaperPositionMode','auto')
    %print -dpng Figures/rmss.png
end





