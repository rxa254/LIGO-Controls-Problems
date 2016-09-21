function filter = make_filter_from_params_DampQuad(fparams,plant_params,debug,set_logspace)
% Decode the Particle Swarm parameters and generate the filter
% 
% Brett Shapiro
% 24 May 2016

param_numbers = plant_params.numbers;

if set_logspace
    fparams = 10.^fparams; % convert logspace to regular space
end

% K_long  = fparams(end);    % pop out the last element as the overall gain
ugf_long  = fparams(end);    % pop out the last element as the ugf
fparams = fparams(1:end-1);

% parameter indices
real_poles_indices = 1:param_numbers.real_poles;
real_zeros_indices = (real_poles_indices(end)+1) : (real_poles_indices(end)+param_numbers.real_zeros);
complex_poles_indices = [];
if param_numbers.complex_poles > 0
    complex_poles_indices = (real_zeros_indices(end)+1) : (real_zeros_indices(end)+param_numbers.complex_poles);
end
complex_zeros_indices = [];
if param_numbers.complex_zeros > 0
    if param_numbers.complex_poles > 0
        complex_zeros_indices = (complex_poles_indices(end)+1) : (complex_poles_indices(end)+param_numbers.complex_zeros);
    else
        complex_zeros_indices = (real_zeros_indices(end)+1) : (real_zeros_indices(end)+param_numbers.complex_zeros);
    end
end

foo_realpoles = fparams(real_poles_indices); % for 1 loop
foo_realzeros = fparams( real_zeros_indices ); % for 1 loop
foo_complex_poles = fparams( complex_poles_indices );
foo_complex_zeros = fparams( complex_zeros_indices );
goo.long  = [];
real_poles = [];
real_zeros = [];

if ~isempty(foo_realpoles)
    real_poles = -transpose(foo_realpoles);
end
if ~isempty(foo_realzeros)
    real_zeros = -transpose(foo_realzeros);
end

% complex poles and zeros
if param_numbers.complex_poles+param_numbers.complex_zeros > 0
    for kk = 1:1
        
        % poles
        pz=[]; 
        for k = 1:2:param_numbers.complex_poles

            f0 = foo_complex_poles(k);
            Q  = foo_complex_poles(k+1);

            res  = sqrt(1 - 4 * Q.^2);
            mag  = f0 / (0.1 + 2 * Q);

            rs   = -[mag*(1 + res); mag*(1 - res)];
            pz   = [pz; rs];
        end
        
        % zeros
        zz=[];
        for k = 1:2:param_numbers.complex_zeros

            f0 = foo_complex_zeros(k);
            Q  = foo_complex_zeros(k+1);

            res  = sqrt(1 - 4 * Q.^2);
            mag  = f0 / (0.1 + 2 * Q);

            rs   = -[mag*(1 + res); mag*(1 - res)];
            zz   = [zz; rs];
        end
        
        if debug
            zz/2/pi
            pz/2/pi
        end
        
        if kk == 1 % first control filter
            goo.long = [zz pz];
    %     elseif kk == 2 % second control filter
    %         goo.long  = [zz pz];
        end
        
    end
end

% make the ZPK obj
% if isempty(goo.long)
%     filter  = zpk(real_zeros,  real_poles, K_long);
% else
%     filter  = zpk([real_zeros;goo.long(:,1)],  [real_poles;goo.long(:,2)], K_long);
% end
if isempty(goo.long)
    filter  = zpk(real_zeros,  real_poles, 1);
else
    filter  = zpk([real_zeros;goo.long(:,1)],  [real_poles;goo.long(:,2)], 1);
end
filter = filter / abs(freqresp(prescale(plant_params.plant*filter,{0.99*ugf_long,1.01*ugf_long}),ugf_long));