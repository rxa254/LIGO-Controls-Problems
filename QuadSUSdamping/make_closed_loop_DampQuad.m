function [damped_quad_model,...
    damping_filter_input_index,...
    damping_filter_output_index] =...
    make_closed_loop_DampQuad(plant_params, damping_filter)

% append the quad model and the damping loop into a single state space system 
% (note, each part is still independent after the append command)
appended_model = append(plant_params.undamped_ss, damping_filter); 

% input index for the damping filter damping_filter
damping_filter_input_index = plant_params.undamped_input_num + 1; 

% Output index for the dmaping filter damping_filter
damping_filter_output_index = plant_params.undamped_output_num + 1; 

% this matrix defines which inputs and outputs are connected to close the damping loop
connection_matrix = [
% input indices              <-         % output indices     
% controller force to top mass drive
plant_params.undamped_in.top.drive.L    damping_filter_output_index

% top mass displacement to controller input
damping_filter_input_index             -plant_params.undamped_out.top.disp.L
];

% list of all the  input indices from the appended state space that 
% we want access to in the closed loop state space
inputs  = 1:damping_filter_input_index;

% list of all the output indices from the appended state space that 
% we want access to in the closed loop state space
outputs = 1:damping_filter_output_index;

% damped model
damped_quad_model = connect(appended_model, connection_matrix, inputs, outputs);