function [damped_quad_model,Hlong_input_index,Hlong_output_index] = make_closed_loop_DampQuad(plant_params,Hlong)

appended_model = append(plant_params.undamped_ss,Hlong);

Hlong_input_index = plant_params.undamped_input_num + 1;
Hlong_output_index = plant_params.undamped_output_num + 1;


connection_matrix = [

% input indices              <-         % output indices     
plant_params.undamped_in.top.drive.L    Hlong_output_index             % controller force to top mass drive

Hlong_input_index                  -plant_params.undamped_out.top.disp.L % top mass displacement to controller input

];

inputs = 1:Hlong_input_index;
outputs = 1:Hlong_output_index;

damped_quad_model = connect(appended_model,connection_matrix,inputs,outputs); % damped model