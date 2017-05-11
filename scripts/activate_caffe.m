function activate_caffe(gpu_id)
% Initialize caffe with the selected gpu
  % set gpu in matlab
  gpuDevice(gpu_id);
  %Initialize Caffe
  cur_dir = pwd;
  caffe_dir = fullfile(pwd, 'caffe', 'matlab', '+caffe');    
  addpath(genpath(caffe_dir));
  cd(caffe_dir);
  caffe.reset_all(); 
  caffe.set_mode_gpu(); 
  caffe.set_device(gpu_id-1);
  cd(cur_dir);
end
