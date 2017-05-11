function run_net()
%This is the primary function that trains and tests 
% The caffe network. 

  clc;
  clear mex;
  %Run from home directory
  startup; 
  %Select GPU and activate Caffe
  gpu_id= auto_select_gpu;
  activate_caffe(gpu_id); 

  % Initialize the network
  opts.model_dir = './models/s2s/'; %Directory where prototxts are stored
  opts.net_model_prototxt = [opts.model_dir 'train_crawl1.prototxt']; %Train prototxt
  opts.net_solver_prototxt = [opts.model_dir 'solver_crawl1.prototxt']; %solver prototxt
  opts.test_net_proto = [opts.model_dir 'test.prototxt'];  %Testing prototxt
  opts.load_from_proto = false; % If false, network will load weights from path in opts.net_weights
  opts.net_weights = [opts.model_dir 'trained_models/ccb_ep7_ccb_crawl_iter_100000.caffemodel'];  
  opts.phase = 'train'; % Train or test 
  opts.db_name = 'seq2seq'; %Prefix for snapshots stored by caffe 
  opts.total_sen = 1; %Total number of sentences in the bitext (.hdf5) 
  opts.batch_size = 10; %Choose a factor of opts.total_sen (not required, train.prototxt can set this)
  opts.max_iter = floor(opts.total_sen/opts.batch_size);  %(not required, solver.prototxt can set this)
  opts.max_test_iter = 6003;%Testing on nm1213 dataset %(not required)
  opts.display_loss = 1000; %Print loss during training %(not required)
  %opts.save_iter = 10000; 
  opts.seed = 1207; 
  % Train Network
  model_path = train_net(opts); 
  opts.phase = 'test';
  %We will test it on training data to begin with
  test_net(opts, model_path); 
  caffe.reset_all();

