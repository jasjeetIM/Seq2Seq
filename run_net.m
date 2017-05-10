function run_net()
  clc;
  clear mex;
  %Run from home directory
  startup; 
  %Select GPU and activate Caffe
  gpu_id= auto_select_gpu;
  activate_caffe(gpu_id); 

  % Initialize the network
  opts.model_dir = './models/s2s/';
  opts.net_model_prototxt = [opts.model_dir 'train_crawl1.prototxt'];
  opts.net_solver_prototxt = [opts.model_dir 'solver_crawl1.prototxt']; 
  opts.load_from_proto = false;
  opts.net_weights = [opts.model_dir 'trained_models/ccb2_1_ep7_singles_ccb2_2_iter_110000.caffemodel'];  
  opts.phase = 'train'; 
  opts.db_name = 'ccb_ep_singles_ccb_crawl'; 
  opts.total_sen = 1200000; 
  opts.batch_size = 10; %Choose a factor of opts.total_sen
  opts.max_iter = floor(opts.total_sen/opts.batch_size); 
  opts.max_test_iter = 6003;%Testing on nm1213 dataset 
  opts.display_loss = 1000; %Print loss during training 
  %opts.save_iter = 10000; 
  opts.seed = 1207; 
  % Train Network
  model_path = train_net(opts); 
  opts.phase = 'test'; 
  %We will test it on training data to begin with
  test_net(opts, model_path); 
  caffe.reset_all();

