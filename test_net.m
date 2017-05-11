function model_path = test_net(opts, model_path, varargin)
% Test network accuracy
  if ~exist(model_path, 'file')
     fprintf('Could not find model to test, quitting...\n'); 
     return;
  end
  
  fprintf('Loading network from file...\n');   

  % init caffe solver
  try
    net = caffe.get_net(opts.test_net_proto, model_path, opts.phase)
  catch
    fprintf('Error: Could not load model from weights file %s', model_path); 
    return; 
  end

  % init log
  timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
  mkdir_if_missing(fullfile(cache_dir, 'log'));
  log_file = fullfile(cache_dir, 'log', ['test_', timestamp, '.txt']);
  diary(log_file);
    
  %% Test
  test_results = [];  
  max_iter = opts.max_test_iter;
  iter_ = 1;  
  while (iter_ < max_iter)
    fprintf('Starting iteration %d\n', iter_); 
    % one forward
    net.forward_prefilled(); 
    acc = net.blobs('accuracy').get_data(); 
    fprintf('Accuracy = %d\n', acc);  
    iter_ = iter_ + 1; 
  end
end




