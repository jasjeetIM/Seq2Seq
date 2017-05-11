function model_path = train_net(opts, varargin)
 %Main function for training the network
  %Cache directory to store model
  cache_dir = fullfile(pwd, 'output', 'cachedir');
  mkdir_if_missing(cache_dir); 
  model_path = fullfile(cache_dir, ['s2s_trained_' opts.db_name '.caffemodel']);
  if exist(model_path, 'file')
     fprintf('Found existing model, not need to train...\n'); 
     return;
  end
  
  fprintf('Generating network and solver...\n'); 
  
  % init log
  timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
  mkdir_if_missing(fullfile(cache_dir, 'log'));
  log_file = fullfile(cache_dir, 'log', ['train_', opts.db_name , '_',  timestamp, '.txt']);
  diary(log_file);
  
  % init caffe solver
  if (opts.load_from_proto)
    net = caffe.get_net(opts.net_model_prototxt, opts.phase); 
    solver = caffe.get_solver(opts.net_solver_prototxt); 
  else
    net = caffe.Net(opts.net_model_prototxt, opts.net_weights, opts.phase); 
    solver = caffe.get_solver(opts.net_solver_prototxt); 
  end

  %% training
  train_results = [];  
  iter_ = solver.iter();
  max_iter = opts.max_iter;
    
  while (iter_ < max_iter)
    % one iter SGD update
    solver.step(1000);
    if mod(iter_, opts.display_loss) == 0
      loss = solver.net.blobs('cross_entropy_loss').get_data(); 
      fprintf('Iteration: %d    Loss:%d\n', iter_, loss)
    end
    iter_ = solver.iter();
  end
  %Save final model
  save_model(solver, model_path); 

end

function save_model(solver, model_path)
  solver.net.save(model_path); 
  fprintf('Saved model at %s\n', model_path)
end


