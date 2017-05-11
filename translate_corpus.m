function translate_corpus()
% Sets up translation options for a source
% file and calls decoder(opts) to translate
% the sentences in the source file to sentences
% in the target language.  
  clc;
  clear mex;
  %Run from home directory
  startup; 
  %Select GPU and activate Caffe
  gpu_id= auto_select_gpu;
  activate_caffe(gpu_id); 

  % Initialize opts for the decoding stage
  opts.model_dir = './models/s2s/';
  opts.net_model_file = [opts.model_dir 'test.prototxt'];
  opts.src_vocab_file = './datasets/vocab_en.txt'; 
  opts.tar_vocab_file = './datasets/vocab_fr.txt'; 
  opts.net_weights = './models/s2s/trained_models/s2s.caffemodel'; 
  opts.phase = 'test'; 
  opts.db_name = 's2s';
  opts.model_old = false; 
  opts.vocab_same = false; 
  opts.in_file = './datasets/dev/ntst14.en'; 
  opts.out_file = ['./datasets/dev/ntst14_' opts.db_name '.fr'];
  opts.beam_width = 4; 
  decoder(opts);  
%  caffe.reset_all();

