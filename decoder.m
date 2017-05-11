function decoder(opts)
%Decode sentences in a src file one at a time
% and write the translations into the target file
% Utilize beam search to choose the best possible 
% translation

  beam_width = opts.beam_width; 
  src_vocab = get_vocab(opts.src_vocab_file); 
  tar_vocab = get_vocab(opts.tar_vocab_file); 
  inv_tar_vocab = containers.Map(tar_vocab.values, tar_vocab.keys); 
  %net = caffe.get_net(opts.net_model_file, opts.net_weights, opts.phase)
  try
    net = caffe.get_net(opts.net_model_file, opts.net_weights, opts.phase)
  catch
    fprintf('Error: Could not load model from weights file %s\n', opts.net_weights); 
    return; 
  end
  fid = fopen(opts.in_file, 'r'); 
  fod = fopen(opts.out_file, 'wt'); 
  counter = 1; 
  while ~feof(fid)
    fprintf('Translating Line : %d\n', counter); 
    sentence = fgetl(fid); 
    
    %Remove punctuation 
    idx = regexp(sentence, '[^.,!?;:()-]'); 
    sentence = sentence(idx);
    sentence = [strsplit(sentence)]; 
  
    %Remove trailing whitespace     
    if strcmp(sentence(end), '')
      sentence = sentence(1:end-1); 
    end

    sentence = fliplr(['<BOS>' sentence '<EOS>']); 
    cont_sen = [0 repmat([1], 1, length(sentence) -1 )]; 
    [sentence, tar_sen] = prep_sen_for_net(sentence, src_vocab, tar_vocab);  

    prediction = predict_beam_search(net, sentence, cont_sen, tar_sen, src_vocab, tar_vocab,inv_tar_vocab, beam_width, opts.model_old, opts.vocab_same);  
    fprintf(fod, '%s\n', prediction); 
    counter = counter + 1; 
  end
  fclose(fid); 
  fclose(fod); 
end
