function prediction = predict_beam_search(net, src_sen, cont_sen, tar_sen, src_vocab, tar_vocab, inv_tar_vocab, beam_width, old_model, vocab_same)
% Translates a source sentence using beam search to prune the 
% search space. 
% Inputs: 
%	net: pre-trained network which will perform the prediction
%	src_sen: encoded source sentence to be translated
%	cont_sen: array of 0 or 1. 
%		  0 for beginning of new sentence and 1 otherwise
%	tar_sen: encoded sentence containing '<PAD>' in target embedding
%		 same as length of src_sen
%	src_vocab: dictionary of source words to indices for one-hot encoding
%	tar_vocab; dictionary of target words to indices for one-hot encoding
%	inv_tar_vocab: dictionary of target one-hot-encoding indices to target words
%	beam_width: width of frontier at every time step 

  %We cannot change the number of timesteps in Caffe
  % Therefore forward one timestep at a time
  [probs, keys] = get_best_words( net, src_sen, cont_sen, tar_sen, beam_width, old_model); 
 
  %Beam search setup
  frontier = {}; 
  decode = true; 
  max_len = length(cont_sen) + floor(length(cont_sen)/2);  
  src_pad = src_vocab('<PAD>'); 
  tar_pad = tar_vocab('<PAD>'); 
  tar_end = tar_vocab('<EOS>'); 

  %Make the top 'beam_width' predicted words the root nodes
  for i=1:beam_width
    root.word = keys(i); 
    root.parent = []; 
    root.log_prob = log(probs(i)); 
    root.sen_len = 1; 
    frontier{i} = root; 
  end 

  %Continue to search until all frontier nodes
  % have either ended with '<EOS>' or reached
  % the maximum allowable sentence length
  while decode
    for i=1:beam_width
      if frontier{i}.word == tar_end
        curr = frontier{i}; 
        frontier = frontier(2:end); 
        frontier{end+1} = curr; 
      else
         %Get a node 
         curr =  frontier{i};
         frontier = frontier(2:end); 
         %Get inputs for network
         [curr_src_sen, curr_cont_sen, curr_tar_sen] = get_curr_input(curr, src_sen, cont_sen, src_pad, tar_pad, vocab_same); 
         [probs, keys] = get_best_words(net, curr_src_sen, curr_cont_sen, curr_tar_sen, beam_width,old_model); 
         frontier = add_to_frontier(frontier, curr, probs, keys, max_len); 
      end
    end
    frontier = sort_and_prune_frontier(frontier, beam_width);
    show_current_frontier(frontier, beam_width, inv_tar_vocab);  
    decode = check_frontier_status(frontier, max_len, tar_end); 
  end
  
  prediction = get_best_prediction(frontier, inv_tar_vocab); 
end

function pred = get_best_prediction(frontier, inv_tar_vocab)
% Converts the best ranked sentence from one hot encoding
% to words in the target language. 

  curr = frontier{1}; 
  pred = [inv_tar_vocab(curr.word)];
  curr = curr.parent;  
  while ~isempty(curr)
    pred = [inv_tar_vocab(curr.word) ' ' pred];
    curr = curr.parent;  
  end
end

function decode = check_frontier_status(frontier, max_len, tar_end)
% Returns true if frontier contains any nodes
% that can be further expanded, and false otherwise.

  decode = true;
  counter = 0; 
  for i=1:length(frontier)
    if (frontier{i}.sen_len == max_len || frontier{i}.word == tar_end)
      counter = counter + 1; 
    end
  end
  if counter == length(frontier)
    decode = false; 
  end
end


function frontier = add_to_frontier(frontier, curr, probs, keys, max_len)
%Add the given encoded words to the frontier. 
   
    %Create objects and add to frontier if applicable
    for i=1:length(keys)
      node.word = keys(i); 
      node.parent = curr; 
      node.log_prob = log(probs(i)) + node.parent.log_prob;
      node.sen_len = node.parent.sen_len + 1; 
      if node.sen_len <= max_len
        frontier{end+1} = node; 
      end
    end

end

function frontier = sort_and_prune_frontier(frontier, beam_width)
% First sort the frontier in descending order by log probabilites
% of the partial sentences. Then prune it to the beam_width. 

  %Sort frontier 
  for i=1:length(frontier)
    max_node_idx = i; 
    for j=i+1:length(frontier)
      if frontier{max_node_idx}.log_prob < frontier{j}.log_prob
        max_node_idx = j; 
      end
    end
    if (max_node_idx ~= i)
      tmp = frontier{i}; 
      frontier{i} = frontier{max_node_idx}; 
      frontier{max_node_idx} = tmp;
    end
  end
  %Prune frontier
  frontier = frontier(1:beam_width);
end
 
function show_current_frontier(frontier, beam_width, inv_tar_vocab)
% Converts the frontier sentences from one hot encoding
% to words in the target language. 

  for i=1:length(frontier)
    curr = frontier{i}; 
    pred = [inv_tar_vocab(curr.word)];
    curr = curr.parent;  
    while ~isempty(curr)
      pred = [inv_tar_vocab(curr.word) ' ' pred];
      curr = curr.parent;  
    end
    fprintf('Partial Sentence %d : %s\n', i, pred); 
  end
end

function [curr_src_sen, curr_cont_sen, curr_tar_sen] = get_curr_input(curr, src_sen, cont_sen,  src_pad, tar_pad, vocab_same); 
% Get all the words in the partial sentence and 
% prepare network inputs to predict the next word 
  
  tar_sen = [curr.word]; 
  parent = curr.parent;
  while ~isempty(parent)
    tar_sen = [parent.word tar_sen]; 
    parent = parent.parent; 
  end
  if ~vocab_same
    curr_src_sen = [src_sen repmat([src_pad], 1, length(tar_sen) )];
  else
    curr_src_sen = [src_sen tar_sen]; 
  end
  curr_cont_sen = [cont_sen repmat([1], 1, length(tar_sen) )]; 
  curr_tar_sen = [repmat([tar_pad], 1, length(src_sen)) tar_sen];   
end


function [probs, keys] = get_best_words(net,src_sen, cont_sen, tar_sen, beam_width, old_model);
  fprintf('Now using src sen: '); 
  src_sen 
  fprintf('tar sen: '); 
  tar_sen 
  fprintf('\n'); 
  %Get output for the last word
  for i=1:length(cont_sen)
    cont = [cont_sen(i)];
    src = [src_sen(i)];
    if ~old_model
      tar = [tar_sen(i)];
      net.blobs('target_train_sentence').set_data(tar); 
    end
    net.blobs('cont_sentence').set_data(cont); 
    net.blobs('input_sentence').set_data( src ); 
    net.forward_prefilled();
    pred = net.blobs('probs').get_data(); 
%    save(sprintf('pred_%d.mat',i), 'pred'); 
  end
  %Get the best 'beam_width' predictions
  [probs, keys] = sort(pred, 'descend');
  fprintf('The top keys are ...\n'); 
  keys = keys(1:beam_width)
  fprintf('Top probs are ...\n'); 
  probs = probs(1:beam_width)
end
