function [src_input, tar_input, cont_input] = make_hdf5(src_vmap, src_corpus, tar_vmap, tar_corpus, cont_corpus, db_file)

  %Currently limit stream size to 1
  N = 1; 
  max_ts = 80; %Set the maximum number of timesteps in the unrolled network 
  %D is the dimensionality of the data
  total_words_src = sum(cellfun(@length, src_corpus)); 
  total_words_cont = sum(cellfun(@length, cont_corpus)); 
  total_words_tar = sum(cellfun(@length, tar_corpus)); 
  num_sen_src = length(src_corpus); 
  num_sen_tar = length(tar_corpus);   

  fprintf('Total src words %d\n', total_words_src); 
  fprintf('Total tar words %d\n', total_words_tar); 
  fprintf('Total cont words %d\n', total_words_cont); 
  %Sentence pairs must be the same
  assert(num_sen_src == num_sen_tar, 'Error: #(source sentences) != #(target sentences)'); 

  %Length of src_sent, tar_sen, and cont_inp 
  % must be the same. 
  assert(total_words_src == total_words_cont, 'Error: src shape != cont shape'); 
  assert(total_words_cont == total_words_tar, 'Error: cont shape != tar shape'); 
 
  sz_src_vocab = size(src_vmap, 1); 
  sz_tar_vocab = size(tar_vmap, 1); 
  
  fprintf('Total size Src Vocab %d\n', sz_src_vocab); 
  fprintf('Total size Tar Vocab %d\n', sz_tar_vocab); 

  %T x N x D datastream. We use N = 1 for simplicity
  src_input = zeros(num_sen_src*max_ts, N, 1);  
  cont_input = ones(num_sen_src*max_ts, N); 
  tar_input = zeros(num_sen_tar*max_ts, N, 1); 
  tar_train_input = zeros(num_sen_tar*max_ts, N, 1); 
  fprintf('Preparing data for HDF5 storage...\n'); 
  for i=1:num_sen_src
    src_sen = src_corpus{i}; 
    tar_sen = tar_corpus{i}; 
    cont_sen = cont_corpus{i};

    %Fill source input 
    for j=1:length(src_sen)
      if isKey(src_vmap, src_sen{j})
        src_input((i-1)*max_ts + j, 1, 1) = src_vmap(src_sen{j}); 
      else
        src_input((i-1)*max_ts + j, 1,1) = src_vmap('<UNK>'); 
      end
    end

   %Fill target input
    for j=1:length(tar_sen)
      if isKey(tar_vmap, tar_sen{j})
        if strcmp(tar_sen{j},'<PAD>') 
          tar_input((i-1)*max_ts + j,1,1) = -1;
        else 
          tar_input((i-1)*max_ts + j,1,1) = tar_vmap(tar_sen{j});
        end
        tar_train_input((i-1)*max_ts + j, 1, 1) = tar_vmap(tar_sen{j}); 
      else
        tar_input((i-1)*max_ts + j,1,1) = tar_vmap('<UNK>');
        tar_train_input((i-1)*max_ts + j, 1, 1) = tar_vmap('<UNK>'); 
      end
    end 
    %Fill cont input
    cont_input((i-1)*max_ts + 1:(i-1)*max_ts + max_ts,1) = cont_sen; 
    if mod(i,1000) == 0
      fprintf('Completed preparing line %d\n', i)
    end
  end

  %Permute data into row major order
  src_input = permute(src_input, [3, 2, 1]); 
  tar_input = permute(tar_input, [3, 2, 1]); 
  tar_train_input = permute(tar_train_input, [3, 2, 1]); 
  cont_input = permute(cont_input, [2, 1]); 
  fprintf('Storing data to HDF5 database...\n'); 
  % files. If the files exist, delete them.
  db_path = fullfile(pwd, 'hdf5/data/', db_file)
  delete(db_path); 
  % First write the train data
  h5create(db_path,'/input_sentence',size(src_input),'Datatype','double');
  h5write(db_path,'/input_sentence', src_input);
  h5create(db_path,'/target_sentence',size(tar_input),'Datatype','double');
  h5write(db_path,'/target_sentence', tar_input);
  h5create(db_path,'/target_train_sentence',size(tar_train_input),'Datatype','double');
  h5write(db_path,'/target_train_sentence', tar_train_input);
  h5create(db_path,'/cont_sentence',size(cont_input),'Datatype','double');
  h5write(db_path,'/cont_sentence', cont_input);
end



