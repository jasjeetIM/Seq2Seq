function [src_vocab, src_corpus, tar_vocab, tar_corpus, cont_corpus] = get_corpus(src_file, tar_file, src_vocab_file, tar_vocab_file, num_sentences)
%function get_corpus(src_file, tar_file, vocab_file, num_sentences)
%This function takes in a corpus of setences from the source and target
%files. It creates datastructures required to store the sentences
% in HDF5. 


    max_ts = 80;  
    fprintf('Importing corpus data\n');  
    fid_src = fopen(src_file,'r');
    fid_tar = fopen(tar_file, 'r'); 
    src_corpus = {}; 
    tar_corpus = {}; 
    cont_corpus = {}; 

    for i=1:num_sentences
      % Get src and tar sentences
      src_line = fgetl(fid_src); 
      tar_line = fgetl(fid_tar);  

      %Remove punctuation 
      src_idx = regexp(src_line, '[^.,!?;:()-]'); 
      tar_idx = regexp(tar_line, '[^.,!?;:()-]'); 
      src_line = src_line(src_idx);
      tar_line = tar_line(tar_idx);  

      src_sen = [strsplit(src_line)]; 
      tar_sen = [strsplit(tar_line)]; 

      %Remove trailing whitespace     
      if strcmp(src_sen(end), '')
        src_sen = src_sen(1:end-1); 
      end

      if strcmp(tar_sen(end), '')
        tar_sen = tar_sen(1:end-1); 
      end

      %Calculate truncate length
      %Truncate equally on both source and target
      if (length(src_sen) + length(tar_sen)) > (max_ts - 3) % -3 for 2 EOS tags and 1 BOS tag.

        trun = length(src_sen) + length(tar_sen) - max_ts + 3; 
        %fprintf('Length Src = %d, Length Tar = %d, Trun = %d\n', length(src_sen), length(tar_sen), trun); 
        %Src sentence is allowed a length of (max_ts/2 - 2) and tar_sentence is allowed (max_ts/2 -1); 
        src_trun = length(src_sen) - max_ts/2 + 2; 
        tar_trun = length(tar_sen) - max_ts/2 + 1; 
        if src_trun > 0
          src_sen = src_sen(1:end - src_trun); 
        end
        if tar_trun > 0
          tar_sen = tar_sen(1:end - tar_trun); 
        end
        %fprintf('New Length Src = %d, New Length Tar = %d\n', length(src_sen), length(tar_sen)); 
      end

      %Update sentences and reverse src sentence
      src_sen = fliplr(['<BOS>' src_sen '<EOS>']); 
      tar_sen = [tar_sen '<EOS>'];     
    
      %Calculate padding for src and tar
      src_padding = [strsplit(repmat(['<PAD> '], 1, max_ts - length(src_sen)))]; 
      tar_padding_pre = [strsplit(repmat(['<PAD> '], 1, length(src_sen)))]; 
      tar_padding_post = [strsplit(repmat(['<PAD> '], 1, max_ts - (length(src_sen) + length(tar_sen))))];

      %Remove the last extra space
      src_padding = src_padding(1:end-1); 
      tar_padding_pre = tar_padding_pre(1:end-1); 
      tar_padding_post = tar_padding_post(1:end-1); 
      %fprintf('%d, %d, %d\n', length(tar_padding_pre), length(tar_sen), length(tar_padding_post)); 
      %Update sentences  
      cont_sen = [0 repmat([1], 1, length(src_sen) + length(tar_sen) -1 ) repmat([0], 1, max_ts - (length(src_sen) + length(tar_sen)) )]; 
      tar_sen = [tar_padding_pre tar_sen tar_padding_post]; 
      src_sen = [src_sen src_padding]; 

      assert(length(src_sen) == max_ts, 'Src sentence has length %d', length(src_sen)); 
      assert(length(cont_sen) == max_ts, 'Cont sentence has length %d', length(cont_sen)); 
      assert(length(tar_sen) == max_ts, 'Target sentence has length %d', length(tar_sen)); 

      cont_corpus{i} = cont_sen;  
      tar_corpus{i} = tar_sen;
      src_corpus{i} = src_sen; 

      if mod(i,1000) == 0
        fprintf('Completed line %d\n', i); 
      end
    end
    fclose(fid_src);
    fclose(fid_tar); 
  
    src_vocab = get_vocab(src_vocab_file); 
    tar_vocab = get_vocab(tar_vocab_file); 
    fprintf('Done importing data\n');
end 
