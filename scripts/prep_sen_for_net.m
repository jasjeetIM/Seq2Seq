function [sentence, tar_sen] = prep_sen_for_net(sen, src_vocab, tar_vocab)
% Prepare an input sentence for the network

  sentence = zeros(1, length(sen));  
  tar_sen = zeros(1, length(sen)); 
  %Fill source input 
  for j=1:length(sen)
    if isKey(src_vocab, sen{j})
      sentence(j) = src_vocab(sen{j}); 
    else
      sentence(j) = src_vocab('<UNK>'); 
    end
    tar_sen(j) = tar_vocab('<PAD>'); 
  end

end



