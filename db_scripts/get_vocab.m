function vocab = get_vocab(vocab_file)
% Read in vocabulary words from a text
% file into a containers.Map object. 
% Word on line i will be assigned value i. 
  fid = fopen(vocab_file, 'r'); 
  vocab = containers.Map; 
  counter = 1;  
  while ~feof(fid)
    word = fgetl(fid); 
    vocab(word) = counter; 
    if mod(counter, 10000) == 0
      fprintf('Completed vocab line %d\n', counter); 
    end
    counter = counter + 1; 
  end
  
  fprintf('Vocab Size = %d\n', length(vocab)); 
  fclose(fid);
end
