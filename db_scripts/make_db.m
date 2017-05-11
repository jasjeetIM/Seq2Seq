function make_db()
  %Makes HDF5 database to be used by the caffe network
  num_sentences = 1; %Enter number of sentence in the bitext here 
  hdf5_db = 'name.hdf5'; %Enter name of .hdf5 file to be stored here 
  src_file = fullfile(pwd, 'datasets/source.en'); %Enter name and location of source bitext here
  tar_file = fullfile(pwd, 'datasets/target.fr'); %Enter name and location of target bitext here
  src_vocab_file = fullfile(pwd, 'datasets/vocab_en.txt'); 
  tar_vocab_file = fullfile(pwd, 'datasets/vocab_fr.txt'); 
  [src_vocab, src_corpus, tar_vocab, tar_corpus, cont_corpus] = get_corpus(src_file, tar_file, src_vocab_file, tar_vocab_file, num_sentences); 
  [src_input, tar_input, cont_input] = make_hdf5(src_vocab, src_corpus, tar_vocab, tar_corpus, cont_corpus, hdf5_db); 

end
