function make_db()
  num_sentences =1200000; 
  hdf5_db = 'crawl_1.hdf5'; 
  src_file = fullfile(pwd, 'datasets/bitexts_selected/crawl_tk_lc_nm_1.en'); 
  tar_file = fullfile(pwd, 'datasets/bitexts_selected/crawl_tk_lc_nm_1.fr'); 
  src_vocab_file = fullfile(pwd, 'datasets/vocab_en.txt'); 
  tar_vocab_file = fullfile(pwd, 'datasets/vocab_fr.txt'); 
  %get_corpus(src_file, tar_file, src_vocab_file, tar_vocab_file, num_sentences); 
  [src_vocab, src_corpus, tar_vocab, tar_corpus, cont_corpus] = get_corpus(src_file, tar_file, src_vocab_file, tar_vocab_file, num_sentences); 
  [src_input, tar_input, cont_input] = make_hdf5(src_vocab, src_corpus, tar_vocab, tar_corpus, cont_corpus, hdf5_db); 

end
