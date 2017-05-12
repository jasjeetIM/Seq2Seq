function startup()
    curdir = fileparts(mfilename('fullpath'));
    scripts_path = fullfile(curdir, 'scripts'); 
    db_scripts_path = fullfile(curdir, 'db_scripts'); 
    addpath(genpath(scripts_path));
    addpath(genpath(db_scripts_path)); 
    mkdir_if_missing(fullfile(curdir, 'datasets'));
    caffe_path = fullfile(curdir,'caffe', 'matlab');
    if exist(caffe_path, 'dir') == 0
        error('matcaffe is missing from external/caffe/matlab; See README.md');
    end
    addpath(genpath(caffe_path));
    mkdir_if_missing(fullfile(curdir, 'output'));
    fprintf('seq2seq startup done\n');
end
