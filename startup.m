function startup()
    curdir = fileparts(mfilename('fullpath'));
    mkdir_if_missing(fullfile(curdir, 'datasets'));
    caffe_path = fullfile(curdir,'caffe', 'matlab');
    if exist(caffe_path, 'dir') == 0
        error('matcaffe is missing from external/caffe/matlab; See README.md');
    end
    addpath(genpath(caffe_path));
    mkdir_if_missing(fullfile(curdir, 'output'));
    mkdir_if_missing(fullfile(curdir, 'models'));
    fprintf('seq2seq startup done\n');
end
