function gpu_id = auto_select_gpu()
%Select the GPU with the most free mem

    gpuDevice([]);
    gpu_id = -1; 
    max_free = -1;
    for i = 1:gpuDeviceCount
        gpu = gpuDevice(i);
        free = gpu.FreeMemory();
        fprintf('GPU %d: has memory:  %d\n', i, free);
        if free > max_free && i ~= 2
            max_free = free; 
            gpu_id = i;
        end
    end
    fprintf('Selecting GPU %d\n', gpu_id);
    gpuDevice([]);
end
