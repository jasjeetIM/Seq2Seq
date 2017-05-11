function made = mkdir_if_missing(path)
%make directory if does not exist
made = false;
if exist(path, 'dir') == 0
  mkdir(path);
  made = true;
end
