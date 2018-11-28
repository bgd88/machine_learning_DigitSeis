%  program run_makeTrainingData.m

% This program sets up files to be analyzed and runs makeTrainData script.

%  October 12, 2018
%  Last Modified:  October 12, 2018

%% default parameters

ldir = true;        % true if processing all .mat files in a given directory

nr = 1000;          % number of row pixels for training data
nc = 2000;          % number of columns for training data
dnr = 500;          % increment in row for stepping
dnc = 1000;          % increment in column for stepping

%% get filenames to be read in and output directory name

cfileList = {};
cfilenames = {};

if ldir  % get directory name and then all the .mat filenames within the directory
    disp('Choose directory where all .mat files to be processed can be found');
    cdir = uigetdir('.','Directory for all .mat files to be read in');
    
    if isfloat(cdir)  % input directory has not been chosen
        return  % end program
    end  % if isfloat(cdir)
    
    % get names of all .mat files within the directory
    cfileList = dir(cdir);
    if isempty(cfileList)  % if something went wrong
        return  % end program
    end  % if isempty(cfileList)
    
    % check to make sure there are a finite number of .mat files
    cfilenames = {cfileList.name};
    indx = find(~cellfun('isempty',regexp(cfilenames,'.mat\>')));
    if isempty(indx)  % no .mat files
        return  % end program
    end  % if isempty(indx)
    
    cfilenames = cfilenames(indx);  % only keep .mat filenames
else  % get a single .mat filename
    [filename cdir] = uigetfile('*.mat');
    
    if isfloat(filename)  % file not chosen
        return
    end   % if isfloat(filename)
    
    cfilenames{1} = filename;
end  % if ldir

if ~strcmpi(cdir(end),'/')  % make sure cdir ends with '/'
    cdir = [cdir '/'];
end  % if ~strcmpi(cdir(end),'/')

% save location information
disp('Choose directory for saving results');
cdirOut = uigetdir('.','Directory for Data to be Saved');

if isfloat(cdirOut)  % output directory has not been chosen
    cdirOut = '';  % empty directory name, i.e., save into current working directory
end  % if isfloat(cdirOut)

%% call makeTrainingData

for k = 1:numel(cfilenames)
    
    clear Iorig S  % clear memory
    
    filename = cfilenames{k};
    disp(['working on file: ', filename]);
    
    load([cdir filename],'Iorig','S');
    
    if isempty(Iorig) || isempty(S)  % data not available
        disp(['Iorig and/or S variables do not exist in ',[cdir filename]])
    else  % data exist
        makeTrainingData(Iorig,S,nr,nc,dnr,dnc,1,cdirOut,filename);
    end  % if isempty(Iorig) || isempty(S)
    
end  % for k = 1:numel(cfilenames)
