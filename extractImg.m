% program extractImg.m
%
%  This program extracts the original image and classified structure so
%  that the relationship between the original image, final binary image,
%  and classified objects can be studied with artificial intelligence.

%  October 06, 2018
%  Last Modified: October 11, 2018

%% initial/extract option parameters

lnoise = true;      % true if binary image includes noise objects 
                    % note that some analyses may have deleted all noise
                    % objects

ldir = true;        % true if processing all .mat files in a given directory

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


%% read and write information into files

for k = 1:numel(cfilenames)
    
    % clear variables
    clear Iorig Ibin S  % make sure needed files do not exist in the work space
    Iorig = []; Ibin = []; S = [];  % empty variables so that we can use them to check
    
    % load data
    filename = cfilenames{k};
    disp(['working on file: ', filename]);
    
    load([cdir filename],'Iorig','S');
    
    if isempty(Iorig) || isempty(S)
        disp(['Iorig and/or S variables do not exist in ',[cdir filename]])
    else  % Iorig and S exist - get the binary image and save
        Ibin = false(size(Iorig));
        if lnoise  % if including noise objects (i.e., all objects)
            indx = 1:numel(S);
        else  % if ignoring noise objects
            indx = find([S.ID]>=0);
        end  % if lnoise
        rpix = cell2mat({S(indx).PixelIdxList}');
        Ibin(rpix) = true;
        
        % save information
        cfile = [filename(1:end-4) '.AI.mat'];
        save([cdirOut '/' cfile],'Iorig','Ibin','S');
    end   % if isempty(Iorig) || isempty(S)
    
end  % for k = 1:numel(cfilenames)

