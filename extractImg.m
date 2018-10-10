% program extractImg.m
%
%  This program extracts the original image and classified structure so
%  that the relationship between the original image, final binary image,
%  and classified objects can be studied with artificial intelligence.

%  October 06, 2018
%  Last Modified: October 09, 2018

%% extract option parameters

lnoise = true;      % true if binary image includes noise objects 
                    % note that some analyses may have deleted all noise
                    % objects
                    
%% read file

[filename cdir] = uigetfile('*.mat');

if isfloat(filename)  % file not chosen
    return
end   % if isfloat(filename)

%% load needed information

clear Iorig S  % make sure needed files do not exist in the work space
Iorig = []; S = [];  % empty variables so that we can use them to check

load([cdir filename],'Iorig','S');

if isempty(Iorig) || isempty(S)
    disp(['Iorig and/or S variables do not exist in ',[cdir filename]])
    return
end   % if isempty(Iorig) || isempty(S)

Ibin = false(size(Iorig));
if lnoise  % if including noise objects (i.e., all objects)
    indx = 1:numel(S);
else  % if ignoring noise objects
    indx = find([S.ID]>=0);
end  % if lnoise
rpix = cell2mat({S(indx).PixelIdxList}');
Ibin(rpix) = true;

%% save information

disp('Choose directory for saving results');
cdirOut = uigetdir('.','Directory for Data to be Saved');

if isfloat(cdirOut)  % output directory has not been chosen
    cdirOut = '';  % empty directory name, i.e., save into current working directory
end  % if isfloat(cdirOut)

cfile = [filename(1:end-4) '.AI.mat'];
save([cdirOut '/' cfile],'Iorig','Ibin','S');