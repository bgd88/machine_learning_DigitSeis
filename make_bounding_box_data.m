% make_bounding_box_data.m
% This script will extract and store data from DigitSeis analysis files for
% training of a neural net to classify areas within bounding boxes as
% timemarks, traces, noise etc...

% Data is output in a .csv with each line corresponding to a bounding box
% and having format:
% ID, X_start, Y_start, Length, Width, HSV_1, HSV_2, ... , HSV_end
% ID is the type (2 for timemarks, 1 for trace, 0 for everthing else)
% X_start, Y_start, Length, Width are all normalized coordinates
% HSV values are row vectors


% Written By: Thomas Lee
% Created On: 22 Oct 2018
% Last Modified: 24 Oct 2018
%   - added in generation of .jpgs in a new directory, added single mat 
%     option, and added diagnotic option

%% init
clear all;

% diagnostics
diag=0;

% single mat or directory read in
singlemat=true;

% csv mode
csv_on=false;

% jpg mode
jpg_on=true;

if singlemat
    %% pick .mat file
    [cfile,cpath]=uigetfile;
    disp([cpath,'/',cfile]);
    cont=dir(cpath);
    bad_idx=[];
    for ii=1:length(cont)
        if ~strcmpi(cont(ii).name,cfile)
            bad_idx=[bad_idx,ii];
        end
    end
    cont(bad_idx)=[];
else
    %% pick directory with .mat files
    cpath=uigetdir; %get directory
    disp(cpath);
    cont=dir(cpath); %list contents
end

%% create .csv file
if csv_on
    csvname=input('Name for .csv file (no ext): ','s');
    fid=fopen([cpath,'/',csvname,'.csv'],'w');
end

%% create .jpg dir
if jpg_on
    dirname=input('Name for .jpg directory: ','s');
    mkdir([cpath,'/',dirname]);
end

%% loop over mat files
for i=1:length(cont) % file level loop
    if length(cont(i).name)>9% skip over . .. .DS_store etc
        if strcmpi(cont(i).name(1:9),'DigitSeis') % check that is it DigitSeis .mat
            disp(['Started ',cont(i).name,' at ',datestr(clock)]);
            %% get content of mat
            load([cpath,'/',cont(i).name])
            
            hw=waitbar(0,cont(i).name);
            %% loop over bounding boxes and write each line
            for j=1:numel(S) % structure level loop
                % check that fields exist
                if ~isempty(S(j).BoundingBox) & ~isempty(S(j).PixelIdxList)...
                        & ~isempty(S(j).ID) & ~isempty(S(j).TraceNum)
                    % get obj type
                    switch S(j).ID
                        case {1 3} % timemark case
                            tmp_line=[2];
                        case {0 2} % trace case
                            tmp_line=[1];
                        otherwise % everything else
                            tmp_line=[0];
                    end
                    % add bounding box
                    tmp_box=[round(S(j).BoundingBox(1)),round(S(j).BoundingBox(2)),...
                        round(S(j).BoundingBox(3)),round(S(j).BoundingBox(4))];
                    [r,c]=size(Iorig);
                    if (tmp_box(1)+tmp_box(3))>c %check for over exceeding indices
                        tmp_box(3)=c-tmp_box(1);
                    end
                    if (tmp_box(2)+tmp_box(4))>r
                        tmp_box(4)=r-tmp_box(2);
                    end
                    tmp_line=[tmp_line,tmp_box];
                    
                    % add HSV values
                    tmp_hsv=Iorig(tmp_box(2):tmp_box(2)+tmp_box(4), tmp_box(1):tmp_box(1)+tmp_box(3));
                    tmp_line=[tmp_line,reshape(tmp_hsv,[1,numel(tmp_hsv)])];
                    
                    % plot image in diagnostic mode
                    if diag
                        imshow(tmp_hsv);
                        pause(1)
                        close
                    end
                    
                    % save image
                    if jpg_on
                        tmp_fname=[num2str(tmp_line(1)),'_',num2str(tmp_box(1)),'_',...
                            num2str(tmp_box(2)),'_',num2str(tmp_box(3)),'_',...
                            num2str(tmp_box(4))];
                        imwrite(tmp_hsv,[cpath,'/',dirname,'/',tmp_fname],'jpg');
                        disp(['Wrote ',tmp_fname,'.jpg to ',cpath,'/',dirname]);
                    end
                    
                    % write to csv
                    if csv_on
                        dlmwrite([cpath,'/',csvname,'.csv'],tmp_line,'-append');
                        clear tmp_line
                    end
                    
                    % update waitbar
                    if mod(j,100)
                        waitbar(j/numel(S),hw)
                    end
                end
            end
            clear S
            clear Iorig
            close(hw);
            disp(['Done with ',cont(i).name,' at ',datestr(clock)]);
        end
    end
end

if csv_on
    fclose(fid);
end
















