function makeTrainingData(Iorig,S,nr,nc,dnr,dnc,iopt,cdirOut,cfileOut)
%  This function takes an image and divides it into smaller sections with
%  information about pixel (x,y) values (normalized with respect to image
%  size), intensity value, and type ID and saves into a .mat file.
%
%  Input: Iorig = original image file
%         S = structure containing classified object information
%         nr = number of rows/training image
%         nc = number of columns/training image
%         dnr = number of rows for stepping
%         dnc = number of columns for stepping
%         iopt = option for treatment at the end
%              = 0 - ignore whatever is left at the end
%              = 1 - calculate from the end and create another training set
%         cdirOut = output directory name
%         cfileOut = output filename (with numbers added for patches of
%                    images)
%  Output: 
%
%  Note: variables that will be stored in the generated training files are
%        x = [xs xe] = normalized starting and ending x position of the patch
%        y = [ys ye] = normazlied starting and ending y position of the
%                      patch
%        imInt = intensity value of the pixel
%        pixID = type associated with this pixel
%              = -1 - noise or background
%              = 0 - traces
%              = 1 - time marks
%              = 2 - overlapping
%        nr = number of rows
%        nc = number of columns
%  imInt and pixId should have number of elements corresponding to nr*nc

%  October 12, 2018
%  Last Modified:  October 12, 2018

%% initial prep

[ImRow ImCol] = size(Iorig);  % size of Iorig matrix

if ~strcmpi(cdirOut(end),'/')  % make sure cdirOut ends with /
    cdirOut = [cdirOut '/'];
end  % if ~strcmpi(cdirOut(end),'/')

% get filename before .mat
if strcmpi(cfileOut(end-3:end),'.mat')  % if ending with .mat, remove
    cfileOut = cfileOut(1:end-4);
end  % if strcmpi(cfileOut(end-3:end),'.mat')

% remove all S objects that belong to noise
indx = find([S.ID]<0);
S(indx) = [];

% get bounding box of each valid objects
sbox = cell2mat({S.BoundingBox}');


%% set up division parameters

% divide rows
itmp = 1:dnr:(ImRow-nr+1);
ndivRs = itmp(1:end-1); ndivRe = ndivRs+nr-1;
if iopt == 1  % come back from the end and add another patch
    ndivRs = unique([ndivRs ImRow-nr+1]);
    ndivRe = unique([ndivRe ImRow]);
end  % if iopt == 1

% divide columns
itmp = 1:dnc:(ImCol-nc+1);
ndivCs = itmp(1:end-1); ndivCe = ndivCs+nc-1;
if iopt == 1  % come back from the end and add another patch
    ndivCs = unique([ndivCs ImCol-nc+1]);
    ndivCe = unique([ndivCe ImCol]);
end  % if iopt == 1

% ndivRs = [1]; ndivRe = ndivRs + nr - 1;
% ndivCs = [1 2001]; ndivCe = ndivCs + nc - 1;

%% cut image up into pieces and process

hwait = waitbar(0,cfileOut);
ntot = numel(ndivRs)*numel(ndivCs);

ncnt = 1;  % initial value for counter used for numbering saved files
for k = 1:numel(ndivRs)  % divisin in the row direction
    irs = ndivRs(k); ire = ndivRe(k);  % row range
    
    % reset variable
    y = [irs ire]/ImRow;  % normalized y range for this patch

    for m = 1:numel(ndivCs)  % divising in the column direction
        ics = ndivCs(m); ice = ndivCe(m);  % column range
        
        % reset variables
        x = [ics ice]/ImCol; % normalized x range for this patch
        imInt = []; 
        pixID = -1*ones([nr*nc 1]); % set everything to noise/background
        
        %disp([' image patch: [', num2str(y(1)), ' ', num2str(y(2)), ';', num2str(x(1)), ' ', num2str(x(2)), ']']);
        
        % set up the image intensity values
        Itmp = Iorig(irs:ire,ics:ice);
        imInt = reshape(Itmp,[nr*nc 1]);
        
        % find S objects that are potentially in this range
        indx = findInside([ics ice],[irs ire],sbox,0);
        
        itmp = find([S(indx).ID]==0 | [S(indx).ID]==2);  % trace objects
        itr = indx(itmp);  % indices of trace objects with respect to the full S array
        
        itmp = find([S(indx).ID]==1 | [S(indx).ID]==3);  % time mark objects
        itm = indx(itmp);  % indices of time mark objects with respect to the full S array
        
        ipixTr = cell2mat({S(itr).PixelIdxList}');  % trace pixels
        ipixTm = cell2mat({S(itm).PixelIdxList}');  % time mark pixels
        ipixD = find_duplicates(ipixTr,ipixTm,[]);  % find overlapping pixels
        
        % remove overlapping pixels from trace and time mark pixels
        if ~isempty(ipixD)  % overlapping pixels exist
            if ~isempty(ipixTr)  % trace pixels exist
                ipixTr = setdiff(ipixTr,ipixD);
            end  % if ~isempty(ipixTr)
            if ~isempty(ipixTm)  % time mark pixels exist
                ipixTm = setdiff(ipixTm,ipixD);
            end  % if ~isempty(ipixTm)
        end  % if ~isempty(ipixD)
        
        % map the indices to new image size and assign type in pixID
        for ii = 1:3  % loop over trace, time mark and duplicates
            if ii == 1  % trace
                ipix = ipixTr;
            elseif ii == 2  % time mark
                ipix = ipixTm;
            else  % duplicates
                ipix = ipixD;
            end  % if ii == 1 ...
            
            if ~isempty(ipix)  % if there are pixels to be looked at
                [ir ic] = ind2sub([ImRow ImCol],ipix);
                ir = ir - irs + 1; ic = ic - ics + 1;  % move to new image frame
                itmp = find(ir<1 | ir > nr | ic < 1 | ic > nc); % indices outside of the box
                if ~isempty(itmp)
                    ir(itmp) = []; ic(itmp) = [];  % remove pixels outside of the box
                end  % if ~isempty(itmp)
                if ~isempty(ir)  % pixels left
                    ipix = sub2ind([nr nc],ir,ic);
                    pixID(ipix) = ii-1;  % set to trace
                end  % if ~isempty(ir)
            end  % if ~isempty(ipix)
        end  % for ii = 1:3
        
        try; waitbar(ncnt/ntot,hwait); end;
        
        % save variables
        ctmp = sprintf('%s.%d.mat',cfileOut,ncnt);
        save([cdirOut ctmp],'x','y','imInt','pixID','nr','nc','-v7.3');
        ncnt = ncnt + 1;  % advance to next number
        
    end  % for m = 1:numel(ndivCs)
end  % for k = 1:numel(ndivRs)

try; delete(hwait); end;

end  % function makeTrainingData