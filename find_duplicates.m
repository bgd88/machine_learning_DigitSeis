function idup = find_duplicates(varargin)
%function idup = find_duplicates(itrace,itm,inoise)
%  This function reads in trace, time mark, and noise pixels and finds
%  overlapping entries, and remove duplicate entries from trace, time mark
%  and noise objects
%
%   Input:  itrace0 = pixels corresponding to trace objects
%           itm0 = pixels corresponding to time mark objects
%           inoise0 = pixels corresonding to noise objects
%   Output: idup = pixeld of overlapping trace or time mark objects
%
%  July 27, 2017
%  Last Modified: October 05, 2018

idup = [];  % indices of duplicates

lorig = true;
if numel(varargin) == 3
    itrace = varargin{1};
    itm = varargin{2};
    inoise = varargin{3};
elseif numel(varargin) == 1
    S = varargin{1};
    lorig = false;
else
    htmp = warndlg('Input parameter issue in find_duplicates.m - should not get here');
    waitfor(htmp);
    return
end  % if numel(varargin) == 1

% disp('in find_duplicates');
% tic

if ~lorig  % if object structure input (uses less memory)
    sbox = cell2mat({S.BoundingBox}');
    for k = 1:numel(S)-1
        xp = sbox(k,1) + [0 sbox(k,3)]; % x patch defining outline of this object
        yp = sbox(k,2) + [0 sbox(k,4)]; % y patch defining outline of this object
        indx = findInside(xp,yp,sbox(k+1:end,:),0); % find sboxes that are inside
        if ~isempty(indx)
            rpix = [S(k).PixelIdxList];  % pixels of interest
            indx = indx + k;  % make sure the index points to index of sbox
            rpixt = cell2mat({S(indx).PixelIdxList}');
            itmp = intersect(rpix,rpixt);
            if ~isempty(itmp)
                itmp = reshape(itmp,[numel(itmp),1]);
                idup = [idup; itmp];
            end % if ~isempty(itmp)
%             for m = 1:numel(indx)
%                 rpixt = [S(indx(m)).PixelIdxList]; % pixels to be checked
%                 itmp = intersect(rpix,rpixt);
%                 if ~isempty(itmp)
%                     itmp = reshape(itmp,[numel(itmp),1]);
%                     idup = [idup; itmp];
%                 end  % if ~isempty(itmp)
%             end  % for m = 1:numel(indx)
        end  % if ~isempty(indx)
    end % for k = 1:numel(S)-1
else
    % first find all duplicates within each group
    [itmU, itmp] = unique(itm);
    if (numel(itm)-numel(itmp)) >0  % duplicate entries exist
        indx = itm; indx(itmp) = [];
        idup = [idup; indx];
    end  % if (numel(itick)-numel(itmp)) > 0
    %disp(['After processing time marks: ', num2str(toc), ' s']);
    [itraceU, itmp] = unique(itrace);
    if (numel(itrace)-numel(itmp)) >0  % duplicate entries exist
        indx = itrace; indx(itmp) = [];
        idup = [idup; indx];
    end  % if (numel(itrace)-numel(itmp)) > 0
    %disp(['After processing traces: ', num2str(toc),' s']);
    
    if nargin == 3  % inoise available
        [inoiseU, itmp] = unique(inoise);
        if (numel(inoise)-numel(itmp)) > 0  % duplicate entries exist
            indx = inoise; indx(itmp) = [];
            idup = [idup; indx];
        end  % if (numel(inoise) - numel(itmp)) > 0
    end  % if (nargin == 3)
    
    % second, find all overlapping values between different groups
    % indx = intersect(irej,itick);
    % if ~isempty(indx)  % if overlapping pixels exist
    %     idup = [idup; indx];
    % end  % if ~isempty(indx)
    indx = intersect(itmU,itraceU);
    if ~isempty(indx)  % if overlapping pixels exist
        idup = [idup; indx];
    end  % if ~isempty(indx)
    %disp(['After finding overlap between time mark and trace: ', num2str(toc),' s']);
    % indx = intersect(itrace,irej);
    % if ~isempty(indx)  % if overlapping pixels exist
    %     idup = [idup; indx];
    % end  % if ~isempty(indx)
    
    if nargin == 3  % inoise variable exists
        indx = intersect(itmU,inoiseU);  % overlap of time mark and noise
        if ~isempty(indx)  % if overlapping pixels exist
            idup = [idup; indx];
        end  % if ~isempty(indx)
        indx = intersect(itraceU,inoiseU);  % overlap of trace and noise
        if ~isempty(indx)  % if overlapping pixels
            idup = [idup; indx];
        end  % if ~isempty(indx)
    end  % if nargin == 3
    
    % finally find unique overlapping values
    if numel(idup)>1  % overlapping pixels exist
        idup = unique(idup);
    end  % if numel(idup)>1
    %disp(['After finding duplicate entries in idup: ', num2str(toc),' s']);
    
    % % remove any noise pixels that are overlapping with trace or time mark
    % inoise = setdiff(inoise,itrace);
    % inoise = setdiff(inoise,itm);
    %
end  % if ~lorig

% toc
end  % function find_duplicates