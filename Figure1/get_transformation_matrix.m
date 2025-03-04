% This script reads a folder (and subfolders) of .czi files and outputs the
% transformation matrix of the zebrafish eye
% Created: 2023-11-30 15:04 am
% Last modified: 2023-05-02 14:37 am
%
% INPUTS
%  - image files (.czi)
%
% OUTPUTS
%  - *_transformation_matrix.mat
clear
close all
clc

% Change the current folder to the folder of this m-file.
if(~isdeployed)
  cd(fileparts(matlab.desktop.editor.getActiveFilename));
end

%% Defining Parameters
% File Paths
% path to data (folder with .czi files)
ops.filedir = '../originals/RAW files x segmentation/'; % folder

% path to saving directory (where the transformation matrices will be stroed)
ops.savedir = '../outputs/RAW files x segmentation/'; % folder

% Other Parameters
% size of pixels (voxels)
ops.sizeX = 0.4883917; % [microns]
ops.sizeY = ops.sizeX; % [microns]
ops.sizeZ = 1; % [microns]

% additional output format for figures
ops.fig_format = '.tif';
%%
if ~exist(ops.savedir, 'dir')
    mkdir(ops.savedir)
end

[~,v] = bfCheckJavaPath();  % added such that path to Bio-Format Toolbox is know

loop_through_folder(ops.filedir, ops);

disp('Done:)')

function loop_through_folder(foldername, ops)
    %% loops through folder and subfolders
    filelist = dir(foldername);
    for i = 1:length(filelist) 
        if strcmp(filelist(i).name,'.') || strcmp(filelist(i).name,'..') 
            continue
        elseif filelist(i).isdir % is a folder
            disp(filelist(i).name);
            loop_through_folder(fullfile(filelist(i).folder,filelist(i).name), ops);
        elseif contains(filelist(i).name, '.czi') % is a .czi
            ops.filefolder = filelist(i).folder;
            ops.filename = filelist(i).name;
            % try
                compute_transformation_matrix(ops);
            % catch
            %     warning('Error. Check file.')
            % end
        end
    end
end

function compute_transformation_matrix(ops)
    %% Load Data
    tic;
    disp(ops.filename);
    data = bfopen(fullfile(ops.filefolder,ops.filename));
    data = data{1,1};
    data = data(:,1);
    
    % dimension order in data: XYCZT
    ops.Nx = size(data{1},2);
    ops.Ny = size(data{1},1);
    ops.Nz = size(data,1);
    
    data = cell2mat(data);
    data = reshape(data,ops.Ny,ops.Nz,ops.Nx);
    data = permute(data, [1,3,2]);
    
    % axes
    ops.x = (0:ops.Nx-1)*ops.sizeX;
    ops.y = (0:ops.Ny-1)*ops.sizeY;
    ops.z = (0:ops.Nz-1)*ops.sizeZ;
    
    % seed points for region growing 
    ops.seedX = ops.Nx/2;
    ops.seedY = ops.Ny/2;
    ops.seedZ = ops.Nz-20;
    
    % interpolation in Z-axis
    ops.z_interp = 0:ops.sizeX:ops.z(end);
    ops.Nz_interp = length(ops.z_interp);
    data_interp = double(data);
    data_interp = permute(data_interp, [3,1,2]);
    data_interp = interp1(ops.z, data_interp, ops.z_interp,'linear');
    data_interp = permute(data_interp, [2,3,1]);
    ops.seedZ = find(ops.z_interp>ops.seedZ, 1,"first");
    
    
    %% 2D plot 
    % smoothing: Gaussian filter
    data_filt = imgaussfilt3(double(data_interp),3)/255;
    
    % adjust contrast
    ref_val = data_filt(ops.seedY, ops.seedX, ops.seedZ);
    data_filt = imadjustn(data_filt,[0 ref_val*3],[0 1]);
    
    %% Segmentation with Thresholding
    buffer = 26;
    ROI = data_filt(ops.seedY-buffer:ops.seedY+buffer, ops.seedX-buffer:ops.seedX+buffer, ops.seedZ:ops.Nz_interp);
    [ref_val, idx] = max(ROI,[],'all');
    % update seed point
    [y, x, z] = ind2sub(size(ROI), idx);
    ops.seedX = ops.seedX-buffer + x -1;
    ops.seedY = ops.seedY-buffer + y -1;
    ops.seedZ = ops.seedZ+z-1;
    
    ops.thres = ref_val/2;
    [stats_3d, RG_mask] = lens_segmentation(ops, data_filt);
    
    RG_mask = zeros(size(RG_mask));
    RG_mask(stats_3d.VoxelIdxList{1,1}) = 1;
    
    %% Axis of lens
    min_area = 500; %[px]
    centroid_2d = zeros(4, ops.Nz_interp); % dimensions: 3 x N; [x; y; z; 1]
    centroid_2d(3,:) = ops.z_interp;
    centroid_2d(4,:) = 1;
    
    for z = ops.Nz_interp:-1:1
        slice = RG_mask(:,:,z);
        stats = regionprops(slice,"Centroid","Area");
        if ~isempty(stats)
            k = find(vertcat(stats.Area) == max(vertcat(stats.Area)));
            stats = stats(k,:); % take the biggest object only
            if z > stats_3d.Centroid(1,3) || stats.Area > min_area
                centroid_2d(1:2,z) = stats.Centroid;
            else
                centroid_2d = centroid_2d(:,z+1:end);
                break
            end
        end
    end
    
    centroid_2d(1,:) = centroid_2d(1,:)*ops.sizeX;
    centroid_2d(2,:) = centroid_2d(2,:)*ops.sizeY;
    
    centroid_2d = rmoutliers(centroid_2d');
    centroid_2d = centroid_2d';

    %% Matrix Transformation
    RG_mask_transform = RG_mask;
    ops.x_t = ops.x;
    ops.y_t = ops.y;
    ops.z_t = ops.z_interp;
    
    T = [1  0  0  0; ...
         0  1  0  0; ...
         0  0  1  0; ...
         0  0  0  1];
    
    % translation of origin to center of image;
    stats_3d = regionprops3(RG_mask_transform,"Centroid");
    Tx = (ops.Nx/2-stats_3d.Centroid(1,1))*2;
    Ty = (ops.Ny/2-stats_3d.Centroid(1,2))*2;
    Tz = (ops.Nz_interp/2-stats_3d.Centroid(1,3))*2;
    Txyz = [Tx Ty Tz];
    RG_mask_transform = imtranslate(RG_mask_transform,Txyz,'OutputView','full');
    
    ops.x_t = (-size(RG_mask_transform,2)/2:1:size(RG_mask_transform,2)/2)*ops.sizeX;
    ops.y_t = (-size(RG_mask_transform,1)/2:1:size(RG_mask_transform,1)/2)*ops.sizeY;
    ops.z_t = (-size(RG_mask_transform,3)/2:1:size(RG_mask_transform,3)/2)*ops.sizeX;
    
    TL = [1  0  0  -stats_3d.Centroid(1,1)*ops.sizeX; ...
          0  1  0  -stats_3d.Centroid(1,2)*ops.sizeY; ...
          0  0  1  -stats_3d.Centroid(1,3)*ops.sizeX; ...
          0  0  0  1];
    
    T = TL*T;
    
    % reflection relative to xy plane
    RG_mask_transform = RG_mask_transform(:,:,end:-1:1);
    RFz = [1  0  0  0; ...
           0  1  0  0; ...
           0  0 -1  0; ...
           0  0  0  1];
    
    T = RFz*T;
    
    % find best-fit line in Y-Z plane
    centroid_2d_t = T*centroid_2d;
    x = centroid_2d_t(3,:); % Z
    y = centroid_2d_t(2,:); % Y
    p_yz = polyfit(x,y,1);
    
    % find best-fit line in X-Z plane
    x = centroid_2d_t(3,:); % Z
    y = centroid_2d_t(1,:); % X
    p_xz = polyfit(x,y,1);
    
    % rotation in y-z plane (about x-axis)
    angled = -atand(p_yz(1));
    angle = atan(p_yz(1));
    RG_mask_transform = imrotate3(RG_mask_transform,angled,[1 0 0],"linear","crop");
    Rx = [1           0            0  0; ...
          0  cos(angle)  -sin(angle)  0; ...
          0  sin(angle)   cos(angle)  0; ...
          0           0            0  1];
    T = Rx*T;
    
    % rotation in x-z plane (about y-axis)
    angled = atand(p_xz(1));
    angle = -atan(p_xz(1));
    RG_mask_transform = imrotate3(RG_mask_transform,angled,[0 1 0],"linear","crop");
    Ry = [ cos(angle)  0  sin(angle)  0; ...
                    0  1           0  0; ...
          -sin(angle)  0  cos(angle)  0; ...
                    0  0           0  1];
    T = Ry*T;

    % translation to center at the projected centroid
    stats_3d = regionprops3(RG_mask_transform,"Centroid");
    I = squeeze(RG_mask_transform(:,round(stats_3d.Centroid(1,1)),:));
    p = regionprops(I,"ConvexHull");
    xData = p.ConvexHull(:,1); % z
    yData = p.ConvexHull(:,2); % y

    P = prctile(xData,50);
    idx = xData>P;
    xData = xData(idx);
    yData = yData(idx);
    
    % circle
    f = @(a) (xData-a(1)).^2 + (yData-a(2)).^2 - a(3).^2;
    a0 = [mean(xData),mean(yData),max(xData)-mean(xData)]; 
    if isnan(a0)
        return
    end
    circFit = lsqnonlin(f,a0);
    % use the circFit parameters to create the fitted circle
    theta = linspace(0,2*pi,100);  % arbitrary spacing
    xFit_circle = circFit(3)*cos(theta) + circFit(1); 
    yFit_circle = circFit(3)*sin(theta) + circFit(2);
    
    shift = size(RG_mask_transform,3)/2 - circFit(1);

    Tx = 0;
    Ty = 0;
    Tz = shift*2;
    Txyz = [Tx Ty Tz];

    xFit_circle = xFit_circle+shift;
    xData = xData+shift;
    xFit_circle = (xFit_circle-size(RG_mask_transform,3)/2)*ops.sizeX;
    xData = (xData-size(RG_mask_transform,3)/2)*ops.sizeX;
      
    yFit_circle = (yFit_circle-size(RG_mask_transform,1)/2)*ops.sizeX;
    yData = (yData-size(RG_mask_transform,1)/2)*ops.sizeX;


    RG_mask_transform = imtranslate(RG_mask_transform,Txyz,'OutputView','full');
    ops.z_t = (-size(RG_mask_transform,3)/2:1:size(RG_mask_transform,3)/2)*ops.sizeX;
    
    TL = [1  0  0  0; ...
          0  1  0  0; ...
          0  0  1  shift*ops.sizeX; ...
          0  0  0  1];
    
    T = TL*T;
    centroid_2d_t = T*centroid_2d;

    % save image
    C = strsplit(ops.filename,'_');
    C = C(1:2);
    fish_name = strjoin(C,'_');
    save(fullfile(ops.savedir,strcat(fish_name,'_transformation_matrix.mat')),"T","ops")
    
    % plotting
    plot_fig(ops, RG_mask_transform, centroid_2d_t,xFit_circle,yFit_circle, xData, yData);
    toc;
end

function plot_fig(ops, im, centroid_2d, varargin)
    if nargin >3
        x1 = varargin{1};
        y1 = varargin{2};
    end
    if nargin > 5
        x2 = varargin{3};
        y2 = varargin{4};
    end


    stats_3d = regionprops3(im,"Centroid");
    % plotting
    figure;
    set(gcf,'Units','normalized','Position',[0 0 .8 .4]); % [0 0 width height]
    
    % X-Z plane
    subplot(1,3,1)
    imagesc(ops.z_t, ops.x_t, squeeze(im(round(stats_3d.Centroid(1,2)),:,:)))
    hold on
    x = centroid_2d(3,:); % Z
    y = centroid_2d(1,:); % X
    scatter(x, y, 'rx');
    yline(0,'w')
    xline(0,'w')
    axis image
    xlabel('Z [μm]')
    ylabel('X [μm]')
    title('X-Z plane')
    
    % Y-Z plane
    subplot(1,3,2)
    imagesc(ops.z_t, ops.y_t, squeeze(im(:,round(stats_3d.Centroid(1,1)),:)))
    hold on
    x = centroid_2d(3,:); % Z
    y = centroid_2d(2,:); % Y
    scatter(x, y, 'rx');
    yline(0,'w')
    xline(0,'w')
    if exist('xFit_circle','var')
        plot(x1,y1,'r-','LineWidth',1)
    end
    if exist('xFit_oval','var')
        plot(x2,y2,'r-','LineWidth',1)
    end
    axis image
    xlabel('Z [μm]')
    ylabel('Y [μm]')
    title('Y-Z plane')
    
    % X-Y plane
    subplot(1,3,3)
    imagesc(ops.x_t, ops.y_t, squeeze(im(:,:,round(stats_3d.Centroid(1,3)))))
    hold on
    x = centroid_2d(1,end); % X
    y = centroid_2d(2,end); % Y
    scatter(x, y, 'rx');
    axis image
    xlabel('X [μm]')
    ylabel('Y [μm]')
    xline(x,'w')
    yline(y,'w')
    title('X-Y plane')

    % save figure
    filename = strcat(ops.filename(1:end-4),'_lens');
    saveas(gcf, strcat(ops.savedir, filesep, filename, '.fig'))
    saveas(gcf, strcat(ops.savedir, filesep, filename, ops.fig_format))
    close all
end

function [stats_3d, RG_mask] = lens_segmentation(ops, data_filt)

    RG_mask = data_filt>ops.thres;
    
    stats_3d = regionprops3(RG_mask,"Centroid","Volume","VoxelIdxList");
    seed_ind = sub2ind([ops.Ny, ops.Nx, ops.Nz_interp],ops.seedY,ops.seedX, ops.seedZ);
    stats_3d = sortrows(stats_3d,"Volume",'descend');
    
    for row = 1:size(stats_3d,1)
        if any(stats_3d(row,:).VoxelIdxList{1,1} == seed_ind)
            k = row;
            break
        end
    end
    stats_3d = stats_3d(k,:); % take the object that contains the seed point
    
end

