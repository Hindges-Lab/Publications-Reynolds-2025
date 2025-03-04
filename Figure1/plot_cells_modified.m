% This script reads an excel for the locations of cells, then applies
% the corresponding tranformation matrix and plots all cells in the same 
% space.
% Created: 2023-12-06 11:19 am
% Last modified: 2023-05-02 15:02 am
%
% INPUTS
%  - cell_list.xlsx
%
% OUTPUTS
%  - output_cell_list_*.xlsx
%  - figures (.fig and .tif)
%       XYZ_view, frontal_view, sagittal_view, transverse_view, polar_plot

%% Defining Parameters
clear
close all
clc

% Change the current folder to the folder of this mlx-file.
if(~isdeployed)
  cd(fileparts(matlab.desktop.editor.getActiveFilename));
end

% File Paths
% path to saving directory with transformation_matrix.mat
loaddir = '../outputs/RAW files x segmentation'; % folder

% excel files
xls_filename = '../originals/RAW files x segmentation/cell_list_complete_Vertical_shortlisted.xlsx'; % file
output_xls_filename = '../outputs/RAW files x segmentation/output_cell_list_Vertical.xlsx'; % file
% 
% xls_filename = '../originals/RAW files x segmentation/cell_list_complete_Horizontal_shortlisted.xlsx'; % file
% output_xls_filename = '../outputs/RAW files x segmentation/output_cell_list_Horizontal.xlsx'; % file
% 
% xls_filename = '../originals/RAW files x segmentation/cell_list_complete_BaselineNSNC_shortlisted.xlsx'; % file
% output_xls_filename = '../outputs/RAW files x segmentation/output_cell_list_BaselineNSNC.xlsx'; % file
% 
% xls_filename = '../originals/RAW files x segmentation/cell_list_complete_BaselineNSChannels_shortlisted.xlsx'; % file
% output_xls_filename = '../outputs/RAW files x segmentation/output_cell_list_BaselineNSChannels.xlsx'; % file

% additional output format for figures (apart from .fig)
fig_format = '.tif'; %'.tif'

orientation = {'all', 'horizontal', 'oblique', 'vertical'}; % 'all', 'horizontal', 'oblique', 'vertical'
plot_DF = true; % set to false to plot only cell body

for n_plotorientation  = 1:numel(orientation)
    if ~strcmp(orientation{n_plotorientation}, 'all')
        plot_orientation = orientation{n_plotorientation};

    end
%%
    cell_list = readtable(xls_filename);
    
    f1 = figure();
    hold on
    
    for i = 1:size(cell_list,1)
        filename = cell_list{i,"FileName"};
        filename = filename{1};
    
        load(strcat(loaddir,filesep,filename,'_transformation_matrix.mat'))
        
        ops.savedir = fullfile(loaddir,orientation{n_plotorientation});
        if ~exist(ops.savedir, 'dir')
            mkdir(ops.savedir)
        end

        cell_orientation = cell_list{i,"orientation"};
        switch cell_orientation{1}
            case 'horizontal'
                color = 'c';
            case 'oblique'
                color = 'g';
            case 'vertical'
                color = "m";
        end
        if exist("plot_orientation", "var") && ~strcmp(cell_orientation, plot_orientation)
            continue;
        end
    
        % normaliztion 
        cellBody = T*[cell_list{i,"cellBodyX"}; cell_list{i,"cellBodyY"}; cell_list{i,"cellBodyZ"}; 1];
        dendriteStart = T*[cell_list{i,"dendriteStartX"}; cell_list{i,"dendriteStartY"}; cell_list{i,"dendriteStartZ"}; 1];
        dendriteEnd = T*[cell_list{i,"dendriteEndX"}; cell_list{i,"dendriteEndY"}; cell_list{i,"dendriteEndZ"}; 1];
    
        cellBody(1:3) = cellBody(1:3)./vecnorm(dendriteStart(1:3));
        dendriteEnd(1:3) = dendriteEnd(1:3)./vecnorm(dendriteEnd(1:3));
        dendriteStart(1:3) = dendriteStart(1:3)./vecnorm(dendriteStart(1:3));
        
        markersize = 20;
        h1 = scatter3(cellBody(1),cellBody(2),cellBody(3),markersize,"MarkerEdgeColor",color,"MarkerFaceColor",color);
        str = cell(1,1);
        for pp = 1:1
            str{pp} = strjoin(strsplit(cell_list.FileName{i},'_'),' ');
        end
        txt = dataTipTextRow('FileName =',str);
        h1.DataTipTemplate.DataTipRows(end+1) = txt;
            
        % save properties to table
        cell_list.cellBodyX(i) = cellBody(1);
        cell_list.cellBodyY(i) = cellBody(2);
        cell_list.cellBodyZ(i) = cellBody(3);
    
        cell_list.dendriteStartX(i) = dendriteStart(1);
        cell_list.dendriteStartY(i) = dendriteStart(2);
        cell_list.dendriteStartZ(i) = dendriteStart(3);
    
        cell_list.dendriteEndX(i) = dendriteEnd(1);
        cell_list.dendriteEndY(i) = dendriteEnd(2);
        cell_list.dendriteEndZ(i) = dendriteEnd(3);
    
        angle_xy = atan2d(dendriteEnd(2)-dendriteStart(2), dendriteEnd(1)-dendriteStart(1));
        angle_xz = atan2d(dendriteEnd(3)-dendriteStart(3), dendriteEnd(1)-dendriteStart(1));
        angle_yz = atan2d(dendriteEnd(3)-dendriteStart(3), dendriteEnd(2)-dendriteStart(2));
    
        cell_list.angle_xy(i) = angle_xy;
        cell_list.angle_xz(i) = angle_xz;
        cell_list.angle_yz(i) = angle_yz;
    
        % cross product to obtain the vector normal to plane    
        C = cross(dendriteStart(1:3),dendriteEnd(1:3));
        
        T = [1  0  0  0; ...
             0  1  0  0; ...
             0  0  1  0; ...
             0  0  0  1];
        
        % rotation to align normal vector with z-axis, i.e. the two points
        % would be in x-y plane after rotation.
        % rotation in x-y plane (about z-axis)
        angle = -atan2(C(2), C(1));
        Rz = [cos(angle)  -sin(angle)  0  0; ...
              sin(angle)   cos(angle)  0  0; ...
                       0            0  1  0; ...
                       0            0  0  1];
        T = Rz*T;
        C = T*[C; 1];
        
        % rotation in x-z plane (about y-axis)
        angle = -atan2(C(1), C(3));
        Ry = [ cos(angle)  0  sin(angle)  0; ...
                        0  1           0  0; ...
              -sin(angle)  0  cos(angle)  0; ...
                        0  0           0  1];
        T = Ry*T;
    
        dendriteStart = T*dendriteStart;
        dendriteEnd = T*dendriteEnd;
    
        % radius of curvature
        r1 = vecnorm(dendriteStart(1:2)); 
        r2 = vecnorm(dendriteEnd(1:2));
        
        theta1 = atan2(dendriteStart(2), dendriteStart(1));
        theta2 = atan2(dendriteEnd(2), dendriteEnd(1));
    
        % save properties to structure
        cell_list.r1(i) = r1;
        cell_list.r2(i) = r2;
    
        % special case: theta1 and theta2 are of different signs
        if theta1*theta2 < 0
            if theta1>theta2
                theta1 = -2*pi()+theta1;
            else % theta2>theta1
                theta2 = -2*pi()+theta2;
            end
        end
    
        n = 10; % number of points in arc
        
        step = (theta2-theta1)/(n-1);
        theta = theta1:step:theta2;
        
        step = (r2-r1)/(n-1);
        r = r1:step:r2;
    
        % x = r.*cos(theta);
        % y = r.*sin(theta);    
        x = cos(theta);
        y = sin(theta);
    
        arc = inv(T)*[x; y; zeros(1,n); ones(1,n)];
        
        % find arc length
        arclength = 0;
        for k = 1:n-1
          arclength = arclength + sqrt( (x(k+1)-x(k))^2 + (y(k+1)-y(k))^2 );
        end
        cell_list.arclength(i) = arclength;
        
        if plot_DF
            h1 = plot3(arc(1,:), arc(2,:), arc(3,:),"Color",color,'LineWidth',1.5);
            str = cell(1,n);
            for pp = 1:n
                str{pp} = strjoin(strsplit(cell_list.FileName{i},'_'),' ');
            end
            txt = dataTipTextRow('FileName =',str);
            h1.DataTipTemplate.DataTipRows(end+1) = txt;
        end
    end
    xlabel('Nasal-Temporal (normalized)')
    ylabel('Ventral-Dorsal (normalized)')
    zlabel('Lateral-Medial (normalized)')
    set(gca,'Color','k')
    set(gca,'FontSize',14)
    axis equal
    
    % show spheres for reference
    % % IPL
    r = 1;
    [X,Y,Z] = sphere;
    surf(X*r,Y*r,Z*r,'FaceColor','w','FaceAlpha',0.4,'EdgeColor','none','EdgeAlpha',0.5)

    % lens
    r = 0.38;
    [X,Y,Z] = sphere;
    surf(X*r,Y*r,Z*r,'FaceColor','w','FaceAlpha',0.7,'EdgeColor','none')

    limit = 1.2;
    xlim([-limit limit])
    ylim([-limit limit])
    zlim([0 limit])
    view([35 20]) % Azimuth and Elevation
    % save figure
    set(gcf, 'InvertHardCopy', 'off'); 
    saveas(gcf, strcat(ops.savedir,filesep, 'XYZ_view.fig'))
    saveas(gcf, strcat(ops.savedir,filesep, 'XYZ_view', fig_format))

    % view in other orientation: X-Y plane
    ax1 = gca;
    f2 = figure();
    ax2 = copyobj(ax1,f2);
    view([180 90]) % Azimuth and Elevation
    set ( gca, 'xdir', 'reverse' )
    set ( gca, 'ydir', 'reverse' )
    title('Sagittal Plane') % X-Y
    % save figure
    set(f2, 'InvertHardCopy', 'off'); 
    filename = 'sagittal_view';
    saveas(f2, strcat(ops.savedir,filesep, filename, '.fig'))
    saveas(f2, strcat(ops.savedir,filesep, filename, fig_format))
    close(f2)

    % view in other orientation: Y-Z plane
    f3 = figure();
    ax3 = copyobj(ax1,f3);
    view([90 0]) % Azimuth and Elevation
    title('Transverse Plane')
    % save figure
    set(f3, 'InvertHardCopy', 'off'); 
    filename = 'transverse_view';
    saveas(f3, strcat(ops.savedir,filesep, filename, '.fig'))
    saveas(f3, strcat(ops.savedir,filesep, filename, fig_format))
    close(f3)

    % view in other orientation: X-Z plane
    f4 = figure();
    ax4 = copyobj(ax1,f4);
    view([0 0]) % Azimuth and Elevation
    title('Frontal Plane')
    % save figure
    set(f4, 'InvertHardCopy', 'off'); 
    filename = 'frontal_view';
    saveas(f4, strcat(ops.savedir,filesep, filename, '.fig'))
    saveas(f4, strcat(ops.savedir,filesep, filename, fig_format))
    close(f4)
    
    % save data
    if strcmp(orientation{n_plotorientation}, 'all')
        disp(cell_list)
        writetable(cell_list,output_xls_filename)
    end
    
    %% polar plots
    
    % plot angles in polar plots
    % f = figure;
    % set(gcf,'Units','normalized','Position',[0 0 .4 .3]); % [0 0 width height]
    % 
    % ThetaTick = 0:90:360;
    % RTick = 0:0.5:1; %0.2:0.2:1;
    % RTickLabel = [""];
    % RLim = [0 max(RTick)];
    % LineColor = [161 179 220]/255;%"#ffb703";
    % LineWidth = 1;
    % LineColor_avg = 'k';%[20 32 57]/255;
    % LineWidth_avg = 3;
    % EdgeAlpha_avg = 0.5;
    % fontsize = 16;
    % 
    % if exist("plot_orientation", "var")
    %     ind = strcmp(cell_list{:,"orientation"}, plot_orientation);
    % end
    % 
    % subplot(1,3,1) % X-Y Sagittal
    % x = cell_list{:,"dendriteEndX"}-cell_list{:,"dendriteStartX"};
    % y = cell_list{:,"dendriteEndY"}-cell_list{:,"dendriteStartY"};
    % if exist("plot_orientation", "var")
    %     x = x(ind);
    %     y = y(ind);
    % end
    % [theta,rho] = cart2pol(x,y);
    % for i = 1:length(theta)
    %     polarplot([0 theta(i)],[0 rho(i)],"Color",LineColor,"LineWidth",LineWidth)
    %     hold on
    % end
    % % get average
    % x = mean(x);
    % y = mean(y);
    % [theta,rho] = cart2pol(x,y);
    % a = polarplot([0 theta],[0 rho],"Color",LineColor_avg,"LineWidth",LineWidth_avg);
    % a.Color =[a.Color(1:3), EdgeAlpha_avg];
    % thetaticks(ThetaTick);
    % thetaticklabels({'T','D','N','V'});
    % rlim(RLim)
    % rticks(RTick);
    % rticklabels(RTickLabel);
    % title('Sagittal Plane') % X-Y
    % set(gca, "FontSize", fontsize)
    % 
    % subplot(1,3,2) % Y-Z transverse
    % x = cell_list{:,"dendriteEndZ"}-cell_list{:,"dendriteStartZ"};
    % y = cell_list{:,"dendriteEndY"}-cell_list{:,"dendriteStartY"};
    % if exist("plot_orientation", "var")
    %     x = x(ind);
    %     y = y(ind);
    % end
    % [theta,rho] = cart2pol(x,y);
    % for i = 1:length(theta)
    %     polarplot([0 theta(i)],[0 rho(i)],"Color",LineColor,"LineWidth",LineWidth)
    %     hold on
    % end
    % % get average
    % x = mean(x);
    % y = mean(y);
    % [theta,rho] = cart2pol(x,y);
    % a = polarplot([0 theta],[0 rho],"Color",LineColor_avg,"LineWidth",LineWidth_avg);
    % a.Color =[a.Color(1:3), EdgeAlpha_avg];
    % thetaticks(ThetaTick);
    % thetaticklabels({'L','D','M','V'});
    % rlim(RLim)
    % rticks(RTick);
    % rticklabels(RTickLabel);
    % title('Transverse plane') % Y-Z
    % set(gca, "FontSize", fontsize)
    % 
    % subplot(1,3,3) % X-Z Frontal
    % x = cell_list{:,"dendriteEndX"}-cell_list{:,"dendriteStartX"};
    % y = cell_list{:,"dendriteEndZ"}-cell_list{:,"dendriteStartZ"};
    % if exist("plot_orientation", "var")
    %     x = x(ind);
    %     y = y(ind);
    % end
    % [theta,rho] = cart2pol(x,y);
    % for i = 1:length(theta)
    %     polarplot([0 theta(i)],[0 rho(i)],"Color",LineColor,"LineWidth",LineWidth)
    %     hold on
    % end
    % % get average
    % x = mean(x);
    % y = mean(y);
    % [theta,rho] = cart2pol(x,y);
    % a = polarplot([0 theta],[0 rho],"Color",LineColor_avg,"LineWidth",LineWidth_avg);
    % a.Color =[a.Color(1:3), EdgeAlpha_avg];
    % thetaticks(ThetaTick);
    % thetaticklabels({'T','M','N','L'});
    % rlim(RLim)
    % rticks(RTick);
    % rticklabels(RTickLabel);
    % title('Frontal plane') % X-Z
    % set(gca, "FontSize", fontsize)
    % 
    % % save figure
    % saveas(f, strcat(ops.savedir,filesep, 'polar_plot.fig'))
    % saveas(f, strcat(ops.savedir,filesep, 'polar_plot', fig_format))
    % 
    close all
end

disp('DONE')