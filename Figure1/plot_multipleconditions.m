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
savedir = '../outputs/RAW files x segmentation/circle-fitted lens/'; % folder

xls_filename_bnsnc = fullfile(savedir, 'output_cell_list_BaselineNSChannels.xlsx'); % file
xls_filename_bnsc = fullfile(savedir, 'output_cell_list_BaselineNSNC.xlsx'); % file
xls_filename_h = fullfile(savedir, 'output_cell_list_Horizontal.xlsx'); % file
xls_filename_v = fullfile(savedir, 'output_cell_list_Vertical.xlsx'); % file

%%
cell_list_bnsnc = readtable(xls_filename_bnsnc);
cell_list_bnsc = readtable(xls_filename_bnsc);
cell_list_h = readtable(xls_filename_h);
cell_list_v = readtable(xls_filename_v);

linewidth = 3;

x_bnsnc = mean(cell_list_bnsnc{:,"dendriteEndX"}-cell_list_bnsnc{:,"dendriteStartX"});
y_bnsnc = mean(cell_list_bnsnc{:,"dendriteEndY"}-cell_list_bnsnc{:,"dendriteStartY"});
z_bnsnc = mean(cell_list_bnsnc{:,"dendriteEndZ"}-cell_list_bnsnc{:,"dendriteStartZ"});

x_bnsc = mean(cell_list_bnsc{:,"dendriteEndX"}-cell_list_bnsc{:,"dendriteStartX"});
y_bnsc = mean(cell_list_bnsc{:,"dendriteEndY"}-cell_list_bnsc{:,"dendriteStartY"});
z_bnsc = mean(cell_list_bnsc{:,"dendriteEndZ"}-cell_list_bnsc{:,"dendriteStartZ"});

x_h = mean(cell_list_h{:,"dendriteEndX"}-cell_list_h{:,"dendriteStartX"});
y_h = mean(cell_list_h{:,"dendriteEndY"}-cell_list_h{:,"dendriteStartY"});
z_h = mean(cell_list_h{:,"dendriteEndZ"}-cell_list_h{:,"dendriteStartZ"});

x_v = mean(cell_list_v{:,"dendriteEndX"}-cell_list_v{:,"dendriteStartX"});
y_v = mean(cell_list_v{:,"dendriteEndY"}-cell_list_v{:,"dendriteStartY"});
z_v = mean(cell_list_v{:,"dendriteEndZ"}-cell_list_v{:,"dendriteStartZ"});

figure;
hold on
grid on
% plot3([0 x_bnsnc], [0 y_bnsnc], [0 z_bnsnc],'LineWidth',linewidth)
% plot3([0 x_bnsc], [0 y_bnsc], [0 z_bnsc],'LineWidth',linewidth)
% plot3([0 x_h], [0 y_h], [0 z_h],'LineWidth',linewidth)
% plot3([0 x_v], [0 y_v], [0 z_v],'LineWidth',linewidth)

quiver3(0, 0, 0, x_bnsnc, y_bnsnc, z_bnsnc, 0, 'LineWidth',linewidth)
quiver3(0, 0, 0, x_bnsc, y_bnsc, z_bnsc, 0, 'LineWidth',linewidth)
quiver3(0, 0, 0, x_h, y_h, z_h, 0, 'LineWidth',linewidth)
quiver3(0, 0, 0, x_v, y_v, z_v, 0, 'LineWidth',linewidth)

legend('Baseline NSNC', 'Baseline NSChannels', 'Horizontal', 'Vertical');


xlabel('Nasal-Temporal (normalized)')
ylabel('Ventral-Dorsal (normalized)')
zlabel('Lateral-Medial (normalized)')
axis equal

