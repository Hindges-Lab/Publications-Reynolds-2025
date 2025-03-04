% first open the figure, then run this script

% Change the current folder to the folder of this mlx-file.
if(~isdeployed)
  cd(fileparts(matlab.desktop.editor.getActiveFilename));
end

OptionZ.FrameRate = 15;
OptionZ.Duration  = 15;
OptionZ.Periodic  = true;

tic;
folder = '../outputs/RAW files x segmentation/circle-fitted lens/';
filename = 'Vertical_all';
CaptureFigVid([0, 20; 180, 20; 360, 20],[folder filename],OptionZ)%; 360, 20
toc