clear; close all; clc
path0 = "C:/Users/24048369/OneDrive - The University of Western Australia/MHW_Project_FHL/Ocetrac/";


addpath(genpath('C:/Users/24048369/OneDrive - The University of Western Australia/MHW_Project_FHL/toolbox_gsw/'))
addpath(genpath('C:/Users/24048369/OneDrive - The University of Western Australia/MHW_Project_FHL/pcolor3/'))

blobs = ncread(path0+ 'merged_blobs_2010-2011_4D.nc', 'Blobs');

mask = ncread(path0 + 'mask_2010-2011_4D.nc', 'thetao');


lon = ncread(path0+ 'merged_blobs_2010-2011_4D.nc', 'longitude');
lat = -(ncread(path0+ 'merged_blobs_2010-2011_4D.nc', 'latitude'));
depth = ncread(path0+ 'merged_blobs_2010-2011_4D.nc', 'depth'); % Depth levels in meters to eference file
time = ncread(path0+ 'merged_blobs_2010-2011_4D.nc', 'time');

cli_start = datenum(2010,9,1);
cli_end = datenum(2011,6,1);
dates = datenum(cli_start):datenum(cli_end);
datesm = datetime(dates, 'ConvertFrom', 'datenum');
% ref_date = datetime(2010, 11, 1);  % Example reference date, adjust according to your dataset
% time_in_dates = ref_date + days(time);


aux = zeros(size(blobs,1),size(blobs,2));



% Define parameters
numFrames = 100; % Number of frames (adjust as needed)
videoFileName = 'myVideo_4D'; % Output video file name

% % Create VideoWriter object
% videoWriter = VideoWriter(videoFileName);
% videoWriter.FrameRate = 1; % Frames per second (adjust as needed)
% open(videoWriter);







for t = 180:180%length(datesm)
% t = 130;
% Crear la figura
    fig = figure('units', 'normalized', 'outerposition', [0 0 1 .9]);
    
    
    blobs_sel = blobs(:,:,:,t);
    %%%%%% test
    [nLat, nLon, nDepth] = size(blobs_sel);
    nanLat = NaN(1, nLon, nDepth);      % One extra row in latitude dimension
    nanLon = NaN(nLat+2, 1, nDepth);    % One extra row in longitude dimension
    nanDepth = NaN(nLat+2, nLon+2, 1);  % One extra row in depth dimension
    blobs_sel = cat(1, nanLat, blobs_sel, nanLat);  % Add NaN in front and back in latitude
    blobs_sel = cat(2, nanLon, blobs_sel, nanLon);  % Add NaN in front and back in longitude
    blobs_sel = cat(3, nanDepth, blobs_sel, nanDepth);  % Add NaN in front and back in depth
    dLat = lat(2) - lat(1);
    new_latitude = [lat(1) - dLat; lat(:); lat(end) + dLat];
    dLon = lon(2) - lon(1);
    new_longitude = [lon(1) - dLon; lon(:); lon(end) + dLon];
    new_depth = [-5; depth(:); 210];

    nanLat = ones(1, nLon, nDepth);      % One extra row in latitude dimension
    nanLon = ones(nLat+2, 1, nDepth);    % One extra row in longitude dimension
    nanDepth = ones(nLat+2, nLon+2, 1);  % One extra row in depth dimension
    mask2 = mask;
    mask2 = cat(1, nanLat, mask2, nanLat);  % Add NaN in front and back in latitude
    mask2 = cat(2, nanLon, mask2, nanLon);  % Add NaN in front and back in longitude
    mask2 = cat(3, nanDepth, mask2, nanDepth);  % Add NaN in front and back in depth

    %%%%%%



    h = isosurface(new_latitude, new_longitude, -new_depth, ~isnan(blobs_sel), 0);
    p = patch(h);
    set(p, 'FaceColor', 'red', 'FaceAlpha', 0.2, 'EdgeColor', 'none'); % Customize the appearance
    camlight; 
    lighting gouraud;
    
    hold on
    x_point = 115.0733;
    z = -new_depth;
    y_point = 31.6;
    plot3(y_point * ones(size(z)), x_point * ones(size(z)), z, 'blue', 'LineWidth', 4);
    scatter3(y_point , x_point , 5, 'blue', 'LineWidth', 5);
    grid off
    
    
    s = isosurface(new_latitude, new_longitude, -new_depth,mask2,0)
    p = patch(s);
    set(p,'FaceColor',[0.5 0.5 0.5]);  
    set(p,'EdgeColor',[0.5 0.5 0.5]);
    camlight;
    lighting gouraud;
    
    title_str = sprintf('Data for %s', datestr(datesm(t), 'dd-mmm-yyyy'));
    title(title_str);
    
    
    azimuth = 50;  % Ángulo en el plano horizontal (en grados)
    elevation = 55; % Ángulo de elevación (en grados)
    view(azimuth, elevation);

    ylim([109,116]);
    zlim([-210,10]);
    xlim([26.5, 35.5]);

    

    xlabel('Latitude[°S]')
    ylabel('Longitude [°E]')
    zlabel('Depth [m]')

%     frame = getframe(fig);
     % Capture the plot as an image
  
%     writeVideo(videoWriter, frame); % Write frame to video
    
    % Close the figure
%     close(fig);
end

% % Close the video file
% close(videoWriter);



%%


fig = figure('units', 'normalized', 'outerposition', [0.1 0.1 0.8 .7]);
    
time_point = 10;
blobs_sel = blobs; %(:,:,:,time_point);
%%%%%% add in front and back of each dimensions another NaN row/column
[nLat, nLon, nDepth,nTimes] = size(blobs_sel);
nanLat = NaN(1, nLon, nDepth,nTimes);      % One extra row in latitude dimension
nanLon = NaN(nLat+2, 1, nDepth,nTimes);    % One extra row in longitude dimension
nanDepth = NaN(nLat+2, nLon+2, 1,nTimes);  % One extra row in depth dimension

blobs_sel = cat(1, nanLat, blobs_sel, nanLat);  % Add NaN in front and back in latitude
blobs_sel = cat(2, nanLon, blobs_sel, nanLon);  % Add NaN in front and back in longitude
blobs_sel = cat(3, nanDepth, blobs_sel, nanDepth);  % Add NaN in front and back in depth
dLat = lat(2) - lat(1);
new_latitude = [lat(1) - dLat; lat(:); lat(end) + dLat];
dLon = lon(2) - lon(1);
new_longitude = [lon(1) - dLon; lon(:); lon(end) + dLon];
new_depth = [-5; depth(:); 210];


nanLat = ones(1, nLon, nDepth);      % One extra row in latitude dimension
nanLon = ones(nLat+2, 1, nDepth);    % One extra row in longitude dimension
nanDepth = ones(nLat+2, nLon+2, 1);  % One extra row in depth dimension
mask2 = mask;
mask2 = cat(1, nanLat, mask2, nanLat);  % Add NaN in front and back in latitude
mask2 = cat(2, nanLon, mask2, nanLon);  % Add NaN in front and back in longitude
mask2 = cat(3, nanDepth, mask2, nanDepth);  % Add NaN in front and back in depth
%%%%%%



h = isosurface(new_latitude, new_longitude, -new_depth, ~isnan(blobs_sel(:,:,:,time_point)), 0);
p = patch(h);
set(p, 'FaceColor', 'red', 'FaceAlpha', 0.2, 'EdgeColor', 'none'); % Customize the appearance
camlight; 
lighting gouraud;
hold on
grid off
x_point = 115.0733;
z = -new_depth;
y_point = 31.6;
plot3(y_point * ones(size(z)), x_point * ones(size(z)), z, 'blue', 'LineWidth', 4);
scatter3(y_point , x_point , 5, 'blue', 'LineWidth', 5);

s = isosurface(new_latitude, new_longitude, -new_depth,mask2,0)
p = patch(s);
set(p,'FaceColor',[0.5 0.5 0.5]);  
set(p,'EdgeColor',[0.5 0.5 0.5]);
camlight;
lighting gouraud;


title_str = sprintf('Time = %s', datestr(datesm(time_point), 'dd-mmm-yyyy'));
title(title_str);


azimuth = 50;  % Ángulo en el plano horizontal (en grados)
elevation = 55; % Ángulo de elevación (en grados)
view(azimuth, elevation);

ylim([109,116]);
zlim([-210,10]);
xlim([26.5, 35.5]);


xlabel('Latitude[°S]')
ylabel('Longitude [°E]')
zlabel('Depth [m]')


[LonGrid, LatGrid, DepthGrid] = meshgrid(new_latitude,new_longitude, new_depth);
% Create a structure to hold the plot data
plot_data.LonGrid =  LonGrid;
plot_data.LatGrid = LatGrid;
plot_data.DepthGrid = DepthGrid;
plot_data.data = blobs_sel;
plot_data.nTime = datesm;

plot_data.datamask = mask2;


% Store the structure with guidata
h = gcf;  % Handle to the current figure
guidata(h, plot_data);





% Add slider control for time selection
slider = uicontrol('Style', 'slider', 'Min', 1, 'Max', length(datesm), 'Value', time_point, ...
               'Position', [100, 50, 300, 20], 'Callback', @update_plot);

% Function to update the plot based on slider value (time)
% Function to update the plot based on slider value (time)
function update_plot(source, ~)
    h = gcf;  % Handle to the current figure
    plot_data = guidata(h);  % Retrieve the structure from guidata
    
    if isstruct(plot_data)  % Ensure that plot_data is a structure
        time_point = round(source.Value);  % Get the current time point from the slider
        
        % Access the fields from the structure
        fv = isosurface(plot_data.LonGrid, plot_data.LatGrid,- plot_data.DepthGrid, ...
                        ~isnan(plot_data.data(:,:,:,time_point)), 0);
        cla;  % Clear the current plot
        p = patch(fv);
        set(p, 'FaceColor', 'red', 'FaceAlpha',0.5,'EdgeColor', 'none');
        camlight;
        lighting gouraud;


        s = isosurface(plot_data.LonGrid, plot_data.LatGrid, ...
            -plot_data.DepthGrid,plot_data.datamask,0)
        x_point = 115.0733;
        z = -squeeze(plot_data.DepthGrid(1,1,:));
        y_point = 31.6;
        plot3(y_point * ones(size(z)), x_point * ones(size(z)), z, 'blue', 'LineWidth', 4);
        scatter3(y_point , x_point , 5, 'blue', 'LineWidth', 5);

        p = patch(s);
        set(p,'FaceColor',[0.5 0.5 0.5]);  
        set(p,'EdgeColor',[0.5 0.5 0.5]);
        camlight;
        lighting gouraud;

    else
        disp('Error: plot_data is not a structure.');
    end
    title_str = sprintf('Time = %s', datestr(plot_data.nTime(time_point), 'dd-mmm-yyyy'));
    title(title_str);
end






