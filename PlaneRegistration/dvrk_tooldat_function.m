function [robot_data_resamp,frame_vec, t_dvrk,time_us]=dvrk_tooldat_function(datapath_video,datapath_robot,plot)

%--------------------<US Video Processing>----------------------
vidReader=VideoReader(datapath_video);
sec_per_frame=vidReader.Duration/(vidReader.NumFrames-1);

time_us=[0:sec_per_frame:vidReader.Duration];
frame_vec=[0:1:vidReader.NumFrames]; %Vector of the frames

%------------------<Robot Data Processing           >------------------------------
Data=readmatrix(datapath_robot);

time_dvrk=Data(3:end,4);
psm1_pos=Data(3:end,6:8);
probe_angle=Data(3:end,end-1);


% Converting dvrk Time To Hours Minutes and Seconds
t_dvrk=datetime(time_dvrk,'ConvertFrom','epochtime','Epoch','1970-01-01','TicksPerSecond',1e6,'Format','HH:mm:ss.SSS');

%Reformatting dvrk time to start at time zero
min_time=min(t_dvrk);
t_dvrk=t_dvrk-min_time;
t_dvrk.Format='hh:mm:ss.SSS';
[~,~,~,~,~,t_dvrk]=datevec(t_dvrk);

%Reformatting dvrk time and tool data to match time in ascending order
robot_data=[psm1_pos,probe_angle];

nan_vals=isnan(t_dvrk)==1; %Logical array where NaN values are at
t_dvrk=t_dvrk(~nan_vals);
robot_data=robot_data(~nan_vals,:);

[t_dvrk,sort_ind]=sort(t_dvrk); %Sorts the dVRK time in ascending order
robot_data=robot_data(sort_ind,:); %Sorts dvrk data correspondingly

% --------------------<Interpolation>-------------------------
[t_dvrk,unique_ind,~]=unique(t_dvrk); %Eliminates duplicate time samples
robot_data=robot_data(unique_ind,:); 
robot_data_resamp=zeros(length(time_us),4);
robot_data_resamp(:,1)=interp1(t_dvrk,robot_data(:,1),time_us,'spline');
robot_data_resamp(:,2)=interp1(t_dvrk,robot_data(:,2),time_us,'spline');
robot_data_resamp(:,3)=interp1(t_dvrk,robot_data(:,3),time_us,'spline');
robot_data_resamp(:,4)=interp1(t_dvrk,robot_data(:,4),time_us,'pchip');
% robot_data_resamp(:,1)=resample(robot_data(:,1),t_dvrk,sec_per_frame);

%---------------------<Plotting Results>------------------------
if plot==1
    figure;
    subplot(2,2,1);
    plot(time_us,robot_data_resamp(:,1),'r',t_dvrk,robot_data(:,1),'b');
    title('x data resampled');

    subplot(2,2,2);
    plot(time_us,robot_data_resamp(:,2),'r',t_dvrk,robot_data(:,2),'b');
    title('y data resampled');
    subplot(2,2,3);
    plot(time_us,robot_data_resamp(:,3),'r',t_dvrk,robot_data(:,3),'b');
    title('z data resampled');
    subplot(2,2,4);
    plot(time_us,robot_data_resamp(:,4),'r',t_dvrk,robot_data(:,4),'b');
    title('roll angle data resampled');

    legend('Resampled','Actual');




end


end







