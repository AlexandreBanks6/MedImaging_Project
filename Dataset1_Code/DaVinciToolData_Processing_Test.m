%% Reading Data and Preprocessing
clear
clc
close all

datapath=['C:\Users\playf\OneDrive\Documents\UBC\Alexandre_UNI_2022_2023\' ...
    'Semester1\ELEC_523_MedImaging\Project\MooreBanks_Results\Trial4\Registration\data.csv'];
Data=readmatrix(datapath);
Data2=readtable(datapath);


frame_no=Data(2:end,2);
time_us=Data2(1:end,3);
time_us=table2array(time_us);
time_dvrk=Data(3:end,4);
psm1_pos=Data(3:end,6:8);
probe_angle=Data(3:end,end-1);

%Removing Gaps
time_us=time_us(~ismissing(time_us));
frame_no=frame_no(~isnan(frame_no));

% Converting dvrk Time To Hours Minutes and Seconds
t_dvrk=datetime(time_dvrk,'ConvertFrom','epochtime','Epoch','1970-01-01','TicksPerSecond',1e6,'Format','HH:mm:ss.SSS');

%Reformatting dvrk time to start at time zero
min_time=min(t_dvrk);
t_dvrk=t_dvrk-min_time;
t_dvrk.Format='hh:mm:ss.SSS';
[~,~,~,~,~,t_dvrk]=datevec(t_dvrk);

%Reformatting US Time to Seconds Vector
t_us=format_time_char(time_us); %Returns a char array where ' ' are changed to '0' and ':' changed to '.'
t_us=convert_chartime_todouble(t_us); %Converts the char array to a double vector
t_us=t_us-t_us(1); %Zeros the time measurements


%% DVRK Position to Frame Correlation
total_frame_num=frame_no(end)-frame_no(1);
robot_data=[]; %Array of x,y,z,rotorangle
time_vec=[]; %array of time vector
nan_vals=find(isnan(t_dvrk)==1);
nan_vals=[0;nan_vals];
for i=1:length(frame_no)-1
    frame_diff=frame_no(i+1)-frame_no(i); %Number of missing frames
    %Finding and re-arranging segment
    time_seg=t_dvrk(nan_vals(i)+1:nan_vals(i+1)-1);
    
    %Sorting Time Segment into increasing Order
    [time_seg,sort_ind]=sort(time_seg);

    %Sorting tool data segment
    robot_data_seg=[psm1_pos(nan_vals(i)+1:nan_vals(i+1)-1,:),probe_angle(nan_vals(i)+1:nan_vals(i+1)-1,:)];
    robot_data_seg=robot_data_seg(sort_ind,:);
    
    %Finding number of steps between existing frames
    frame_steps=floor(length(time_seg)/frame_diff);
    
    for j=[1:frame_diff] %Loops for the number of frames between missing frames
        time_vec=[time_vec;time_seg((j-1)*frame_steps+1)];
        robot_data=[robot_data;robot_data_seg((j-1)*frame_steps+1,:)];

    end

end

%Adding final segment
time_seg=t_dvrk(nan_vals(i)+1:nan_vals(i+1)-1);
[time_seg,sort_ind]=sort(time_seg);
robot_data_seg=[psm1_pos(nan_vals(i)+1:nan_vals(i+1)-1,:),probe_angle(nan_vals(i)+1:nan_vals(i+1)-1,:)];
robot_data_seg=robot_data_seg(sort_ind,:);


%Return Values

time_vec=[time_vec;time_seg(1)];
robot_data=[robot_data;robot_data_seg(1,:)];
frame_vec=[frame_no(1):frame_no(end)]';




%{
To Convert from UNIX to current time do:
Ut=...; %Unix Time
d =
datetime(UnixTime1,'ConvertFrom','epochtime','Epoch','1970-01-01','TicksPerSecond',1e6,'Format','dd-MMM-yyyy HH:mm:ss.SSS');

%Then to extract hours, minutes, seconds, do the following:
[h,m,s]=hms(d);


%}



%% Function Definitions


function t_new=format_time_char(time_us)
t_new=cell2mat(time_us); %converts US time to char array
t_new=t_new(:,[end-5:end]);
[row,col]=size(t_new);
for i=[1:row]
    for j=[1:col]
        if t_new(i,j)==' '
            t_new(i,j)='0';
        end
        if t_new(i,j)==':'
            t_new(i,j)='.';
        end
    end

end



end

function t_new=convert_chartime_todouble(time_us)

[row,col]=size(time_us);
t_new=zeros(row,1);
for i=[1:row]
    t_new(i)=str2double(time_us(i,:));
end
end