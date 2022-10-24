clear
clc

close all

%% Reading in Video
vidReader=VideoReader('BallsRolling_Trim.mp4'); %Create an object with the video

%% Figure Windo for optical flow vectors
h=figure;
movegui();
hViewPanel=uipanel(h,'Position',[0 0 1 1],'Title','Plot of Optical Flow Vectors');
hPlot = axes(hViewPanel);


%% Parameters
VelThresh=0.075; %Threshold for velocity magnitudes that are not considered movement
MinObArea=25; %Used to remove all objects with less than this value of connected pixels
radmin=30;
radmax=100;
HughSensit=0.8;

%% Creating a optical flow object using Lucas-Kanade Method
opticFlow=opticalFlowLK('NoiseThreshold',0.001); %Increasing number means movement of object has less impact on calculation 
%opticFlow=opticalFlowHS('MaxIteration',10,'VelocityDifference',1);
%opticFlow=opticalFlowHS('MaxIteration',5);
%% Looping Through Video Frames

i=1;
Centroids={}; %Empty cell of centroid positions
while hasFrame(vidReader)
    frameRGB=readFrame(vidReader);
    frameGray=im2gray(frameRGB);
    flow=estimateFlow(opticFlow,frameGray);
    GrayNew=imopen(flow.Magnitude,strel('disk',1)); %Performs erosion then dilation (morph op)
    Mask=imbinarize(GrayNew,VelThresh); %Sets Pixels with velocity above threshold to 1
    
    Mask=imclose(Mask,strel('disk',3)); %Performs a dilation then an erosion (morph op)
    Mask=bwareaopen(Mask,MinObArea); %Any objects with area smaller than MinObArea are discarded
    [centers,radii,metric]=imfindcircles(Mask,[radmin radmax],'Sensitivity',HughSensit);
    
    imshow(Mask);
    viscircles(centers,radii,'EdgeColor','b');
    pause(0.001);

    % Insert Center into centroid cell array
    CentSizeNew=size(centers,1); %Number of rows are the number of centers of the new row
    if i==1
        %We are in the first frame, so we just plug in the centroid values
        for k=1:CentSizeNew
            Centroids{1,k}=centers(k,:);
        end
    else
        if CentSizeOld<CentSizeNew  %We have new centroids detected
            DistMeas=zeros(1,CentSizeNew);
            for k=1:CentSizeOld %Looping through each old centroid
                for j=1:CentSizeNew     %We check each new center against one old center
                    DistMeas(j)=sqrt((centers(j,1)-Centroids{i-1,k}(1))^2+(centers(j,2)-Centroids{i-1,k}(2))^2);
                end
                minVal=min(DistMeas);
                minIndex=find(DistMeas==minVal);
                Centroids{i,minIndex}=centers(minIndex,:);
            end
            Centroids{i,CentSizeOld+1:CentSizeNew}=[0 0];

        else %Number of new centroids equals old centroids
            DistMeas=zeros(1,CentSizeNew);
            for k=1:CentSizeOld %Looping through each old centroid
                for j=1:CentSizeNew     %We check each new center against one old center
                    DistMeas(j)=sqrt((centers(j,1)-Centroids{i-1,k}(1))^2+(centers(j,2)-Centroids{i-1,k}(2))^2);
                end
                minVal=min(DistMeas);
                minIndex=find(DistMeas==minVal);
                Centroids{i,minIndex}=centers(minIndex,:);
            end
        end


    end

    CentSizeOld=CentSizeNew;

    % Code below here is just me testing things

    %frameGray=122-frameGray; %This is for this particular dataset
    %imshow(frameGray);
    %frameGray=padarray(frameGray,[3,3],"both"); %Padding optical flow vector field (maybe can change this)
    
    %GrayNew=imclose(GrayNew,strel('disk',6));

    %Preprocessing Image
    %flow.Magnitude(flow.Magnitude<VelThresh)=0; 
    

    %[B,L]=bwboundaries(Mask,'noholes'); %Calculates the boundary of objects in the image
    %Mask=imfill(Mask,"holes");
    
    %Mask=imclose(Mask,strel('disk',2));
    %Mask=imclose(Mask,strel('rectangle',[3 3])); %Completes a dilation followed by an erosion
    %Mask=bwareaopen(Mask,MinObArea);
    %Mask=imclose(Mask,strel('disk',5));
    %Mask=imfill(Mask,"holes");
    %subplot(1,2,1);
    %imshow(Mask);
    %Mask=
    %Mask=imclose(Mask,strel('rectangle',[10 10])); %Completes a dilation followed by an erosion
    %Mask=imfill(Mask,'holes');
    %[centers,radii]=imfindcircles(Mask,[radmin radmax]);
    
    %Retain the 5 strongest circles
    %subplot(1,2,2);
    %imshow(Mask);
    %{
    hold on
    for k=1:length(B)
        boundary=B{k};
        plot(boundary(:,2),boundary(:,1),'b','LineWidth',2)
    end
    hold off
    %}
    %viscircles(centers, radii,'EdgeColor','b');
    %Now we use blob detection to find centroid of flow field components



    disp(i); i=i+1;
end

