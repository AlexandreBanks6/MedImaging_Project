clear
clc
close all
pathname=['C:/Users/playf/OneDrive/Documents/UBC/Alexandre_UNI_2022_2023/Semester1'...
    '/ELEC_523_MedImaging/Project/MooreBanks_Results/Trial4/Registration/Recorder_2_Nov11_20-01-30.mp4'];
vidReader=VideoReader(pathname);
opticFlow=opticalFlowLK('NoiseThreshold',0.001); %Increasing number means movement of object has less impact on calculation
VelThresh=1; %Threshold for velocity magnitudes that are not considered movement
radmin=8;
radmax=30;
HughSensit=0.8;
MinObArea=30;
while hasFrame(vidReader)
    frame=readFrame(vidReader);
    %[frameCropped,rectout]=imcrop(frame); %Uncomment to adjust crop region
    %manually
    CropRec=[1607.51,85.51,883.98,740.98];
    frameCropped=imcrop(frame,CropRec); %Crops just the US portion of the image
    frameGray=rgb2gray(frameCropped);
    %Tubules=fibermetric(frameGray,100,"ObjectPolarity","bright"); %Uses Vesselness FIlter to Find lines
    %Tubules=imbinarize(Tubules);
    flow=estimateFlow(opticFlow,frameGray);
    Mask=imbinarize(flow.Magnitude,VelThresh);
    %Mask=imfill(Mask,'holes');
    %Mask=imclose(Mask,strel('disk',2));
    %Mask2=bwareaopen(Mask,MinObArea);  %Any objects with area smaller than MinObArea are discarded
    %Mask=imfill(Mask,'holes');
    Mask=imopen(Mask,strel('disk',2));
    %Mask=bwareaopen(Mask,floor((2*pi*(radmin^2))/2));
    [centroids,radii,metric]=imfindcircles(Mask,[radmin radmax],'Sensitivity',HughSensit);
    
    %imshow(labeloverlay(frameGray,Tubules));
    imshow(Mask);
    viscircles(centroids,radii);
    pause(0.01);
    %montage([Mask,Mask2,Mask3]);

end

