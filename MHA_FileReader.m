%Script for reading in MHA files into MATLAB 
%Run script to: load video data into workspace as V (where the first two
%columns are image info, and column 3 represents time index), or use this
%script to save the video to a .mp4 file 
mhaFile = "UltrasoundTest_ToolPoke.mha"; %filename
data = mha_read_header(mhaFile);
V = mha_read_volume(data);
[x, y, z] = size(V); 

%to access a single frame at frame t 
t = 1
Im = V(:,:,t)
imshow(flip(imrotate(Im, -90), 2)); %produces the correct orientation as seen in the image 

%write to video file MP4 
 vidfile = VideoWriter('UltrasoundTest_ToolPoke.mp4','MPEG-4');
 vidfile.FrameRate = 10;
 open(vidfile);
 for i = 1:400
    
    im = flip(imrotate(V(:,:,i), -90), 2); %produces the correct orientation as seen in the image 
    writeVideo(vidfile, im);

 end
close(vidfile)



