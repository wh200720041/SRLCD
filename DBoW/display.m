clear;
close;

ground_truth_path = 'C:\\Users\\owner\\Desktop\\Dataset\\KITTI\\sequence00\\GroundTruth\\';
is_KITTI = true;
text_path ='saliency3.txt';
video_para.frameNum = 4540;


if(is_KITTI == true)
    txt_end ='.txt';
    groundtruth = [];
    for i =1:video_para.frameNum

        file_name =strcat(ground_truth_path,sprintf('%010d', i),txt_end);
        fid = fopen(file_name);
        line = textscan(fid,'%f %f');
        line = [line{:}];
        y_temp(i) = line(1,1)*10000;
        x_temp(i) = line(1,2)*10000;
        fclose(fid);

    end

    mean(x_temp)
    x = x_temp' - mean(x_temp)';
    y = y_temp' - mean(y_temp)';
    figure;
    plot(x,y);


    if(exist(text_path,'file')==2)
        hold on;
        loop_closure_file = fopen(text_path);
        points_pair = textscan(loop_closure_file,'t1=%d\tcoincides with t2=%d\r\n');
        points_pair_old = points_pair{1,2};
        points_pair_new = points_pair{1,1};
        num = size(points_pair_new,1);
        fclose(loop_closure_file);

        for i=1:num
            plot(x(points_pair_new(i)),y(points_pair_new(i)),'r','Marker','o','MarkerSize',7);
            %new points pair is the loop closure place
            %plot(x(points_pair_old(i)),y(points_pair_old(i)),'r','Marker','x','MarkerSize',7);
        end
        %{
        for i=1570:1635
            plot(x(i),y(i),'k','Marker','o','MarkerSize',7);
        end
        for i=2345:2460
            plot(x(i),y(i),'k','Marker','o','MarkerSize',7);
        end
        for i=3288:3845
            plot(x(i),y(i),'k','Marker','o','MarkerSize',7);
        end
        for i=4450:4520
            plot(x(i),y(i),'k','Marker','o','MarkerSize',7);
        end
        %}
        %plot(x(3131),y(3131),'b','Marker','x','MarkerSize',7);
        hold off;
    end

elseif(is_TUM == true)
    fileName = strcat(ground_truth_path,'groundtruth.txt');
    fid = fopen(fileName);
    line = textscan(fid,'%f %f %f %f %f %f %f %f');
    line = [line{:}];
    [ground_truth_num,col] = size(line);
    if(video_para.frameNum>ground_truth_num)
        error_msg = 'video number less than ground truth number'
    else
        for k =1:ground_truth_num
            x_temp(k) = line(k,2);
            y_temp(k) = line(k,3);
            z_temp(k) = line(k,4);
            if(x_temp(k) == NaN ||y_temp(k) == NaN||z_temp(k) == NaN)
                x_temp(k) = 0;
                y_temp(k) = 0;
                z_temp(k) = 0;
            end
        end
        plot(x_temp,y_temp);
        grid on;
        hold on;
        %5649-5400

        %74
        if(exist(text_path,'file')==2)
            loop_closure_file = fopen(text_path);
            points_pair = textscan(loop_closure_file,'t1=%f\tcoincides with t2=%f\tframe1:%d\tframe2:%d with img1=%d img2=%d\r\n');
            points_pair_old = points_pair{1,4};
            points_pair_new = points_pair{1,3};
            num = size(points_pair_new,1);
            fclose(loop_closure_file);
            for i=1:num
                plot(x_temp(uint16(points_pair_new(i)*ground_truth_num/video_para.frameNum)),y_temp(uint16(points_pair_new(i)*ground_truth_num/video_para.frameNum)),'r','Marker','o','MarkerSize',7);
                plot(x_temp(uint16(points_pair_old(i)*ground_truth_num/video_para.frameNum)),y_temp(uint16(points_pair_old(i)*ground_truth_num/video_para.frameNum)),'r','Marker','x','MarkerSize',7);
            end
            %plot(x_temp(2000),y_temp(2000),'b','Marker','x','MarkerSize',7);
        end
    hold off;


    end
elseif(is_ETH == true)
    filename = strcat(ground_truth_path,'GroundTruthAGL.csv');
    delimiter = ',';
    startRow = 2;
    formatSpec = '%f%f%f%f%f%f%f%f%f%f%[^\n\r]';
    fileID = fopen(filename,'r');
    dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines' ,startRow-1, 'ReturnOnError', false);
    fclose(fileID);
    %imgid = dataArray{:, 1};
    x_gt = dataArray{:, 2};
    y_gt = dataArray{:, 3};
    z_gt = dataArray{:, 4};
    x_temp = x_gt - mean(x_gt);
    y_temp = y_gt - mean(y_gt);
    z_temp = z_gt - mean(z_gt);
    %omega_gt = dataArray{:, 5};
    %phi_gt = dataArray{:, 6};
    %kappa_gt = dataArray{:, 7};
    %x_gps = dataArray{:, 8};
    %y_gps = dataArray{:, 9};
    %z_gps = dataArray{:, 10};
    %clearvars filename delimiter startRow formatSpec fileID dataArray ans;
    %plot3(x_temp, y_temp, z_temp, '.')
    [ground_truth_num,col] = size(x_temp);
    plot(x_temp, y_temp);
    grid on;
    hold on;
    if(exist(text_path,'file')==2)
        loop_closure_file = fopen(text_path);
        points_pair = textscan(loop_closure_file,'t1=%f\tcoincides with t2=%f\tframe1:%d\tframe2:%d with img1=%d img2=%d\r\n');
        points_pair_old = points_pair{1,4};
        points_pair_new = points_pair{1,3};
        num = size(points_pair_new,1);
        fclose(loop_closure_file);
        for i=1:num
            plot(x_temp(uint16(points_pair_new(i)*ground_truth_num/video_para.frameNum/30)),y_temp(uint16(points_pair_new(i)*ground_truth_num/video_para.frameNum/30)),'r','Marker','o','MarkerSize',7);
            plot(x_temp(uint16(points_pair_old(i)*ground_truth_num/video_para.frameNum/30)),y_temp(uint16(points_pair_old(i)*ground_truth_num/video_para.frameNum/30)),'r','Marker','x','MarkerSize',7);
        end
    end
    hold off;

elseif(is_CC == true)
    file_name =strcat(ground_truth_path,'result.mat');
    load(file_name,'GPS');
    num = size(GPS,1);
    for i=1:video_para.frameNum*2
        if(mod(i,2)==1)
            x_temp((i+1)/2) = GPS(i,1);
            y_temp((i+1)/2) = GPS(i,2);
        end
    end
    x = x_temp(:) - mean(x_temp(:));
    y = y_temp(:) - mean(y_temp(:));

    loop_closure_file = fopen(text_path);
    points_pair = textscan(loop_closure_file,'t1=%f\tcoincides with t2=%f\tframe1:%d\tframe2:%d with img1=%d img2=%d\r\n');
    points_pair_old = points_pair{1,4};
    points_pair_new = points_pair{1,3};
    point_num = size(points_pair_new,1);
    fclose(loop_closure_file);
    %{
    680-731
    795-1210
    %}

    figure;
    plot(x,y);
    hold on;
    for i=1:point_num
        plot(x(points_pair_new(i)),y(points_pair_new(i)),'r','Marker','o','MarkerSize',7);
        plot(x(points_pair_old(i)),y(points_pair_old(i)),'r','Marker','x','MarkerSize',7);
    end
    %plot(x(240),y(240),'b','Marker','x','MarkerSize',7);
    hold off;

elseif(is_NC == true)
    file_name =strcat(ground_truth_path,'result.mat');
    load(file_name,'GPS');
    num = size(GPS,1);
    for i=1:video_para.frameNum*2
        if(mod(i,2)==1)
            x_temp((i+1)/2) = GPS(i,1);
            y_temp((i+1)/2) = GPS(i,2);
        end
    end
    x = x_temp(:) - mean(x_temp(:));
    y = y_temp(:) - mean(y_temp(:));

    loop_closure_file = fopen(text_path);
    points_pair = textscan(loop_closure_file,'t1=%f\tcoincides with t2=%f\tframe1:%d\tframe2:%d with img1=%d img2=%d\r\n');
    points_pair_old = points_pair{1,4};
    points_pair_new = points_pair{1,3};
    point_num = size(points_pair_new,1);
    fclose(loop_closure_file);
    %{
    680-731
    795-1210
    %}

    figure;
    plot(x,y);
    hold on;
    for i=1:point_num
        plot(x(points_pair_new(i)),y(points_pair_new(i)),'r','Marker','o','MarkerSize',7);
        plot(x(points_pair_old(i)),y(points_pair_old(i)),'r','Marker','x','MarkerSize',7);
    end
    %plot(x(240),y(240),'b','Marker','x','MarkerSize',7);
    hold off;

end

