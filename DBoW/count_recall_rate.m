
text_path ='saliency2.txt';
%text_path ='result_set/KITTI_2/result.txt';
%text_path ='result_set/CityCenter_R/result.txt';
%text_path ='result_set/CityCenter_L/result.txt';
%text_path ='result_set/NewCollege_L/result.txt';
%text_path ='result_set/NewCollege_L/result.txt';
loop_closure_file = fopen(text_path);
%points_pair = textscan(loop_closure_file,'t1=%f\tcoincides with t2=%f\tframe1:%d\tframe2:%d\r\n');
points_pair = textscan(loop_closure_file,'t1=%d\tcoincides with t2=%d\r\n');
points_pair_old = points_pair{1,2};
points_pair_new = points_pair{1,1};
num = size(points_pair_new,1);
final_num=1;
for i=2:num     
    if(points_pair_new(i)~=points_pair_new(i-1))
        final_num=final_num+1;
    end
end
fclose(loop_closure_file);
final_num


%{
3390-1440 / 3 =

some stats
Quater 0.1
final num 497 
964s

Quarter 0.08
final num 460 
835s

Quarter 0.05
final num 413 
572s

Quarter 0.03
final num 345 
402s
%}