function [ID, CP, HP, stardust, level, cir_center] = pokemon_stats (img, model)
% Please DO NOT change the interface
% INPUT: image; model(a struct that contains your classification model, detector, template, etc.)
% OUTPUT: ID(pokemon id, 1-201); level(the position(x,y) of the white dot in the semi circle); cir_center(the position(x,y) of the center of the semi circle)
img_path_1 = './train/';
img_dir_1 = dir([img_path_1,'*CP*']);
img_num_1 = length(img_dir_1);

img_path = './val/';
img_dir = dir([img_path,'*CP*']);
img_num = length(img_dir);

ID_gt = zeros(img_num_1,1);
CP_gt = zeros(img_num_1,1);
HP_gt = zeros(img_num_1,1);
stardust_gt = zeros(img_num,1);
ID = zeros(img_num,1);
CP = zeros(img_num,1);
HP = zeros(img_num,1);
stardust = zeros(img_num,1);

% Replace these with your code
im_gray = rgb2gray(img);
pikachu = load('model.mat');
%raichu = load('model2.mat');
%snorlax = load('model3.mat');
%pidgey = load('model_pokemon.mat');
%squirtle = load('modelSD.mat');
%bulbasaur = load('model_svm_id.mat');

%kakuna = load('modelSD1.mat');
%dratini = load('modelHP.mat');
%dragonite = load('modelHP1.mat');
%ratata = load('modelCP.mat');
%raticate = load('modelCP1.mat');

%svmStruct = fitcecoc(pikachu.feature_histogram,raichu.ID_gt);
points = detectSURFFeatures(im_gray);
points_strongest = selectStrongest(points,500);
[features,valid_points] = extractFeatures(im_gray,points_strongest,'Method','SURF','SURFSize',64);
features1_img = features;
A = pikachu.feature_histogram;
B = pikachu.ID_gt;
%
feature_histogram_test = zeros(1,100);
Mdl=fitcknn(A,B);
for p=1:size(features1_img)
f=features1_img(p,:);
dist = pdist2(f,pikachu.codeBook1,'euclidean');
[minimumdist,codebook_id]=min(dist);
feature_histogram_test(1,codebook_id)=feature_histogram_test(1,codebook_id)+1;
end


Mdl1 = fitcknn(pikachu.feature_histogramSD,pikachu.stardust_gt);
%ID = predict(svmStruct,(feature_img));
Mdl2 = fitcknn(pikachu.feature_histogramHP,pikachu.HP_gt);

Mdl3 = fitcknn(ratata.feature_histogramCP,pikachu.CP_gt);

feature_histogram1=zeros(1,100);
feature_histogram2=zeros(1,100);
feature_histogram3=zeros(1,100);


r_gray = rgb2gray(img);
%imshow(r);
l=imresize(r_gray,[720 350]);
%imshow(l);
k=imcrop(l,[180 567 100 50] );
points = detectSURFFeatures(k);
[features1 ,valid_points1] = extractFeatures(k,points);
features1_img1 = features1;


l_hp = imcrop(l,[127 354 100 50]);
points_hp=detectSURFFeatures(l_hp);
[features_hp, valid_points_hp] = extractFeatures(l_hp,points_hp);
features1_hp = features_hp;

c_CP =imcrop(l,[90 7 300 100]);
points_cp = detectSurfFeatures(c_CP);
[features_img_CP, validpointsCP] = extractFeatures(c_CP, points_cp);
features1_CP = features_img_CP;

for q=1:size(features1_img1)
f1=features1_img(q,:);
dist1 = pdist2(f1,pikachu.codeBookSD,'euclidean');
[minimumdist1,codebook_id1]=min(dist1);
feature_histogram1(1,codebook_id1)=feature_histogram1(1,codebook_id1)+1;
end


for r=1:size(features1_hp)
f2=features1_hp(r,:);
dist1 = pdist2(f2,pikachu.codeBookHP,'euclidean');
[minimumdist2,codebook_id2]=min(dist1);
feature_histogram2(1,codebook_id2)=feature_histogram2(1,codebook_id2)+1;
end


for s=1:size(features1_hp)
f3=features1_CP(s,:);
dist1 = pdist2(f3,pikachu.codeBookHP,'euclidean');
[minimumdist3,codebook_id3]=min(dist1);
feature_histogram3(1,codebook_id3)=feature_histogram3(1,codebook_id3)+1;
end


predicted_label = predict(Mdl,feature_histogram_test);


predicted_label_1 = predict(Mdl1,feature_histogram1);

predicted_label_2 = predict(Mdl2,feature_histogram2);

predicted_label_3 = predict(Mdl3,feature_histogram3);

[x_axis ,y_axis, dimension] = size(img);
img_crop = imcrop(r_gray,[6 0 344 350];
[centers,radii]=imfindcircles(img_crop);


ID =120 - predicted_label;
CP = predicted_label_3;
HP = predicted_label_2;
stardust = predicted_label_1;
level = [centers(1,1) ,centers(1,2)];
cir_center = [x_axis/2,y_axis/3];

end
