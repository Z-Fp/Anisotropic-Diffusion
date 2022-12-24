%1399/08/18
%using LSP features to detect real and fake fingerprint
%120 samples for training and 400 samples for testing
close all
clear all
clc
%read input images(for train)
imagepath5='train';
filelist5=dir(fullfile(imagepath5,'*.bmp'));
list5={filelist5.name};
for i=1:length(list5)
    img5{i,1}=imresize(imread(fullfile(imagepath5,list5{i})),[96 96]); 
    
end
data_train=[img5];
for i=1:120
    input1=data_train{i,1};
    inputt=double(input1);
    xx=[];
%bigin anisotropic diffusion algorithm
%set the number of updates of the AD image
    for iter=1:25
        %compute gradient image
        %north gradient
        k=40;
        north=zeros(size(input1,1),size(input1,2));
        north(2:end,1:end)=input1(1:end-1,1:end);
        north(1,:)=input1(1,:);
        deltan=north-inputt;
        deltann{i,1}=deltan;
        %south gradient
        south=zeros(size(input1,1),size(input1,2));
        south(1:end-1,1:end)=input1(2:end,1:end);
        south(end,:)=input1(end,:);
        deltas=south-inputt;
        deltass{i,1}=deltas;
        %west gradient
        west=zeros(size(input1,1),size(input1,2));
        west(:,2:end)=input1(:,1:end-1);
        west(:,1)=input1(:,1);
        deltaw=west-inputt;
        deltaww{i,1}=deltaw;
        %east gradient
        east=zeros(size(input1,1),size(input1,2));
        east(:,1:end-1)=input1(:,2:end);
        east(:,end)=input1(:,end);
        deltae=east-inputt;
        deltaee{i,1}=deltae;
        %calculate diffusion coefficients
        cn{i,1}=exp(-(deltann{i,1}./k).^2);
        cs{i,1}=exp(-(deltass{i,1}./k).^2);
        cw{i,1}=exp(-(deltaww{i,1}./k).^2);
        ce{i,1}=exp(-(deltaee{i,1}./k).^2);
        %update the image on this iteration
        input_plus_1{i,1}=inputt+(cn{i,1}.*deltann{i,1}+cs{i,1}.*deltass{i,1}+cw{i,1}.*deltaww{i,1}+ce{i,1}.*deltaee{i,1});
        %update input
        input1=input_plus_1{i,1};
    end
end
for i=1:120
resize=imresize(input_plus_1{i,1},[96 96]);
input2{i,1}=resize;
end
for i=1:120
    xx=[xx,reshape(input2{i,1},9216,1)];  
end
xdata=[xx]';
%label
for q=1:60
    group{q,1}='real';
end
for q=61:120
    group{q,1}='fake';
end
%svm struct
svmStruct= svmtrain(xdata,group,'kernel_function','rbf','rbf_sigma',207,'showplot',false);
%testing 
%read input images
%input images for testing
imagepath1='test';
filelist1=dir(fullfile(imagepath1,'*.bmp'));
list1={filelist1.name};
for i=1:length(list1)
    img1{i,1}=imresize(imread(fullfile(imagepath1,list1{i})),[96 96]); 
    
end
data_test=[img1];
for i=1:400
%produce distort images
Mask1=(rand(96,96)<0.15);
MissImage1{i,1}=double(img1{i,1}).*double(Mask1);
end
for i=1:400
     input11=MissImage1{i,1};
    inputt=double(input11);
    zz=[];
%bigin anisotropic diffusion algorithm
%set the number of updates of the AD image
    for iter=1:25
        %compute gradient image
        %north gradient
        k=40;
       north=zeros(size(input11,1),size(input11,2));
        north(2:end,1:end)=input11(1:end-1,1:end);
        north(1,:)=input11(1,:);
        deltan=north-inputt;
        deltann{i,1}=deltan;
        %south gradient
        south=zeros(size(input11,1),size(input11,2));
        south(1:end-1,1:end)=input11(2:end,1:end);
        south(end,:)=input11(end,:);
        deltas=south-inputt;
        deltass{i,1}=deltas;
        %west gradient
        west=zeros(size(input11,1),size(input11,2));
        west(:,2:end)=input11(:,1:end-1);
        west(:,1)=input11(:,1);
        deltaw=west-inputt;
        deltaww{i,1}=deltaw;
        %east gradient
        east=zeros(size(input11,1),size(input11,2));
        east(:,1:end-1)=input11(:,2:end);
        east(:,end)=input11(:,end);
        deltae=east-inputt;
        deltaee{i,1}=deltae;
        %calculate diffusion coefficients
        cn{i,1}=exp(-(deltann{i,1}./k).^2);
        cs{i,1}=exp(-(deltass{i,1}./k).^2);
        cw{i,1}=exp(-(deltaww{i,1}./k).^2);
        ce{i,1}=exp(-(deltaee{i,1}./k).^2);
        %update the image on this iteration
        input_plus_11{i,1}=inputt+(cn{i,1}.*deltann{i,1}+cs{i,1}.*deltass{i,1}+cw{i,1}.*deltaww{i,1}+ce{i,1}.*deltaee{i,1});
        %update input
        input11=input_plus_11{i,1};
    end
end
for i=1:400
resize11=imresize(input_plus_11{i,1},[96 96]);
input22{i,1}=resize11;
end
for i=1:400
   zz=[zz,reshape(input22{i,1},9216,1)];  
end
sample=[zz]';
%svm test
Test = svmclassify(svmStruct,sample,'showplot',false);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%f-score
for i=1:300
    actual{i,1}=[0];
end
for i=301:400
    actual{i,1}=[1];
end
for i=1:400
    if Test{i,1}=='real'
        predicted{i,1}=[0];
    else
        predicted{i,1}=[1];
    end
end
ACTUAL=(cell2mat(actual));
PREDICTED=(cell2mat(predicted));
result=fscore(ACTUAL,PREDICTED);
