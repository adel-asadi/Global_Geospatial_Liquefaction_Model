% All Sectiones of Codes

%%
% Geospatial Liquefaction Model - Data Preparation
% Adel Asadi

%%
% Reading CSV file as a table

EQ_Data_All=readtable('C:\Users\aasadi01\Box\MTU Home Drive\Downloads\Geospatial_Liquefaction\02_Codes\Final_Complete_Data\All_EQ_Data.csv');

%%
% Loading MAT variables

load('EQ_Regions')

%%
% Modifying Dataset

%EQ_Data_All(:,1)=EQ_Data_All(:,1)+1;
EQ_ID=zeros(299117,1);
EQ_Region=zeros(299117,1);
EQ_Region_Cat=zeros(299117,1);

for i=1:54
    i
    for j=1:299117
        if strcmpi(table2cell(EQ_Data_All(j,"eqname")),table2cell(EQ_Regions(i,"EQ_Name")))==1
            EQ_ID(j)=i;
            %EQ_Region(j)=table2cell(EQ_Regions(i,"Region"));
            EQ_Region_Cat(j)=table2array(EQ_Regions(i,"Region_Category"));
        end
    end
end

%strcmpi(EQ_Data_All(1,"eqname"),EQ_Regions(1,"EQ_Name"))

%%
% Adding new variables

Full_Table = addvars(EQ_Data_All,EQ_ID,EQ_Region_Cat,'Before','emag');

%%
% Adding coastal/non-coastal index variable

EQ_coastal=zeros(299117,1);

for i=1:54
    eq_ind=find(EQ_ID==i);
    EQ_coastal(eq_ind)=table2array(EQ_Regions(i,"Coastal"));
end

Full_Table = addvars(Full_Table,EQ_coastal,'Before','emag');

%%
% Fixing categorical variables

landform_cat=zeros(299117,1);

cat_var_33=unique(Full_Table.landform);
for i=1:15
    cat_ind=find(Full_Table.landform==cat_var_33(i));
    landform_cat(cat_ind)=i;
end

Full_Table = addvars(Full_Table,landform_cat,'Before','landform');

%%
% Removing unuseful variables

Full_Table.id=Full_Table.id+1;

Full_Table.elev_std = [];
Full_Table.PGA_std = [];
Full_Table.PGV_std = [];
Full_Table.eqname = [];
Full_Table.landform = [];

%%
% Writing final data into excel sheet

filename = 'Final_Data_MAT.xlsx';
writetable(Full_Table,filename,'Sheet',1,'Range','A1')

%%
% Loading data

load('EQ_Regions')
load('Full_Table')
load('Var_names')

%%
% Variable names

var_names=Full_Table.Properties.VariableNames;

for i = 1:35
  %eval(['var_' num2str(i) '=' table2array(Full_Table(:,i))]);
  eval(['Var_' num2str(i) '=table2array(Full_Table(:,i));']);
end

%%
% Array data creation

Full_Array=table2array(Full_Table);

%%
%Z-score data normalization

Z1 = zscore(Full_Array(:,8:end));
Z2 = zscore(Full_Array(:,8:end),1);

%%
% outlier detection and removal

[B1,TFrm1] = rmoutliers(Z1,"gesd","MaxNumOutliers",3000);
B_2 = rmoutliers(Full_Array(:,8:end),"grubbs","ThresholdFactor",0.01);
B_3 = rmoutliers(Full_Array(:,8:end),"movmedian",1000);
B_4 = rmoutliers(Full_Array(:,8:end),"percentiles",[1,99]);
B_5 = rmoutliers(Full_Array(:,8:end),"median","ThresholdFactor",5);

%%
%Testing outlier impact

T_Z1=Full_Array(:,7);
T_B1=Full_Array(~TFrm1,7);

EQ_ID_Out=Full_Array(~TFrm1,2);
EQ_ID_In=Full_Array(TFrm1,2);
EQ_ID_All=Full_Array(:,2);

%%
% Defining data indicators

Full_Array=table2array(Full_Table);

sample_loc_ID_global=Full_Array(:,1:7);
input_vars_global=Full_Array(:,8:end);
label_liq_global=Full_Array(:,7);

%%
% Global data indicators

for i=1:54
    global_eq=find(sample_loc_ID_global(:,2)==i);
    global_ind_eq{i}=sample_loc_ID_global(global_eq,1);
    global_etc=find(sample_loc_ID_global(:,2)~=i);
    global_ind_etc{i}=sample_loc_ID_global(global_etc,1);
end

%%
% Coastal & non-coastal

coastal_eq_ind=find(Full_Array(:,4)==1);
coastal_eq=Full_Array(coastal_eq_ind,:);

sample_loc_ID_coastal=Full_Array(coastal_eq_ind,1:7);
input_vars_coastal=Full_Array(coastal_eq_ind,8:end);
label_liq_coastal=Full_Array(coastal_eq_ind,7);

noncoastal_eq_ind=find(Full_Array(:,4)==0);
noncoastal_eq=Full_Array(noncoastal_eq_ind,:);

sample_loc_ID_noncoastal=Full_Array(noncoastal_eq_ind,1:7);
input_vars_noncoastal=Full_Array(noncoastal_eq_ind,8:end);
label_liq_noncoastal=Full_Array(noncoastal_eq_ind,7);

%%
%Coastal & non-coastal indicators

for i=1:54
    coast_etc=find(sample_loc_ID_coastal(:,2)~=i);
    coastal_ind_etc{i}=sample_loc_ID_coastal(coast_etc,1);
    coast_eq=find(sample_loc_ID_coastal(:,2)==i);
    coastal_ind_eq{i}=sample_loc_ID_coastal(coast_eq,1);
end

for i=1:54
    noncoast_etc=find(sample_loc_ID_noncoastal(:,2)~=i);
    noncoastal_ind_etc{i}=sample_loc_ID_noncoastal(noncoast_etc,1);
    noncoast_eq=find(sample_loc_ID_noncoastal(:,2)==i);
    noncoastal_ind_eq{i}=sample_loc_ID_noncoastal(noncoast_eq,1);
end

%%
% Regional indicators

for i=1:54
    EQ_ind=find(Full_Array(:,2)==i);
    EQ_coords=Full_Array(EQ_ind,[5,6]);
    EQ_cntr_loc{i}=mean(EQ_coords);
    EQ_ind_etc=find(Full_Array(:,2)~=i);
    EQ_coords_etc=Full_Array(EQ_ind_etc,[1,5,6]);
    Dist_etc=[];
    for j=1:length(EQ_ind_etc(:,1))
        latlon1=EQ_cntr_loc{i};
        latlon2=EQ_coords_etc(j,[2,3]);
        [d1km d2km]=lldistkm(latlon1,latlon2);
        Dist_etc(j)=d1km;
    end
    close_points=find(Dist_etc<=4000);
    close_points_ind{i}=EQ_coords_etc(close_points,1);
end

%%
% Global data partitioning

large_liq_EQs_ID=find(sample_loc_ID_global(:,2)==5 | sample_loc_ID_global(:,2)==11 | sample_loc_ID_global(:,2)==12 | sample_loc_ID_global(:,2)==48);
large_liq_EQs_liq=find(sample_loc_ID_global(:,7)==1);
large_liq_EQs=intersect(large_liq_EQs_ID,large_liq_EQs_liq);
large_liq_EQs_ind=sample_loc_ID_global(large_liq_EQs,1);

for i=1:54
    liq_label_eq=label_liq_global(global_ind_etc{i});
    num_liq_extra{i}=sum(liq_label_eq)-(112666-length(find(sample_loc_ID_global(global_ind_eq{i},7)==0)));
    large_liq_EQs_mutual = intersect(large_liq_EQs_ind,global_ind_etc{i});
    if length(large_liq_EQs_mutual)<((2*num_liq_extra{i})-2)
        ind_removal1{i}=sample_loc_ID_global([large_liq_EQs_mutual(1:2:end);global_ind_eq{i}],1);
        ind_removal2{i}=sample_loc_ID_global([large_liq_EQs_mutual(2:2:end);global_ind_eq{i}],1);
    else
        ind_removal1{i}=sample_loc_ID_global([large_liq_EQs_mutual(1:2:(2*num_liq_extra{i}));global_ind_eq{i}],1);
        ind_removal2{i}=sample_loc_ID_global([large_liq_EQs_mutual(2:2:(2*num_liq_extra{i}));global_ind_eq{i}],1);
    end
end

%%
% Coastal data partitioning

for i=1:54

    liq_label_coast=label_liq_global(coastal_ind_etc{i});
    liq_label_noncoast=label_liq_global(noncoastal_ind_etc{i});
    num_liq_extra_coast{i}=sum(liq_label_coast)-(79410-length(find(sample_loc_ID_global(coastal_ind_eq{i},7)==0)));
    num_liq_extra_noncoast{i}=sum(liq_label_noncoast)-(33256-length(find(sample_loc_ID_global(noncoastal_ind_eq{i},7)==0)));
    large_liq_EQs_mutual_coast = intersect(large_liq_EQs_ind,coastal_ind_etc{i});
    large_liq_EQs_mutual_noncoast = intersect(large_liq_EQs_ind,noncoastal_ind_etc{i});

    if length(large_liq_EQs_mutual_coast)<((2*num_liq_extra_coast{i})-2)
        ind_removal1_coast{i}=sample_loc_ID_global([large_liq_EQs_mutual_coast(1:2:end);coastal_ind_eq{i}],1);
        ind_removal2_coast{i}=sample_loc_ID_global([large_liq_EQs_mutual_coast(2:2:end);coastal_ind_eq{i}],1);
    else
        ind_removal1_coast{i}=sample_loc_ID_global([large_liq_EQs_mutual_coast(1:2:(2*num_liq_extra_coast{i}));coastal_ind_eq{i}],1);
        ind_removal2_coast{i}=sample_loc_ID_global([large_liq_EQs_mutual_coast(2:2:(2*num_liq_extra_coast{i}));coastal_ind_eq{i}],1);
    end

    if length(large_liq_EQs_mutual_noncoast)<((2*num_liq_extra_noncoast{i})-2)
        ind_removal1_noncoast{i}=sample_loc_ID_global([large_liq_EQs_mutual_noncoast(1:2:end);noncoastal_ind_eq{i}],1);
        ind_removal2_noncoast{i}=sample_loc_ID_global([large_liq_EQs_mutual_noncoast(2:2:end);noncoastal_ind_eq{i}],1);
    else
        ind_removal1_noncoast{i}=sample_loc_ID_global([large_liq_EQs_mutual_noncoast(1:2:(2*num_liq_extra_noncoast{i}));noncoastal_ind_eq{i}],1);
        ind_removal2_noncoast{i}=sample_loc_ID_global([large_liq_EQs_mutual_noncoast(2:2:(2*num_liq_extra_noncoast{i}));noncoastal_ind_eq{i}],1);
    end
end

%%
% Regional data partitioning

for i=1:54
    liq_label_regional=label_liq_global(close_points_ind{i});
    num_liq_regional(i)=sum(liq_label_regional);
    num_nonliq_regional(i)=length(find(liq_label_regional==0));
    num_liq_extra_regional(i)=sum(liq_label_regional)-length(find(liq_label_regional==0));
end

%%
% Histogram comparison

Variable_col = 16;

Liq_var=Full_Array(1:186451,Variable_col);
NonLiq_var=Full_Array(186452:end,Variable_col);

figure;
%'BinWidth',1000
histogram(((Liq_var)));
hold on
histogram(((NonLiq_var)));
legend('Liquefaction','Non-Liquefaction')
axis tight
title('Landform Histogram Comparison')
%xlim([0 200])

%%
% Kernel Density

Variable_col = 23;

Liq_var=Full_Array(1:186451,Variable_col);
NonLiq_var=Full_Array(186452:end,Variable_col);

figure
[f1,xi1] = ksdensity(Liq_var);
[f2,xi2] = ksdensity(NonLiq_var);
plot(xi1,f1,xi2,f2)
legend('Liq','NonLiq');
title('TPI KDE')

%%
% Box plots

Variable_col = 31;

Liq_var=Full_Array(1:186451,Variable_col);
NonLiq_var=Full_Array(186452:end,Variable_col);
data_bp = Full_Array(:,Variable_col);

figure
g1 = repmat({'Liq'},length(Liq_var),1);
g2 = repmat({'NonLiq'},length(NonLiq_var),1);
g = [g1;g2];
boxplot(data_bp,g)
title('Magnitude Box Plot')

%%
% Scatter plots by group

Variable_cols=[19,18,29,30,31];

X = Full_Array(:,Variable_cols);

figure
gplotmatrix(X,[],Full_Array(:,7),['b' 'r'],[],[],false);

%%
% Categorical variables

Variable_col = 32;

tbl1_liq = tabulate(Full_Array(1:186451,Variable_col));
tbl2_nonliq = tabulate(Full_Array(186452:end,Variable_col));
t1 = array2table(tbl1_liq,'VariableNames',{'Value','Count','Percent'});
t2 = array2table(tbl2_nonliq,'VariableNames',{'Value','Count','Percent'});
t1.Value = categorical(t1.Value);
t2.Value = categorical(t2.Value);

figure
bar(t1.Value,t1.Count)
xlabel('Category')
ylabel('Frequency')
hold on
bar(t2.Value,t2.Count)
legend('Liq','NonLiq')

%%
% Correlation plots

%Variable_cols=[8,34,35]; % Load
Variable_cols=[17,18,19,20,21,22,23,24,25,26,27,28,29,30]; % Saturation
%Variable_cols=[9,10,11,12,13,14,15,16,31]; % Density

%corrplot(data_all_ga(:,[1,2,5,6,8,10,11,12,14]))
[rho,pval] = corr(Full_Array(:,Variable_cols));

figure
matvisual(rho, 'annotation')

%%
% Plotting histograms loop

for i=8:35
    Variable_col = i;
    Liq_var=Full_Array(1:186451,Variable_col);
    NonLiq_var=Full_Array(186452:end,Variable_col);
    figure(i);
    histogram(((Liq_var)));
    hold on
    histogram(((NonLiq_var)));
    legend('Liquefaction','Non-Liquefaction')
    hold off
end

%%
% Descriptive Statistics

Variable_col = 16;

a_min_liq=min(Full_Array(1:186451,Variable_col));
b_max_liq=max(Full_Array(1:186451,Variable_col));
c_mean_liq=mean(Full_Array(1:186451,Variable_col));
d_median_liq=median(Full_Array(1:186451,Variable_col));
e_mode_liq=mode(Full_Array(1:186451,Variable_col));
f_std_liq=std(Full_Array(1:186451,Variable_col));
g_skew_liq=skewness(Full_Array(1:186451,Variable_col));

a_min_nonliq=min(Full_Array(186452:end,Variable_col));
b_max_nonliq=max(Full_Array(186452:end,Variable_col));
c_mean_nonliq=mean(Full_Array(186452:end,Variable_col));
d_median_nonliq=median(Full_Array(186452:end,Variable_col));
e_mode_nonliq=mode(Full_Array(186452:end,Variable_col));
f_std_nonliq=std(Full_Array(186452:end,Variable_col));
g_skew_nonliq=skewness(Full_Array(186452:end,Variable_col));

%%
% Descriptive Statistics - PGA/PGV

Variable_col = 35;

exp_liq=exp(Full_Array(1:186451,Variable_col));
exp_nonliq=exp(Full_Array(186452:end,Variable_col));

a_min_liq=min(exp_liq);
b_max_liq=max(exp_liq);
c_mean_liq=mean(exp_liq);
d_median_liq=median(exp_liq);
e_mode_liq=mode(exp_liq);
f_std_liq=std(exp_liq);
g_skew_liq=skewness(exp_liq);

a_min_nonliq=min(exp_nonliq);
b_max_nonliq=max(exp_nonliq);
c_mean_nonliq=mean(exp_nonliq);
d_median_nonliq=median(exp_nonliq);
e_mode_nonliq=mode(exp_nonliq);
f_std_nonliq=std(exp_nonliq);
g_skew_nonliq=skewness(exp_nonliq);

%%
% Number of points per class/earthquake

for i=1:54
    EQ_class_num(i,1)=i;
    EQ_class_num(i,2)=length(find(Full_Array(1:186451,2)==i));
    EQ_class_num(i,3)=length(find(Full_Array(186452:end,2)==i));
end

mean_liq_num=round(mean(EQ_class_num(:,2)));
mean_nonliq_num=round(mean(EQ_class_num(:,3)));

%%
% Thresholds indexing

Prediction_with_threshold=ones(299117,1).*-999;

Vs30_ind_pred=find(Full_Array(:,9)<min(Full_Array(1:186451,9)));
elev_ind_pred=find(Full_Array(:,10)>max(Full_Array(1:186451,10)));
slope_ind_pred=find(Full_Array(:,11)>max(Full_Array(1:186451,11)));
PGA_ind_pred=find((Full_Array(:,34)>max(Full_Array(1:186451,34))) | (Full_Array(:,34)<min(Full_Array(1:186451,34))));
PGV_ind_pred=find((Full_Array(:,35)>max(Full_Array(1:186451,35))) | (Full_Array(:,35)<min(Full_Array(1:186451,35))));
TPI_ind_pred=find((Full_Array(:,13)>max(Full_Array(1:186451,13))) | (Full_Array(:,13)<min(Full_Array(1:186451,13))));
TRI_ind_pred=find(Full_Array(:,14)>max(Full_Array(1:186451,14)));
roughness_ind_pred=find(Full_Array(:,12)>max(Full_Array(1:186451,12)));
CTI_ind_pred=find(Full_Array(:,16)<min(Full_Array(1:186451,16)));
dc1_ind_pred=find(Full_Array(:,18)>max(Full_Array(1:186451,18)));
dr1_ind_pred=find(Full_Array(:,19)>max(Full_Array(1:186451,19)));
DL_ind_pred=find(Full_Array(:,20)>max(Full_Array(1:186451,20)));
dwb2_ind_pred=find(Full_Array(:,24)>max(Full_Array(1:186451,24)));
WBE_ind_pred=find(Full_Array(:,22)>max(Full_Array(1:186451,22)));
hwb2_ind_pred=find((Full_Array(:,25)>max(Full_Array(1:186451,25))) | (Full_Array(:,25)<min(Full_Array(1:186451,25))));
HAND_ind_pred=find(Full_Array(:,27)>max(Full_Array(1:186451,27)));
WTD_ind_pred=find(Full_Array(:,28)>max(Full_Array(1:186451,28)));
AI_ind_pred=find(Full_Array(:,29)>max(Full_Array(1:186451,29)));

magnitude_ind_pred=find(Full_Array(:,8)<5);

CTI_ind_pred_liq=find(Full_Array(:,16)>max(Full_Array(186452:end,16)));

nonliq_pre_threshold_ind=union(Vs30_ind_pred,union(elev_ind_pred,union(WBE_ind_pred,union(dr1_ind_pred,union(DL_ind_pred,union(dc1_ind_pred,union(slope_ind_pred,union(roughness_ind_pred,union(PGA_ind_pred,union(PGV_ind_pred,union(magnitude_ind_pred,union(WTD_ind_pred,union(HAND_ind_pred,union(hwb2_ind_pred,union(dwb2_ind_pred,union(CTI_ind_pred,union(TRI_ind_pred,union(TPI_ind_pred,AI_ind_pred))))))))))))))))));
Prediction_with_threshold(nonliq_pre_threshold_ind)=0;
Prediction_with_threshold(CTI_ind_pred_liq)=1;
label_temp_pre_thresholds=Full_Array(nonliq_pre_threshold_ind,7);

%%
% Feature transformations

Full_Array_trans=Full_Array;

Full_Array_trans(:,9)=real(log(Full_Array(:,9))); % Vs30
Full_Array_trans(:,24)=sqrt(Full_Array(:,24)); % DWB
Full_Array_trans(:,28)=sqrt(Full_Array(:,28)); % WTD
Full_Array_trans(:,13)=sqrt(abs(Full_Array(:,13))); % TPI
Full_Array_trans(:,14)=sqrt(abs(Full_Array(:,14))); % TRI
Full_Array_trans(:,27)=sqrt(Full_Array(:,27)); % HAND
Full_Array_trans(:,10)=sqrt(Full_Array(:,10)); % Elevation
Full_Array_trans(:,20)=sqrt(Full_Array(:,20)); % DL
Full_Array_trans(:,25)=sqrt(abs(Full_Array(:,25))); % HWB

%%
% Feature removal

Full_Array_trans_FR=Full_Array_trans;
Full_Array_trans_FR(:,[11,12,15,17,21,23,26,30,34])=[];

Full_Table_FR=Full_Table;
Full_Table_FR = removevars(Full_Table_FR,[11,12,15,17,21,23,26,30,34]);

%%
% Categforical variables

Full_Array_trans_FR(:,[23,24])=categorical(Full_Array_trans_FR(:,[23,24]));

%%
% Data normalization

Full_Array_trans_FR_norm=Full_Array_trans_FR;
Full_Array_trans_FR_norm(:,8:end)=normalize(Full_Array_trans_FR(:,8:end));
Full_Array_trans_FR_norm_sampled=Full_Array_trans_FR_norm;

%%
% Sampling for Feature Analysis

nonliq_num=112666-length(nonliq_pre_threshold_ind);
liq_num=186451-length(CTI_ind_pred_liq);
surplus_liq_num=liq_num-nonliq_num;

r=rand(1,surplus_liq_num);
[aa,c]=sort(r);
cc=round(c.*((146640-1)/surplus_liq_num));

ind_removal_fs=union(large_liq_EQs_ind(cc),nonliq_pre_threshold_ind);
Full_Array_trans_FR_norm_sampled(ind_removal_fs,:)=[];
label_sampled_fs=Full_Array_trans_FR_norm_sampled(:,7);

%%
% Saving variable names

Var_names_cell=table(Full_Table_FR.Properties.VariableNames);

for i=1:26
    Var_names{i}=Var_names_cell.Var1{1,i};
end

%%
% NCA feature selection

mdl = fscnca(Full_Array_trans_FR_norm_sampled(:,8:end),Full_Array_trans_FR_norm_sampled(:,7));

figure()
plot((mdl.FeatureWeights)','ro')
grid on
xlabel('Feature Index')
ylabel('Feature Weight')

%%
% Lambda parameter tuning

%Xtrain=Full_Array_trans_FR_norm_sampled(1:5:end,[9,10,11,12,13,23]); % Soil Density
Xtrain=Full_Array_trans_FR_norm_sampled(1:5:end,[14,15,16,17,18,19,20,21,22]); % Saturation

ytrain=Full_Array_trans_FR_norm_sampled(1:5:end,7);

cvp = cvpartition(ytrain,'kfold',5);
numvalidsets = cvp.NumTestSets;

n = length(ytrain);
lambdavals = linspace(0,20,20)/n;
lossvals = zeros(length(lambdavals),numvalidsets);

for i = 1:(length(lambdavals)/2)
    i
    for k = 1:numvalidsets
        X = Xtrain(cvp.training(k),:);
        y = ytrain(cvp.training(k),:);
        Xvalid = Xtrain(cvp.test(k),:);
        yvalid = ytrain(cvp.test(k),:);

        nca = fscnca(X,y,'FitMethod','exact', ...
             'Solver','sgd','Lambda',lambdavals(i*2), ...
             'IterationLimit',30,'GradientTolerance',1e-4, ...
             'Standardize',true);
                  
        lossvals(i,k) = loss(nca,Xvalid,yvalid,'LossFunction','classiferror');
    end
end

meanloss = mean(lossvals,2);

figure()
plot(lambdavals(2:2:end),meanloss(1:10),'ro-')
xlabel('Lambda')
ylabel('Loss (MSE)')
grid on

[~,idx] = min(meanloss(1:10)); % Find the index
bestlambda = lambdavals(idx*2); % Find the best lambda value

%%
% NCA full analysis

%Xtrain=Full_Array_trans_FR_norm_sampled(:,[9,10,11,12,13,23]); % Soil Density
Xtrain=Full_Array_trans_FR_norm_sampled(:,[14,15,16,17,18,19,20,21,22]); % Saturation

ytrain=Full_Array_trans_FR_norm_sampled(:,7);

ncaMdl = fscnca(Xtrain,ytrain,'FitMethod','average','NumPartitions',5,'Verbose',1,'Lambda',bestlambda,'Solver','sgd','Standardize',true);

figure()
plot(mean((ncaMdl.FeatureWeights)'),'ro')
grid on
hold on
%plot(0:6,ones(7)*0.5,'r','LineWidth',1) % Soil Density
%plot(0:9,ones(10)*0.5,'r','LineWidth',1) % Saturation
xlabel('Feature Index')
ylabel('Feature Weight')
%title('Soil Density Feature Selection by NCA Method')
title('Saturation Feature Selection by NCA Method')
hold off
%set(gca,'xtick',[1:14],'xticklabel',vars_1(:,1))

%%
% Selected features index

Features_selected_index=[1,2,3,4,5,6,7,8,9,10,14,15,22,24,26];

%%
% Data after feature selection

Full_Array_trans_FR_norm_FS=Full_Array_trans_FR_norm(:,Features_selected_index);
Full_Table_FR_FS=Full_Table_FR;
Full_Table_FR_FS = removevars(Full_Table_FR_FS,[11,12,13,16,17,18,19,20,21,23,25]);

%%
% Test Earthquake ID

%Test_EQ_ID = 5; % Bhuj
%Test_EQ_ID = 8; % Chi Chi
Test_EQ_ID = 11; % Christchurch
%Test_EQ_ID = 15; % Emilia
%Test_EQ_ID = 18; % Hokkaido
%Test_EQ_ID = 22; % Iwate
%Test_EQ_ID = 23; % Kobe
%Test_EQ_ID = 25; % Kumamoto
%Test_EQ_ID = 26; % Loma Prieta
%Test_EQ_ID = 32; % Nepal
%Test_EQ_ID = 33; % Nigata 1964
%Test_EQ_ID = 38; % Northridge
%Test_EQ_ID = 42; % Puget Sound 1949
%Test_EQ_ID = 48; % Tohoku

%%
% Training data index preparation

Test_EQ_ind=global_ind_eq{Test_EQ_ID};
test_predicted_label=ones(length(Test_EQ_ind),1).*-999;
test_predifined_nonliq = ismember(Test_EQ_ind,nonliq_pre_threshold_ind);
test_predicted_label(test_predifined_nonliq)=0;
Test_EQ_ind_remained=Test_EQ_ind(~test_predifined_nonliq);

Test_EQ_ind_etc=global_ind_etc{Test_EQ_ID};
test_predifined_nonliq_etc = ismember(Test_EQ_ind_etc,nonliq_pre_threshold_ind);
Test_EQ_ind_etc_remained=Test_EQ_ind_etc(~test_predifined_nonliq_etc);

%%
% Global balanced training data preparation

test_etc_label=Full_Array_trans_FR_norm_FS(Test_EQ_ind_etc_remained,7);
liq_num=sum(test_etc_label);
nonliq_num=length(test_etc_label)-liq_num;
surplus_liq_num=liq_num-nonliq_num;
number_needed_large=146640-surplus_liq_num;
num_classifiers_global=round(surplus_liq_num/number_needed_large)+1;

%%
% Global training and testing data

for i=1:num_classifiers_global
%r=rand(1,surplus_liq_num);
%[aa,c]=sort(r);
%cc{i}=round(c.*((146640-1)/surplus_liq_num));
%rng(i)
%r=rand(1,number_needed_large);
%[aa,c]=sort(r);
%cc{i}=round(c.*((146640-1)/number_needed_large));
large_liq_EQs_ind_etc=large_liq_EQs_ind;
large_liq_EQs_ind_etc(i:num_classifiers_global:end)=[];
%large_liq_EQs_ind_etc(cc{i})=[];
ind_removal_test_EQ{i}=union(large_liq_EQs_ind_etc,union(Test_EQ_ind_remained,nonliq_pre_threshold_ind));
Full_Array_trans_FR_norm_FS_training=Full_Array_trans_FR_norm_FS;
Full_Array_trans_FR_norm_FS_training(ind_removal_test_EQ{i},:)=[];
Full_Array_trans_FR_norm_FS_training_sets{i}=Full_Array_trans_FR_norm_FS_training;
label_test_eq_training_set{i}=Full_Array_trans_FR_norm_FS_training(:,7);
end

Full_Array_trans_FR_norm_FS_testing=Full_Array_trans_FR_norm_FS;
Full_Array_trans_FR_norm_FS_testing=Full_Array_trans_FR_norm_FS_testing(Test_EQ_ind_remained,:);

%%
% Global training sets

Full_Array_trans_FR_norm_FS_training_sets_1=Full_Array_trans_FR_norm_FS_training_sets{1};
Full_Array_trans_FR_norm_FS_training_sets_2=Full_Array_trans_FR_norm_FS_training_sets{2};
%Full_Array_trans_FR_norm_FS_training_sets_3=Full_Array_trans_FR_norm_FS_training_sets{3};
%Full_Array_trans_FR_norm_FS_training_sets_4=Full_Array_trans_FR_norm_FS_training_sets{4};

%%
% Regional training set

regional_ind_test=close_points_ind{Test_EQ_ID};
regional_repetition_ind=ismember(regional_ind_test,union(Test_EQ_ind_remained,nonliq_pre_threshold_ind));
Full_Array_trans_FR_norm_FS_training_sets_R=Full_Array_trans_FR_norm_FS(regional_ind_test(~regional_repetition_ind),:);
liq_num_R=sum(Full_Array_trans_FR_norm_FS_training_sets_R(:,7));
nonliq_num_R=length(Full_Array_trans_FR_norm_FS_training_sets_R(:,7))-liq_num_R;
surplus_liq_num_R=liq_num_R-nonliq_num_R;
r=rand(1,surplus_liq_num_R);
[aa,c]=sort(r);
%cc=round(c.*((liq_num_R-1)/surplus_liq_num));
liq_ind_R=find(Full_Array_trans_FR_norm_FS_training_sets_R(:,7)==1);
Full_Array_trans_FR_norm_FS_training_sets_balanced_R=Full_Array_trans_FR_norm_FS_training_sets_R;
Full_Array_trans_FR_norm_FS_training_sets_balanced_R(liq_ind_R(c),:)=[];

%%
% Event/Class capped sample number

cap_number_liq=3500;
cap_number_nonliq=2000;

for i=1:54
    if EQ_class_num(i,2)>cap_number_liq
        surplus_liq_event(i)=EQ_class_num(i,2)-cap_number_liq;
    else
        surplus_liq_event(i)=0;
    end
    if EQ_class_num(i,3)>cap_number_nonliq
        surplus_nonliq_event(i)=EQ_class_num(i,3)-cap_number_nonliq;
    else
        surplus_nonliq_event(i)=0;
    end
end

sum_surplus_liq=sum(surplus_liq_event);
sum_surplus_nonliq=sum(surplus_nonliq_event);

liq_rem_cap=186451-sum_surplus_liq;
nonliq_rem_cap=299117-186451-sum_surplus_nonliq;

%%
% Datasets creation - Liquefaction

Bhuj_liq_ind=find(Full_Array_trans_FR_norm_FS(1:186451,2)==5);
Bhuj_num_sets=round(EQ_class_num(5,2)/cap_number_liq);

for i=1:Bhuj_num_sets
    Bhuj_ind_sets{i}=Bhuj_liq_ind(i:Bhuj_num_sets:end);
end

Tohoku_liq_ind=find(Full_Array_trans_FR_norm_FS(1:186451,2)==48);
Tohoku_num_sets=round(EQ_class_num(48,2)/cap_number_liq);

for i=1:Tohoku_num_sets
   Tohoku_ind_sets{i}=Tohoku_liq_ind(i:Tohoku_num_sets:end);
end 

Darfield_liq_ind=find(Full_Array_trans_FR_norm_FS(1:186451,2)==12);
Darfield_num_sets=round(EQ_class_num(12,2)/cap_number_liq);

for i=1:Darfield_num_sets
   Darfield_ind_sets{i}=Darfield_liq_ind(i:Darfield_num_sets:end);
end

Christchurch_liq_ind=find(Full_Array_trans_FR_norm_FS(1:186451,2)==11);
Christchurch_num_sets=round(EQ_class_num(11,2)/cap_number_liq);

for i=1:Christchurch_num_sets
   Christchurch_ind_sets{i}=Christchurch_liq_ind(i:Christchurch_num_sets:end);
end

Nigata1964_liq_ind=find(Full_Array_trans_FR_norm_FS(1:186451,2)==33);
Nigata1964_num_sets=round(EQ_class_num(33,2)/cap_number_liq);

for i=1:Nigata1964_num_sets
   Nigata1964_ind_sets{i}=Nigata1964_liq_ind(i:Nigata1964_num_sets:end);
end

Nihonkai_liq_ind=find(Full_Array_trans_FR_norm_FS(1:186451,2)==36);
Nihonkai_num_sets=round(EQ_class_num(36,2)/cap_number_liq);

for i=1:Nihonkai_num_sets
   Nihonkai_ind_sets{i}=Nihonkai_liq_ind(i:Nihonkai_num_sets:end);
end

Kobe_liq_ind=find(Full_Array_trans_FR_norm_FS(1:186451,2)==23);
Kobe_num_sets=round(EQ_class_num(23,2)/cap_number_liq);

for i=1:Kobe_num_sets
   Kobe_ind_sets{i}=Kobe_liq_ind(i:Kobe_num_sets:end);
end

Nigata2004_liq_ind=find(Full_Array_trans_FR_norm_FS(1:186451,2)==34);
Nigata2004_num_sets=round(EQ_class_num(34,2)/cap_number_liq);

for i=1:Nigata2004_num_sets
   Nigata2004_ind_sets{i}=Nigata2004_liq_ind(i:Nigata2004_num_sets:end);
end

%%
% Datasets creation - Non-Liquefaction

Tohoku_nonliq_ind=find(Full_Array_trans_FR_norm_FS(186452:end,2)==48)+186451;
Tohoku_nonliq_num_sets=round(EQ_class_num(48,3)/cap_number_nonliq);

for i=1:Tohoku_nonliq_num_sets
   Tohoku_nonliq_ind_sets{i}=Tohoku_nonliq_ind(i:Tohoku_nonliq_num_sets:end);
end

Nigata1964_nonliq_ind=find(Full_Array_trans_FR_norm_FS(186452:end,2)==33)+186451;
Nigata1964_nonliq_num_sets=round(EQ_class_num(33,3)/cap_number_nonliq);

for i=1:Nigata1964_nonliq_num_sets
   Nigata1964_nonliq_ind_sets{i}=Nigata1964_nonliq_ind(i:Nigata1964_nonliq_num_sets:end);
end

Hokkaido_nonliq_ind=find(Full_Array_trans_FR_norm_FS(186452:end,2)==18)+186451;
Hokkaido_nonliq_num_sets=round(EQ_class_num(18,3)/cap_number_nonliq);

for i=1:Hokkaido_nonliq_num_sets
   Hokkaido_nonliq_ind_sets{i}=Hokkaido_nonliq_ind(i:Hokkaido_nonliq_num_sets:end);
end

PS1949_nonliq_ind=find(Full_Array_trans_FR_norm_FS(186452:end,2)==42)+186451;
PS1949_nonliq_num_sets=round(EQ_class_num(42,3)/cap_number_nonliq);

for i=1:PS1949_nonliq_num_sets
   PS1949_nonliq_ind_sets{i}=PS1949_nonliq_ind(i:PS1949_nonliq_num_sets:end);
end

PS1965_nonliq_ind=find(Full_Array_trans_FR_norm_FS(186452:end,2)==43)+186451;
PS1965_nonliq_num_sets=round(EQ_class_num(43,3)/cap_number_nonliq);

for i=1:PS1965_nonliq_num_sets
   PS1965_nonliq_ind_sets{i}=PS1965_nonliq_ind(i:PS1965_nonliq_num_sets:end);
end

Tokachi_nonliq_ind=find(Full_Array_trans_FR_norm_FS(186452:end,2)==49)+186451;
Tokachi_nonliq_num_sets=round(EQ_class_num(49,3)/cap_number_nonliq);

for i=1:Tokachi_nonliq_num_sets
   Tokachi_nonliq_ind_sets{i}=Tokachi_nonliq_ind(i:Tokachi_nonliq_num_sets:end);
end

Wenchuan_nonliq_ind=find(Full_Array_trans_FR_norm_FS(186452:end,2)==53)+186451;
Wenchuan_nonliq_num_sets=round(EQ_class_num(53,3)/cap_number_nonliq);

for i=1:Wenchuan_nonliq_num_sets
   Wenchuan_nonliq_ind_sets{i}=Wenchuan_nonliq_ind(i:Wenchuan_nonliq_num_sets:end);
end

Maule_nonliq_ind=find(Full_Array_trans_FR_norm_FS(186452:end,2)==27)+186451;
Maule_nonliq_num_sets=round(EQ_class_num(27,3)/cap_number_nonliq);

for i=1:Maule_nonliq_num_sets
   Maule_nonliq_ind_sets{i}=Maule_nonliq_ind(i:Maule_nonliq_num_sets:end);
end

Denali_nonliq_ind=find(Full_Array_trans_FR_norm_FS(186452:end,2)==13)+186451;
Denali_nonliq_num_sets=round(EQ_class_num(13,3)/cap_number_nonliq);

for i=1:Denali_nonliq_num_sets
   Denali_nonliq_ind_sets{i}=Denali_nonliq_ind(i:Denali_nonliq_num_sets:end);
end

%%
% Datasets preparation

X_ind_liq=Full_Array_trans_FR_norm_FS(1:186451,2);
X_ind_nonliq=Full_Array_trans_FR_norm_FS(186452:end,2);
ind_non_majority_eqs_liq=find(X_ind_liq~=5 & X_ind_liq~=48 & X_ind_liq~=12 & X_ind_liq~=11 & X_ind_liq~=33 & X_ind_liq~=36 & X_ind_liq~=23 & X_ind_liq~=34);
ind_non_majority_eqs_nonliq=find(X_ind_nonliq~=48 & X_ind_nonliq~=33 & X_ind_nonliq~=18 & X_ind_nonliq~=42 & X_ind_nonliq~=43 & X_ind_nonliq~=49 & X_ind_nonliq~=53 & X_ind_nonliq~=27 & X_ind_nonliq~=13);
ind_real_non_majority_eqs_nonliq=Full_Array_trans_FR_norm_FS(186452:end,1);
ind_real_non_majority_eqs_nonliq=ind_real_non_majority_eqs_nonliq(ind_non_majority_eqs_nonliq);

DS1=[Bhuj_ind_sets{1};Tohoku_ind_sets{1};Darfield_ind_sets{1};Christchurch_ind_sets{1};Nigata1964_ind_sets{1};Nihonkai_ind_sets{1};Kobe_ind_sets{1};Nigata2004_ind_sets{1};Tohoku_nonliq_ind_sets{1};Hokkaido_nonliq_ind_sets{1};PS1965_nonliq_ind_sets{1};PS1949_nonliq_ind_sets{1};Tokachi_nonliq_ind_sets{1};Wenchuan_nonliq_ind_sets{1};Maule_nonliq_ind_sets{1};Denali_nonliq_ind_sets{1};ind_non_majority_eqs_liq;ind_real_non_majority_eqs_nonliq];
DS2=[Bhuj_ind_sets{2};Tohoku_ind_sets{2};Darfield_ind_sets{2};Christchurch_ind_sets{2};Nigata1964_ind_sets{2};Nihonkai_ind_sets{2};Kobe_ind_sets{2};Nigata2004_ind_sets{2};Tohoku_nonliq_ind_sets{2};Hokkaido_nonliq_ind_sets{2};PS1965_nonliq_ind_sets{2};PS1949_nonliq_ind_sets{2};Tokachi_nonliq_ind_sets{2};Wenchuan_nonliq_ind_sets{2};Maule_nonliq_ind_sets{2};Denali_nonliq_ind_sets{2};ind_non_majority_eqs_liq;ind_real_non_majority_eqs_nonliq];
DS3=[Bhuj_ind_sets{3};Tohoku_ind_sets{3};Darfield_ind_sets{3};Christchurch_ind_sets{3};Nigata1964_ind_sets{3};Nihonkai_ind_sets{3};Kobe_ind_sets{1};Nigata2004_ind_sets{1};Tohoku_nonliq_ind_sets{3};Hokkaido_nonliq_ind_sets{1};PS1965_nonliq_ind_sets{3};PS1949_nonliq_ind_sets{3};Tokachi_nonliq_ind_sets{1};Wenchuan_nonliq_ind_sets{3};Maule_nonliq_ind_sets{3};Denali_nonliq_ind_sets{1};ind_non_majority_eqs_liq;ind_real_non_majority_eqs_nonliq];
DS4=[Bhuj_ind_sets{4};Tohoku_ind_sets{4};Darfield_ind_sets{4};Christchurch_ind_sets{4};Nigata1964_ind_sets{1};Nihonkai_ind_sets{1};Kobe_ind_sets{2};Nigata2004_ind_sets{2};Tohoku_nonliq_ind_sets{1};Hokkaido_nonliq_ind_sets{2};PS1965_nonliq_ind_sets{1};PS1949_nonliq_ind_sets{4};Tokachi_nonliq_ind_sets{2};Wenchuan_nonliq_ind_sets{4};Maule_nonliq_ind_sets{4};Denali_nonliq_ind_sets{2};ind_non_majority_eqs_liq;ind_real_non_majority_eqs_nonliq];
DS5=[Bhuj_ind_sets{5};Tohoku_ind_sets{5};Darfield_ind_sets{5};Christchurch_ind_sets{5};Nigata1964_ind_sets{2};Nihonkai_ind_sets{2};Kobe_ind_sets{1};Nigata2004_ind_sets{1};Tohoku_nonliq_ind_sets{2};Hokkaido_nonliq_ind_sets{1};PS1965_nonliq_ind_sets{2};PS1949_nonliq_ind_sets{1};Tokachi_nonliq_ind_sets{1};Wenchuan_nonliq_ind_sets{1};Maule_nonliq_ind_sets{5};Denali_nonliq_ind_sets{1};ind_non_majority_eqs_liq;ind_real_non_majority_eqs_nonliq];
DS6=[Bhuj_ind_sets{6};Tohoku_ind_sets{6};Darfield_ind_sets{6};Christchurch_ind_sets{6};Nigata1964_ind_sets{3};Nihonkai_ind_sets{3};Kobe_ind_sets{2};Nigata2004_ind_sets{2};Tohoku_nonliq_ind_sets{3};Hokkaido_nonliq_ind_sets{2};PS1965_nonliq_ind_sets{3};PS1949_nonliq_ind_sets{2};Tokachi_nonliq_ind_sets{2};Wenchuan_nonliq_ind_sets{2};Maule_nonliq_ind_sets{6};Denali_nonliq_ind_sets{2};ind_non_majority_eqs_liq;ind_real_non_majority_eqs_nonliq];
DS7=[Bhuj_ind_sets{7};Tohoku_ind_sets{7};Darfield_ind_sets{7};Christchurch_ind_sets{1};Nigata1964_ind_sets{1};Nihonkai_ind_sets{1};Kobe_ind_sets{1};Nigata2004_ind_sets{1};Tohoku_nonliq_ind_sets{1};Hokkaido_nonliq_ind_sets{1};PS1965_nonliq_ind_sets{1};PS1949_nonliq_ind_sets{3};Tokachi_nonliq_ind_sets{1};Wenchuan_nonliq_ind_sets{3};Maule_nonliq_ind_sets{7};Denali_nonliq_ind_sets{1};ind_non_majority_eqs_liq;ind_real_non_majority_eqs_nonliq];
DS8=[Bhuj_ind_sets{8};Tohoku_ind_sets{8};Darfield_ind_sets{8};Christchurch_ind_sets{2};Nigata1964_ind_sets{2};Nihonkai_ind_sets{2};Kobe_ind_sets{2};Nigata2004_ind_sets{2};Tohoku_nonliq_ind_sets{2};Hokkaido_nonliq_ind_sets{2};PS1965_nonliq_ind_sets{2};PS1949_nonliq_ind_sets{4};Tokachi_nonliq_ind_sets{2};Wenchuan_nonliq_ind_sets{4};Maule_nonliq_ind_sets{1};Denali_nonliq_ind_sets{2};ind_non_majority_eqs_liq;ind_real_non_majority_eqs_nonliq];
DS9=[Bhuj_ind_sets{9};Tohoku_ind_sets{9};Darfield_ind_sets{9};Christchurch_ind_sets{3};Nigata1964_ind_sets{3};Nihonkai_ind_sets{3};Kobe_ind_sets{1};Nigata2004_ind_sets{1};Tohoku_nonliq_ind_sets{3};Hokkaido_nonliq_ind_sets{1};PS1965_nonliq_ind_sets{3};PS1949_nonliq_ind_sets{1};Tokachi_nonliq_ind_sets{1};Wenchuan_nonliq_ind_sets{1};Maule_nonliq_ind_sets{2};Denali_nonliq_ind_sets{1};ind_non_majority_eqs_liq;ind_real_non_majority_eqs_nonliq];
DS10=[Bhuj_ind_sets{10};Tohoku_ind_sets{10};Darfield_ind_sets{1};Christchurch_ind_sets{4};Nigata1964_ind_sets{1};Nihonkai_ind_sets{1};Kobe_ind_sets{2};Nigata2004_ind_sets{2};Tohoku_nonliq_ind_sets{1};Hokkaido_nonliq_ind_sets{2};PS1965_nonliq_ind_sets{1};PS1949_nonliq_ind_sets{2};Tokachi_nonliq_ind_sets{2};Wenchuan_nonliq_ind_sets{2};Maule_nonliq_ind_sets{3};Denali_nonliq_ind_sets{2};ind_non_majority_eqs_liq;ind_real_non_majority_eqs_nonliq];
DS11=[Bhuj_ind_sets{11};Tohoku_ind_sets{11};Darfield_ind_sets{3};Christchurch_ind_sets{5};Nigata1964_ind_sets{2};Nihonkai_ind_sets{2};Kobe_ind_sets{1};Nigata2004_ind_sets{1};Tohoku_nonliq_ind_sets{2};Hokkaido_nonliq_ind_sets{1};PS1965_nonliq_ind_sets{2};PS1949_nonliq_ind_sets{3};Tokachi_nonliq_ind_sets{1};Wenchuan_nonliq_ind_sets{3};Maule_nonliq_ind_sets{4};Denali_nonliq_ind_sets{1};ind_non_majority_eqs_liq;ind_real_non_majority_eqs_nonliq];
DS12=[Bhuj_ind_sets{12};Tohoku_ind_sets{12};Darfield_ind_sets{5};Christchurch_ind_sets{6};Nigata1964_ind_sets{3};Nihonkai_ind_sets{3};Kobe_ind_sets{2};Nigata2004_ind_sets{2};Tohoku_nonliq_ind_sets{3};Hokkaido_nonliq_ind_sets{2};PS1965_nonliq_ind_sets{3};PS1949_nonliq_ind_sets{4};Tokachi_nonliq_ind_sets{2};Wenchuan_nonliq_ind_sets{4};Maule_nonliq_ind_sets{5};Denali_nonliq_ind_sets{2};ind_non_majority_eqs_liq;ind_real_non_majority_eqs_nonliq];
DS13=[Bhuj_ind_sets{13};Tohoku_ind_sets{13};Darfield_ind_sets{7};Christchurch_ind_sets{2};Nigata1964_ind_sets{1};Nihonkai_ind_sets{1};Kobe_ind_sets{1};Nigata2004_ind_sets{1};Tohoku_nonliq_ind_sets{1};Hokkaido_nonliq_ind_sets{1};PS1965_nonliq_ind_sets{1};PS1949_nonliq_ind_sets{1};Tokachi_nonliq_ind_sets{1};Wenchuan_nonliq_ind_sets{1};Maule_nonliq_ind_sets{6};Denali_nonliq_ind_sets{1};ind_non_majority_eqs_liq;ind_real_non_majority_eqs_nonliq];
DS14=[Bhuj_ind_sets{14};Tohoku_ind_sets{7};Darfield_ind_sets{9};Christchurch_ind_sets{4};Nigata1964_ind_sets{3};Nihonkai_ind_sets{3};Kobe_ind_sets{2};Nigata2004_ind_sets{2};Tohoku_nonliq_ind_sets{3};Hokkaido_nonliq_ind_sets{2};PS1965_nonliq_ind_sets{3};PS1949_nonliq_ind_sets{3};Tokachi_nonliq_ind_sets{2};Wenchuan_nonliq_ind_sets{3};Maule_nonliq_ind_sets{7};Denali_nonliq_ind_sets{2};ind_non_majority_eqs_liq;ind_real_non_majority_eqs_nonliq];

Label1=Full_Array_trans_FR_norm_FS(DS1,7);
Label14=Full_Array_trans_FR_norm_FS(DS14,7);
DS={DS1,DS2,DS3,DS4,DS5,DS6,DS7,DS8,DS9,DS10,DS11,DS12,DS13,DS14};

%%
% Leave-one-out data preparation and balancing

for i=1:54
    ind_leave_eq=find(Full_Array_trans_FR_norm_FS(:,2)~=i);
    for j=1:14
        C{i,j} = intersect(DS{j},ind_leave_eq);
        C_label=Full_Array_trans_FR_norm_FS(C{i,j},7);
        C_liq_num=sum(C_label);
        C_nonliq_num=length(C_label)-C_liq_num;
        C_surplus_liq_num(i,j)=C_liq_num-C_nonliq_num;
        %C_balanced_1{i}=
    end
end

%%
% Test Earthquake ID

%Test_EQ_ID = 5; % Bhuj
%Test_EQ_ID = 8; % Chi Chi
Test_EQ_ID = 11; % Christchurch
%Test_EQ_ID = 15; % Emilia
%Test_EQ_ID = 18; % Hokkaido
%Test_EQ_ID = 22; % Iwate
%Test_EQ_ID = 23; % Kobe
%Test_EQ_ID = 25; % Kumamoto
%Test_EQ_ID = 26; % Loma Prieta
%Test_EQ_ID = 32; % Nepal
%Test_EQ_ID = 33; % Nigata 1964
%Test_EQ_ID = 38; % Northridge
%Test_EQ_ID = 42; % Puget Sound 1949
%Test_EQ_ID = 48; % Tohoku

%%
% App data preparation

App_1=Full_Array_trans_FR_norm_FS(C{11,1},:);
App_2=Full_Array_trans_FR_norm_FS(C{11,2},:);
App_3=Full_Array_trans_FR_norm_FS(C{11,3},:);
App_4=Full_Array_trans_FR_norm_FS(C{11,4},:);
App_5=Full_Array_trans_FR_norm_FS(C{11,5},:);
App_6=Full_Array_trans_FR_norm_FS(C{11,6},:);
App_7=Full_Array_trans_FR_norm_FS(C{11,7},:);
App_8=Full_Array_trans_FR_norm_FS(C{11,8},:);
App_9=Full_Array_trans_FR_norm_FS(C{11,9},:);
App_10=Full_Array_trans_FR_norm_FS(C{11,10},:);
App_11=Full_Array_trans_FR_norm_FS(C{11,11},:);
App_12=Full_Array_trans_FR_norm_FS(C{11,12},:);
App_13=Full_Array_trans_FR_norm_FS(C{11,13},:);
App_14=Full_Array_trans_FR_norm_FS(C{11,14},:);

%%
%Test data

ind_leave_eq_test=find(Full_Array_trans_FR_norm_FS(:,2)==11);

%%
% Model prediction

yfit1 = trainedModel_11_1.predictFcn(Full_Array_trans_FR_norm_FS(ind_leave_eq_test,8:end));
yfit2 = trainedModel_11_4.predictFcn(Full_Array_trans_FR_norm_FS(ind_leave_eq_test,8:end));
yfit3 = trainedModel_11_8.predictFcn(Full_Array_trans_FR_norm_FS(ind_leave_eq_test,8:end));
yfit4 = trainedModel_11_12.predictFcn(Full_Array_trans_FR_norm_FS(ind_leave_eq_test,8:end));
yfit5 = trainedModel_11_6.predictFcn(Full_Array_trans_FR_norm_FS(ind_leave_eq_test,8:end));

yfit_avg=(yfit1+yfit2+yfit3+yfit4+yfit5)/5;

%%

%yfit = trainedModel_overal.predictFcn(Full_Array_trans_FR_norm_FS_testing(:,8:end));

%%
% Label comparison

%test_predicted_label(~test_predifined_nonliq)=round(yfit_avg);
predicted_label=round(yfit_avg);
original_label=Full_Array_trans_FR_norm_FS(ind_leave_eq_test,7);

%%
% Accuracy calculation

[ STATS ] = modelperf(original_label,test_predicted_label);

%%
% Mapping results

figure
lon = Full_Array_trans_FR_norm_FS(Test_EQ_ind,5);
lat = Full_Array_trans_FR_norm_FS(Test_EQ_ind,6);

A = test_predicted_label;
%A = original_label;

%b=ones(length(LomaPrietaliq),1);
%c=zeros(length(LomaPrietanonliq),1);
%A=[b;c];
for i=1:length(A)
    if A(i)==1
        A(i)=20;
        %C(i,:)=[0 0 1];
        C(i,:)=[1 0 0];
    else
        A(i)=1;
        C(i,:)=[0 0 0];
    end
end
geoscatter(lat,lon,A,C,'filled')
geobasemap topographic
title('Christchurch 2011 Earthquake (6.3 M_w) - Predicted Liquefaction Labels')
legend('Liquefaction','Non-Liquefaction')

%%
% Overal data modeling

all_label=Full_Array_trans_FR_norm_FS(:,7);
liq_num=sum(all_label);
nonliq_num=length(all_label)-liq_num;
surplus_liq_num=liq_num-nonliq_num;
r=rand(1,surplus_liq_num);
[aa,c]=sort(r);
cc=round(c.*((146640-1)/surplus_liq_num));
large_liq_EQs_ind_etc=large_liq_EQs_ind;
removal_large=large_liq_EQs_ind_etc(cc);
Full_Array_trans_FR_norm_FS_training=Full_Array_trans_FR_norm_FS;
Full_Array_trans_FR_norm_FS_training(removal_large,:)=[];

%%

yfit = trainedModel_ALL2.predictFcn(Full_Array_trans_FR_norm_FS(:,8:end));
yfit(nonliq_pre_threshold_ind)=0;
Label_original=Full_Array_trans_FR_norm_FS(:,7);
[ STATS ] = modelperf(Label_original,yfit);

%%
% Mapping results

EQ_NO=3;

figure
lon = Full_Array_trans_FR_norm_FS(global_ind_eq{EQ_NO},5);
lat = Full_Array_trans_FR_norm_FS(global_ind_eq{EQ_NO},6);

%A = Label_original(global_ind_eq{EQ_NO});
A = yfit(global_ind_eq{EQ_NO});

for i=1:length(A)
    if A(i)==1
        A(i)=5;
        %C(i,:)=[0 0 1];
        C(i,:)=[1 0 0];
    else
        A(i)=5;
        C(i,:)=[0 0 1];
    end
end

geoscatter(lat,lon,A,C,'filled')
geobasemap topographic
title('Arequipa 2001 Earthquake (8.4 M_w) - Predicted Liquefaction Labels')
legend('Liquefaction','Non-Liquefaction')

%%

[ STATS_eq ] = modelperf(Label_original(global_ind_eq{EQ_NO}),yfit(global_ind_eq{EQ_NO}));





