% Test Earthquake ID

%Test_EQ_ID = 1; % Achaia
%Test_EQ_ID = 2; % Aquila
%Test_EQ_ID = 3; % Arequipa
%Test_EQ_ID = 4; % Baja California
%Test_EQ_ID = 6; % Cephalonia
%Test_EQ_ID = 7; % Chi Chi
%Test_EQ_ID = 8; % Chiba
%Test_EQ_ID = 10; % Christchurch
%Test_EQ_ID = 11; % Darfield
%Test_EQ_ID = 12; % Denali
%Test_EQ_ID = 13; % Duzce
%Test_EQ_ID = 14; % Emilia
%Test_EQ_ID = 15; % Haiti
%Test_EQ_ID = 17; % Hokkaido
%Test_EQ_ID = 18; % Honduras
%Test_EQ_ID = 19; % Illapel
%Test_EQ_ID = 20; % Iquique
%Test_EQ_ID = 22; % Kobe
%Test_EQ_ID = 23; % Kocaeli
%Test_EQ_ID = 24; % Kumamoto
%Test_EQ_ID = 25; % Loma Prieta
%Test_EQ_ID = 26; % Maule
%Test_EQ_ID = 27; % Meinong
%Test_EQ_ID = 28; % Miyagi Ken
%Test_EQ_ID = 29; % Muisne
%Test_EQ_ID = 30; % Napa
%Test_EQ_ID = 31; % Nepal (Gorkha)
%Test_EQ_ID = 32; % Nigata 1964
%Test_EQ_ID = 33; % Nigata 2004
%Test_EQ_ID = 34; % Nigata 2007
%Test_EQ_ID = 35; % Nihonkai
%Test_EQ_ID = 36; % Nisqually
Test_EQ_ID = 37; % Northridge
%Test_EQ_ID = 38; % Oklahoma
%Test_EQ_ID = 40; % Pisco
%Test_EQ_ID = 41; % Puget Sound 1949
%Test_EQ_ID = 42; % Puget Sound 1965
%Test_EQ_ID = 43; % Samara
%Test_EQ_ID = 44; % San Simeon
%Test_EQ_ID = 45; % Tecoman
%Test_EQ_ID = 46; % Talire Limon
%Test_EQ_ID = 47; % Tohoku
%Test_EQ_ID = 48; % Takachi
%Test_EQ_ID = 49; % Tottori
%Test_EQ_ID = 50; % VanTab
%Test_EQ_ID = 51; % Virginia
%Test_EQ_ID = 52; % Wenchun
%Test_EQ_ID = 5; % Central Italy
%Test_EQ_ID = 9; % Chino Hills
%Test_EQ_ID = 16; % Hector Mine
%Test_EQ_ID = 21; % Iwate
%Test_EQ_ID = 39; % Piedmont
%Test_EQ_ID = 53; % Yountville

Global_1=Full_Array_trans_FR_norm(balanced_vec_id_global_1,:);
Global_1_rmv=find(Global_1(:,2)==Test_EQ_ID);
Global_1(Global_1_rmv,:)=[];

Global_2=Full_Array_trans_FR_norm(balanced_vec_id_global_2,:);
Global_2_rmv=find(Global_2(:,2)==Test_EQ_ID);
Global_2(Global_2_rmv,:)=[];

Global_3=Full_Array_trans_FR_norm(balanced_vec_id_global_3,:);
Global_3_rmv=find(Global_3(:,2)==Test_EQ_ID);
Global_3(Global_3_rmv,:)=[];

if table2array(EQ_Regions_new(Test_EQ_ID,5))==1

Coastal_1=Full_Array_trans_FR_norm(balanced_vec_id_coastal_1,:);
Coastal_1_rmv=find(Coastal_1(:,2)==Test_EQ_ID);
Coastal_1(Coastal_1_rmv,:)=[];

Coastal_2=Full_Array_trans_FR_norm(balanced_vec_id_coastal_2,:);
Coastal_2_rmv=find(Coastal_2(:,2)==Test_EQ_ID);
Coastal_2(Coastal_2_rmv,:)=[];

else

NonCoastal_1=Full_Array_trans_FR_norm(balanced_vec_id_noncoastal_1,:);
NonCoastal_1_rmv=find(NonCoastal_1(:,2)==Test_EQ_ID);
NonCoastal_1(NonCoastal_1_rmv,:)=[];

NonCoastal_2=Full_Array_trans_FR_norm(balanced_vec_id_noncoastal_2,:);
NonCoastal_2_rmv=find(NonCoastal_2(:,2)==Test_EQ_ID);
NonCoastal_2(NonCoastal_2_rmv,:)=[];

end

if table2array(EQ_Regions_new(Test_EQ_ID,4))==1

Regional=Full_Array_trans_FR_norm(balanced_indices_regional_1,:);
Regional_1_rmv=find(Regional(:,2)==Test_EQ_ID);
Regional(Regional_1_rmv,:)=[];

elseif table2array(EQ_Regions_new(Test_EQ_ID,4))==2

Regional=Full_Array_trans_FR_norm(balanced_indices_regional_2,:);
Regional_2_rmv=find(Regional(:,2)==Test_EQ_ID);
Regional(Regional_2_rmv,:)=[];

elseif table2array(EQ_Regions_new(Test_EQ_ID,4))==3

Regional=Full_Array_trans_FR_norm(balanced_indices_regional_3,:);
Regional_3_rmv=find(Regional(:,2)==Test_EQ_ID);
Regional(Regional_3_rmv,:)=[];

elseif table2array(EQ_Regions_new(Test_EQ_ID,4))==4

Regional=Full_Array_trans_FR_norm(balanced_indices_regional_4,:);
Regional_4_rmv=find(Regional(:,2)==Test_EQ_ID);
Regional(Regional_4_rmv,:)=[];

elseif table2array(EQ_Regions_new(Test_EQ_ID,4))==5

Regional=Full_Array_trans_FR_norm(balanced_indices_regional_5,:);
Regional_5_rmv=find(Regional(:,2)==Test_EQ_ID);
Regional(Regional_5_rmv,:)=[];

elseif table2array(EQ_Regions_new(Test_EQ_ID,4))==6

Regional=Full_Array_trans_FR_norm(balanced_indices_regional_6,:);
Regional_6_rmv=find(Regional(:,2)==Test_EQ_ID);
Regional(Regional_6_rmv,:)=[];

end

disp(EQ_Regions_new(Test_EQ_ID,5));
disp(EQ_Regions_new(Test_EQ_ID,4));

ind_test_tr=find(Full_Array_trans_FR_norm_sampled(:,2)==Test_EQ_ID);
%training_LOO=Full_Array_trans_FR_norm_sampled(:,[7,8,9,13,14,21,25,26]);
training_LOO=Full_Array_trans_FR_norm_sampled(:,:);
training_LOO(ind_test_tr,:)=[];
ind_test_tst=find(Full_Array_trans_FR_norm(:,2)==Test_EQ_ID);
%testing_LOO=Full_Array_trans_FR_norm(ind_test_tst,[7,8,9,13,14,21,25,26]);
testing_LOO=Full_Array_trans_FR_norm(ind_test_tst,:);

[trainedModel_global_1, validationAccuracy1] = trainClassifier_global_1(Global_1);
[trainedModel_global_2, validationAccuracy2] = trainClassifier_global_2(Global_1);
[trainedModel_global_3, validationAccuracy3] = trainClassifier_global_3(Global_1);

if table2array(EQ_Regions_new(Test_EQ_ID,5))==1
    [trainedModel_cnc_4, validationAccuracy4] = trainClassifier_cnc_4(Coastal_1);
    [trainedModel_cnc_5, validationAccuracy5] = trainClassifier_cnc_5(Coastal_2);
else
    [trainedModel_cnc_4, validationAccuracy4] = trainClassifier_cnc_4(NonCoastal_1);
    [trainedModel_cnc_5, validationAccuracy5] = trainClassifier_cnc_5(NonCoastal_2);
end

[trainedModel_regional, validationAccuracy6] = trainClassifier_regional(Regional);

A=nonliq_pre_threshold_ind;
B=testing_LOO(:,1);
[CC,ia,ib] = intersect(A,B);

[yfit_global_1,scores_Global_1] = trainedModel_global_1.predictFcn(testing_LOO(:,1:25));
[yfit_global_2,scores_Global_2] = trainedModel_global_2.predictFcn(testing_LOO(:,1:25));
[yfit_global_3,scores_Global_3] = trainedModel_global_3.predictFcn(testing_LOO(:,1:25));
[yfit_cnc_1,scores_coastal_1] = trainedModel_cnc_4.predictFcn(testing_LOO(:,1:25));
[yfit_cnc_2,scores_coastal_2] = trainedModel_cnc_5.predictFcn(testing_LOO(:,1:25));
[yfit_regional,scores_regional] = trainedModel_regional.predictFcn(testing_LOO(:,1:25));

vote_eq_binary=round((yfit_global_1+yfit_global_2+yfit_global_3+yfit_cnc_1+yfit_cnc_2+yfit_regional)./6);
vote_eq_prob=((yfit_global_1+yfit_global_2+yfit_global_3+yfit_cnc_1+yfit_cnc_2+yfit_regional)./6);

vote_eq_binary(ib)=0;
vote_eq_prob(ib)=0;

original_label=testing_LOO(:,26);

if length(unique(original_label))>1
cp_eq = classperf(testing_LOO(:,26),vote_eq_binary,'Positive',1,'Negative',0);
end

if length(unique(original_label))>1
cm_eq = confusionmat(testing_LOO(:,26), vote_eq_binary);
precision_eq = cm_eq(2,2) / sum(cm_eq(:,2));
recall_eq = cm_eq(2,2) / sum(cm_eq(2,:));
f1Score_eq = 2 * (precision_eq * recall_eq) / (precision_eq + recall_eq);
%STATS_eq10 = modelperf(testing_LOO_eq25(:,26),vote_eq25_binary);
end

if length(unique(original_label))>1
[X,Y,T,AUC_eq] = perfcurve(testing_LOO(:,26),vote_eq_prob,1);

[X,Y,T,AUC_1] = perfcurve(testing_LOO(:,26),yfit_global_1,1);
[X,Y,T,AUC_2] = perfcurve(testing_LOO(:,26),yfit_global_2,1);
[X,Y,T,AUC_3] = perfcurve(testing_LOO(:,26),yfit_global_3,1);
[X,Y,T,AUC_4] = perfcurve(testing_LOO(:,26),yfit_cnc_1,1);
[X,Y,T,AUC_5] = perfcurve(testing_LOO(:,26),yfit_cnc_2,1);
[X,Y,T,AUC_6] = perfcurve(testing_LOO(:,26),yfit_regional,1);
end

%test_predicted_label=vote_eq_binary;
%test_predicted_label=double(vote_eq_prob>=0.15); % Probability Threshold
%test_predicted_label=yfit_cnc_2; % Single model outpu
test_predicted_label=yfit_global_2;
%test_predicted_label=yfit_regional;
visual_coords=testing_LOO;

figure
%a=[LomaPrietaliq;LomaPrietanonliq];
%test=data_eq{21};
B=original_label;

lon = visual_coords(:,5);
lat = visual_coords(:,6);

clear C

%A = testing_data(:,end);
A = test_predicted_label;
%b=ones(length(LomaPrietaliq),1);
%c=zeros(length(LomaPrietanonliq),1);
%A=[b;c];
for i=1:length(A)
    if A(i)==1 && B(i)==1
        A(i)=30;
        %C(i,:)=[0 0 1];
        C(i,:)=[1 0 0];
    elseif A(i)==1 && B(i)==0
        A(i)=30;
        %C(i,:)=[0 0 1];
        C(i,:)=[0.9290 0.6940 0.1250];
    elseif A(i)==0 && B(i)==1
        A(i)=30;
        %C(i,:)=[0 1 1];
        C(i,:)= [0 0 1];
    elseif A(i)==0 && B(i)==0
        A(i)=30;
        C(i,:)=[0 0.5 0];
        %C(i,:)=[0 1 0];
    %else
        %A(i)=1;
        %C(i,:)=[0 0 0];
    end
end
geoscatter(lat,lon,A,C,'filled')
geobasemap topographic
%title('Predicted Liquefaction for 6.9 M_w Loma Prieta 1989 Earthquake')
%title('Predicted Liquefaction for 7.6 M_w Tohoku 1991 Earthquake')
%title('Predicted Liquefaction for 6.9 M_w Kobe 1995 Earthquake')
%title('Predicted Liquefaction for 6.1 M_w Christchurch 2011 Earthquake')
%legend('Liquefaction','Non-Liquefaction')
