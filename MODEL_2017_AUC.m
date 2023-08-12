%% MODEL_2017_AUC

%%

% Sample table
T_EVENT = DataExtraction2017rasters;

% Specify the specific text you want to filter by
%target_name = "LomaPrieta";
%target_name = "Kobe";
%target_name = "Tottori";
%target_name = "Christchurch";
%target_name = "Miyagi";
%target_name = "SanSimeon";
%target_name = "Darfield";
%target_name = "Nisqually";
%target_name = "Nigata2004";
%target_name = "Tohoku";
%target_name = "Hokkaido";
%target_name = "Nigata1964";
%target_name = "Nihonkai";
%target_name = "Chiba";
%target_name = "ChiChi";
%target_name = "PugetSound1949";
%target_name = "Tokachi";
%target_name = "PugetSound1965";
%target_name = "Northridge";
%target_name = "Wenchuan";
%target_name = "Achaia";
%target_name = "Arequipa";
%target_name = "BajaCalifornia";
%target_name = "Cephalonia";
%target_name = "Denali";
%target_name = "Duzce";
%target_name = "Emilia";
%target_name = "Haiti";
%target_name = "Honduras";
%target_name = "Illapel";
%target_name = "Kobe";
%target_name = "Kocaeli";
%target_name = "Kumamoto";
%target_name = "Maule";
%target_name = "Meinong";
%target_name = "Muisne";
%target_name = "Napa";
%target_name = "Nepal";
%target_name = "Nigata2007";
%target_name = "Oklahoma";
%target_name = "Pisco";
%target_name = "Samara";
%target_name = "Tecoman";
%target_name = "TelireLimon";
%target_name = "Aquila";
%target_name = "Iquique";
%target_name = "VanTab";
%target_name = "TelireLimon";
%target_name = "Virginia";
%target_name = "CentralItaly";
target_name = "Iwate";

% Logical indexing to extract rows with the target name
filtered_table_EVENT = T_EVENT((T_EVENT.Earthquake == target_name),["Latitude","Longitude","global_vs30_1b","global_dc_1b","dist_river_1b","global_wtd_fil1_1b","global_precip_fil_1b","PGV","liq"]);

array_filtered_table_EVENT = table2array(filtered_table_EVENT);

% Example matrix
matrix_EVENT = array_filtered_table_EVENT;

% Find rows containing the value -9999
rows_to_remove_EVENT = any(matrix_EVENT <= -500, 2);

% Remove corresponding rows
cleaned_data_EVENT = matrix_EVENT(~rows_to_remove_EVENT, :);

% Find rows with NaN elements
rows_with_nan_EVENT = any(isnan(cleaned_data_EVENT), 2);

% Remove rows with NaN elements
cleaned_data_EVENT2 = cleaned_data_EVENT(~rows_with_nan_EVENT, :);

%for i =1:6
%sample_range = cleaned_data_EVENT2(:,i+2);
%min(sample_range)
%max(sample_range)
%figure
%histogram(sample_range)
%end

% X = 8.801 + (0.334 * log(PGV)) + (-1.918 * log(Vs30)) + (0.0005408 *
% Precip) - (0.2054 * DW) - (0.0333 * WTD)
% P_LIQ = 1 / (1 + exp(-X));

DW_data = cleaned_data_EVENT2(:,[4,5]);
DW = min(DW_data,[],2);
DR = cleaned_data_EVENT2(:,5);
DC = cleaned_data_EVENT2(:,4);
PGV = cleaned_data_EVENT2(:,8);
%ln_PGV = array_filtered_table_Europe_balanced(:,8);
Vs30 = cleaned_data_EVENT2(:,3);
Precip = cleaned_data_EVENT2(:,7);
WTD = cleaned_data_EVENT2(:,6);

%EVENT_balanced_transformed_coastal = [log(PGV),log(Vs30),Precip,DW,WTD,cleaned_data_EVENT2(:,end)];
%EVENT_balanced_transformed_noncoastal = [log(PGV),log(Vs30),Precip,DW,WTD,cleaned_data_EVENT2(:,end)];
%Europe_balanced_transformed = [ln_PGV,log(Vs30),Precip,DW,WTD,array_filtered_table_Europe_balanced(:,end)];

X_noncoastal = 8.801 + (0.334 * log(PGV)) + (-1.918 * log(Vs30)) + (0.0005408 * Precip) + (-0.2054 * DW) + (-0.0333 * WTD);
X_coastal = 12.435 + (0.301 * log(PGV)) + (-2.615 * log(Vs30)) + (0.0005556 * Precip) + (-0.0287 * sqrt(DC)) + (0.0666 * DR) + (-0.0369 * (DR .* sqrt(DC)));

P_LIQ_noncoastal = 1 ./ (1 + exp(-X_noncoastal));
P_LIQ_coastal = 1 ./ (1 + exp(-X_coastal));

ground_truth = cleaned_data_EVENT2(:,end);

% Calculate the AUC
%[fpr1, tpr1, thresholds1, AUC_2017_noncoastal] = perfcurve(ground_truth, P_LIQ_noncoastal, 1);
%[fpr, tpr, thresholds, AUC_2017_coastal] = perfcurve(ground_truth, P_LIQ_coastal, 1);

% Assuming you have binary predictions 'predicted_labels' (0 or 1) and true binary labels 'true_labels' (0 or 1)

%predicted_labels = predicted_class;
predicted_labels = round(P_LIQ_noncoastal);

% Get indices of positive class (class label 1)
positive_indices = find(ground_truth == 1);

% Get predictions and true labels for the positive class
positive_predicted_labels = predicted_labels(positive_indices);
positive_true_labels = ground_truth(positive_indices);

% Calculate sensitivity (True Positive Rate)
sensitivity_2017_noncoastal = sum(positive_predicted_labels == positive_true_labels) / sum(positive_true_labels);

% Get indices of negative class (class label 0)
negative_indices = find(ground_truth == 0);

% Get predictions and true labels for the negative class
negative_predicted_labels = predicted_labels(negative_indices);
negative_true_labels = ground_truth(negative_indices);

% Calculate specificity (True Negative Rate)
specificity_2017_noncoastal = sum(negative_predicted_labels == negative_true_labels) / numel(negative_true_labels);

% Calculate overall accuracy
accuracy_2017_noncoastal = sum(predicted_labels(:) == ground_truth(:)) / numel(ground_truth(:));

accuracy_results = [AUC_2017_coastal;accuracy_2017_noncoastal;AUC_2017_noncoastal;specificity_2017_noncoastal;sensitivity_2017_noncoastal];





