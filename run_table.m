clear;

%% download the movielens zip file;
dir_exist = exist('ml-10m', 'dir');  % check the existence of the folder ml-10m
dir_exist_7 = dir_exist - 7;
if (dir_exist_7)
    if ~isfile('ml-10m.zip')    % check the existence of the zip file ml-10m.zip
        url = 'http://files.grouplens.org/datasets/movielens/ml-10m.zip';
        filename = 'ml-10m.zip';
        options = weboptions('Timeout',Inf);
        websave(filename, url, options);
        clear options
        
        fprintf('ml-10m.zip downloaded. \n')
    end 
        unzip('ml-10m.zip', 'ml-10m');
        fprintf('ml-10m.zip unzipped. \n');
end

%% read rating.dat;
%%%   This data reading part is from read_movielens_10M.m file of InFaceExtendedFW-MatrixCompletion
M = dlmread('ml-10m/ml-10M100K/ratings.dat');

irow = M(:, 1); % users
jcol = M(:, 3); % movies

%%% fix irow, jcol
[unique_rows, irow_ind, unique_rows_ind] = unique(irow);
[unique_cols, jcol_ind, unique_cols_ind] = unique(jcol);

nrow = length(unique_rows);
ncol = length(unique_cols);

irow = (1:nrow)';
irow = irow(unique_rows_ind);
jcol = (1:ncol)';
jcol = jcol(unique_cols_ind);

Omega = sub2ind([nrow,ncol],irow,jcol);

Xobs_vec = M(:, 5);
no_obs = length(Xobs_vec);

%%% center the data
centerMatRows = [1:no_obs, 1:no_obs]';
centerMatCols = [irow; nrow + jcol];
centerMat = sparse(centerMatRows, centerMatCols, ones(2*no_obs, 1));

alphabeta = lsqr(centerMat, Xobs_vec, 10^-6, 1000);
alpha = alphabeta(1:nrow);
beta = alphabeta(nrow+1:nrow+ncol);

Xobs_vec = Xobs_vec - alpha(irow) - beta(jcol);
scale = norm(Xobs_vec, 2);
Xobs_vec = Xobs_vec/norm(Xobs_vec, 2);

save movielens_10M Omega irow jcol  Xobs_vec;  

fprintf('The user/movie data processed. \n')

%% add pathes for running InFaceExtendedFW
addpath('.\InFaceExtendedFW-MatrixCompletion\experiments')
addpath('.\InFaceExtendedFW-MatrixCompletion\experiments\paper-experiments')
addpath('.\InFaceExtendedFW-MatrixCompletion\solver')


%%  Split the dataset into training and testting sets; 
%%% This part is from movielens10M_run.m of InFaceExtendedFW-MatrixCompletion
rng(345);
warning('error', 'MATLAB:eigs:NoEigsConverged');

load('movielens_10M.mat');

[train_data, test_data] = split_matcomp_instance(Omega, irow, jcol, Xobs_vec, 0.7);

mat_comp_instance = struct();
mat_comp_instance.X_obs_vec = train_data.Xobs_vec;
mat_comp_instance.irow = train_data.irow;
mat_comp_instance.jcol = train_data.jcol;

mat_comp_instance.X_test_vec = test_data.Xobs_vec;
mat_comp_instance.irow_test = test_data.irow;
mat_comp_instance.jcol_test = test_data.jcol;

%% value of sigma in the models; named as delta in Freund's paper and codes
mat_comp_instance.delta = 2.5932;

%% run the algorithms
time_limit = 3600;          % maximum cputime
a = clock;

%%% The InFace direction method
options = struct();
options.verbose = 1;
options.rel_opt_TOL = -Inf;
options.abs_opt_TOL = -Inf;
options.bound_slack = 10^-6;
options.time_limit = time_limit;
options.max_iter = 40000;
options.prox_grad_tol = 10^-5;

options.gamma_1 = 0;
options.gamma_2 = Inf;
options.last_toward = 0;
options.rank_peak = 0;

options.pre_start_full = 0;

options.test_error_basic = 1;

options.svd_options.large_type = 'eigs';
options.svd_options.maxiter = 5000;
options.svd_options.tol = 10^-8;
options.svd_options.vector_stopping = 0;
options.svd_options.svd_test = 0;

options.alg_type = 'InFace';
fprintf('Start of the InFaceExtendedFW method.\n')
tstart1 = tic;
[final_solnIF_0Inf, historyIF_0Inf] = InFace_Extended_FW_sparse(mat_comp_instance, @Away_step_standard_sparse, @update_svd, options);
t_IF_0Inf = toc(tstart1);

matfilename_IF = ['results_IF_time_limit' '-' int2str(options.time_limit) '-date-' date '-' int2str(a(4)) '-' int2str(a(5)) '.mat'];
save(matfilename_IF,  'time_limit',  'final_solnIF_0Inf', 'historyIF_0Inf', 't_IF_0Inf');

%%% The nonconvex FW and AFW methods
clear options
m = 69878;
n = 10677;

X_train_vec = mat_comp_instance.X_obs_vec;
irow_obs = mat_comp_instance.irow;
jcol_obs = mat_comp_instance.jcol;

mu = 0.5;
sigma = 2.5932;

options = struct();
options.tol = 1e-6;
options.maxiter = 40000;
options.maxtime = time_limit;
options.dsp = 1;

%%% The boosting FW method
options.isafw = 0;
options.boosting = 1;
fprintf('Start of the FW-boosting method.\n')
tstart2 = tic;
[Xk_fw_bst, iter_fw_bst, fval_fw_bst, history_fw_bst] = FW_nuc(mat_comp_instance, m, n, mu, sigma, options);
t_fw_bst = toc(tstart2);

%%% The away-step FW method with boosting
options.isafw = 1;
options.boosting = 1;
fprintf('Start of the AFW-boosting method.\n')
tstart3 = tic;
[Xk_afw_bst, iter_afw_bst, fval_afw_bst, history_afw_bst] = FW_nuc(mat_comp_instance, m, n, mu, sigma, options);
t_afw_bst = toc(tstart3);

matfilename_FW = ['results_FW_time_limit' '-' int2str(options.maxtime) '-date' date '-' int2str(a(4)) '-' int2str(a(5)) '.mat'];
save(matfilename_FW, 'time_limit',  'Xk_fw_bst' , 'Xk_afw_bst', 'iter_fw_bst', 'iter_afw_bst', 'fval_fw_bst', ...
    'fval_afw_bst', 'history_fw_bst',  'history_afw_bst',  't_fw_bst',  't_afw_bst');

%% form the table
fname = ['Results\run-table'  '-'  date  '-'  int2str(a(4))  '-'  int2str(a(5))  '.txt'];
fid = fopen(fname, 'w');
fprintf(fid, ' %6s & %20s & %20s & %20s \n', ...
    ' ', ' IF-(0,inf)', 'FW-ncvx', 'AFW-ncvx');
fprintf(fid, ' %6s & %8s & %8s & %8s & %8s & %8s & %8s & %8s & %8s & %8s  \\\\ \n', ...
    'time_ub',  'gap_rel', 'rank', 'RMSE', 'gap_rel', 'rank', 'RMSE', 'gap_rel', 'rank', 'RMSE');

timeframes = [1000; 1500; 2000; 2500; 3000];

for tt = 1:length(timeframes)
    time_ub =  timeframes(tt);
    
    iter_IF_up = sum(historyIF_0Inf.cputimes(1:historyIF_0Inf.num_iters) <= time_ub);         % the iter number before reaching the time limit
    iter_fw_up = sum(history_fw_bst.cputimes <= time_ub);
    iter_afw_up = sum(history_afw_bst.cputimes <= time_ub);
    
    RMSE_IF_0Inf = sqrt( historyIF_0Inf.test_set_errors(iter_IF_up) *2*scale^2 /  length(mat_comp_instance.X_test_vec) );% compute the root mean squared error
    RMSE_fw = sqrt( history_fw_bst.test_errs(iter_fw_up) *2*scale^2 /  length(mat_comp_instance.X_test_vec) );
    RMSE_afw = sqrt( history_afw_bst.test_errs(iter_afw_up) *2*scale^2 /  length(mat_comp_instance.X_test_vec) );
    
    gap_rel_IF = historyIF_0Inf.bound_gaps_nooffset(iter_IF_up)/historyIF_0Inf.lowerbnds(iter_IF_up);
    gap_rel_fw = history_fw_bst.fw_gaps_rel(iter_fw_up);
    gap_rel_afw = history_afw_bst.fw_gaps_rel(iter_afw_up);
    
    data_tt = [time_ub, gap_rel_IF, historyIF_0Inf.ranks(iter_IF_up), RMSE_IF_0Inf,...
        gap_rel_fw, history_fw_bst.ranks(iter_fw_up),RMSE_fw,...
        gap_rel_afw, history_afw_bst.ranks(iter_afw_up),RMSE_afw];
    fprintf(fid, '%d &  %5.1e &  %5d & %6.4f &  %5.1e &  %5d & %6.4f &   %5.1e &  %5d & %6.4f   \\\\ \n', data_tt);
    
end
fclose(fid);

