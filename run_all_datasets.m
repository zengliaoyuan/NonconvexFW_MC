    clear;

%% add pathes for running InFaceExtendedFW
addpath('.\InFaceExtendedFW-MatrixCompletion\experiments')
addpath('.\InFaceExtendedFW-MatrixCompletion\experiments\paper-experiments')
addpath('.\InFaceExtendedFW-MatrixCompletion\solver')

%% parameters used in the testing
Data_set = {'movielens_10M','movielens_20M','movielens_32M', 'netflix_100M'};
M = [69878  138493  200971  480189];                     % dimension of m for dataset
N = [10677  26744   84349   17770];                      % dimension of n for dataset
Sigma = [2.6054  3.6576  4.4592  5.9402];                % sigma determined by the file find_delta_asIF.m
Time_limit = [3000 13000 50000 50000] + 500;             % the maximum time allowed
Timeframes = [1000 1500 2000 2500 3000                   % the timeframe for writting the table
              1000 4000 7000 10000 13000
              10000    20000   30000   40000   50000
              10000    20000   30000   40000   50000]; 




Mu = [0.25 0.5 0.75];

Data_IF = cell(length(index),1);
Data_FW = cell(length(index),length(Mu));
Data_AFW = cell(length(index),length(Mu));
Data_Dim = zeros(length(index),1);
Data_Scale = zeros(length(index),1);




%% run the IF-FW of Freund's, FW and AFW with mu = 0.25 for all chosen datasets
mu = Mu(1);

sigma = Sigma(1);
time_limit = Time_limit(1);
[historyIF_0Inf,history_fw_bst,history_afw_bst,scale] = runcode_movLens_10M(sigma,mu,time_limit);
Data_IF{1,1} = historyIF_0Inf;
Data_FW{1,1} = history_fw_bst;
Data_AFW{1,1} = history_afw_bst;
Data_Scale(1,1) = scale;

sigma = Sigma(2);
time_limit = Time_limit(2);
[historyIF_0Inf,history_fw_bst,history_afw_bst,scale] = runcode_movLens_20M(sigma,mu,time_limit);
Data_IF{2,1} = historyIF_0Inf;
Data_FW{2,1} = history_fw_bst;
Data_AFW{2,1} = history_afw_bst;
Data_Scale(2,1) = scale;

sigma = Sigma(3);
time_limit = Time_limit(3);
[historyIF_0Inf,history_fw_bst,history_afw_bst,scale] = runcode_movLens_32M(sigma,mu,time_limit);
Data_IF{3,1} = historyIF_0Inf;
Data_FW{3,1} = history_fw_bst;
Data_AFW{3,1} = history_afw_bst;
Data_Scale(3,1) = scale;

sigma = Sigma(4);
time_limit = Time_limit(4);
[historyIF_0Inf,history_fw_bst,history_afw_bst,scale] = runcode_netflix_100M(sigma,mu,time_limit);
Data_IF{4,1} = historyIF_0Inf;
Data_FW{4,1} = history_fw_bst;
Data_AFW{4,1} = history_afw_bst;
Data_Scale(4,1) = scale;






%% run FW and AFW for the chosen dataset for mu = 0.5 and 0.75

options = struct();
options.tol = 1e-6;
options.maxiter = 40000;
options.dsp = 1;
options.boosting = 1;

for i = 1 : length(Data_set)
    
    data_name = [Data_set{i} '.mat']
    load(data_name)
    m = M(i)
    n = N(i)
    sigma = Sigma(i)
    

    %% Split the dataset into training and testting sets; 
    %%% This part is from movielens10M_run.m of InFaceExtendedFW-MatrixCompletion
    rng(345);

    [train_data, test_data] = split_matcomp_instance(Omega, irow, jcol, Xobs_vec, 0.7);
    mat_comp_instance = struct();
    mat_comp_instance.X_obs_vec = train_data.Xobs_vec;
    mat_comp_instance.irow = train_data.irow;
    mat_comp_instance.jcol = train_data.jcol;

    mat_comp_instance.X_test_vec = test_data.Xobs_vec;
    mat_comp_instance.irow_test = test_data.irow;
    mat_comp_instance.jcol_test = test_data.jcol;

    Data_Dim(i,1) = length(mat_comp_instance.X_test_vec);
    time_limit = Time_limit(1,i);

    %% run FW and AFW
    for j = 2 : length(Mu)

        mu = Mu(1,j);
        %%% The FW method with boosting
        options.isafw = 0;
        options.maxtime = time_limit;
        fprintf('Start of the FW-boosting method for dataset %d and mu=%4.2f.\n', i,mu)
        tstart = tic;
        [Xk_fw_bst, iter_fw_bst, fval_fw_bst, history_fw_bst] = FW_nuc(mat_comp_instance, m, n, mu, sigma, options);
        t_fw_bst = toc(tstart);
        Data_FW{i,j} = history_fw_bst;

        %%% The away-step FW method with boosting
        options.isafw = 1;
        options.maxtime = time_limit;
        fprintf('Start of the AFW-boosting method for dataset %d and mu=%4.2f.\n', i,mu)
        tstart = tic;
        [Xk_afw_bst, iter_afw_bst, fval_afw_bst, history_afw_bst] = FW_nuc(mat_comp_instance, m, n, mu, sigma, options);
        t_afw_bst = toc(tstart);
        Data_AFW{i,j} = history_afw_bst;


        % data storage
        a = clock;
        datname = ['FWandAFWResults' '_' Data_set{1,i} '_mu'   num2str(mu*100) '_'  'timelimit' num2str(time_limit) '_date' date '-' int2str(a(4)) '-' int2str(a(5)) '.mat'];
        save(datname, 'Xk_fw_bst',  'Xk_afw_bst', 'iter_fw_bst', 'iter_afw_bst', 'fval_fw_bst', 'fval_afw_bst', 'history_fw_bst', 'history_afw_bst', 't_fw_bst', 't_afw_bst','scale');
       
        
        %% plot the figure for dataset i and mu j
        % plot sum squared error on training set and testing set
        historyIF_0Inf = Data_IF{i,1};

        iterscosts_fw = history_fw_bst.cputimes;
        iterscosts_afw = history_afw_bst.cputimes;
        iterscosts_IF = historyIF_0Inf.cputimes(1:historyIF_0Inf.num_iters);

        fvals_fw = history_fw_bst.fvals;
        fvals_afw = history_afw_bst.fvals;
        fvals_IF = historyIF_0Inf.objvals(1:historyIF_0Inf.num_iters);

        test_errs_fw = history_fw_bst.test_errs;
        test_errs_afw = history_afw_bst.test_errs;
        test_errs_IF = historyIF_0Inf.test_set_errors(1:historyIF_0Inf.num_iters);

        fig1 = figure;
        plot(iterscosts_fw, fvals_fw, '-', 'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2); hold on
        plot(iterscosts_afw, fvals_afw, '-', 'Color', [174 221 129]/255,'LineWidth', 2); hold on
        plot(iterscosts_IF, fvals_IF, '-', 'Color', [146 172 209]/255, 'LineWidth', 2);
        legend('trainErr-FW', 'trainErr-AFW', 'trainErr-IF');
        y_max = max([max(fvals_fw), max(fvals_afw), max(fvals_IF)]);
        y_min = min([min(fvals_fw), min(fvals_afw), min(fvals_IF)]);
        yaxis_up = y_max + (y_max - y_min)/100;
        yaxis_bottom = y_min - (y_max - y_min)/100;
        axis([0 time_limit yaxis_bottom yaxis_up])
        xlabel('time(s)');
        ylabel('Train Error')
        a = clock;
        figname = [Data_set{1,i}   '-trainingErrs-mu' num2str(mu*100)  '-' date '-' num2str(a(4)) '-'  num2str(a(5))];
        savefig(fig1, figname, 'Figures');

        fig2 = figure;
        plot(iterscosts_fw, test_errs_fw, '-', 'Color', [147, 147, 145]/255,'LineWidth', 2); hold on
        plot(iterscosts_afw, test_errs_afw, '-', 'Color', [238,130,238]/255, 'LineWidth', 2); hold on
        plot(iterscosts_IF, test_errs_IF, '-', 'Color', [226, 17, 0]/255, 'LineWidth', 2); hold off
        legend( 'testErr-FW', 'testErr-AFW', 'testErr-IF');
        y_max = max([max(test_errs_fw), max(test_errs_afw), max(test_errs_IF)]);
        y_min = min([min(test_errs_fw), min(test_errs_afw), min(test_errs_IF)]);
        yaxis_up = y_max + (y_max - y_min)/100;
        yaxis_bottom = y_min - (y_max - y_min)/100;
        axis([0 time_limit yaxis_bottom yaxis_up])
        xlabel('time(s)');
        ylabel('Test Error')
        a = clock;
        figname = [ Data_set{1,i}  '-testErrs-mu'  num2str(mu*100) '-' date '-' num2str(a(4)) '-'  num2str(a(5))];
        savefig(fig2, figname, 'Figures');
        close all;

         

    end


end





%% form the table
a = clock;
fname = ['Results\run-table'  '-'  date  '-'  int2str(a(4))  '-'  int2str(a(5))  '.txt'];
fid = fopen(fname, 'w');
fprintf(fid, ' %6s & %20s & %20s & %20s \n', ...
    ' ', ' IF-(0,inf)', 'FW-ncvx', 'AFW-ncvx');
fprintf(fid, ' %6s & %8s & %8s & %8s & %8s & %8s & %8s & %8s & %8s & %8s & %8s & %8s & %8s & %8s & %8s & %8s & %8s & %8s & %8s & %8s & %8s & %8s  \\\\ \n', ...
             'time_ub',  'gap_rel', 'rank', 'RMSE', 'gap_rel', 'rank', 'RMSE', 'gap_rel', 'rank', 'RMSE',...
             'gap_rel', 'rank', 'RMSE', 'gap_rel', 'rank', 'RMSE', 'gap_rel', 'rank', 'RMSE', 'gap_rel', 'rank', 'RMSE');



for t = 1 : length(Data_set)
      
    scale = Data_Scale(t,1);
    historyIF_0Inf = Data_IF{t,1};
    timeframes = Timeframes(t,:);

    for tt = 1:length(timeframes)
        
        time_ub =  timeframes(tt);

        iter_IF_up = sum(historyIF_0Inf.cputimes(1:historyIF_0Inf.num_iters) <= time_ub);         % the iter number before reaching the time limit
        RMSE_IF_0Inf = sqrt( historyIF_0Inf.test_set_errors(iter_IF_up) *2*scale^2 /  Data_Dim(t,1) );% compute the root mean squared error
        gap_rel_IF = historyIF_0Inf.bound_gaps_nooffset(iter_IF_up)/historyIF_0Inf.lowerbnds(iter_IF_up);
        data = [time_ub,gap_rel_IF, historyIF_0Inf.ranks(iter_IF_up), RMSE_IF_0Inf];


        for ttt = 1 : length(Mu)
            
            history_fw_bst = Data_FW{t,ttt};
            history_afw_bst = Data_AFW{t,ttt};
            iter_fw_up = sum(history_fw_bst.cputimes <= time_ub);
            iter_afw_up = sum(history_afw_bst.cputimes <= time_ub);    
            RMSE_fw = sqrt( history_fw_bst.test_errs(iter_fw_up) *2*scale^2 /  Data_Dim(t,1) );
            RMSE_afw = sqrt( history_afw_bst.test_errs(iter_afw_up) *2*scale^2 /  Data_Dim(t,1) );
            gap_rel_fw = history_fw_bst.fw_gaps_rel(iter_fw_up);
            gap_rel_afw = history_afw_bst.fw_gaps_rel(iter_afw_up);
    
            data = [data, gap_rel_fw, history_fw_bst.ranks(iter_fw_up), RMSE_fw, gap_rel_afw, history_afw_bst.ranks(iter_afw_up), RMSE_afw];

        end

        fprintf(fid, ['%d &  %5.1e &  %5d & %6.4f &  %5.1e &  %5d & %6.4f &   %5.1e &  %5d & %6.4f' ...
                                                 '&  %5.1e &  %5d & %6.4f &   %5.1e &  %5d & %6.4f' ...
                                                 '&  %5.1e &  %5d & %6.4f &   %5.1e &  %5d & %6.4f  \\\\ \n'], data);

    end

    fprintf(fid,'\n');
   
    
end

fclose(fid);

