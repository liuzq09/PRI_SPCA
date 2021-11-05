% This matlab script contains all the experiments performed in our main paper; incuding: 
% The experiments considering the effect of sample size, sparsity, and noise

% NOTE: Authors of the NeurIPS submission will be listed here in any
% final/public version of the code.  For files that we downloaded from a
% public repository, we have kept any author names listed in the code 
% unchanged, and added a link to the repository.


%% phase retrieval for sparse signals 
%stores signal models (.mat) and results (.mat/fig/jpg). not committed to git repo.
if ~exist('results','dir')
    mkdir('results')
end
addpath('utils','ThWF','SPARTA','CoPRAM','results','plotFuncs','GRQI')
close all; clear all; clc;

%% parameters that are always fixed
n = 1000; % ambient dimension
trials_M = 50; % the number of random trials 

l = 1; u = 5; % parameters for calculating our weighted empirical covariance matrix V
GRQI_iter = 100; % no. of total iterations in GRQI
GRQI_power_iter = 100; % no. of power iterations in GRQI
deflation_param = 0.2; % the deflation parameter of GRQI
GRQI_thr = 1e-6; % the accuracy threshold of GRQI

%% parameters used in the experiment for Figure 1 (Sample size effect)
kspan = [10,20]; % sparsity level
kl = length(kspan);
mspan = 100:100:3000; % no. of measurements
ml = length(mspan);
sigma = 0; % noise level

%% the matrices to record the numerical results (Sample size effect)
err_mat1 = zeros(kl * trials_M,ml); % record the relative error for CoPRAM
err_mat2 = zeros(kl * trials_M,ml); % record the relative error for PR-SPCA
err_mat3 = zeros(kl * trials_M,ml); % record the relative error for PR-SPCA-NT
err_mat4 = zeros(kl * trials_M,ml); % record the relative error for ThWF
err_mat5 = zeros(kl * trials_M,ml); % record the relative error for SparAF

%% the main iteration in the experiment for Figure 1 (Sample size effect)
for k_iter = 1:kl
    for m_iter = 1:ml
        s = kspan(k_iter); % current sparsity level
        m = mspan(m_iter); % current no. of samples
        for tr = 1:trials_M % iteration for different random trials
            count_num = (k_iter - 1) * trials_M + tr; 
            
            fprintf('\nTrial no. :%d\nNo. of measurements M :%d\nSparsity K :%d\n',tr,m,s);
            %% generate signal and measurements
            [z,z_ind] =  generate_signal(n,s); % generate the signal 
            znorm = norm(z); % the norm of the signal
            A = randn(m,n); % generate the sensing matrix
            y  = abs(A * z) + sigma * znorm * randn(m,1); 
            
            % calcuting our empirical matrices V and \tilde{V}
            lambda = sqrt(pi/2) * sum(y(:)) / numel(y(:)); % calculating lambda
            ytr = y.* ((y > l * lambda) & (y < u * lambda)); % truncated version
            V = A' * diag(ytr) * A / m; % calculating V
            V_noTrunc = A' * diag(y) * A / m; % calculating the non-truncated version of V
            
            % calculating the observation vector of ThWF (quadratic measurements)
            y_twf = y.^2; % calculating the observation vector of ThWF

            %% use CoPRAM/PR-SPCA/PR-SPCA-NT/ThWF/SPARTA to recover x1/x2/x3/x4/x5 
            fprintf('\nRunning CoPRAM . . .\n');
            x1 =  CoPRAM_init(y,A,s);   
            
            fprintf('\nRunning PR-SPCA . . .\n');
            x2 = GRQI(V,s,1,GRQI_iter,deflation_param,GRQI_power_iter,GRQI_thr); % 1 means that we only calculate the leading sparse eigenvector
            x2 = lambda * x2; % the output of GRQI is a unit vector; we need to multiply the estimated norm lambda
            
            fprintf('\nRunning PR-SPCA-NT . . .\n');
            x3 = GRQI(V_noTrunc,s,1,GRQI_iter,deflation_param,GRQI_power_iter,GRQI_thr); % 1 means that we only calculate the leading sparse eigenvector
            x3 = lambda * x3; % the output of GRQI is a unit vector; we need to multiply the estimated norm lambda
            
            fprintf('\nRunning Thresholded Wirtinger Flow . . .\n');
            x4 = Thresholded_WF_init(y_twf,A);
            
            fprintf('\nRunning Sparse Truncated Amplitude Flow . . .\n');
            x5 = SparTAF_init(y,A,s);
            
            % relative error w.r.t ground truth
            err_mat1(count_num,m_iter) = approx_err(x1,z);
            err_mat2(count_num,m_iter) = approx_err(x2,z);
            err_mat3(count_num,m_iter) = approx_err(x3,z);
            err_mat4(count_num,m_iter) = approx_err(x4,z);
            err_mat5(count_num,m_iter) = approx_err(x5,z);
        end
    end
end

%% save results for our first experiment (Sample size effect)
cd('results')
str_now=datestr(now,30);

strr1 = ['Exp1_CoPRAM_','n',num2str(n),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.mat'];
save(strr1,'err_mat1','n','kspan','mspan','trials_M');

strr2 = ['Exp1_errors_PR-SPCA_','n',num2str(n),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.mat'];
save(strr2,'err_mat2','n','kspan','mspan','trials_M');

strr3 = ['Exp1_errors_PR-SPCA-NT_noTrunc_','n',num2str(n),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.mat'];
save(strr3,'err_mat3','n','kspan','mspan','trials_M');

strr4 = ['Exp1_errors_ThWF_','n',num2str(n),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.mat'];
save(strr4,'err_mat4','n','kspan','mspan','trials_M');

strr5 = ['Exp1_errors_SparTA_','n',num2str(n),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.mat'];
save(strr5,'err_mat5','n','kspan','mspan','trials_M');
cd('..')

%% Plot the figures for our first experiment (Sample size effect)
for k=1:length(kspan)
    close all;
    figure; hold on;
    
    mean_val1 = mean(err_mat1(((k-1)*trials_M+1) : (k*trials_M),:)); std_val1 = std(err_mat1(((k-1)*trials_M+1) : (k*trials_M),:));
    mean_val2 = mean(err_mat2(((k-1)*trials_M+1) : (k*trials_M),:)); std_val2 = std(err_mat2(((k-1)*trials_M+1) : (k*trials_M),:));
    mean_val3 = mean(err_mat3(((k-1)*trials_M+1) : (k*trials_M),:)); std_val3 = std(err_mat3(((k-1)*trials_M+1) : (k*trials_M),:));
    mean_val4 = mean(err_mat4(((k-1)*trials_M+1) : (k*trials_M),:)); std_val4 = std(err_mat4(((k-1)*trials_M+1) : (k*trials_M),:));
    mean_val5 = mean(err_mat5(((k-1)*trials_M+1) : (k*trials_M),:)); std_val5 = std(err_mat5(((k-1)*trials_M+1) : (k*trials_M),:));
    
    ha = errorbar(mspan,mean_val1,std_val1,'s-','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','b'); hold on;
    hb = errorbar(mspan,mean_val2,std_val2,'d-','MarkerSize',10,'MarkerEdgeColor','r','MarkerFaceColor','r'); hold on;
    hc = errorbar(mspan,mean_val3,std_val3,'o-','MarkerSize',10,'MarkerEdgeColor','g','MarkerFaceColor','g'); hold on;
    hd = errorbar(mspan,mean_val4,std_val4,'+-','MarkerSize',10,'MarkerEdgeColor','m','MarkerFaceColor','m'); hold on;
    he = errorbar(mspan,mean_val5,std_val5,'p-','MarkerSize',10,'MarkerEdgeColor','c','MarkerFaceColor','c'); hold on;
    
    legend('CoPRAM','PRI-SPCA','PRI-SPCA-NT','ThWF','SPARTA','Location', 'Northeast')

    min_relerr = min(min([mean_val1-std_val1;mean_val2-std_val2;mean_val3-std_val3;mean_val4-std_val4;mean_val5-std_val5]));
    max_relerr = max(max([mean_val1+std_val1;mean_val2+std_val2;mean_val3+std_val3;mean_val4+std_val4;mean_val5+std_val5]));
    
    xlabel(['No. of measurements $m$; $s = $ ', num2str(kspan(k))],'Interpreter','latex')
    ylabel('Initial relative error')
    axis([min(mspan) max(mspan) min_relerr max_relerr])
    box on
    grid on
    set(gcf,'color','w');
    set(gca,'FontSize',18);
    filename=['Figures/Exp1_n',num2str(n),'_s',num2str(kspan(k)),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.pdf'];
    export_fig(gcf,'Color','Transparent',filename);
end



%% Next, we consider reproducing Figure 2 (Sparsity effect)

%% parameters used in the experiment for Figure 2 (Sparsity effect)
kspan = 5:5:50; % sparsity level
kl = length(kspan);
mspan = [1000,2000]; % no. of measurements
ml = length(mspan);
sigma = 0; % noise level

%% the matrices to record the numerical results (Sparsity effect)
err_mat1 = zeros(trials_M * ml,kl); % record the relative error for CoPRAM
err_mat2 = zeros(trials_M * ml,kl); % record the relative error for PR-SPCA
err_mat3 = zeros(trials_M * ml,kl); % record the relative error for PR-SPCA-NT
err_mat4 = zeros(trials_M * ml,kl); % record the relative error for ThWF
err_mat5 = zeros(trials_M * ml,kl); % record the relative error for SparAF

%% the main iteration in the experiment for Figure 2 (Sparsity effect)
for k_iter = 1:kl
    for m_iter = 1:ml
        s = kspan(k_iter);
        m = mspan(m_iter);
        for tr = 1:trials_M
            count_num = (m_iter - 1) * trials_M + tr;
            
            fprintf('\nTrial no. :%d\nNo. of measurements M :%d\nSparsity K :%d\n',tr,m,s);
            %% generate signal and measurements
            [z,z_ind] =  generate_signal(n,s); % generate the signal
            znorm = norm(z); % the norm of the signal
            A = randn(m,n); % generate the sensing matrix
            y  = abs(A * z) + sigma * znorm * randn(m,1);
            
            % calcuting our empirical matrices V and \tilde{V}
            lambda = sqrt(pi/2) * sum(y(:)) / numel(y(:)); % calculating lambda
            ytr = y.* ((y > l * lambda) & (y < u * lambda)); % truncated version
            V = A' * diag(ytr) * A / m; % calculating V
            V_noTrunc = A' * diag(y) * A / m; % calculating the non-truncated version of V
            
            % calculating the observation vector of ThWF (quadratic measurements)
            y_twf = y.^2; % calculating the observation vector of ThWF
            
            %% use CoPRAM/PR-SPCA/PR-SPCA-NT/ThWF/SPARTA to recover x1/x2/x3/x4/x5
            fprintf('\nRunning CoPRAM . . .\n');
            x1 =  CoPRAM_init(y,A,s);
            
            fprintf('\nRunning PR-SPCA . . .\n');
            x2 = GRQI(V,s,1,GRQI_iter,deflation_param,GRQI_power_iter,GRQI_thr); % 1 means that we only calculate the leading sparse eigenvector
            x2 = lambda * x2; % the output of GRQI is a unit vector; we need to multiply the estimated norm lambda
            
            fprintf('\nRunning PR-SPCA-NT . . .\n');
            x3 = GRQI(V_noTrunc,s,1,GRQI_iter,deflation_param,GRQI_power_iter,GRQI_thr); % 1 means that we only calculate the leading sparse eigenvector
            x3 = lambda * x3; % the output of GRQI is a unit vector; we need to multiply the estimated norm lambda
            
            fprintf('\nRunning Thresholded Wirtinger Flow . . .\n');
            x4 = Thresholded_WF_init(y_twf,A);
            
            fprintf('\nRunning Sparse Truncated Amplitude Flow . . .\n');
            x5 = SparTAF_init(y,A,s);
            
            % relative error w.r.t ground truth
            err_mat1(count_num,k_iter) = approx_err(x1,z);
            err_mat2(count_num,k_iter) = approx_err(x2,z);
            err_mat3(count_num,k_iter) = approx_err(x3,z);
            err_mat4(count_num,k_iter) = approx_err(x4,z);
            err_mat5(count_num,k_iter) = approx_err(x5,z);
        end
    end
end

%% save results for our second experiment (Sparsity effect)
cd('results')
str_now=datestr(now,30);

strr1 = ['Exp2_CoPRAM_','n',num2str(n),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.mat'];
save(strr1,'err_mat1','n','kspan','mspan','trials_M');

strr2 = ['Exp2_PR-SPCA_','n',num2str(n),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.mat'];
save(strr2,'err_mat2','n','kspan','mspan','trials_M');

strr3 = ['Exp2_PR-SPCA-NT_noTrunc_','n',num2str(n),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.mat'];
save(strr3,'err_mat3','n','kspan','mspan','trials_M');

strr4 = ['Exp2_ThWF_','n',num2str(n),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.mat'];
save(strr4,'err_mat4','n','kspan','mspan','trials_M');

strr5 = ['Exp2_SparTA_','n',num2str(n),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.mat'];
save(strr5,'err_mat5','n','kspan','mspan','trials_M');
cd('..')

%% Plot the figures for our second experiment (Sparsity effect)
for k=1:length(kspan)
    close all;
    figure; hold on;
    
    mean_val1 = mean(err_mat1(((k-1)*trials_M+1) : (k*trials_M),:)); std_val1 = std(err_mat1(((k-1)*trials_M+1) : (k*trials_M),:));
    mean_val2 = mean(err_mat2(((k-1)*trials_M+1) : (k*trials_M),:)); std_val2 = std(err_mat2(((k-1)*trials_M+1) : (k*trials_M),:));
    mean_val3 = mean(err_mat3(((k-1)*trials_M+1) : (k*trials_M),:)); std_val3 = std(err_mat3(((k-1)*trials_M+1) : (k*trials_M),:));
    mean_val4 = mean(err_mat4(((k-1)*trials_M+1) : (k*trials_M),:)); std_val4 = std(err_mat4(((k-1)*trials_M+1) : (k*trials_M),:));
    mean_val5 = mean(err_mat5(((k-1)*trials_M+1) : (k*trials_M),:)); std_val5 = std(err_mat5(((k-1)*trials_M+1) : (k*trials_M),:));
    
    ha = errorbar(kspan,mean_val1,std_val1,'s-','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','b'); hold on;
    hb = errorbar(kspan,mean_val2,std_val2,'d-','MarkerSize',10,'MarkerEdgeColor','r','MarkerFaceColor','r'); hold on;
    hc = errorbar(kspan,mean_val3,std_val3,'o-','MarkerSize',10,'MarkerEdgeColor','g','MarkerFaceColor','g'); hold on;
    hd = errorbar(kspan,mean_val4,std_val4,'+-','MarkerSize',10,'MarkerEdgeColor','m','MarkerFaceColor','m'); hold on;
    he = errorbar(kspan,mean_val5,std_val5,'p-','MarkerSize',10,'MarkerEdgeColor','c','MarkerFaceColor','c'); hold on;
    
    legend('CoPRAM','PRI-SPCA','PRI-SPCA-NT','ThWF','SPARTA','Location', 'Northeast')

    min_relerr = min(min([mean_val1-std_val1;mean_val2-std_val2;mean_val3-std_val3;mean_val4-std_val4;mean_val5-std_val5]));
    max_relerr = max(max([mean_val1+std_val1;mean_val2+std_val2;mean_val3+std_val3;mean_val4+std_val4;mean_val5+std_val5]));
    
    xlabel(['Sparisty level $s$; $m = $ ', num2str(m)],'Interpreter','latex')
    ylabel('Initial relative error')
    axis([min(kspan) max(kspan) min_relerr max_relerr])
    box on
    grid on
    set(gcf,'color','w');
    set(gca,'FontSize',18);
    filename=['Figures/Exp2_n',num2str(n),'_m',num2str(mspan(k)),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.pdf'];
    export_fig(gcf,'Color','Transparent',filename);
end


%% Next, we consider reproducing Figure 3 (Noise effect)

%% parameters used in the experiment for Figure 3 (Noise effect)
kspan = [10,20]; % sparsity level
kl = length(kspan);
m = 3000; % no. of measurements
sigmaSpan = 0.1:0.1:1; % noise level
sigmal = length(sigmaSpan);

%% the matrices to record the numerical results (Noise effect)
err_mat1 = zeros(kl * trials_M,sigmal); % record the relative error for CoPRAM
err_mat2 = zeros(kl * trials_M,sigmal); % record the relative error for PR-SPCA
err_mat3 = zeros(kl * trials_M,sigmal); % record the relative error for PR-SPCA-NT
err_mat5 = zeros(kl * trials_M,sigmal); % record the relative error for SparAF

%% the main iteration in the experiment for Figure 3 (Noise effect)
for k_iter = 1:kl
    for sigma_iter = 1:sigmal
        s = kspan(k_iter);
        sigma = sigmaSpan(sigma_iter);
        for tr = 1:trials_M
            count_num = (k_iter - 1) * trials_M + tr;
            
            fprintf('\nTrial no. :%d\nNo. of measurements M :%d\nSparsity K :%d\n',tr,m,s);
            %% generate signal and measurements
            [z,z_ind] =  generate_signal(n,s); % generate the signal 
            znorm = norm(z); % the norm of the signal
            A = randn(m,n); % generate the sensing matrix
            y  = abs(A * z) + sigma * znorm * randn(m,1); 
            
            % calcuting our empirical matrices V and \tilde{V}
            lambda = sqrt(pi/2) * sum(y(:)) / numel(y(:)); % calculating lambda
            ytr = y.* ((y > l * lambda) & (y < u * lambda)); % truncated version
            V = A' * diag(ytr) * A / m; % calculating V
            V_noTrunc = A' * diag(y) * A / m; % calculating the non-truncated version of V

            %% use CoPRAM/PR-SPCA/PR-SPCA-NT/SPARTA to recover x1/x2/x3/x5 
            fprintf('\nRunning CoPRAM . . .\n');
            x1 =  CoPRAM_init(y,A,s);   
            
            fprintf('\nRunning PR-SPCA . . .\n');
            x2 = GRQI(V,s,1,GRQI_iter,deflation_param,GRQI_power_iter,GRQI_thr); % 1 means that we only calculate the leading sparse eigenvector
            x2 = lambda * x2; % the output of GRQI is a unit vector; we need to multiply the estimated norm lambda
            
            fprintf('\nRunning PR-SPCA-NT . . .\n');
            x3 = GRQI(V_noTrunc,s,1,GRQI_iter,deflation_param,GRQI_power_iter,GRQI_thr); % 1 means that we only calculate the leading sparse eigenvector
            x3 = lambda * x3; % the output of GRQI is a unit vector; we need to multiply the estimated norm lambda
            
            fprintf('\nRunning Sparse Truncated Amplitude Flow . . .\n');
            x5 = SparTAF_init(y,A,s);

            % relative error w.r.t ground truth
            err_mat1(count_num,sigma_iter) = approx_err(x1,z);
            err_mat2(count_num,sigma_iter) = approx_err(x2,z);
            err_mat3(count_num,sigma_iter) = approx_err(x3,z);
            err_mat5(count_num,sigma_iter) = approx_err(x5,z);
        end
    end
end

%% save results for our third experiment (Noise effect)
cd('results')
str_now=datestr(now,30);

strr1 = ['Exp3_CoPRAM_','n',num2str(n),'_trials',num2str(trials_M),'_m',num2str(m),'_',str_now(1:12),'.mat'];
save(strr1,'err_mat1','n','m','kspan','sigmaSpan','trials_M');

strr2 = ['Exp3_PR-SPCA_','n',num2str(n),'_trials',num2str(trials_M),'_m',num2str(m),'_',str_now(1:12),'.mat'];
save(strr2,'err_mat2','n','m','kspan','sigmaSpan','trials_M');

strr3 = ['Exp3_PR-SPCA-NT_noTrunc_','n',num2str(n),'_trials',num2str(trials_M),'_m',num2str(m),'_',str_now(1:12),'.mat'];
save(strr3,'err_mat3','n','m','kspan','sigmaSpan','trials_M');

strr5 = ['Exp3_SparTA_','n',num2str(n),'_trials',num2str(trials_M),'_m',num2str(m),'_',str_now(1:12),'.mat'];
save(strr5,'err_mat5','n','m','kspan','sigmaSpan','trials_M');
cd('..')

%% Plot the figures for our third experiment (Noise effect)
for k=1:length(kspan)
    close all;
    figure; hold on;
    
    mean_val1 = mean(err_mat1(((k-1)*trials_M+1) : (k*trials_M),:)); std_val1 = std(err_mat1(((k-1)*trials_M+1) : (k*trials_M),:));
    mean_val2 = mean(err_mat2(((k-1)*trials_M+1) : (k*trials_M),:)); std_val2 = std(err_mat2(((k-1)*trials_M+1) : (k*trials_M),:));
    mean_val3 = mean(err_mat3(((k-1)*trials_M+1) : (k*trials_M),:)); std_val3 = std(err_mat3(((k-1)*trials_M+1) : (k*trials_M),:));
    mean_val5 = mean(err_mat5(((k-1)*trials_M+1) : (k*trials_M),:)); std_val5 = std(err_mat5(((k-1)*trials_M+1) : (k*trials_M),:));
    
    ha = errorbar(sigmaSpan,mean_val1,std_val1,'s-','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','b'); hold on;
    hb = errorbar(sigmaSpan,mean_val2,std_val2,'d-','MarkerSize',10,'MarkerEdgeColor','r','MarkerFaceColor','r'); hold on;
    hc = errorbar(sigmaSpan,mean_val3,std_val3,'o-','MarkerSize',10,'MarkerEdgeColor','g','MarkerFaceColor','g'); hold on;
    he = errorbar(sigmaSpan,mean_val5,std_val5,'p-','MarkerSize',10,'MarkerEdgeColor','c','MarkerFaceColor','c'); hold on;
    
    legend('CoPRAM','PRI-SPCA','PRI-SPCA-NT','SPARTA')

    min_relerr = min(min([mean_val1-std_val1;mean_val2-std_val2;mean_val3-std_val3;mean_val5-std_val5]));
    max_relerr = max(max([mean_val1+std_val1;mean_val2+std_val2;mean_val3+std_val3;mean_val5+std_val5]));
    
    xlabel(['Noise level $\sigma$; $s = $ ', num2str(kspan(k))],'Interpreter','latex')
    ylabel('Initial relative error')
    axis([min(sigmaSpan) max(sigmaSpan) min_relerr max_relerr])
    box on
    grid on
    set(gcf,'color','w');
    set(gca,'FontSize',18);
    filename=['Figures/Exp3_n',num2str(n),'_s',num2str(kspan(k)),'_trials',num2str(trials_M),'_m',num2str(m),'_',str_now(1:12),'.pdf'];
    export_fig(gcf,'Color','Transparent',filename);
end
    