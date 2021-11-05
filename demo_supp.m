% This matlab script contains all the experiments performed in our supplementray material.

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
batch_size = 50; % the number of random trials each time
batch_num = 10; % the number of repeated times
trials_M = batch_size * batch_num; % the number of total random trials; to produce meaningful error bars for empirical success rates

l = 1; u = 5; % parameters for calculating our weighted empirical covariance matrix V
GRQI_iter = 100; % no. of total iterations in GRQI
GRQI_power_iter = 100; % no. of power iterations in GRQI
deflation_param = 0.2; % the deflation parameter of GRQI
GRQI_thr = 1e-6; % the accuracy threshold of GRQI

iter = 100; % maximum iterations for the subsequent iterative algorithm of CoPRAM 

%% parameters used in the experiment for Figures 4 and 5 
kspan = [10,20]; % sparsity level
kl = length(kspan);
mspan = 100:50:1000; % no. of measurements
ml = length(mspan);
sigma = 0; % noise level

%% the matrices to record the numerical results for Figures 4 and 5 
err_mat1 = zeros(kl * trials_M,ml); % record the relative error for CoPRAM
err_mat2 = zeros(kl * trials_M,ml); % record the relative error for PR-SPCA
err_mat3 = zeros(kl * trials_M,ml); % record the relative error for PR-SPCA-NT
err_mat4 = zeros(kl * trials_M,ml); % record the relative error for ThWF
err_mat5 = zeros(kl * trials_M,ml); % record the relative error for SparAF
err_mat6 = zeros(kl * trials_M,ml); % record the relative error for RandInit

%% the main iteration in the experiment for Figures 4 and 5 
for k_iter = 1:kl
    for m_iter = 1:ml
        s = kspan(k_iter);
        m = mspan(m_iter);
        for tr = 1:trials_M
            count_num = (k_iter - 1) * trials_M + tr;
            
            fprintf('\nTrial no. :%d\nNo. of measurements M :%d\nSparsity K :%d\n',tr,m,s);
            %% generate signal and measurements
            [z,z_ind] =  generate_signal(n,s); % generate the signal
            znorm = norm(z); % the norm of the signal
            A = randn(m,n); % generate the sensing matrix
            y  = abs(A * z) + sigma * znorm * randn(m,1); % the observed vector
            
            % calcuting our empirical matrices V and \tilde{V}
            lambda = sqrt(pi/2) * sum(y(:)) / numel(y(:)); % calculating lambda
            ytr = y.* ((y > l * lambda) & (y < u * lambda)); % truncated version
            V = A' * diag(ytr) * A / m; % calculating V
            V_noTrunc = A' * diag(y) * A / m; % calculating the non-truncated version of V
            
            % calculating the observation vector of ThWF (quadratic measurements)
            y_twf = y.^2; % calculating the observation vector of ThWF
            
            %% use CoPRAM/PR-SPCA/PR-SPCA-NT/ThWF/SPARTA/randInit - recover x1/x2/x3/x4/x5/x6
            fprintf('\nRunning CoPRAM . . .\n');
            x1 =  CoPRAM_init(y,A,s);
            x1_GD = CoPRAM_GD(y,x1,A,s,iter);
            
            
            fprintf('\nRunning PR-SPCA . . .\n');
            x2 = GRQI(V,s,1,GRQI_iter,deflation_param,GRQI_power_iter,GRQI_thr); % 1 means that we only calculate the leading sparse eigenvector
            x2 = lambda * x2; % the output of GRQI is a unit vector; we need to multiply the estimated norm lambda
            x2_GD = CoPRAM_GD(y,x2,A,s,iter);
            
            
            fprintf('\nRunning PR-SPCA-NT . . .\n');
            x3 = GRQI(V_noTrunc,s,1,GRQI_iter,deflation_param,GRQI_power_iter,GRQI_thr); % 1 means that we only calculate the leading sparse eigenvector
            x3 = lambda * x3; % the output of GRQI is a unit vector; we need to multiply the estimated norm lambda
            x3_GD = CoPRAM_GD(y,x3,A,s,iter);
            
            
            fprintf('\nRunning Thresholded Wirtinger Flow . . .\n');
            x4 = Thresholded_WF_init(y_twf,A);
            x4_GD = CoPRAM_GD(y,x4,A,s,iter);
            
            
            fprintf('\nRunning Sparse Truncated Amplitude Flow . . .\n');
            x5 = SparTAF_init(y,A,s);
            x5_GD = CoPRAM_GD(y,x5,A,s,iter);
            
            fprintf('\nRunning RandInit. . .\n');
            x6 = randn(n,1);
            x6 = lambda * x6 / norm(x6); % random initial vector
            x6_GD = CoPRAM_GD(y,x6,A,s,iter);
            
            %error w.r.t ground truth
            err_mat1(count_num,m_iter) = approx_err(x1_GD,z);
            err_mat2(count_num,m_iter) = approx_err(x2_GD,z);
            err_mat3(count_num,m_iter) = approx_err(x3_GD,z);
            err_mat4(count_num,m_iter) = approx_err(x4_GD,z);
            err_mat5(count_num,m_iter) = approx_err(x5_GD,z);
            err_mat6(count_num,m_iter) = approx_err(x6_GD,z);
        end
    end
end

%% save results for our Figures 4 and 5
cd('results')
str_now=datestr(now,30);

strr1 = ['Exp4_GD_CoPRAM_','n',num2str(n),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.mat'];
save(strr1,'err_mat1','n','kspan','mspan','trials_M');

strr2 = ['Exp4_GD_PR-SPCA_','n',num2str(n),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.mat'];
save(strr2,'err_mat2','n','kspan','mspan','trials_M');

strr3 = ['Exp4_GD_PR-SPCA-NT_','n',num2str(n),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.mat'];
save(strr3,'err_mat3','n','kspan','mspan','trials_M');

strr4 = ['Exp4_GD_ThWF_','n',num2str(n),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.mat'];
save(strr4,'err_mat4','n','kspan','mspan','trials_M');

strr5 = ['Exp4_GD_SparTA_','n',num2str(n),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.mat'];
save(strr5,'err_mat5','n','kspan','mspan','trials_M');

strr6 = ['Exp4_GD_randInit_','n',num2str(n),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.mat'];
save(strr6,'err_mat6','n','kspan','mspan','trials_M');
cd('..')
    

%% Plot Figures 4 and 5 
for k=1:length(kspan)
    close all;
    mean_val1 = mean(err_mat1(((k-1)*trials_M+1) : (k*trials_M),:)); std_val1 = std(err_mat1(((k-1)*trials_M+1) : (k*trials_M),:));
    mean_val2 = mean(err_mat2(((k-1)*trials_M+1) : (k*trials_M),:)); std_val2 = std(err_mat2(((k-1)*trials_M+1) : (k*trials_M),:));
    mean_val3 = mean(err_mat3(((k-1)*trials_M+1) : (k*trials_M),:)); std_val3 = std(err_mat3(((k-1)*trials_M+1) : (k*trials_M),:));
    mean_val4 = mean(err_mat4(((k-1)*trials_M+1) : (k*trials_M),:)); std_val4 = std(err_mat4(((k-1)*trials_M+1) : (k*trials_M),:));
    mean_val5 = mean(err_mat5(((k-1)*trials_M+1) : (k*trials_M),:)); std_val5 = std(err_mat5(((k-1)*trials_M+1) : (k*trials_M),:));
    mean_val6 = mean(err_mat6(((k-1)*trials_M+1) : (k*trials_M),:)); std_val6 = std(err_mat6(((k-1)*trials_M+1) : (k*trials_M),:));
    
    err_bar1_mat = zeros(batch_num,ml); err_bar2_mat = zeros(batch_num,ml);
    err_bar3_mat = zeros(batch_num,ml); err_bar4_mat = zeros(batch_num,ml);
    err_bar5_mat = zeros(batch_num,ml); err_bar6_mat = zeros(batch_num,ml);
    for batch_iter = 1:batch_num
        ind_set = (((batch_iter-1)*batch_size + 1):(batch_iter*batch_size));
        err_bar1_mat(batch_iter,:) = mean(err_mat1(ind_set,:));
        err_bar2_mat(batch_iter,:) = mean(err_mat2(ind_set,:));
        err_bar3_mat(batch_iter,:) = mean(err_mat3(ind_set,:));
        err_bar4_mat(batch_iter,:) = mean(err_mat4(ind_set,:));
        err_bar5_mat(batch_iter,:) = mean(err_mat5(ind_set,:));
        err_bar6_mat(batch_iter,:) = mean(err_mat6(ind_set,:));
    end
    err_bar1_vec = std(err_bar1_mat); err_bar2_vec = std(err_bar2_mat); 
    err_bar3_vec = std(err_bar3_mat); err_bar4_vec = std(err_bar4_mat); 
    err_bar5_vec = std(err_bar5_mat); err_bar6_vec = std(err_bar6_mat); 
    
    
    ha = errorbar(mspan,mean_val1,err_bar1_vec,'s-','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','b'); hold on;
    hb = errorbar(mspan,mean_val2,err_bar2_vec,'d-','MarkerSize',10,'MarkerEdgeColor','r','MarkerFaceColor','r'); hold on;
    hc = errorbar(mspan,mean_val3,err_bar3_vec,'o-','MarkerSize',10,'MarkerEdgeColor','g','MarkerFaceColor','g'); hold on;
    hd = errorbar(mspan,mean_val4,err_bar4_vec,'+-','MarkerSize',10,'MarkerEdgeColor','m','MarkerFaceColor','m'); hold on;
    he = errorbar(mspan,mean_val5,err_bar5_vec,'p-','MarkerSize',10,'MarkerEdgeColor','c','MarkerFaceColor','c'); hold on;
    hf = errorbar(mspan,mean_val6,err_bar6_vec,'x-','MarkerSize',10,'MarkerEdgeColor','y','MarkerFaceColor','y'); hold on;
    
    legend('CoPRAM','PRI-SPCA','PRI-SPCA-NT','ThWF','SPARTA','RandInit','Location', 'Northeast')
    min_relerr = min(min([mean_val1-err_bar1_vec;mean_val2-err_bar2_vec;mean_val3-err_bar3_vec;mean_val4-err_bar4_vec;mean_val5-err_bar5_vec;mean_val6+err_bar6_vec]));
    max_relerr = max(max([mean_val1+err_bar1_vec;mean_val2+err_bar2_vec;mean_val3+err_bar3_vec;mean_val4+err_bar4_vec;mean_val5+err_bar5_vec;mean_val6+err_bar6_vec]));
        
    xlabel(['No. of measurements $m$; $s = $ ', num2str(kspan(k))],'Interpreter','latex')
    ylabel('Relative error')
    axis([min(mspan) max(mspan) min_relerr max_relerr])
    box on
    grid on
    set(gcf,'color','w');
    set(gca,'FontSize',18);
    filename=['Figures/Exp4_GD_n',num2str(n),'_s',num2str(kspan(k)),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.pdf'];
    export_fig(gcf,'Color','Transparent',filename);
    
    %% plot success rate
    close all;
    rel_err_thre = 0.01; % a trial is claimed success if the relative error is less than rel_err_thre
    success_vec1 = sum(err_mat1(((k-1)*trials_M+1) : (k*trials_M),:)<rel_err_thre)/trials_M;
    success_vec2 = sum(err_mat2(((k-1)*trials_M+1) : (k*trials_M),:)<rel_err_thre)/trials_M;
    success_vec3 = sum(err_mat3(((k-1)*trials_M+1) : (k*trials_M),:)<rel_err_thre)/trials_M;
    success_vec4 = sum(err_mat4(((k-1)*trials_M+1) : (k*trials_M),:)<rel_err_thre)/trials_M;
    success_vec5 = sum(err_mat5(((k-1)*trials_M+1) : (k*trials_M),:)<rel_err_thre)/trials_M;
    success_vec6 = sum(err_mat6(((k-1)*trials_M+1) : (k*trials_M),:)<rel_err_thre)/trials_M;
    
    err_bar1s_mat = zeros(batch_num,ml); err_bar2s_mat = zeros(batch_num,ml);
    err_bar3s_mat = zeros(batch_num,ml); err_bar4s_mat = zeros(batch_num,ml);
    err_bar5s_mat = zeros(batch_num,ml); err_bar6s_mat = zeros(batch_num,ml);
    for batch_iter = 1:batch_num
        ind_set = (((batch_iter-1)*batch_size + 1):(batch_iter*batch_size));
        err_bar1s_mat(batch_iter,:) = sum(err_mat1(ind_set,:)<rel_err_thre)/batch_size;
        err_bar2s_mat(batch_iter,:) = sum(err_mat2(ind_set,:)<rel_err_thre)/batch_size;
        err_bar3s_mat(batch_iter,:) = sum(err_mat3(ind_set,:)<rel_err_thre)/batch_size;
        err_bar4s_mat(batch_iter,:) = sum(err_mat4(ind_set,:)<rel_err_thre)/batch_size;
        err_bar5s_mat(batch_iter,:) = sum(err_mat5(ind_set,:)<rel_err_thre)/batch_size;
        err_bar6s_mat(batch_iter,:) = sum(err_mat6(ind_set,:)<rel_err_thre)/batch_size;
    end
    err_bar1s_vec = std(err_bar1s_mat); err_bar2s_vec = std(err_bar2s_mat); 
    err_bar3s_vec = std(err_bar3s_mat); err_bar4s_vec = std(err_bar4s_mat); 
    err_bar5s_vec = std(err_bar5s_mat); err_bar6s_vec = std(err_bar6s_mat); 
    
    ha = errorbar(mspan,success_vec1,err_bar1s_vec,'s-','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','b'); hold on;
    hb = errorbar(mspan,success_vec2,err_bar2s_vec,'d-','MarkerSize',10,'MarkerEdgeColor','r','MarkerFaceColor','r'); hold on;
    hc = errorbar(mspan,success_vec3,err_bar3s_vec,'o-','MarkerSize',10,'MarkerEdgeColor','g','MarkerFaceColor','g'); hold on;
    hd = errorbar(mspan,success_vec4,err_bar4s_vec,'+-','MarkerSize',10,'MarkerEdgeColor','m','MarkerFaceColor','m'); hold on;
    he = errorbar(mspan,success_vec5,err_bar5s_vec,'p-','MarkerSize',10,'MarkerEdgeColor','c','MarkerFaceColor','c'); hold on;
    hf = errorbar(mspan,success_vec6,err_bar6s_vec,'x-','MarkerSize',10,'MarkerEdgeColor','y','MarkerFaceColor','y'); hold on;
    
    legend('CoPRAM','PRI-SPCA','PRI-SPCA-NT','ThWF','SPARTA','RandInit','Location', 'Northwest')
    
    xlabel(['No. of measurements $m$; $s = $ ', num2str(kspan(k))],'Interpreter','latex')
    ylabel('Empirical success rate')
    axis([min(mspan) max(mspan) 0 1])
    box on
    grid on
    set(gcf,'color','w');
    set(gca,'FontSize',18);
    filename=['Figures/Exp4_rate_GD_n',num2str(n),'_s',num2str(kspan(k)),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.pdf'];
    export_fig(gcf,'Color','Transparent',filename);
end



%% Next, we consider reproducing Figure 6 (Relative error vs running time)

%% parameters used in the experiment for Figure 6 (Relative error vs running time)
m = 500; % no. of measurements
s = 20; % sparsity level
sigmaSpan = [0.1,0.2]; % noise level
sigmal = length(sigmaSpan);

%% the matrices to record the numerical results (Relative error vs running time)
% for the noisy case, we do not compare with ThWF as it considers quadratic measurements
time_mat1 = zeros(trials_M * sigmal,iter); % record the running time for CoPRAM
time_mat2 = zeros(trials_M * sigmal,iter); % record the running time for PR-SPCA
time_mat3 = zeros(trials_M * sigmal,iter); % record the running time for PR-SPCA-NT
time_mat5 = zeros(trials_M * sigmal,iter); % record the running time for SparAF
time_mat6 = zeros(trials_M * sigmal,iter); % record the running time for randInit

rel_err_mat1 = zeros(trials_M * sigmal,iter); % record the relative error for CoPRAM
rel_err_mat2 = zeros(trials_M * sigmal,iter); % record the relative error for PR-SPCA
rel_err_mat3 = zeros(trials_M * sigmal,iter); % record the relative error for PR-SPCA-NT
rel_err_mat5 = zeros(trials_M * sigmal,iter); % record the relative error for SparAF
rel_err_mat6 = zeros(trials_M * sigmal,iter); % record the relative error for randInit

%% the main iteration in the experiment for Figure 6 (Relative error vs running time)
for tr = 1:trials_M
    for sigma_iter = 1:sigmal
        sigma = sigmaSpan(sigma_iter);
        count_num = (sigma_iter - 1) * trials_M + tr;
        
        fprintf('\nTrial no. :%d\nNo. of measurements M :%d\nSparsity K :%d\n',tr,m,s);
        %% generate signal and measurements
        [z,z_ind] =  generate_signal(n,s); % generate the signal
        znorm = norm(z); % the norm of the signal
        A = randn(m,n); % generate the sensing matrix
        y  = abs(A * z) + sigma * znorm * randn(m,1); % the observed vector
        
        % calcuting our empirical matrices V and \tilde{V}
        lambda = sqrt(pi/2) * sum(y(:)) / numel(y(:)); % calculating lambda
        ytr = y.* ((y > l * lambda) & (y < u * lambda)); % truncated version
        V = A' * diag(ytr) * A / m; % calculating V
        V_noTrunc = A' * diag(y) * A / m; % calculating the non-truncated version of V
        
        %% use CoPRAM/PR-SPCA/PR-SPCA-NT/ThWF/SPARTA - recover x1/x2/x3/x5/x6
        fprintf('\nRunning CoPRAM . . .\n');
        tic;
        x1 =  CoPRAM_init(y,A,s); 
        t1 = toc; % record the running time for initialization 
        [x1_GD,err_vec1,time_vec1] =  CoPRAM_GD_time(y,x1,A,s,iter,z); % the subsequent iterative algorithm
        time_vec1 = t1 + time_vec1; % the time cost is the initialization time plus the running time of subsequent iterations
        time_mat1(count_num,:) = time_vec1;
        rel_err_mat1(count_num,:) = err_vec1;
        
        
        fprintf('\nRunning PR-SPCA . . .\n');
        tic;
        x2 = GRQI(V,s,1,GRQI_iter,deflation_param,GRQI_power_iter,GRQI_thr); % 1 means that we only calculate the leading sparse eigenvector
        x2 = lambda * x2;
        t2 = toc;  % record the running time for initialization 
        [x2_GD,err_vec2,time_vec2] =  CoPRAM_GD_time(y,x2,A,s,iter,z);  % the subsequent iterative algorithm
        time_vec2 = t2 + time_vec2; % the time cost is the initialization time plus the running time of subsequent iterations
        time_mat2(count_num,:) = time_vec2; 
        rel_err_mat2(count_num,:) = err_vec2;
        
        
        fprintf('\nRunning PR-SPCA-NT . . .\n');
        tic;
        x3 = GRQI(V_noTrunc,s,1,GRQI_iter,deflation_param,GRQI_power_iter,GRQI_thr); % 1 means that we only calculate the leading sparse eigenvector
        x3 = lambda * x3; % the output of GRQI is a unit vector; we need to multiply the estimated norm lambda
        t3 = toc;  % record the running time for initialization
        [x3_GD,err_vec3,time_vec3] =  CoPRAM_GD_time(y,x3,A,s,iter,z);  % the subsequent iterative algorithm
        time_vec3 = t3 + time_vec3; % the time cost is the initialization time plus the running time of subsequent iterations
        time_mat3(count_num,:) = time_vec3;
        rel_err_mat3(count_num,:) = err_vec3;
        
        fprintf('\nRunning Sparse Truncated Amplitude Flow . . .\n');
        tic;
        x5 = SparTAF_init(y,A,s);
        t5 = toc;  % record the running time for initialization 
        [x5_GD,err_vec5,time_vec5] =  CoPRAM_GD_time(y,x5,A,s,iter,z);  % the subsequent iterative algorithm
        time_vec5 = t5 + time_vec5; % the time cost is the initialization time plus the running time of subsequent iterations
        time_mat5(count_num,:) = time_vec5;
        rel_err_mat5(count_num,:) = err_vec5;
        
        fprintf('\nRunning RandInit. . .\n');
        tic;
        x6 = randn(n,1);
        x6 = lambda * x6 / norm(x6);
        t6 = toc;  % record the running time for initialization 
        [x6_GD,err_vec6,time_vec6] =  CoPRAM_GD_time(y,x6,A,s,iter,z);  % the subsequent iterative algorithm
        time_vec6 = t1 + time_vec6; % the time cost is the initialization time plus the running time of subsequent iterations
        time_mat6(count_num,:) = time_vec6;
        rel_err_mat6(count_num,:) = err_vec6;
    end
end

%% save results for Figure 6 (Relative error vs running time)

cd('results')
str_now=datestr(now,30);

strr1 = ['Exp6_relErrorVsTime_CoPRAM_','n',num2str(n),'_m',num2str(m),'_s',num2str(s),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.mat'];
save(strr1,'rel_err_mat1','time_mat1','n','m','s','sigmaSpan','trials_M','sigma');

strr2 = ['Exp6_relErrorVsTime_PR-SPCA_','n',num2str(n),'_m',num2str(m),'_s',num2str(s),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.mat'];
save(strr2,'rel_err_mat2','time_mat2','n','m','s','sigmaSpan','trials_M');

strr3 = ['Exp6_relErrorVsTime_PR-SPCA-NT_noTrunc_','n',num2str(n),'_m',num2str(m),'_s',num2str(s),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.mat'];
save(strr3,'rel_err_mat3','time_mat3','n','m','s','sigmaSpan','trials_M');

strr5 = ['Exp6_relErrorVsTime_SparTA_','n',num2str(n),'_m',num2str(m),'_s',num2str(s),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.mat'];
save(strr5,'rel_err_mat5','time_mat5','n','m','s','sigmaSpan','trials_M');

strr6 = ['Exp6_relErrorVsTime_randInit_','n',num2str(n),'_m',num2str(m),'_s',num2str(s),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.mat'];
save(strr6,'rel_err_mat6','time_mat6','n','m','s','sigmaSpan','trials_M');
cd('..')

%% Create the matrices of relative errors corresponding to tspan (Relative error vs running time)

tspan = 0.1:0.02:0.5;
tl = length(tspan);

tt_mat1 = zeros(trials_M * sigmal,tl); % record the relative error corresponding to tspan
tt_mat2 = zeros(trials_M * sigmal,tl); % record the relative error corresponding to tspan
tt_mat3 = zeros(trials_M * sigmal,tl); % record the relative error corresponding to tspan
tt_mat5 = zeros(trials_M * sigmal,tl); % record the relative error corresponding to tspan
tt_mat6 = zeros(trials_M * sigmal,tl); % record the relative error corresponding to tspan

for t_iter = 1:tl
    for tr = 1:(trials_M * sigmal)
        [~,ind] = min(abs(time_mat1(tr,:)-tspan(t_iter)));
        tt_mat1(tr,t_iter) = rel_err_mat1(tr,ind);
        
        [~,ind] = min(abs(time_mat2(tr,:)-tspan(t_iter)));
        tt_mat2(tr,t_iter) = rel_err_mat2(tr,ind);
        
        [~,ind] = min(abs(time_mat3(tr,:)-tspan(t_iter)));
        tt_mat3(tr,t_iter) = rel_err_mat3(tr,ind);
        
        [~,ind] = min(abs(time_mat5(tr,:)-tspan(t_iter)));
        tt_mat5(tr,t_iter) = rel_err_mat5(tr,ind);
        
        [~,ind] = min(abs(time_mat6(tr,:)-tspan(t_iter)));
        tt_mat6(tr,t_iter) = rel_err_mat6(tr,ind);
    end
end

%% Plot Figure 6

for k = 1:sigmal
    current_ind = ((k-1)*trials_M + 1):(k*trials_M);
    
    rel_err_vec1 = mean(tt_mat1(current_ind,:));
    rel_err_vec2 = mean(tt_mat2(current_ind,:));
    rel_err_vec3 = mean(tt_mat3(current_ind,:));
    rel_err_vec5 = mean(tt_mat5(current_ind,:));
    rel_err_vec6 = mean(tt_mat6(current_ind,:));
    
    err_bar1_mat = zeros(batch_num,tl); 
    err_bar2_mat = zeros(batch_num,tl);
    err_bar3_mat = zeros(batch_num,tl);
    err_bar5_mat = zeros(batch_num,tl); 
    err_bar6_mat = zeros(batch_num,tl);
    for batch_iter = 1:batch_num
        ind_set = ((k-1)*trials_M + (batch_iter-1)*batch_size + 1):((k-1)*trials_M + batch_iter*batch_size);
        err_bar1_mat(batch_iter,:) = mean(tt_mat1(ind_set,:));
        err_bar2_mat(batch_iter,:) = mean(tt_mat2(ind_set,:));
        err_bar3_mat(batch_iter,:) = mean(tt_mat3(ind_set,:));
        err_bar5_mat(batch_iter,:) = mean(tt_mat5(ind_set,:));
        err_bar6_mat(batch_iter,:) = mean(tt_mat6(ind_set,:));
    end
    rel_std_vec1 = std(err_bar1_mat); 
    rel_std_vec2 = std(err_bar2_mat);
    rel_std_vec3 = std(err_bar3_mat);
    rel_std_vec5 = std(err_bar5_mat); 
    rel_std_vec6 = std(err_bar6_mat);
    
    
    close all;
    ha = errorbar(tspan,rel_err_vec1,rel_std_vec1,'s-','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','b'); hold on;
    hb = errorbar(tspan,rel_err_vec2,rel_std_vec2,'d-','MarkerSize',10,'MarkerEdgeColor','r','MarkerFaceColor','r'); hold on;
    hc = errorbar(tspan,rel_err_vec3,rel_std_vec3,'o-','MarkerSize',10,'MarkerEdgeColor','g','MarkerFaceColor','g'); hold on;
    he = errorbar(tspan,rel_err_vec5,rel_std_vec5,'p-','MarkerSize',10,'MarkerEdgeColor','c','MarkerFaceColor','c'); hold on;
    hf = errorbar(tspan,rel_err_vec6,rel_std_vec6,'x-','MarkerSize',10,'MarkerEdgeColor','y','MarkerFaceColor','y'); hold on;
    
    legend('CoPRAM','PRI-SPCA','PRI-SPCA-NT','SPARTA','RandInit','Location', 'Northeast')
    
    xlabel(['Time cost $t$; $\sigma = $ ', num2str(sigmaSpan(k))],'Interpreter','latex')
    ylabel('Relative error')
    
    axis([min(tspan) max(tspan) (min(rel_err_vec2)-0.05) (max(rel_err_vec6)+0.1)])
    box on
    grid on
    set(gcf,'color','w');
    set(gca,'FontSize',18);
    str_now=datestr(now,30);
    filename=['Figures/Exp6_GD_n',num2str(n),'_m',num2str(m),', $s = $ ',num2str(s),'_trials',num2str(trials_M),'_sigma',num2str(sigma),'_',str_now(1:12),'.pdf'];
    export_fig(gcf,'Color','Transparent',filename);
end