% somre flags to save speed/do plots
rbm_params.do_dropconnect = logical(1);                     % flag to perform dropconnect
rbm_params.do_early_stopping=logical(1);                    % flag to perform early stopping     
rbm_params.do_plots=logical(0);                             % flag to do plots

% initialise network parameters

rbm_params.n_hid = 200;
rbm_params.max_iter = 10000;
rbm_params.learning.n_cd = 1;
rbm_params.batch_size = 5;
rbm_params.weight_stddev = 1;
rbm_params.classweight_stddev = 0.001;
rbm_params.half_cycle_length = 500;
rbm_params.dropconnect_prob = 0.5;

% learning rates
lrate_scale=1;
rbm_params.lrate.h = 0.05*lrate_scale;
rbm_params.lrate.v = 0.05*lrate_scale;
rbm_params.lrate.w = 0.025*lrate_scale;
rbm_params.lrate.d = 0.05*lrate_scale;
rbm_params.lrate.u = 0.025*lrate_scale;

% regularisation parameters
rbm_params.l1_strength = 0.005;% 6e-4;%5E-3;
rbm_params.increase_penalty = 2;                            % increase L1 penalty X times in latter training stages
rbm_params.nonneg_strength = 0.5;                           % non-negativity
rbm_params.hreg_strength = 0.005;                            % hidden bias

% ancillary parameters
rbm_params.prune_iter = 3000;                               % training iteration at which weight pruning starts
rbm_params.prune_check = 100;                               % check features to prune every ...
rbm_params.plot_iters = 50;                                 % number of iterations between each plot
rbm_params.early_stopping_trackback = 10;

% vestigial from development version but included for interest
rbm_params.class_proportion = 0.0;                          % proportion of hidden unit probs from class side of network

%%% load in the data %%%

pca_data = importdata('./processed_159data_2020_04_24.txt','\t')

% sort to ensure the samples are in the same order

X=pca_data.data;
patient_ids = pca_data.textdata(2:end,1);
inputnames = pca_data.textdata(1,2:end);

nsamples = size(X,1);

N_LABELS = 1;

leave_out_percentage = 20; % leave out percentage of the data set each iteration
N_FOLDS=10;
num_to_leave_out=floor(nsamples*leave_out_percentage/100);

idx_train=zeros(N_FOLDS, num_to_leave_out);

for i=1:N_FOLDS

    idx_train(i,:)=randperm(nsamples, num_to_leave_out);
    
end
		
prob_train = cell(1, N_FOLDS);
prob_test = cell(1, N_FOLDS);

inst_enrbm_arr = cell(1, N_FOLDS);
inst_lr_arr = cell(1, N_FOLDS);

fprintf(1, 'Running RBM single layer training...\n');

meta_features={};

% use the outocme data to form classes so we can try to balance minibatches
prog_or_deceased = find(strcmp(inputnames,'progressed')|strcmp(inputnames,'deceased'));
relapse_free=find(strcmp(inputnames,'relapse free'));
y = [double(sum(X(:,prog_or_deceased),2)>0) X(:,relapse_free)];
X(:,[prog_or_deceased relapse_free])=[];
inputnames([prog_or_deceased relapse_free])=[];

weight_mat={};
ifold=1;

weight_mat{N_FOLDS}=[];
num_cores=5;
localpool=parpool(num_cores);
start_total_time=tic;

parfor ifold = 1:N_FOLDS

    % find the 
    indices=ones(nsamples,1);
    indices(idx_train(ifold,:))=0;
    X_train = X(logical(indices),:);  
    Y_train = y(logical(indices),:);
    X_valid = X(idx_train(ifold,:),:);
    
    % run until we get a solution (early stopping restarts with same data)
    while isempty(weight_mat{ifold})
        start_loop_time=tic;
        disp(['Running iteration ' num2str(ifold)]) 
        [weight_mat{ifold}] = rbm_feature_learning(rbm_params, X_train, Y_train, X_valid);
        end_loop_time=toc(start_loop_time);
        disp(['Iteration ', num2str(ifold), ', Time taken: ' num2str(end_loop_time/60)])
    end

end
end_total_time=toc(start_total_time);
disp(['Total Time: ',  char(duration([0, 0, end_total_time])), '. Average time per core: ', char(duration([0, 0, num_cores*end_total_time/N_FOLDS]))])
delete(localpool)

% amalgamate the features
[amalgamated_weights feature_names]=amalgamate_features(weight_mat, inputnames);

