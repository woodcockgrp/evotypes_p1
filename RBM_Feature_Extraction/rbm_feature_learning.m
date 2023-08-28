function [input_weights, hprob_mean, h_bias, v_bias, class_weights, d_bias]= rbm_feature_learning(rbm_params, X, y, X_valid)

%%% RBM_FEATURE_LEARNING uses a non-negative Restricted Boltzmann Machine to
%%% identify feature sets in mixed-type data. Based on ideas from Tran et al (2015)
%
% INPUT:
%   rbm_params      - Network parameters
%   X               - training data, scaled to [0,1]
%   y               - class targets
%   X_valid         - test data for early stopping, scaled to [0,1]
%
% OUTPUT:
%   input_weights   - non-negative weight matrix of input-feature mapping
%   hprob_mean      - mean of hidden unit activations in converged phase
%   h_bias          - hidden unit bias values
%   v_bias          - visible unit bias values
%   class_weights   - non-negative weight matrix of feature-class mapping
%   d_bias          - class unit bias values      

%%% remember the original input values
X_orig=X;
y_orig=y;

%%% initalise network parameters
n_hid=rbm_params.n_hid;                                                     % number of hidden units
[n_patients, n_inputs] = size(X);                                           % number of data points
n_classes = size(y,2);                                                      % number of classes

%%% initalise weights and biases
input_weights = abs(randn(n_inputs, n_hid))*rbm_params.weight_stddev;          % weights
input_weights(input_weights<0)=0;
h_bias = -1*ones(1,n_hid);                                                  % hidden unit biases
v_bias = log(mean(X).*(1-mean(X)));                                         % visible unit biases by proportion active in data
v_bias(~isfinite(v_bias))=min(v_bias(isfinite(v_bias)));                    % avoid any logarithmic strangeness
class_weights = randn(n_hid, n_classes)*rbm_params.classweight_stddev;      % class weights
d_bias = -1*ones(1,n_classes);                                             % class biases

%%% initialise gradient matrices
hgradient = zeros(1, n_hid);
vgradient = zeros(1, n_inputs);
wgradient = zeros(n_inputs, n_hid);
dgradient = zeros(1, n_classes);
ugradient = zeros(n_hid, n_classes);

%%% initialise training parameters
n_iter = 0; %
max_iter = rbm_params.max_iter;

n_hidmats=rbm_params.prune_check;  % after the final prune check, start taking aaverages
hprob_mean{n_hidmats}=[];

%%% initialise learning rate parameters
lrate.h = 0;
lrate.v = 0;
lrate.w = 0;
lrate.d = 0;
lrate.u = 0;

l1_strength_init=rbm_params.l1_strength;

%%% initialise mini batch variables
target_minibatch_size = rbm_params.batch_size;                              % target minibatch size
target_n_batches = ceil(n_patients/target_minibatch_size);                  % number of minibatches per iteration

% identify the two classes
class1 = find(y(:,1));
class2 = find(y(:,1)==0);
n_class1 = length(class1);
n_class2 = length(class2);

%n_batches=n_class1;  % desired number of batches
n_batches = floor(n_class2/ceil(n_class2/n_class1)); % number of batches

% calculate the number of each class in each batch
nclass1inbatch = floor(n_class1/n_batches);
%n_batches = nclass1inbatch*target_n_batches;
nclass2inbatch = floor(n_class2/n_batches);
% adjust the minibatch size and number of batched
mini_batch_size = nclass1inbatch + nclass2inbatch;
%n_batches = ceil(n_patients/mini_batch_size)-1; % the -1 ensures size last batch > mini_batch_size

curr_prune_index=[];

%%% early stopping initalisation
if rbm_params.do_early_stopping
    
    % subsample X to be same size as test set to calculate free energy
    nsamples_in_test = size(X_valid,1);                                     % number of samples in validation set
    num_train_subsamples = 10;                                              % number of training set subsamples to use
    
    % initialise storage
    X_train_inds=zeros(num_train_subsamples, nsamples_in_test);
    free_energy_diffs=zeros(max_iter, num_train_subsamples);
    
    % subsample the training set
    for j=1:num_train_subsamples
        
        X_train_inds(j,:)=randperm(size(X,1), nsamples_in_test);            % select indices at random from training set
        
    end
      
end
    

%%% begin network training
if rbm_params.do_plots
    fprintf(1, '%10s %15s %15s %15s %15s %15s\n', 'Iteration', 'Rec Error', 'Class Error', 'Learning Rate', 'Free Energy', 'Time');
end

% determine data types (currently only {0,1} and [0,1])
use_trunc = any(X~=1&X~=0);
n_trunc = sum(use_trunc);
n_exp = sum(~use_trunc);


start = tic();                                                              % start the stopwatch
for iter=n_iter+1:max_iter
    
    % shuffle X at each iteration to break local correaltions -
    % (Targets required to try and ensure at least one of each class per batch
    % not used otherwise)
    
    % reset the matrices to the original orders
    X=X_orig;
    y=y_orig;   
    
    % shuffle the indices
    shuf_class1 = class1(randperm(n_class1));
    shuf_class2 = class2(randperm(n_class2));
    
    % fill up the index labels
    new_inds = zeros(1,n_patients);
    for k=1:n_batches % only works on two classes
        
        if k < n_batches % create minibatches with contribution from each class
            
            try
            shuf_classes = [shuf_class1(((k-1)*nclass1inbatch+1):(nclass1inbatch*k));
                shuf_class2(((k-1)*nclass2inbatch+1):(nclass2inbatch*k))];
            catch ME
                break;
            end
            new_inds(((k-1)*mini_batch_size+1):(mini_batch_size*k)) = shuf_classes(randperm(length(shuf_classes)));
            
        else % add whatever is left to last batch
            
            shuf_classes = [shuf_class1(((k-1)*nclass1inbatch+1):end);
                shuf_class2(((k-1)*nclass2inbatch+1):end)];
            
            new_inds(((k-1)*mini_batch_size+1):end)=shuf_classes(randperm(length(shuf_classes)));
            
        end
        
    end
    
    % reorder the inputs
    [~, old_inds]=sort(new_inds);
    X=X(new_inds,:);
    y=y(new_inds,:);
    
    recon_err = 0;
    vprob_mat = zeros(size(X));
    hprob_mat = zeros(n_patients, n_hid);
    yprob_mat = zeros(size(y));
    
    % perform dropconnect in first part of training
    if rbm_params.do_dropconnect %& iter < rbm_params.prune_iter
        % perform dropconnect with the probability of the connection
        % being present proportional to the magnitude of the weight (with a
        % minimum probability) aross all patients.

        dropconnect_mask = rand(size(input_weights)) < max(abs(input_weights)./repmat(max(abs(input_weights),[],2),1,size(input_weights,2)),(1-iter/rbm_params.prune_iter)*rbm_params.dropconnect_prob);

    else
        dropconnect_mask = 1;
    end
    
    % set the weight matrix for this iteration
    curr_weights=input_weights.*dropconnect_mask;
    
    for batch_iter=1:mini_batch_size:n_patients  % CHECK THIS LOOPS THROUGH ALL THE DATA
        
        max_batch_size = min(n_patients, batch_iter + target_minibatch_size - 1);
        curr_batch_size = max_batch_size - batch_iter + 1;                  % number of samples in mini-batch
        batch_data = X(batch_iter:max_batch_size,:);                        % data mini-batch
        batch_targets = y(batch_iter:max_batch_size,:);                     % target mini-batch
        %batch_u = u(ibatch:ibatch_max,:);
        
        % clamp to hidden layer probabilities fixed on the data
        hprob = sigmoid(bsxfun(@plus, (1-rbm_params.class_proportion)*batch_data*curr_weights + rbm_params.class_proportion*batch_targets*class_weights', h_bias));		% compute hidden probabilities
        
        % calculate the positive gradients (data estimates)
        posgrad_h = mean(hprob);
        posgrad_v = mean(batch_data);
        posgrad_w = (batch_data'*hprob)/curr_batch_size;
        posgrad_d = mean(batch_targets);
        posgrad_u = (hprob'*batch_targets)/curr_batch_size;
        
        %---- free phase ----
        % contrastive divergence
        for icd=1:rbm_params.learning.n_cd
            
            hsample = double(rand(curr_batch_size, n_hid) < hprob);         % sample Bernoulli hidden units
            
            %%% update hybrid visible layer - sample the different data types separately %%%
            vactivations = bsxfun(@plus, hsample*curr_weights', v_bias);                    % calculate the visible activations
            vsample = double(rand(curr_batch_size, n_inputs) < sigmoid(vactivations));		% sample binary visible units
            % sample continuous visible units
            vsample(:,use_trunc) = log(1-rand(curr_batch_size,n_trunc).*(1-exp(vactivations(:,use_trunc))))./vactivations(:,use_trunc);
            
            %%% sample binary classification labels using softmax
            yexp = exp(repmat(d_bias,curr_batch_size,1)+hsample*class_weights);
            yprob = yexp./(repmat(sum(yexp,2),1,2));
            ysample = double(cumsum(rand(curr_batch_size,2) < cumsum(yprob,2),2)==1);
            
            % calculate the hidden unit probabilities again
            hprob = sigmoid(bsxfun(@plus, (1-rbm_params.class_proportion)*vsample*curr_weights + rbm_params.class_proportion*ysample*class_weights', h_bias));  
            
        end
        
        % fill in the full matrices for each layer 
        vprob_mat(batch_iter:max_batch_size,:)=vsample;
        hprob_mat(batch_iter:max_batch_size,:)=hprob;
        yprob_mat(batch_iter:max_batch_size,:)=yprob;
        
        % calculate the negative gradients (model estimates)
        neggrad_h = mean(hprob);
        neggrad_v = mean(vsample);
        neggrad_w = (vsample'*hprob)/curr_batch_size;
        neggrad_d = mean(ysample);
        neggrad_u = (hprob'*ysample)/curr_batch_size;
        
        %%% update gradients
        delta_C = sum(exp(-batch_data*curr_weights-h_bias)./abs(1+exp(-batch_data*curr_weights-h_bias)).^(3/2)); % L1/2 norm (Cui 2015)
        hgradient = lrate.h*(posgrad_h - neggrad_h - rbm_params.hreg_strength*delta_C);
        vgradient = lrate.v*(posgrad_v - neggrad_v);
        wgradient = dropconnect_mask.*(lrate.w*(posgrad_w - neggrad_w)) - lrate.w*((rbm_params.l1_strength*sign(curr_weights>0))); %  - lrate.w*rbm_params.l1_strength*sign(curr_weights));
        dgradient = lrate.d*(posgrad_d-neggrad_d);
        ugradient = lrate.u*(posgrad_u-neggrad_u);
        
        %%% nonnegative constraint
        idx = (curr_weights < 0);
        wgradient(idx) = wgradient(idx) - rbm_params.lrate.w*rbm_params.nonneg_strength*curr_weights(idx); % use the max learning rate to drive nonnegativity
        idx = (class_weights<=0);
        ugradient(idx) = ugradient(idx) - lrate.u*rbm_params.nonneg_strength*class_weights(idx);
        
        %%% update network varaibles
        h_bias = h_bias + hgradient;
        v_bias = v_bias + vgradient;
        input_weights = input_weights + wgradient;
        d_bias = d_bias + dgradient;
        class_weights = class_weights + ugradient;
        
        % rescale the weights for each feature by the maximum if above a threshold
        max_weight = 20;
        curr_max_weight = max(input_weights);
        length_index = curr_max_weight > max_weight; % find weights greater than the max allowed
        if any(length_index)
            norm_const=curr_max_weight(length_index)/max_weight;
            input_weights(:,length_index) = bsxfun(@rdivide, input_weights(:,length_index), norm_const);
        end
        
        
        max_weight_length = 200^2;
        class_weight_length = sum(class_weights.^2,2);
        length_index = class_weight_length > max_weight_length;
        if any(length_index)
            norm_const=sqrt(class_weight_length(length_index)/(max_weight_length));
            class_weights(length_index,:) = bsxfun(@rdivide, class_weights(length_index,:), norm_const);
        end

        recon_err = recon_err + sum(sum(abs(batch_data-vsample)));          % compute reconstruction error
        
    end
    
    %%% early stopping %%%
    if rbm_params.do_early_stopping
        
        % initalise storage
        Xtrain_free_energy = zeros(1,num_train_subsamples);
        % subsample the training set and calculate free energy
        for j=1:num_train_subsamples
            
            X_train = X(X_train_inds(j,:),:);                       % use same indices each time
            % calculate the average free energy in the subsampled training set data
            Xtrain_free_energy(j) = mean(-X_train*v_bias'-sum(log(1+exp(X_train*input_weights+repmat(h_bias,size(X_train,1),1))),2));% + R.class_proportion*ysample*u', h)
            
        end
        % calculate the average free energy in the validation set data
        Xvalid_free_energy = mean(-X_valid*v_bias'-sum(log(1+exp(X_valid*input_weights+repmat(h_bias,size(X_valid,1),1))),2));
        
        % calculate differences between the energies
        free_energy_diffs(iter,:) = Xtrain_free_energy - Xvalid_free_energy;
        
        num_train_energy_higher=0;
        % if there are enough iterations gone to detect if the avg free energy difference is rising
        if iter == rbm_params.early_stopping_trackback
            % create an estimate for how big the baseline energy differential is (i.e. the gap size)
            energy_differential = mean(mean(free_energy_diffs(1:rbm_params.early_stopping_trackback,:)));
            
        end
        if iter > rbm_params.early_stopping_trackback
            
            % if Xvalid energies are consistently outside the range of Xtrain energies then stop
            num_train_energy_higher = sum((free_energy_diffs((iter-rbm_params.early_stopping_trackback):iter,:)-energy_differential)<0,2);
            
            % if the free energy of all the subsamples in trainin set rise
            % on average over the iterations then stop
            
            % if all the differences are positive (i.e. the gap is growing) then stop      
            if (all(num_train_energy_higher == 1))%|all(free_energy_deviations(:) == -1))
                disp('Danger of overtraining, training stopped early')
                return;
            end
        end
        
    end
    
    % prune neurons in second half of training
    
    if iter == rbm_params.prune_iter  % set a parameters to clean up the weights
        
        % increase the strength of regularisation to push low weights to zero
        rbm_params.l1_strength=rbm_params.increase_penalty*l1_strength_init;
        
    end
    
    if mod(iter,rbm_params.prune_check)==0 && iter >= rbm_params.prune_iter
        
        
        prev_prune_index=curr_prune_index;
        %%% prune low influence features %%%
        curr_prune_index=find(var(hprob_mat)<0.0001); % dead units
        prev_prune_index=curr_prune_index;  % remember the dead units for the next iteration
        curr_prune_index=intersect(prev_prune_index, curr_prune_index);  % still dead
           
        % remove these features from all the relevant matrices
        input_weights(:,curr_prune_index)=[];
        wgradient(:,curr_prune_index)=[];
        h_bias(curr_prune_index)=[];
        hgradient(curr_prune_index)=[];
        class_weights(curr_prune_index,:) = [];
        ugradient(curr_prune_index,:) = [];
        
        n_hid=n_hid-numel(curr_prune_index);                      %  recalculate the number of hidden units
        
    end
    
    % reconstruction error
    recon_err = recon_err / (n_patients*n_inputs);
    
    % classification accuracy
    class_err = sum(abs(y(:,1) - yprob_mat(:,1)))/length(y);
    
    %%% cyclical learning rate parameter adjustment %%%
    
    % determine where we are in the cycle
    cycle=floor(1+(iter)/(2*rbm_params.half_cycle_length));
    adap_step=abs(iter/rbm_params.half_cycle_length - 2*cycle +1);
    
    % update the new learning rates
    lrate.h = rbm_params.lrate.h*(max(0,1-adap_step));
    lrate.v = rbm_params.lrate.v*(max(0,1-adap_step));
    lrate.w = rbm_params.lrate.w*(max(0,1-adap_step));
    lrate.d = rbm_params.lrate.d*(max(0,1-adap_step));
    lrate.u = rbm_params.lrate.u*(max(0,1-adap_step));
    
  
end

end


