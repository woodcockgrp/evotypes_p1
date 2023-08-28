function [feature_representation, class_weights, hprob_mean]= rbm_feature_scores(rbm_params, X, y, input_weights)

%%% RBM_FEATURE_SCORES uses a non-negative Restricted Boltzmann Machine to
%%% identify feature sets in mixed-type data scaled to [0,1]. Based on Tran et al (2015)

%%% remember the original input values
dead_inputs=all(input_weights'==0);
X(:,dead_inputs)=[];
input_weights(dead_inputs,:)=[];
X_orig=X;
y_orig=y;

%%% initalise network parameters
n_hid=size(input_weights,2);                                                % number of hidden units
[n_patients, n_inputs] = size(X);                                           % number of data points
n_classes = size(y,2);                                                      % number of classes

%%% initalise weights and biases
h_bias = -1*ones(1,n_hid);                                                  % hidden unit biases
v_bias = log(mean(X).*(1-mean(X)));                                         % visible unit biases by proportion active in data
v_bias(~isfinite(v_bias))=min(v_bias(isfinite(v_bias)));                    % avoid any logarithmic strangeness
%v_bias = -20*ones(1,n_inputs);
class_weights = randn(n_hid, n_classes)*rbm_params.classweight_stddev;      % class weights
d_bias = -1*ones(1,n_classes); % class biases

%%% initialise gradient matrices
hgradient = zeros(1, n_hid);
vgradient = zeros(1, n_inputs);
dgradient = zeros(1, n_classes);
ugradient = zeros(n_hid, n_classes);

%%% initialise training parameters
n_iter = 0; %
max_iter = rbm_params.max_iter;

n_hprobs=1000; % atart taking aaverages
hprob_mean=zeros(n_patients, size(input_weights,2), n_hprobs);

%%% initialise learning rate parameters
lrate.h = 0;
lrate.v = 0;
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
    
    % reorder the
    [~, old_inds]=sort(new_inds);
    X=X(new_inds,:);
    y=y(new_inds,:);
    
    recon_err = 0;
    vprob_mat = zeros(size(X));
    hprob_mat = zeros(n_patients, n_hid);
    yprob_mat = zeros(size(y));
        
    % set the weight matrix for this iteration
        % perform dropconnect in first part of training
    if rbm_params.do_dropconnect %& iter < rbm_params.prune_iter
        % perform dropconnect with the probability of the connection
        % being present proportional to the magnitude of the weight (with a
        % minimum probability)  % n
        dropconnect_mask = rand(size(input_weights)) < max(abs(input_weights)./repmat(max(abs(input_weights),[],2),1,size(input_weights,2)),rbm_params.dropconnect_prob);
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
            yexp = exp(hsample*class_weights+repmat(d_bias,curr_batch_size,1));
            yprob = yexp./(repmat(sum(yexp,2),1,2));
            ysample = double(cumsum(rand(curr_batch_size,2) < cumsum(yprob,2),2)==1);
            
            % calculate the hidden unit probabilities again
            hprob = sigmoid(bsxfun(@plus, (1-rbm_params.class_proportion)*vsample*curr_weights + rbm_params.class_proportion*ysample*class_weights', h_bias));  
            
        end
        
        % fill in the full matrices for each layer for plotting
        vprob_mat(batch_iter:max_batch_size,:)=vsample;  
        hprob_mat(batch_iter:max_batch_size,:)=hprob;
        yprob_mat(batch_iter:max_batch_size,:)=yprob;
        
        % calculate the negative gradients (model estimates)
        neggrad_h = mean(hprob);
        neggrad_v = mean(vsample);
        neggrad_d = mean(ysample);
        neggrad_u = (hprob'*ysample)/curr_batch_size;
        
        %%% update gradients
        delta_C = sum(exp(-batch_data*curr_weights-h_bias)./abs(1+exp(-batch_data*curr_weights-h_bias)).^(3/2)); % L1/2 norm (Cui 2015)
        hgradient = lrate.h*(posgrad_h - neggrad_h - rbm_params.hreg_strength*delta_C);
        vgradient = lrate.v*(posgrad_v - neggrad_v);
        if iter>rbm_params.class_iter
        dgradient = lrate.d*(posgrad_d-neggrad_d);
        ugradient = lrate.u*(posgrad_u-neggrad_u - rbm_params.l1_strength*sign(class_weights));

        %%% nonnegative constraint
        idx = (class_weights<=0);
        ugradient(idx) = ugradient(idx) - lrate.u*rbm_params.nonneg_strength*class_weights(idx);
        end
        %%% update network varaibles
        h_bias = h_bias + hgradient;
        v_bias = v_bias + vgradient;
        d_bias = d_bias + dgradient;
        class_weights = class_weights + ugradient;
        
        % rescale the weights for each feature by the maximum if above a threshold
        
        max_weight_length = 10^2;
        class_weight_length = sum(class_weights.^2,2);
        length_index = class_weight_length > max_weight_length;
        if any(length_index)
           norm_const=sqrt(class_weight_length(length_index)/(max_weight_length));
            class_weights(length_index,:) = bsxfun(@rdivide, class_weights(length_index,:), norm_const);
        end

        recon_err = recon_err + sum(sum(abs(batch_data-vsample)));          % compute reconstruction error
        
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
    lrate.d = rbm_params.lrate.d*(max(0,1-adap_step));
    lrate.u = rbm_params.lrate.u*(max(0,1-adap_step));
    
    % create a mean hprob matrix at the end of the run
    
    % if the mean thing isn't too long then we can cut the previous version
    % alone with the new one to make the matrix fit
    
    % 
%     if iter == rbm_params.class_iter % when the hidden units have 'burnt in'
%         rbm_params.class_proportion = 0.5;
%     end
    if iter > max_iter - n_hprobs
        hprob_mean(:,:,iter - max_iter + n_hprobs) = hprob_mat(old_inds,:);
    end
    
    % do plotting (for debugging)
    if rbm_params.do_plots
        
        % print the output
        if exist('free_energy_diffs')
            energy_out = mean(free_energy_diffs(iter,:),2);
        else
            energy_out = 0;
        end
        fprintf(1, '%10d %15.4f %15.4f %15.4f %15.2f %15.2f\n', iter, recon_err, class_err, lrate.u, energy_out, toc(start));
        
        if mod(iter,rbm_params.plot_iters)==0
            
            subplot(2,3,1)
            imagesc(X_orig)
            xlabel('Input number','FontSize',16)
            ylabel('Patient number','FontSize',16)
            title('Original data', 'FontSize',16)
            colorbar
            
            subplot(2,3,2)
            imagesc(input_weights');
            xlabel('Input number','FontSize',16)
            ylabel('Hidden unit number','FontSize',16)
            title('Weight matrix', 'FontSize',16)
            colorbar
            
            subplot(2,3,4)
            imagesc(vprob_mat(old_inds,:))
            xlabel('Input number','FontSize',16)
            ylabel('Patient number','FontSize',16)
            title('Reconstruction', 'FontSize',16)
            colorbar
            
            subplot(2,3,5)
            imagesc(hprob_mat(old_inds,:))
            xlabel('Hidden unit number','FontSize',16)
            ylabel('Patient number','FontSize',16)
            title('Hidden layer probabilities', 'FontSize',16)
            colorbar%     subplot(2,3,5)
            
            subplot(2,3,3)
            imagesc(class_weights');%-repmat(mean(w,2),1,size(w,2)))
            ylabel('Class number','FontSize',16)
            xlabel('Hidden unit number','FontSize',16)
            title('Classification weights', 'FontSize',16)
            colorbar
            
            subplot(4,3,9)
            imagesc(y_orig');%-repmat(mean(w,2),1,size(w,2)))
            ylabel('Class number','FontSize',16)
            xlabel('Patient number','FontSize',16)
            title('Classification data', 'FontSize',16)
            colorbar
            
            subplot(4,3,12)
            imagesc(yprob_mat(old_inds,:)');%-repmat(mean(w,2),1,size(w,2)))
            ylabel('Class number','FontSize',16)
            xlabel('Patient number','FontSize',16)
            title('Classification reconstruction', 'FontSize',16)
            colorbar
            
            drawnow
            
        end
    end
    
end

% calculate the feature representation
feature_representation = sigmoid(bsxfun(@plus, (1-rbm_params.class_proportion)*X*input_weights + rbm_params.class_proportion*y*class_weights', h_bias));		% compute hidden probabilities

end


