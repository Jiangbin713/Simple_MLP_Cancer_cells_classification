close all
clear all
%% Generating dataset
mu_1 = [0,0];
sigma_1 = [1,0;0,1];
mu_2 = [4,0];
sigma_2 = [4,0;0,4];

class_1 = (mvnrnd(mu_1, sigma_1, 1650))';
class_2 = (mvnrnd(mu_2, sigma_2, 1650))';

x = [class_1,class_2]; %input
t = zeros(2,3300);
t(:,1:1650) = [ones(1,1650);zeros(1,1650)];
t(:,1651:end) = [zeros(1,1650);ones(1,1650)];%target
%% Bayes boundary
r_x = -2/3;
r_y = 0;
r_r = 2.34;
theta = 0:2*pi/3600:2*pi;
r_1 = r_x+r_r*cos(theta);
r_2 = r_y+r_r*sin(theta);

%% Plot dataset & Bayes boundary
plot(class_1(1,:),class_1(2,:),'*')
hold on
plot(class_2(1,:),class_2(2,:),'r*')


%% Nodes & Epochs setting
nodes =[2];
epochs=[4000];
nodes_length=length(nodes);
epochs_length=length(epochs);
repeat_times = 1;
classifier_num = 1 ; %3 11 19 25

%% Variable
error_matrix =zeros(nodes_length*epochs_length,repeat_times);
error_matrix_train=zeros(nodes_length*epochs_length,repeat_times);
error_matrix_test=zeros(nodes_length*epochs_length,repeat_times);

y_candidates = zeros(classifier_num,3300);
y_candidates_train=zeros(classifier_num,300);
y_candidates_test=zeros(classifier_num,3000);

y_voted = zeros(2,3300);
y_training_voted = zeros(2,300);
y_testing_voted = zeros(2,3000);


%% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm'; 

%% Start of the looping for 30 times

counter_nodes = 1; %nodes
counter_epochs = 1; %epochs
counter_times = 1; %repeat_times
counter_rows = 0; %rows of the error matrix
counter_coloums = 1; %coloum of the error matrix
counter_classifiers = 0; %classifiers
tic
for counter_nodes = 1:nodes_length %nodes
    
    for counter_epochs= 1:epochs_length %epochs
  
        counter_rows = counter_rows + 1;
        
        for counter_times = 1:repeat_times
            counter_coloums = counter_times;

            for counter_classifiers = 1:classifier_num
                             notification_1 = ['running:', ' node number = ', num2str( nodes(counter_nodes) ), ...
                       '    ', 'epoch number = ', num2str( epochs(counter_epochs) ), ...
                       '    ', 'reapeating time :', num2str( counter_times),...
                       '    ', '#classifier :', num2str(counter_classifiers)];
                
                disp(notification_1);
%% Create a Pattern Recognition Network
                net = patternnet(nodes(counter_nodes), trainFcn);
                net.trainParam.epochs = epochs(counter_epochs);

%% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
                net.input.processFcns = {'removeconstantrows','mapminmax'};

%% set fixed trainInd & testInd
trainInd=[1:150,1651:1800];
testInd=[151:1650,1801:3300];
%% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivision
                net.divideFcn = 'divideind';  % Divide data randomly
                net.divideParam.trainInd = trainInd;
                net.divideParam.testInd = testInd;
                %net.divideMode = 'sample';  % Divide up every sample
                %net.divideParam.trainRatio = 50/100;
                %net.divideParam.valRatio = 0;%15/100;
                %net.divideParam.testRatio = 50/100;

%% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
                net.performFcn = 'crossentropy';  % Cross-Entropy
                %net.performFcn = 'mse'
%% Choose Plot Functions
% For a list of all plot functions type: help nnplot
                net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
                'plotconfusion', 'plotroc'};

                
                
%% Train the Network
                %[net,tr] = train(net,x,t,'usegpu','yes'); %tr too much information inside
                [net,tr] = train(net,x,t);
                y = net(x); %output result
%% Voted result: y_candidates big matrix contains all repeated time output (only first row)
                
                y_candidates(counter_classifiers,:)=y(1,:);
                
                y_candidates_train(counter_classifiers,:)= y(1,tr.trainInd);
                
                y_candidates_test(counter_classifiers,:)= y(1,tr.testInd);
               
                % return y matrix's row and coloum size
                [r,c]=size(y_candidates);
                [r1,c1] = size(y_candidates_train);
                [r2,c2] = size(y_candidates_test);
                
                %%%%%%% Finally voted results
                for temp1= 1:c %overall  %coloums
                    if sum(y_candidates(:,temp1) > 0.5) > (classifier_num-1)/2 %only see the first row
                        y_voted(:,temp1) = [1;0];
                    else
                        y_voted(:,temp1) = [0;1];
                    end
                end
                
                y_training_voted = y_voted(:,tr.trainInd);
                y_testing_voted = y_voted(:,tr.testInd);
                          
%% Voted Error rate
                % Target to be compared
                t_training = [t(1,tr.trainInd);t(2,tr.trainInd)];
                t_testing = [t(1,tr.testInd);t(2,tr.testInd)];
                
                %%%%%%%%% voted overall error rate
                tind = vec2ind(t); % target
                yind_vote = vec2ind(y_voted);
                percentErrors_voted = sum(tind ~= yind_vote)/numel(tind);

                %%%%%%%% voted training error rate
                tind_train = vec2ind(t_training);
                yind_train_voted = vec2ind(y_training_voted);
                percentErrors_train_voted = sum(tind_train ~= yind_train_voted)/numel(tind_train);

                %%%%%%% voted testing error rate
                tind_test = vec2ind(t_testing);
                yind_test_voted = vec2ind(y_testing_voted);
                percentErrors_test_voted = sum(tind_test ~= yind_test_voted)/numel(tind_test);
            end                
%% Error rates ¡ª¡ª Test the Network
               
                e = gsubtract(t,y); %target - output
                performance = perform(net,t,y);
                yind = vec2ind(y);
                percentErrors = sum(tind ~= yind)/numel(tind); % Overall Errors rates 
                
                 % output to be compared
                y_training = [y(1,tr.trainInd);y(2,tr.trainInd)];
                y_testing = [y(1,tr.testInd);y(2,tr.testInd)];
                %%%%%%% training error rate
                yind_train = vec2ind(y_training);
                percentErrors_train = sum(tind_train ~= yind_train)/numel(tind_train);
                
                %%%%%%% testing error rate          
                yind_test = vec2ind(y_testing);
                percentErrors_test = sum(tind_test ~= yind_test)/numel(tind_test);

%% Recalculate Training, Validation and Test Performance
                trainTargets = t .* tr.trainMask{1};
                valTargets = t .* tr.valMask{1};
                testTargets = t .* tr.testMask{1};
                trainPerformance = perform(net,trainTargets,y);
                valPerformance = perform(net,valTargets,y);
                testPerformance = perform(net,testTargets,y);

%% End of the loop Error matrix Un-voted & Voted
                
                error_matrix_voted(counter_rows,counter_coloums)=percentErrors_voted;
                error_matrix_train_voted(counter_rows,counter_coloums)=percentErrors_train_voted;
                error_matrix_test_voted(counter_rows,counter_coloums)=percentErrors_test_voted;
                error_matrix(counter_rows,counter_coloums)=percentErrors;
                error_matrix_train(counter_rows,counter_coloums)=percentErrors_train;
                error_matrix_test(counter_rows,counter_coloums)=percentErrors_test;
            

           
        end
    
    end
    
end


toc

plotpc(net.iw{1},net.b{1});
hold on
plot(r_1,r_2,'k')
weight = net.iw{1};
bias = net.b{1};
% plotpc(weight(1,:),bias(1))
% plotpc(weight(2,:),bias(2))
% plotpc(weight(3,:),bias(3))