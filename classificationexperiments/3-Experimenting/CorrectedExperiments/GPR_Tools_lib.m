

% @infVB  @meanOne  @covConst  @likSech2
% @infLaplace  @meanOne  @covConst  @likSech2


function GPR_Tools_lib(Train_features, Test_features, Train_Labels, Test_Labels)

    Num_of_Classes = size(Train_Labels, 1);
    Num_of_Images = size(Train_Labels, 2);
    
%     meanFunction = {@meanMask @meanProd @meanOne @meanScale @meanConst @meanPoly @meanSum @meanLinear @meanPow @meanZero};
%     meanFunction_strings = {'@meanMask' '@meanProd' '@meanOne' '@meanScale' '@meanConst' '@meanPoly' '@meanSum' '@meanLinear' '@meanPow' '@meanZero'};
%     likelihoodFunction = {@likGaussWarp @likSech2 @likGumbel @likT @likBeta @likInvGauss @likUni @likErf @likLaplace @likWeibull @likExp @likLogistic @likGamma @likMix @likGauss @likPoisson};
%     likelihoodFunction_strings = {'@likGaussWarp' '@likSech2' '@likGumbel' '@likT' '@likBeta' '@likInvGauss' '@likUni' '@likErf' '@likLaplace' '@likWeibull' '@likExp' '@likLogistic' '@likGamma' '@likMix' '@likGauss' '@likPoisson'};
%     covarianceFunction = {@covMask @covProd @covMaternard @covRQard @covADD @covMaterniso @covRQiso @covConst @covNNone @covSEard @covCos @covNoise @covSEiso @covFITC @covPERard @covSEisoU @covGaborard @covPERiso @covSM @covGaboriso @covPPard @covScale @covLIN @covPPiso @covSum @covLINard @covPeriodic @covLINiso @covPeriodicNoDC @covLINone @covPoly};
%     covarianceFunction_strings = {'@covMask' '@covProd' '@covMaternard' '@covRQard' '@covADD' '@covMaterniso' '@covRQiso' '@covConst' '@covNNone' '@covSEard' '@covCos' '@covNoise' '@covSEiso' '@covFITC' '@covPERard' '@covSEisoU' '@covGaborard' '@covPERiso' '@covSM' '@covGaboriso' '@covPPard' '@covScale' '@covLIN' '@covPPiso' '@covSum' '@covLINard' '@covPeriodic' '@covLINiso' '@covPeriodicNoDC' '@covLINone' '@covPoly'};
%     inferenceFunction = {@infFITC_EP @infMCMC @infFITC_Laplace @infVB @infEP @infKL @infExact @infLOO @infFITC @infLaplace};
%     inferenceFunction_strings = {'@infFITC_EP' '@infMCMC' '@infFITC_Laplace' '@infVB' '@infEP' '@infKL' '@infExact' '@infLOO' '@infFITC' '@infLaplace'};


%     meanFunction = {@meanOne};
%     meanFunction_strings = {'@meanOne'};
%     likelihoodFunction = {@likSech2};
%     likelihoodFunction_strings = {'@likSech2'};
%     covarianceFunction = {@covConst};
%     covarianceFunction_strings = {'@covConst'};
%     inferenceFunction = {@infVB};
%     inferenceFunction_strings = {'@infVB'};

    meanFunction = {@meanOne};
    meanFunction_strings = {'@meanOne'};
    likelihoodFunction = {@likSech2};
    likelihoodFunction_strings = {'@likSech2'};
    covarianceFunction = {@covConst};
    covarianceFunction_strings = {'@covConst'};
    inferenceFunction = {@infLaplace};
    inferenceFunction_strings = {'@infLaplace'};

    Final_estimate = zeros(Num_of_Classes, Num_of_Images);
    
    mfunction = meanFunction{1};
    inffunction = inferenceFunction{1};
    cfunction = covarianceFunction{1};
    lfunction = likelihoodFunction{1};

    for i = 1:Num_of_Classes
        
        
        hyp2 = AssignHyperParameter(inferenceFunction_strings{1}, meanFunction_strings{1}, covarianceFunction_strings{1}, likelihoodFunction_strings{1});
        disp('Pre-minimize...');
        printHyp(hyp2);
        
        hyp2 = minimize(hyp2, @gp, -100, inffunction, mfunction, cfunction, lfunction, Train_features', Train_Labels(i,:)');
        disp('Post-minimize...');
        printHyp(hyp2);
        
        tic
        [m s2] = gp(hyp2, inffunction, mfunction, cfunction, lfunction, Train_features', Train_Labels(i,:)', Test_features');
        toc
        
        Final_estimate(i,:) = m;
    end
    threshold = 0:0.0001:1;

    Kfolds = 5;
    [Tr_features, Te_features, Tr_Labels, Te_Labels] = Make_Kfolds(Train_features, Train_Labels, Kfolds);
    
    acc = zeros(length(threshold), Kfolds);
    sen = zeros(length(threshold), Kfolds);
    spe = zeros(length(threshold), Kfolds);
    
    
    for x=1:Kfolds
        fprintf('Kfold = %d\n', x);
        
        for i = 1:Num_of_Classes
            hyp2_kf = AssignHyperParameter(inferenceFunction_strings{1}, meanFunction_strings{1}, covarianceFunction_strings{1}, likelihoodFunction_strings{1});
            hyp2_kf = minimize(hyp2_kf, @gp, -100, inffunction, mfunction, cfunction, lfunction, Tr_features{x}', Tr_Labels{x}(i,:)');
            [m_kf s2_kf] = gp(hyp2_kf, inffunction, mfunction, cfunction, lfunction, Tr_features{x}', Tr_Labels{x}(i,:)', Te_features{x}');
            ft_estimated(i,:) = m_kf;
%         ft_softnet = trainSoftmaxLayer(double(Tr_features{x}),Tr_Labels{x}, 'ShowProgressWindow', false);
%         ft_estimated = ft_softnet(double(Te_features{x}));
        end
        
        for i=1:length(threshold)
            ft_final_estimated = ft_estimated;
        
            ft_final_estimated(ft_final_estimated < threshold(i)) = 0;
            ft_final_estimated(ft_final_estimated > 0           ) = 1;

            evals = Evaluate(Te_Labels{x}(:),ft_final_estimated(:));
            
            acc(i, x) = evals(1);
            sen(i, x) = evals(2);
            spe(i, x) = evals(3);
            
        end
    end
    
    final_acc = mean(acc');
%     final_sen = mean(sen');
%     final_spe = mean(spe');
    
%     plot(threshold, final_acc, threshold, final_sen, threshold, final_spe);
%     hold on;
%     legend('Accuracy', 'Sensitivity', 'Specificity');
%     hold off;
%     pause;
    index = find(final_acc == max(final_acc), 1);
%     index = input('Put value: ');
    estimated = Final_estimate;
        
    estimated(estimated < threshold(index)) = 0;
%     estimated(estimated < 0.000001) = 0;
    estimated(estimated > 0               ) = 1;

    evals = Evaluate(Test_Labels(:), estimated(:));

%     fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));
    disp('------------');
    fprintf('Threshold = %f\n\n', threshold(index));
    average = (evals(2) + evals(3))/2;
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nAverage\t\t%f\n',evals(1), evals(2), evals(3), average);

end



function printHyp(hyp2)

    fprintf('hyp.lik\t\t\t');
    for j = 1:length(hyp2.lik)
        fprintf('%f\t\t', hyp2.lik(j));
    end
    fprintf('\n');
    

    fprintf('hyp.mean\t\t\t');
    for j = 1:length(hyp2.mean)
        fprintf('%f\t\t', hyp2.mean(j));
    end
    fprintf('\n');
    
    fprintf('hyp.cov\t\t\t');
    for j = 1:length(hyp2.cov)
        fprintf('%f\t\t', hyp2.cov(j));
    end
    fprintf('\n');
    
end

function hyp = AssignHyperParameter(InferenceFunction, MeanFunction, CovarianceFunction, LiklihoodFunction)
    

    hyp.lik = AssignLiklihoodParameter(LiklihoodFunction);
    hyp.mean = AssignMeanParameter(MeanFunction);
    hyp.cov = AssignCovarianceParameter(CovarianceFunction);
    
end

function hyp = AssignLiklihoodParameter(functionName)
% -------------------------------------------------------------------------
    n = 5; f = randn(n,1);       % create random latent function values

    % set up simple classification likelihood functions
    yc = sign(f);
    lc0 = {'likErf'};     hypc0 = [];   % no hyperparameters are needed
    lc1 = {@likLogistic}; hypc1 = [];    % also function handles are OK
    lc2 = {'likUni'};     hypc2 = [];
    lc3 = {'likMix',{'likUni',@likErf}}; hypc3 = log([1;2]); %mixture

    % set up simple regression likelihood functions
    yr = f + randn(n,1)/20;
    sn = 0.1;                                % noise standard deviation
    lr0 = {'likGauss'};   hypr0 = log(sn);
    lr1 = {'likLaplace'}; hypr1 = log(sn);
    lr2 = {'likSech2'};   hypr2 = log(sn);
    nu = 4;                              % number of degrees of freedom
    lr3 = {'likT'};       hypr3 = [log(nu-1); log(sn)];
%     lr4 = {'likMix',{lr0,lr1}}; hypr4 = [log([1,2]);hypr0;hypr1];

    a = 1; % set up warped Gaussian with g(y) = y + a*sign(y).*y.^2
    lr5 = {'likGaussWarp',['poly2']}; hypr5 = log([a;sn]);
    lr6 = {'likGumbel','+'}; hypr6 = log(sn);

    % set up Poisson regression
    yp = fix(abs(f)) + 1;
    lp0 = {@likPoisson,'logistic'}; hypp0 = [];
    lp1 = {@likPoisson,'exp'};      hypp1 = [];

    % set up other GLM likelihoods for positive or interval regression
    lg1 = {@likGamma,'logistic'}; al = 2;    hyp.lik = log(al);
    lg2 = {@likInvGauss,'exp'};   lam = 1.1; hyp.lik = log(lam);
    lg3 = {@likBeta,'expexp'};    phi = 2.1; hyp.lik = log(phi);
    lg4 = {@likBeta,'logit'};     phi = 4.7; hyp.lik = log(phi);
% -------------------------------------------------------------------------    
    switch (functionName)
%         @likGaussWarp @likSech2 @likGumbel @likT @likBeta @likInvGauss @likUni @likErf @likLaplace @likWeibull @likExp @likLogistic @likGamma @likMix @likGauss @likPoisson
        case '@likGaussWarp'
            lik = lc0; hyp = hypr5; y = yc;
        case '@likSech2'
            lik = lr2; hyp = hypr2; y = yc;
        case '@likGumbel'
            lik = lr6; hyp = hypr6; y = yc;
        case '@likT'
            lik = lr3; hyp = hypr3; y = yc;
        case '@likBeta'
            phi = 2.1;
            lik = lg3; hyp = log(phi); y = yc;
%             more options too
        case '@likInvGauss'
            lam = 1.1;
            lik = lg2; hyp = log(lam); y = yc;
        case '@likUni'
            lik = lc2; hyp = hypc2; y = yc;
        case '@likErf'
            lik = lc0; hyp = hypc0; y = yc;
        case '@likLaplace'
            lik = lr1; hyp = hypr1; y = yc;
        case '@likWeibull'
%             lik = lc0; hyp = hypc0; y = yc;
        case '@likExp'
%             lik = lc0; hyp = hypc0; y = yc;
        case '@likLogistic'
            lik = lc1; hyp = hypc1; y = yc;
        case '@likGamma'
            al = 2;
            lik = lg1; hyp = log(al);; y = yc;
        case '@likMix'
            lik = lc3; hyp = hypc3; y = yc;
        case '@likGauss'
            lik = lr0; hyp = hypr0; y = yc;
        case '@likPoisson'
            lik = lp0; hyp = hypp0; y = yc;
    end
end
function hyp = AssignMeanParameter(functionName)
% -------------------------------------------------------------------------
    n = 5; D = 2; x = randn(n,D);            % create a random data set

    % set up simple mean functions
    m0 = {'meanZero'};  hyp0 = [];      % no hyperparameters are needed
    m1 = {'meanOne'};   hyp1 = [];      % no hyperparameters are needed
    mc = {@meanConst};  hypc = 2;  % also function handles are possible
    ml = {@meanLinear}; hypl = [2;3];              % m(x) = 2*x1 + 3*x2
    mp = {@meanPoly,2}; hypp = [1;1;2;3];  % m(x) = x1+x2+2*x1^2+3*x2^2
    mn = {@meanNN,[1,0; 0,1],[0.9,0.5]}; hypn = [];  % nearest neighbor
    s = 12; hypd = randn(s,1);           % discrete mean with 12 hypers
    md = {'meanDiscrete',s};
    hyp.cov = [0;0]; hypg = [];                    % GP predictive mean
    xt = randn(2*n,D); yt = sign(xt(:,1)-xt(:,2));      % training data
    mg = {@meanGP,hyp,@infEP,@meanZero,@covSEiso,@likErf,xt,yt};
    hype = [0;0; log(0.1)];             % regression GP predictive mean
    xt = randn(2*n,D); yt = xt(:,1).*xt(:,2);           % training data
    me = {@meanGPexact,@meanZero,@covSEiso,xt,yt};

    % set up composite mean functions
    msc = {'meanScale',{m1}};      hypsc = [3; hyp1];      % scale by 3
    msu = {'meanSum',{m0,mc,ml}};  hypsu = [hyp0; hypc; hypl];    % sum
    mpr = {@meanProd,{mc,ml}};     hyppr = [hypc; hypl];      % product
    mpo = {'meanPow',3,msu};       hyppo = hypsu;         % third power
    mask = [false,true];     % mask excluding all but the 2nd component
    mma = {'meanMask',mask,ml};    hypma = hypl(mask);
    mpf = {@meanPref,ml};          hyppf = 2;  % linear pref with slope
% -------------------------------------------------------------------------    
    switch (functionName)
%         @meanMask @meanProd @meanOne @meanScale @meanConst @meanPoly @meanSum @meanLinear @meanPow @meanZero
        case '@meanMask'
            mean = mma; 
            hyp = hypma;
        case '@meanProd'
            mean = mpr; 
            hyp = hyppr;
        case '@meanOne'
            mean = m1; 
            hyp = hyp1;
        case '@meanScale'
            mean = msc; 
            hyp = hypsc;
        case '@meanConst'
            mean = mc; 
            hyp = hypc;
        case '@meanPoly'
            mean = mp; 
            hyp = hypp;
        case '@meanSum'
            mean = msu; 
            hyp = hypsu;
        case '@meanLinear'
            mean = ml; 
            hyp = hypl;
        case '@meanPow'
            mean = mpo; 
            hyp = hyppo;
        case '@meanZero'
            mean = m0; 
            hyp = hyp0;
    end
    feval(mean{:});
    feval(mean{:},hyp,x);
    i = 2; feval(mean{:},hyp,x,i);
end
function hyp = AssignCovarianceParameter(functionName)
% -------------------------------------------------------------------------    
    n = 5; D = 3; x = randn(n,D); xs = randn(3,D);  % create a data set

    % set up simple covariance functions
    cn  = {'covNoise'}; sn = .1;  hypn = log(sn);  % one hyperparameter
    cc  = {@covConst};   sf = 2;  hypc = log(sf); % function handles OK
    ce  = {@covEye};              hype = [];                 % identity
    cl  = {@covLIN};              hypl = []; % linear is parameter-free
    cla = {'covLINard'}; L = rand(D,1); hypla = log(L);  % linear (ARD)
    cli = {'covLINiso'}; l = rand(1);   hypli = log(l);    % linear iso
    clo = {@covLINone}; ell = .9; hyplo = log(ell);  % linear with bias
    cp  = {@covPoly,3}; c = 2; hypp = log([c;sf]);   % third order poly
    cga = {@covSEard};   hypga = log([L;sf]);       % Gaussian with ARD
    cgi = {'covSEiso'};  hypgi = log([ell;sf]);    % isotropic Gaussian
    cgu = {'covSEisoU'}; hypgu = log(ell);   % isotropic Gauss no scale
    cra = {'covRQard'}; al = 2; hypra = log([L;sf;al]); % ration. quad.
    cri = {@covRQiso};          hypri = log([ell;sf;al]);   % isotropic
    cma = {@covMaternard,5};  hypma = log([ell;sf]); % Matern class d=5
    cmi = {'covMaterniso',3}; hypmi = log([ell;sf]); % Matern class d=3
    cnn = {'covNNone'}; hypnn = log([L;sf]);           % neural network
    cpe = {'covPeriodic'}; p = 2; hyppe = log([ell;p;sf]);   % periodic
    cpn = {'covPeriodicNoDC'}; p = 2; hyppe = log([ell;p;sf]); % w/o DC
    cpc = {'covCos'}; p = 2; hypcpc = log([p;sf]);         % cosine cov
    cca = {'covPPard',3}; hypcc = hypgu;% compact support poly degree 3
    cci = {'covPPiso',2}; hypcc = hypgi;% compact support poly degree 2
    cgb = {@covGaboriso}; ell = 1; p = 1.2; hypgb=log([ell;p]); % Gabor
    Q = 2; w = ones(Q,1)/Q; m = rand(D,Q); v = rand(D,Q);
    csm = {@covSM,Q}; hypsm = log([w;m(:);v(:)]);    % Spectral Mixture
    cvl = {@covSEvlen,{@meanLinear}}; hypvl = [1;2;1; 0]; % var lenscal
    s = 12; cds = {@covDiscrete,s};      % discrete covariance function
    L = randn(s); L = chol(L'*L); L(1:(s+1):end) = log(diag(L));
    hypds = L(triu(true(s))); xd = randi([1,s],[n,1]); xsd = [1;3;6];
    cfa = {@covSEfact,2}; hypfa = randn(D*2,1);       % factor analysis

    % set up composite i.e. meta covariance functions
    csc = {'covScale',{cgu}};    hypsc = [log(3); hypgu];  % scale by 9
    csu = {'covSum',{cn,cc,cl}}; hypsu = [hypn; hypc; hypl];      % sum
    cpr = {@covProd,{cc,cci}};   hyppr = [hypc; hypcc];       % product
    mask = [0,1,0]; %   binary mask excluding all but the 2nd component
    cma = {'covMask',{mask,cgi{:}}}; hypma = hypgi;
    % isotropic periodic rational quadratic
    cpi = {'covPERiso',{@covRQiso}};
    % periodic Matern with ARD
    cpa = {'covPERard',{@covMaternard,3}};
    % additive based on SEiso using unary and pairwise interactions
    cad = {'covADD',{[1,2],'covSEiso'}};
    % preference covariance with squared exponential base covariance
    cpr = {'covPref',{'covSEiso'}}; hyppr = [0;0];
    xp = randn(n,2*D); xsp = randn(3,2*D);
% -------------------------------------------------------------------------    
    switch (functionName)
%         @covMask @covProd @covMaternard @covRQard @covADD @covMaterniso @covRQiso @covConst @covNNone @covSEard @covCos @covNoise @covSEiso @covFITC @covPERard @covSEisoU @covGaborard @covPERiso @covSM @covGaboriso @covPPard @covScale @covLIN @covPPiso @covSum @covLINard @covPeriodic @covLINiso @covPeriodicNoDC @covLINone @covPoly
        case '@covMask'
            cov = cma; hyp = hypma; x = xd; xs = xsd;
        case '@covProd'
            cov = cpr; hyp = hyppr; x = xd; xs = xsd;
        case '@covMaternard'
            cov = cma; hyp = hypma; x = xd; xs = xsd;
        case '@covRQard'
            cov = cra; hyp = hypra; x = xd; xs = xsd;
        case '@covADD'
            cov = cad; hyp = hyppr; x = xd; xs = xsd;
        case '@covMaterniso'
            cov = cmi; hyp = hypmi; x = xd; xs = xsd;
        case '@covRQiso'
            cov = cpi; hyp = hyppr; x = xd; xs = xsd;
        case '@covConst'
            cov = cc; hyp = hypc; x = xd; xs = xsd;
        case '@covNNone'
            cov = cnn; hyp = hypnn; x = xd; xs = xsd;
        case '@covSEard'
            cov = cga; hyp = hypga; x = xd; xs = xsd;
        case '@covCos'
            cov = cpc; hyp = hypcpc; x = xd; xs = xsd;
        case '@covNoise'
            cov = cn; hyp = hypn; x = xd; xs = xsd;
        case '@covSEiso'
            cov = cgi; hyp = hypgi; x = xd; xs = xsd;
        case '@covFITC'
%             cov = cds; hyp = hypds; x = xd; xs = xsd;
        case '@covPERard'
            cov = cpa; hyp = hyppr; x = xd; xs = xsd;
        case '@covSEisoU'
            cov = cgu; hyp = hypgu; x = xd; xs = xsd;
        case '@covGaborard'
%             cov = cds; hyp = hypds; x = xd; xs = xsd;
        case '@covPERiso'
            cov = cpi; hyp = hypma; x = xd; xs = xsd;
        case '@covSM'
            cov = csm; hyp = hypsm; x = xd; xs = xsd;
        case '@covLINard'
            cov = cla; hyp = hypla; x = xd; xs = xsd;
        case '@covPeriodic'
            cov = cpe; hyp = hyppe; x = xd; xs = xsd;
        case '@covLINiso'
            cov = cli; hyp = hypli; x = xd; xs = xsd;
        case '@covPeriodicNoDC'
            cov = cpn; hyp = hyppe; x = xd; xs = xsd;
        case '@covLINone'
            cov = clo; hyp = hyplo; x = xd; xs = xsd;
        case '@covPoly'
            cov = cp; hyp = hypp; x = xd; xs = xsd;
    end
end
