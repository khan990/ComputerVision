




function GPR_Tool_Grand_Experimentation(Train_features, Test_features, Train_Labels, Test_Labels)
    diary('GPR_Tool_Grand_Experimentation_Output.txt');
%     meanFunction = {@meanMask @meanProd @meanOne @meanScale @meanConst @meanPoly @meanSum @meanLinear @meanPow @meanZero};
%     meanFunction_strings = {'@meanMask' '@meanProd' '@meanOne' '@meanScale' '@meanConst' '@meanPoly' '@meanSum' '@meanLinear' '@meanPow' '@meanZero'};
%     likelihoodFunction = {@likGaussWarp @likSech2 @likGumbel @likT @likBeta @likInvGauss @likUni @likErf @likLaplace @likWeibull @likExp @likLogistic @likGamma @likMix @likGauss @likPoisson};
%     likelihoodFunction_strings = {'@likGaussWarp' '@likSech2' '@likGumbel' '@likT' '@likBeta' '@likInvGauss' '@likUni' '@likErf' '@likLaplace' '@likWeibull' '@likExp' '@likLogistic' '@likGamma' '@likMix' '@likGauss' '@likPoisson'};
%     covarianceFunction = {@covMask @covProd @covMaternard @covRQard @covADD @covMaterniso @covRQiso @covConst @covNNone @covSEard @covCos @covNoise @covSEiso @covFITC @covPERard @covSEisoU @covGaborard @covPERiso @covSM @covGaboriso @covPPard @covScale @covLIN @covPPiso @covSum @covLINard @covPeriodic @covLINiso @covPeriodicNoDC @covLINone @covPoly};
%     covarianceFunction_strings = {'@covMask' '@covProd' '@covMaternard' '@covRQard' '@covADD' '@covMaterniso' '@covRQiso' '@covConst' '@covNNone' '@covSEard' '@covCos' '@covNoise' '@covSEiso' '@covFITC' '@covPERard' '@covSEisoU' '@covGaborard' '@covPERiso' '@covSM' '@covGaboriso' '@covPPard' '@covScale' '@covLIN' '@covPPiso' '@covSum' '@covLINard' '@covPeriodic' '@covLINiso' '@covPeriodicNoDC' '@covLINone' '@covPoly'};
%     inferenceFunction = {@infFITC_EP @infMCMC @infFITC_Laplace @infVB @infEP @infKL @infExact @infLOO @infFITC @infLaplace};
%     inferenceFunction_strings = {'@infFITC_EP' '@infMCMC' '@infFITC_Laplace' '@infVB' '@infEP' '@infKL' '@infExact' '@infLOO' '@infFITC' '@infLaplace'};

    meanFunction = {@meanPoly @meanSum @meanLinear @meanPow @meanZero};
    meanFunction_strings = {'@meanPoly' '@meanSum' '@meanLinear' '@meanPow' '@meanZero'};
    likelihoodFunction = {@likGaussWarp @likSech2 @likGumbel @likT @likBeta @likInvGauss @likUni @likErf @likLaplace @likWeibull @likExp @likLogistic @likGamma @likMix @likGauss @likPoisson};
    likelihoodFunction_strings = {'@likGaussWarp' '@likSech2' '@likGumbel' '@likT' '@likBeta' '@likInvGauss' '@likUni' '@likErf' '@likLaplace' '@likWeibull' '@likExp' '@likLogistic' '@likGamma' '@likMix' '@likGauss' '@likPoisson'};
    covarianceFunction = {@covMask @covProd @covMaternard @covRQard @covADD @covMaterniso @covRQiso @covConst @covNNone @covSEard @covCos @covNoise @covSEiso @covFITC @covPERard @covSEisoU @covGaborard @covPERiso @covSM @covGaboriso @covPPard @covScale @covLIN @covPPiso @covSum @covLINard @covPeriodic @covLINiso @covPeriodicNoDC @covLINone @covPoly};
    covarianceFunction_strings = {'@covMask' '@covProd' '@covMaternard' '@covRQard' '@covADD' '@covMaterniso' '@covRQiso' '@covConst' '@covNNone' '@covSEard' '@covCos' '@covNoise' '@covSEiso' '@covFITC' '@covPERard' '@covSEisoU' '@covGaborard' '@covPERiso' '@covSM' '@covGaboriso' '@covPPard' '@covScale' '@covLIN' '@covPPiso' '@covSum' '@covLINard' '@covPeriodic' '@covLINiso' '@covPeriodicNoDC' '@covLINone' '@covPoly'};
    inferenceFunction = {@infFITC_EP @infMCMC @infFITC_Laplace @infVB @infEP @infKL @infExact @infLOO @infFITC @infLaplace};
    inferenceFunction_strings = {'@infFITC_EP' '@infMCMC' '@infFITC_Laplace' '@infVB' '@infEP' '@infKL' '@infExact' '@infLOO' '@infFITC' '@infLaplace'};
    
    Num_of_MeanFunctions = length(meanFunction);
    Num_of_LiklihoodFunctions = length(likelihoodFunction);
    Num_of_CovarianceFunctions = length(covarianceFunction);
    Num_of_InferenceFunctions = length(inferenceFunction);
    
    Num_of_Images = size(Train_Labels, 2);
    Num_of_Classes = size(Train_Labels, 1);
    
%     Mean Iteration
    for M = 1:Num_of_MeanFunctions
%         Liklihood Iterations
        for L = 1:Num_of_LiklihoodFunctions
%             Covariance Function Iterations
            for C = 1:Num_of_CovarianceFunctions
%                 Inference Function Iterations
                for I = 1:Num_of_InferenceFunctions
                    
                    Final_estimate = zeros(Num_of_Classes, Num_of_Images);
                    try
                        for i = 1:Num_of_Classes
                            mfunction = meanFunction{M};
                            inffunction = inferenceFunction{I};
                            cfunction = covarianceFunction{C};
                            lfunction = likelihoodFunction{L};
                            fprintf('---------------------------------------------\n');
                            fprintf('Class = %d\n%s\t\t%s\t\t%s\t\t%s\n', i, inferenceFunction_strings{I}, meanFunction_strings{M}, covarianceFunction_strings{C}, likelihoodFunction_strings{L});
                            hyp2 = AssignHyperParameter(inferenceFunction_strings{I}, meanFunction_strings{M}, covarianceFunction_strings{C}, likelihoodFunction_strings{L});
                            disp('Pre-minimize...');
                            printHyp(hyp2);
    %                         hyp2 = minimize(hyp2, @gp, -100, inferenceFunction{I}, meanFunction{M}, covarianceFunction{C}, likelihoodFunction{L}, Train_features', Train_Labels(i,:)');
    %                         hyp2 = minimize(hyp2, @gp, -100, inferenceFunction{I}, [], covarianceFunction{C}, likelihoodFunction{L}, Train_features', Train_Labels(i,:)');
                            hyp2 = minimize(hyp2, @gp, -100, inffunction, mfunction, cfunction, lfunction, Train_features', Train_Labels(i,:)');
                            disp('Post-minimize...');
                            printHyp(hyp2);

    %                         [m s2] = gp(hyp2, inferenceFunction{I}, meanFunction{M}, covarianceFunction{C}, likelihoodFunction{L}, Train_features', Train_Labels(i,:)', Test_features');
    %                         [m s2] = gp(hyp2, inferenceFunction{I}, [], covarianceFunction{C}, likelihoodFunction{L}, Train_features', Train_Labels(i,:)', Test_features');
                            [m s2] = gp(hyp2, inffunction, mfunction, cfunction, lfunction, Train_features', Train_Labels(i,:)', Test_features');
                            Final_estimate(i,:) = m;
                        end
                        Threshold_range = 0:0.0001:1;

                        acc = zeros(length(Threshold_range), 1);
                        sen = zeros(length(Threshold_range), 1);
                        spe = zeros(length(Threshold_range), 1);

                        x = 1;
                        for threshold_value = Threshold_range
                            Final_estimate_2 = Final_estimate;
                            Final_estimate_2(Final_estimate_2 < threshold_value) = 0;
                            Final_estimate_2(Final_estimate_2 > 0        ) = 1;
                            evals = Evaluate(Test_Labels(:), Final_estimate_2(:));

                            acc(x) = evals(1);
                            sen(x) = evals(2);
                            spe(x) = evals(3);
                            x = x + 1;
                        end
                        figure;
                        hold on;
                        plot(Threshold_range, acc, Threshold_range, sen, Threshold_range, spe);
                        legend('Accuracy', 'Sensitivity', 'Specificity');
                        hold off;

                        NewValue = input('Please input threshold value:   ');

                        Final_estimate_2 = Final_estimate;
                        Final_estimate_2(Final_estimate_2 < NewValue) = 0;
                        Final_estimate_2(Final_estimate_2 > 0        ) = 1;

                        evals = Evaluate(Test_Labels(:), Final_estimate_2(:));
                        fprintf('\n');
                        fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));
                    catch
                        warning('Ignore Last combination...');
                    end
                end
            end
        end
    end
    diary off;
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

% function hyp = AssignHyperParameter(InferenceFunction, MeanFunction, CovarianceFunction, LiklihoodFunction)
%     
% 
%     hyp.lik = [];
%     hyp.mean = [];
%     hyp.cov = [];
%     
%     switch (InferenceFunction)
% %         @infFITC_EP @infMCMC @infFITC_Laplace @infVB @infEP @infKL @infExact @infLOO @infFITC @infLaplace
%         case '@infFITC_EP'
% %             disp('infFITC_EP');
%             
%         case '@infMCMC'
% %             disp('infMCMC');
%             
%         case '@infFITC_Laplace'
% %             disp('infFITC_Laplace');
%             
%         case '@infVB'
% %             disp('infVB');
%             
%         case '@infEP'
% %             disp('infEP');
%             
%         case '@infKL'
% %             disp('infKL');
%             
%         case '@infExact'
% %             disp('infExact');
%             
%         case '@infLOO'
% %             disp('infLOO');
%             
%         case '@infFITC'
% %             disp('infFITC');
%             
%         case '@infLaplace'
% %             disp('infLaplace');
% 
%     end
%     
%     switch (MeanFunction)
% %         @meanMask @meanProd @meanOne @meanScale @meanConst @meanPoly @meanSum @meanLinear @meanPow @meanZero
%         case '@meanMask'
% %             disp('meanMask');
% %             mask = [0,1,0]; %   binary mask excluding all but the 2nd component
% %             mma = {'meanMask',mask,mpo}; 
% %             hypma = hyppo;
% 
%             n = 5; D = 2; x = randn(n,D);            % create a random data set
% 
%             % set up simple mean functions
%             m0 = {'meanZero'};  hyp0 = [];      % no hyperparameters are needed
%             m1 = {'meanOne'};   hyp1 = [];      % no hyperparameters are needed
%             mc = {@meanConst};  hypc = 2;  % also function handles are possible
%             ml = {@meanLinear}; hypl = [2;3];              % m(x) = 2*x1 + 3*x2
%             mp = {@meanPoly,2}; hypp = [1;1;2;3];  % m(x) = x1+x2+2*x1^2+3*x2^2
% 
%             % set up composite mean functions
%             msc = {'meanScale',{m1}};      hypsc = [3; hyp1];      % scale by 3
%             msu = {'meanSum',{m0,mc,ml}};  hypsu = [hyp0; hypc; hypl];    % sum
%             mpr = {@meanProd,{mc,ml}};     hyppr = [hypc; hypl];      % product
%             mpo = {'meanPow',3,msu};       hyppo = hypsu;         % third power
%             mask = [0,1,0]; %   binary mask excluding all but the 2nd component
%             mma = {'meanMask',mask,mpo}; hypma = hyppo;
%             
%             hyp0 = [];
%             hypc = 2;
%             hypl = [2;3];
%             hypsu = [hyp0; hypc; hypl];
%             hyp.mean = hypsu;
%             
%             mean = mma;
%             feval(mean{:})
%             feval(mean{:},hyp.mean,x)
%             i = 2; feval(mean{:},hyp.mean,x,i);
%             
%             
%         case '@meanProd'
% %             disp('meanProd');
% %             mpr = {@meanProd,{mc,ml}};     
% %             hyppr = [hypc; hypl];      % product
%             hypc = 2;
%             hypl = [2;3];
%             hyppr = [hypc; hypl];
%             hyp.mean = hyppr;
%             
%         case '@meanOne'
% %             disp('meanOne');
% %             m1 = {'meanOne'};   
%             hyp1 = [];      % no hyperparameters are needed
%             hyp.mean = hyp1;
%             
%         case '@meanScale'
% %             disp('meanScale');
% %             msc = {'meanScale',{m1}};      
% %             hypsc = [3; hyp1];      % scale by 3
%             hyp1 = [];
%             hypsc = [3; hyp1];
%             hyp.mean = hypsc;
%             
%         case '@meanConst'
% %             disp('meanConst');
% %             mc = {@meanConst};  
%             hypc = 2;  % also function handles are possible
%             hyp.mean = hypc;
%             
%         case '@meanPoly'
% %             disp('meanPoly');
% %             mp = {@meanPoly,2}; 
%             hypp = [1;1;2;3];  % m(x) = x1+x2+2*x1^2+3*x2^2
%             hyp.mean = hypp;
%             
%         case '@meanSum'
% %             disp('meanSum');
% %             msu = {'meanSum',{m0,mc,ml}};  
% %             hypsu = [hyp0; hypc; hypl];    % sum
%             hyp0 = [];
%             hypc = 2;
%             hypl = [2;3];
%             hypsu = [hyp0; hypc; hypl];
%             hyp.mean = hypsu;
%             
%         case '@meanLinear'
% %             disp('meanLinear');
% %             ml = {@meanLinear}; 
%             hypl = [2;3];              % m(x) = 2*x1 + 3*x2
%             hyp.mean = hypl;
%             
%         case '@meanPow'
% %             disp('meanPow');
% %             mpo = {'meanPow',3,msu};       
% %             hyppo = hypsu;         % third power
%             hyp0 = [];
%             hypc = 2;
%             hypl = [2;3];
%             hypsu = [hyp0; hypc; hypl];
%             hyp.mean = hypsu;
%             
%         case '@meanZero'
% %             disp('meanZero');
% %             m0 = {'meanZero'};  
%             hyp0 = [];      % no hyperparameters are needed
%             hyp.mean = hyp0;
% 
%     end
%     
%     sf = 2;
%     sn = .1;
%     ell = .9;
%     
%     D = 3;
%     L = rand(D,1);
%     l = rand(1);
%     c = 2;
%     al = 2;
%     p = 2;
%     
%     n = 5; 
%     D = 3; 
%     x = randn(n,D); 
%     xs = randn(3,D);
%     
%     switch (CovarianceFunction)
% %         @covMask @covProd @covMaternard @covRQard @covADD @covMaterniso @covRQiso @covConst @covNNone @covSEard @covCos @covNoise @covSEiso @covFITC @covPERard @covSEisoU @covGaborard @covPERiso @covSM @covGaboriso @covPPard @covScale @covLIN @covPPiso @covSum @covLINard @covPeriodic @covLINiso @covPeriodicNoDC @covLINone @covPoly
%         case '@covMask'
% %             disp('covMask');
% %             cgi = {'covSEiso'};  
% %             hypgi = log([ell;sf]);    % isotropic Gaussian
%             
% %             mask = [0,1,0]; %   binary mask excluding all but the 2nd component
% %             cma = {'covMask',{mask,cgi{:}}}; 
% %             hypma = hypgi;
%             
%             hypgi = log([ell;sf]);
%             hyp.cov = hypgi;
% 
%         case '@covProd'
% %             disp('covProd');
% %             cpr = {@covProd,{cc,ccc}};   
% %             hyppr = [hypc; hypcc];       % product
%             sf = 2;
%             hypc = log(sf);
%             hypcc = log([ell;sf]);
%             hyppr = [hypc; hypcc];
%             hyp.cov = hyppr;
%             
%         case '@covMaternard'
% %             disp('covMaternard');
% %             cma = {@covMaternard,5};  
%             sf = 2;
%             ell = .9;
%             hypma = log([ell;sf]); % Matern class d=5
%             hyp.cov = hypma;
%             
%         case '@covRQard'
% %             disp('covRQard');
% %             cra = {'covRQard'}; 
%             L = rand(D,1);
%             al = 2; 
%             sf = 2;
%             hypra = log([L;sf;al]); % ration. quad.
%             hyp.cov = hypra;
%             
%         case '@covADD'
% %             disp('covADD');
% %             cad = {'covADD',{[1,2],'covSEiso'}};
%             
%         case '@covMaterniso'
% %             disp('covMaterniso');
% %             cmi = {'covMaterniso',3}; 
%             ell = .9;
%             sf = 2;
%             hypmi = log([ell;sf]); % Matern class d=3
%             hyp.cov = hypmi;
%             
%         case '@covRQiso'
% %             disp('covRQiso');
% %             cri = {@covRQiso};          
%             sf = 2;
%             ell = .9;
%             al = 2;
%             hypri = log([ell;sf;al]);   % isotropic
%             hyp.cov = hypri;
%             
%         case '@covConst'
% %             disp('covConst');
% %             cc  = {@covConst};   
%             sf = 2;  
%             hypc = log(sf); % function handles OK
%             hyp.cov = hypc;
%             
%         case '@covNNone'
% %             disp('covNNone');
% %             cnn = {'covNNone'}; 
%             sf = 2;
%             hypnn = log([L;sf]);           % neural network
%             hyp.cov = hypnn;
%             
%         case '@covSEard'
% %             disp('covSEard');
% %             cga = {@covSEard};   
%             sf = 2;
%             hypga = log([L;sf]);       % Gaussian with ARD
%             hyp.cov = hypga;
%             
%         case '@covCos'
% %             disp('covCos');
% %             cpc = {'covCos'}; 
%             p = 2; 
%             sf = 2;
%             hypcpc = log([p;sf]);         % cosine cov
%             hyp.cov = hypcpc;
%             
%         case '@covNoise'
% %             disp('covNoise');
% %             cn  = {'covNoise'}; 
%             sn = .1;  
%             hypn = log(sn);  % one hyperparameter
%             hyp.cov = hypn;
%             
%         case '@covSEiso'
% %             disp('covSEiso');
% %             cgi = {'covSEiso'};  
%             ell = .9;
%             sf = 2;
%             hypgi = log([ell;sf]);    % isotropic Gaussian
%             hyp.cov = hypgi;
%             
%         case '@covFITC'
% %             disp('covFITC');
%             
%         case '@covPERard'
% %             disp('covPERard');
% %             cpa = {'covPERard',{@covMaternard,3}};
%             
%         case '@covSEisoU'
% %             disp('covSEisoU');
% %             cgu = {'covSEisoU'}; 
%             ell = .9;
%             hypgu = log(ell);   % isotropic Gauss no scale
%             hyp.cov = hypgu;
%             
%         case '@covGaborard'
% %             disp('covGaborard');
%             
%         case '@covPERiso'
% %             disp('covPERiso');
% %             cpi = {'covPERiso',{@covRQiso}};
%             
%         case '@covSM'
% %             disp('covSM');
% %             csm = {@covSM,4}; 
% %             hypsm = log([w;m(:);v(:)]);    % Spectral Mixture
%             
%         case '@covLINard'
% %             disp('covLINard');
% %             cla = {'covLINard'}; 
% %             L = rand(D,1); 
%             hypla = log(L);  % linear (ARD)
%             hyp.cov = hypla;
%             
%         case '@covPeriodic'
% %             disp('covPeriodic');
% %             cpe = {'covPeriodic'}; 
%             p = 2; 
%             ell = .9;
%             sf = 2;
%             hyppe = log([ell;p;sf]);   % periodic
%             hyp.cov = hyppe;
%             
%         case '@covLINiso'
% %             disp('covLINiso');
% %             cli = {'covLINiso'}; 
% %             l = rand(1);   
%             hypli = log(l);    % linear iso
%             hyp.cov = hypli;
%             
%         case '@covPeriodicNoDC'
% %             disp('covPeriodicNoDC');
% %             cpn = {'covPeriodicNoDC'}; 
%             p = 2; 
%             ell = .9;
%             sf = 2;
%             hyppe = log([ell;p;sf]); % w/o DC
%             hyp.cov = hyppe;
%             
%         case '@covLINone'
% %             disp('covLINone');
% %             clo = {@covLINone}; 
%             ell = .9; 
%             hyplo = log(ell);  % linear with bias
%             hyp.cov = hyplo;
%             
%         case '@covPoly'
% %             disp('covPoly');
% %             cp  = {@covPoly,3}; 
%             c = 2; 
%             sf = 2;
%             hypp = log([c;sf]);   % third order poly
%             hyp.cov = hypp;
%             
%     end
%     
%     n = 5; 
%     f = randn(n,1);
%     yr = f + randn(n,1)/20;
%     yc = sign(f);
%     sn = 0.1; 
%     nu = 4; 
%     a = 1;
%     
%     switch (LiklihoodFunction)
% %         @likGaussWarp @likSech2 @likGumbel @likT @likBeta @likInvGauss @likUni @likErf @likLaplace @likWeibull @likExp @likLogistic @likGamma @likMix @likGauss @likPoisson
%         case '@likGaussWarp'
% %             disp('likGaussWarp');
% %             lr5 = {'likGaussWarp',['poly2']}; 
%             hypr5 = log([a;sn]);
%             hyp.lik = hypr5;
%             
%         case '@likSech2'
% %             disp('likSech2');
% %             lr2 = {'likSech2'};   
%             hypr2 = log(sn);
%             hyp.lik = hypr2;
%             
%         case '@likGumbel'
% %             disp('likGumbel');
% %             lr6 = {'likGumbel','+'}; 
%             hypr6 = log(sn);
%             hyp.lik = hypr6;
%             
%         case '@likT'
% %             disp('likT');
% %             lr3 = {'likT'};       
%             hypr3 = [log(nu-1); log(sn)];
%             hyp.lik = hypr3;
%         
%         case '@likBeta'
% %             disp('likBeta');
% %             lg3 = {@likBeta,'expexp'};    
%             phi = 2.1; 
%             hyp.lik = log(phi);
%             
% %             lg4 = {@likBeta,'logit'};     
% %             phi = 4.7; 
% %             hyp.lik = log(phi);
%             
%         case '@likInvGauss'
% %             disp('likInvGauss');
% %             lg2 = {@likInvGauss,'exp'};   
%             lam = 1.1; 
%             hyp.lik = log(lam);
%             
%         case '@likUni'
% %             disp('likUni');
% %             lc2 = {'likUni'};     
%             hypc2 = [];
%             hyp.lik = hypc2;
%             
%         case '@likErf'
% %             disp('likErf');
% %             lc0 = {'likErf'};     
%             hypc0 = [];   % no hyperparameters are needed
%             hyp.lik = hypc0;
%             
%         case '@likLaplace'
% %             disp('likLaplace');
% %             lr1 = {'likLaplace'}; 
%             hypr1 = log(sn);
%             hyp.lik = hypr1;
%             
%         case '@likWeibull'
% %             disp('likWeibull');
%             
%         case '@likExp'
% %             disp('likExp');
%             
%         case '@likLogistic'
% %             disp('likLogistic');
% %             lc1 = {@likLogistic}; 
%             hypc1 = [];    % also function handles are OK
%             hyp.lik = hypc1;
%             
%         case '@likGamma'
% %             disp('likGamma');
% %             lg1 = {@likGamma,'logistic'}; 
%             al = 2;    
%             hyp.lik = log(al);
%             
%         case '@likMix'
% %             disp('likMix');
% %             lc3 = {'likMix',{'likUni',@likErf}}; 
%             hypc3 = log([1;2]); %mixture
%             hyp.lik = hypc3;
%             
%         case '@likGauss'
% %             disp('likGauss');
% %             lr0 = {'likGauss'};   
%             hypr0 = log(sn);
%             hyp.lik = hypr0;
%             
%         case '@likPoisson'
% %             disp('likPoisson');
% %             lp0 = {@likPoisson,'logistic'}; 
%             hypp0 = [];
%             hyp.lik = hypp0;
%             
% %             lp1 = {@likPoisson,'exp'};      
% %             hypp1 = [];
%                                                                                         
%     end
% end

% @covMask @covProd @covMaternard @covRQard @covADD @covMaterniso @covRQiso @covConst @covNNone @covSEard @covCos @covNoise @covSEiso @covFITC @covPERard @covSEisoU @covGaborard @covPERiso @covSM @covGaboriso @covPPard @covScale @covLIN @covPPiso @covSum @covLINard @covPeriodic @covLINiso @covPeriodicNoDC @covLINone @covPoly