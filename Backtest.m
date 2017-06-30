%Statistical Arbitrage Topic
%Objective: The goal is to run a backtest simulation using some classical alphas. 
%The backtest is intended to be realistic due to the inclusion of transaction costs.
%% Section 1: Loading the Data


load(strcat(path, 'database.mat'));

final_database = struct ( 'shrink', {},'alpharev',{},'alpharec',{},...
    'alphaval',{},'alphamom',{},'alphablend',{},'lambda',[],'mu',[],...
    't0',[],'trade',[],'back_weight',[],'pnl',[],'booksize',[],...
    'tradesize',[],'sharpe',[],'longest_dd',[],'deepest_dd',[]);

%% Section 2: Risk Model: 
% Every month, for the stocks in the universe where isactivenow is equal to one on the first day of the month, 
% compute the shrinkage estimator of the covariance matrix using the past year of daily returns. 
% By ?returns?, I mean specifically the arithmetic returns computed using DataStream?s Total Return Index (variable named tri). 
% As shrinkage target, take the identity matrix properly scaled. 
% You will use this matrix for all the portfolio optimizations conducted that month. 
% Store the optimal shrinkage intensities that you have estimated in a vector called shrink of dimension (60?1).

N = size(allstocks,2);
days = size(myday,1);
months = 72;
% Variable first_day keeps the row of each first day of the month
first_day = ones(months,1);
j = 2;
for i = 2:days
    if month(myday(i,:)) ~= month(myday(i-1,:))
        first_day(j) = i;
        j = j+1;
    end
end

%Valid months
vmonths = months-12;
%Returns
returns = [zeros(1,N); tri(2:end,:)./tri(1:(end-1),:)-1];
returns(isnan(returns)) = 0;

sigma = zeros(vmonths,1); %shrinkage target
shrink = zeros(vmonths,1); %shrinkage slope
nt = zeros(vmonths,1); %number of stocks active each month
T = zeros(vmonths,1); %number of trading days in each moving year
omega = zeros(vmonths,1); %estimation error (squared)
delta = zeros(vmonths,1); %dispersion (squared)
%active = [];
S = [];
covmat = zeros(N,N,vmonths);
for i = 13:months
    m1 = first_day(i-12);
    m2 = first_day(i)-1;
    T(i-12) = m2-m1+1;
    nt(i-12) = sum(isactivenow(m2+1,:));
    Xt = returns(m1:m2,:)';
    sigma(i-12) = sum(nanvar(Xt,1,2))/nt(i-12);
    S{i-12} = zeros(N);
    for j = 1:T(i-12)
        S{i-12} = S{i-12}+1/T(i-12)*Xt(:,j)*Xt(:,j)';
    end
    for j = 1:T(i-12)
        omega(i-12) = omega(i-12)+1/(T(i-12)*(T(i-12)-1))*sum(sum((Xt(:,j)*Xt(:,j)'-S{i-12}).^2));
    end
    delta(i-12) = sum(sum((S{i-12}-sigma(i-12)*eye(N)).^2))-omega(i-12);
    shrink(i-12) = delta(i-12)/(omega(i-12)+delta(i-12));
    covmat(:,:,i-12) = (1-shrink(i-12))*sigma(i-12)*eye(N)+shrink(i-12)*S{i-12};
        
end

final_database(1).shrink = shrink;


%% Section 3: Alphas

% a: Short-term contrarian (TECHNICAL):
industry = [];
for i = 1:N
    industry = [industry; allstocks(i).industrylist.industry];
end
uindustry = unique(industry,'rows');
rho = size(uindustry,1);
R = zeros(N,rho);
for i = 1:rho
    R(:,i) = strcmp(uindustry(i,:),cellstr(industry))*1;
end
alpharev = zeros(days,N);
alpha1 = eye(N)-R*((R'*R)^-1)*R';
%Triangular decay weights for 21 returns
w21 = repmat(1/11-1/231*(20:-1:0)',1,N);
for i = 2:days
    if i<22
        %Triangular decay asociated to i-2 lagged returns
        wi = repmat(2/i-2/(i*(i-1))*((i-2):-1:0)',1,N);
        alpharev(i,:) = -sum(returns(2:i,:).*wi,1)*alpha1;
    else
        alpharev(i,:) = -sum(returns((i-20):i,:).*w21,1)*alpha1;
    end
end
%Cross-section demean and Standarize
alpharev = (alpharev-repmat(mean(alpharev,2),1,N))./repmat(std(alpharev,0,2),1,N);
%Winsorize
alpharev(alpharev>3) = 3;
alpharev(alpharev<-3) = -3;
%No return first day
alpharev(1,:) = 0;
final_database(1).alpharev = alpharev;

%b: Short-term procyclical(FUNDAMENTAL):
alpharec = zeros(days,N);
for i = 1:days
    if i<45
        alpharec(i,:) = sum(rec(1:i,:),1)/i;
    else
        alpharec(i,:) = sum(rec((i-44):i,:),1)/45;
    end
end
%Cross-section demean and Standarize
alpharec = (alpharec-repmat(mean(alpharec,2),1,N))./repmat(std(alpharec,0,2),1,N);
%Winsorize
alpharec(alpharec>3) = 3;
alpharec(alpharec<-3) = -3;
%No recomendation first day
alpharev(1,:) = 0;
final_database(1).alpharec = alpharec;

%c: Long-term contrarian: (FUNDAMENTAL):
alphaval = 1./mtbv;
%Cross-section demean and Standarize
alphaval = (alphaval-repmat(nanmean(alphaval,2),1,N))./repmat(nanstd(alphaval,0,2),1,N);
%Winsorize
alphaval(alphaval>3) = 3;
alphaval(alphaval<-3) = -3;
final_database(1).alphaval = alphaval;

%d: Long-term procyclical (TECHNICAL):
alphamom = zeros(days,N);
for i = 253:days
    alphamom(i,:) = tri(i-21,:)./tri(i-252,:)-1;
end
%Cross-section demean and Standarize
alphamom = (alphamom-repmat(nanmean(alphamom,2),1,N))./repmat(nanstd(alphamom,0,2),1,N);
%Winsorize
alphamom(alphamom>3) = 3;
alphamom(alphamom<-3) = -3;
final_database(1).alphamom = alphamom;

%alphablend
w = [0.5 0.25 0.15 0.1];
alphablend = w(1)*alpharev+w(2)*alpharec+w(3)*alphaval+w(4)*alphamom;
%Cross-section demean and Standarize
alphablend = (alphablend-repmat(nanmean(alphablend,2),1,N))./repmat(nanstd(alphablend,0,2),1,N);
%Winsorize
alphablend(alphablend>3) = 3;
alphablend(alphablend<-3) = -3;
final_database(1).alphablend = alphablend;

%% Section 4: Optimizer

treturns = returns .';
mu = 1;
lambda = 1;

max_trade_size =  0.0015;% 1% of 15M(15/100) average daily volume
max_position_size = 0.0060; % liquidate every 3 days

% W: initial weights( 0 ), Nx1
t0 = 246;
tend = days;
weights = zeros(N,tend-t0+1);
trades = zeros(size(weights));

% Variables for Beta
marketReturn = nanmean(returns,2);
betaDays = 245;

%Variable to change cov matrix
num_cov = find(first_day > t0,1);
first_day = [first_day ; 0];
for day_index = t0:tend
    
    daycount = day_index-t0+1;
    activenow_idx = isactivenow(day_index,:)>0;
    N_active = sum(activenow_idx);
    sum_w = sum(weights(:, daycount));
    
    %Beta
    betaStocks = marketReturn(day_index-betaDays:day_index-1)\returns(day_index-betaDays:day_index-1,activenow_idx);
    
    if(day_index == t0)
        w = zeros(N_active,1);
    else
        w= weights(activenow_idx,daycount);    
        % start liquidating inactive
        trades(~activenow_idx, daycount) = -weights(~activenow_idx,daycount)*0.5; %liquidate over 10 days
        sum_w = sum_w + sum(trades(~activenow_idx, daycount)); 
    end    
    %
    theta = repmat(max_trade_size,N_active,1); %to follow notes notation
    pi_opt = repmat(max_position_size, N_active,1);    
    
    %num_cov = max(find(first_day <= day_index));
    cov_this = covmat(activenow_idx,activenow_idx,num_cov-13);
    if first_day(num_cov) == day_index
        num_cov = num_cov+1;
        cov_this = covmat(activenow_idx,activenow_idx,num_cov-13);
    end
    %cov_this = cov_mat(activenow_idx,activenow_idx);
    
    %   Matrix H
    H = 2* mu * [cov_this -cov_this; -cov_this cov_this];

    % g___
    alphaopt = alphablend .' ;
    alphaopt = alphaopt(activenow_idx,day_index-1);
    alphaopt(isnan(alphaopt)) = 0;
    
    %tcost
    tcostopt = tcost .';
    tcostopt = tcostopt(activenow_idx,day_index-1);
    tcostopt(isnan(tcostopt)) = 0;
    
    
    g = [2*mu*cov_this*w - alphaopt+lambda*tcostopt ; -2*mu*cov_this*w + alphaopt+lambda*tcostopt];
    
    %     %No equality contrains
    %     if day_index == t0
    %         eq_const=[];
    %         eq_to_const = [];
    %     else
    %         eq_const=[ones(1,N_active) ones(1,N_active);]; % Net position is 0
    %         eq_to_const = 0.15;
    %     end
    
    %eq_const and eq_to_const (Beta)
    eq_const = [betaStocks -betaStocks];
    eq_to_const = -betaStocks*w;
    
    %Upper and lower bound
    LB = zeros(2*N_active,1);
    UB = [max(0,min(theta, pi_opt - w)); max(0,min(theta, pi_opt + w))] ;
    
    options = optimset('Algorithm','interior-point-convex');
    options = optimset(options,'Display','iter');
    
    [u,fval,exitflag,output] = quadprog(H,g,[],[],eq_const,eq_to_const,LB,UB,[],options);
    %[u,fval,exitflag,output] = quadprog(H,g,A,b,eq_const,eq_to_const,LB,UB,[],options);
    
    y = u(1:N_active);
    z = u(N_active+1:end);
    x = w+y-z;
    trades(activenow_idx, day_index-t0+1) = y-z;    
    weights(:,daycount+1) = trades(:,daycount) +  weights(:,daycount);
end

final_database(1).lambda = lambda;
final_database(1).mu = mu;
final_database(1).t0 = t0;
final_database(1).trade = [zeros(t0-1,N);trades']*100000000;

%% Section 5
final_wts = weights(:, 1:end);
back_rets = treturns(:, t0:tend);
stock_positions = final_wts(:,1:(end-1)) .* (1+back_rets(:,1:end)) + trades ;
weight_positions = stock_positions;
stock_positions= stock_positions* 100000000;
back_weight = stock_positions;

%tcosts on trades
tcosts_bystock = tcost(t0:tend,:);
tcosts_bktest=nansum(abs(trades) .' .* tcosts_bystock,2)*100000000;
%
pnl=cumsum(sum(stock_positions .* back_rets(:,1:end)  )  - tcosts_bktest.');
figure(1)
plot(pnl)
title('PnL from Jan 1st, 1998 - Dec 31st, 2002');
xlabel('T')
ylabel('PnL')
saveas(gcf,'PS3_pnl.png')
booksize=sum(abs(final_wts) )*100000000;

final_database(1).back_weight = [zeros(t0-1,N);back_weight'];
final_database(1).pnl = [zeros(t0-1,1);pnl'];
final_database(1).booksize = [zeros(t0-2,1);booksize'];
final_database(1).tradesize = [zeros(t0-1,1);sum(trades)']*100000000;

%% Section 6
sharpe_rets = (sum(stock_positions .* back_rets(:,1:end)  )  - tcosts_bktest.' )/100000000;
daily_exp = mean(sharpe_rets);
daily_var = var(sharpe_rets);
sharpe = daily_exp/sqrt(daily_var) * sqrt(252);
%longest drawdown
max_watermark = 0;
counter_longestdd = 0;
dd_max = 0;
deep_dd = 0;
all_pnls = (sum(stock_positions .* back_rets(:,1:end)  )  - tcosts_bktest.' );
for i = 1:length(pnl)
    if(pnl(i)>max_watermark)
        max_watermark = pnl(i);
        counter_longestdd = 0;
    else
        if(max_watermark - pnl(i) > deep_dd)
             deep_dd = max_watermark - pnl(i);
        end
        counter_longestdd = counter_longestdd + 1;
        if(counter_longestdd > dd_max)
            dd_max = counter_longestdd;
        end
    end
end
longest_dd = dd_max;
deepest_dd = deep_dd;

final_database(1).sharpe = sharpe;
final_database(1).longest_dd = longest_dd;
final_database(1).deepest_dd = deepest_dd;