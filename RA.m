% 取前N个约束和manifold比较，比较速度
clc;
clear;
N = [50:50:500];
%s = rng(2);
transmit_antenna =32;
loop = 1;
result_RA = zeros(length(N),1);
result_MO = zeros(length(N),1);

time_RA      = zeros(length(N),1);
time_MO   = zeros(length(N),1);

result_RA_aver  = zeros(length(N),1);
result_MO_aver  = zeros(length(N),1);

time_RA_aver      = zeros(length(N),1);
time_MO_aver   = zeros(length(N),1);
%% algorithm
for k=1:loop
    for j =1:length(N)      
        %% R1 generator, Rician cascaded
        p = transmit_antenna;% transmitter antennas
        n = N(j);
        K_1 = 100;
        K_2 = 10;
        
        H_w1 = (randn(n,p)+1i*randn(n,p))/sqrt(2);
        
        % eig_M = [sqrt(0.5) ];% spike equals to 1+l+c(1+l)/l, and right bound is (1+sqrt(c))^2
        % M = diag([(eig_M+1i*eig_M)/sqrt(2), zeros(1,p-length(eig_M))]);
        eig_M = 1;
        a  = sqrt(eig_M);
        M_1    = (ones(n,1)+1i*ones(n,1))/sqrt(2);
        M_2    = (ones(p,1)+1i*ones(p,1))/sqrt(2);
        M_first = a*(M_1*M_2');
        M_first = sqrt(p/trace(M_first*M_first'))*M_first; % nomalized
        H_1 = sqrt(1/(K_1+1))*H_w1 + sqrt(K_1/(K_1+1))*M_first;
        
        H_w2= (randn(n,1)+1i*randn(n,1))/sqrt(2);
        M_second= (ones(n,1)+1i*ones(n,1))/sqrt(2);
        M_second= sqrt(n/trace(M_second*M_second'))*M_second; % nomalized
        H_2 = sqrt(1/(K_2+1))*H_w2 + sqrt(K_2/(K_2+1))*M_second;
        
        Phi_total = diag(H_2')*H_1;
        
        R1 = Phi_total*Phi_total';
    %% RA
        M = N(j);
        RA_tic = tic;
        [V,D]=eig(R1);
        theta_RA    = angle(V(:,M));
        time_RA(j)      = toc(RA_tic);   
        result_RA(j) = exp(1i*theta_RA)'*R1*exp(1i*theta_RA);
    %% MO
        MO_tic = tic;
        manifold     = complexcirclefactory(N(j));
        problem.M    = manifold;
        problem.cost = @(w) -w'*R1*w; %代价函数，一般是最小，因为我们是最大化问题，所以要加负号。
        problem.grad = @(w) manifold.egrad2rgrad(w,-2*R1*w);
        %problem.grad = @(w) manifold.egrad2rgrad(w,-R.'*conj(w));%这里的映射是一阶的，正交投影的方式。具体见“https://www.manopt.org/tutorial.html”
        [w,wcost,info,options] = steepestdescent(problem); % at a random point on the manifold
        time_MO(j)       = toc(MO_tic);
        result_MO(j)     = w'*R1*w;
    end
    result_RA_aver = result_RA_aver + result_RA;
    result_MO_aver = result_MO_aver + result_MO;
    
    time_RA_aver      = time_RA_aver      + time_RA;
    time_MO_aver   = time_MO_aver   + time_MO;
end
result_RA_aver = result_RA_aver/loop;
result_MO_aver = result_MO_aver/loop;

time_RA_aver      = time_RA_aver/loop;
time_MO_aver   = time_MO_aver/loop;
%% gain
figure(1);
plot(N,10*log10(real(result_RA_aver)),'-ob','LineWidth',2);
hold on;
plot(N,10*log10(real(result_MO_aver)),'-*k','LineWidth',2);
%plot(N,10*log(real(result_sdr_average)),'-b','LineWidth',2);
legend('RA','MO');
xlabel('N');
ylabel('gain/dB');
grid on;
grid minor;
%% time
figure(2);
plot(N,10*log10(time_RA_aver),'--b','LineWidth',2);
hold on;
plot(N,10*log10(time_MO_aver),'-b','LineWidth',2);
%plot(N,time_sdr_average,'-k','LineWidth',2);
%legend('relax','manifold')
legend('RA','MO');
xlabel('N');
ylabel('seconds/dB');
grid on;
grid minor;