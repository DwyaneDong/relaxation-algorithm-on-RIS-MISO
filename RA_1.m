% relaxation和manifold比较，比较time和gain, rank stay unchanged ,  N increasing
clc;
clear;
N = [50:50:500];
%s = rng(2);
rankaa =[1,4,8,16];
loop = 100;
result_manifold = zeros(length(N),1);
result_verify = zeros(length(N),1);
time_manifold   = zeros(length(N),1);
time_verify      = zeros(length(N),1);
result_manifold_average  = zeros(length(N),1);
result_verify_average  = zeros(length(N),1);
time_manifold_average   = zeros(length(N),1);
time_verify_average      = zeros(length(N),1);
result_sdr=zeros(length(N),1);
time_sdr=zeros(length(N),1);
result_sdr_average=zeros(length(N),1);
time_sdr_average=zeros(length(N),1);
%rank4
result_manifold_4 = zeros(length(N),1);
result_verify_4 = zeros(length(N),1);
time_manifold_4   = zeros(length(N),1);
time_verify_4      = zeros(length(N),1);
result_manifold_average_4  = zeros(length(N),1);
result_verify_average_4  = zeros(length(N),1);
time_manifold_average_4   = zeros(length(N),1);
time_verify_average_4      = zeros(length(N),1);
result_sdr_4=zeros(length(N),1);
time_sdr_4=zeros(length(N),1);
result_sdr_average_4=zeros(length(N),1);
time_sdr_average_4=zeros(length(N),1);
%rank8
result_manifold_8 = zeros(length(N),1);
result_verify_8 = zeros(length(N),1);
time_manifold_8   = zeros(length(N),1);
time_verify_8      = zeros(length(N),1);
result_manifold_average_8  = zeros(length(N),1);
result_verify_average_8  = zeros(length(N),1);
time_manifold_average_8   = zeros(length(N),1);
time_verify_average_8      = zeros(length(N),1);
result_sdr_8=zeros(length(N),1);
time_sdr_8=zeros(length(N),1);
result_sdr_average_8=zeros(length(N),1);
time_sdr_average_8=zeros(length(N),1);
%rank16
result_manifold_16 = zeros(length(N),1);
result_verify_16 = zeros(length(N),1);
time_manifold_16   = zeros(length(N),1);
time_verify_16      = zeros(length(N),1);
result_manifold_average_16  = zeros(length(N),1);
result_verify_average_16  = zeros(length(N),1);
time_manifold_average_16   = zeros(length(N),1);
time_verify_average_16      = zeros(length(N),1);
result_sdr_16=zeros(length(N),1);
time_sdr_16=zeros(length(N),1);
result_sdr_average_16=zeros(length(N),1);
time_sdr_average_16=zeros(length(N),1);
%% algorithm
for k=1:loop
    for j =1:length(N)      
        R1 = GEM_R(N(j),rankaa(1));
        R4 = GEM_R(N(j),rankaa(2));
        R8 = GEM_R(N(j),rankaa(3));
        R16 = GEM_R(N(j),rankaa(4));
        M         =   N(j);
    %% verify the mathematic principle 
        verify_tic = tic;
        [V,D]=eig(R1);
        theta_FR_verify    = angle(V(:,M));
        time_verify(j)      = toc(verify_tic);   
        result_verify(j) = exp(1i*theta_FR_verify)'*R1*exp(1i*theta_FR_verify);
        for i=1:10000
            fprintf('k=%f,j=%f,k=%f,j=%f,k=%f\n',k,j,k,j,k);
        end
        %rank4
        verify_tic_4 = tic;
        [V4,D4]=eig(R4);
        theta_FR_verify_4    = angle(V4(:,M));
        time_verify_4(j)      = toc(verify_tic_4);   
        result_verify_4(j) = exp(1i*theta_FR_verify_4)'*R4*exp(1i*theta_FR_verify_4);
        for i=1:10000
             fprintf('k=%f,j=%f,k=%f,j=%f,k=%f\n',k,j,k,j,k);
        end
        %rank8
        verify_tic_8 = tic;
        [V8,D8]=eig(R8);
        theta_FR_verify_8    = angle(V8(:,M));
        time_verify_8(j)      = toc(verify_tic_8);   
        result_verify_8(j) = exp(1i*theta_FR_verify_8)'*R8*exp(1i*theta_FR_verify_8);
        for i=1:10000
             fprintf('k=%f,j=%f,k=%f,j=%f,k=%f\n',k,j,k,j,k);
        end
        %rank4
        verify_tic_16 = tic;
        [V16,D16]=eig(R16);
        theta_FR_verify_16    = angle(V16(:,M));
        time_verify_16(j)      = toc(verify_tic_16);   
        result_verify_16(j) = exp(1i*theta_FR_verify_16)'*R16*exp(1i*theta_FR_verify_16);
        for i=1:10000
             fprintf('k=%f,j=%f,k=%f,j=%f,k=%f\n',k,j,k,j,k);
        end
    %% manifold optimization
        manifold_tic = tic;
        manifold     = complexcirclefactory(N(j));
        problem.M    = manifold;
        problem.cost = @(w) -w'*R1*w; %代价函数，一般是最小，因为我们是最大化问题，所以要加负号。
        problem.grad = @(w) manifold.egrad2rgrad(w,-2*R1*w);
        [w,wcost,info,options] = steepestdescent(problem); % at a random point on the manifold
        %[w,wcost,info,options] = conjugategradient(problem);%共轭梯度下降法
        %[w,wcost,info,options] = barzilaiborwein(problem);  %Barzilai Borwein梯度法
        time_manifold(j)       = toc(manifold_tic);
        result_manifold(j)     = w'*R1*w;
        %rank4
        manifold_tic_4 = tic;
        manifold_4     = complexcirclefactory(N(j));
        problem_4.M    = manifold_4;
        problem_4.cost = @(w4) -w4'*R4*w4; %代价函数，一般是最小，因为我们是最大化问题，所以要加负号。
        problem_4.grad = @(w4) manifold.egrad2rgrad(w4,-2*R4*w4);
        [w4,wcost,info,options] = steepestdescent(problem_4); % at a random point on the manifold
        %[w,wcost,info,options] = conjugategradient(problem);%共轭梯度下降法
        %[w,wcost,info,options] = barzilaiborwein(problem);  %Barzilai Borwein梯度法
        time_manifold_4(j)       = toc(manifold_tic_4);
        result_manifold_4(j)     = w4'*R4*w4;
        %rank8
        manifold_tic_8 = tic;
        manifold_8     = complexcirclefactory(N(j));
        problem_8.M    = manifold_8;
        problem_8.cost = @(w8) -w8'*R8*w8; %代价函数，一般是最小，因为我们是最大化问题，所以要加负号。
        problem_8.grad = @(w8) manifold.egrad2rgrad(w8,-2*R8*w8);
        [w8,wcost,info,options] = steepestdescent(problem_8); % at a random point on the manifold
        %[w,wcost,info,options] = conjugategradient(problem);%共轭梯度下降法
        %[w,wcost,info,options] = barzilaiborwein(problem);  %Barzilai Borwein梯度法
        time_manifold_8(j)       = toc(manifold_tic_8);
        result_manifold_8(j)     = w8'*R8*w8;
        %rank16
        manifold_tic_16 = tic;
        manifold_16     = complexcirclefactory(N(j));
        problem_16.M    = manifold_16;
        problem_16.cost = @(w16) -w16'*R16*w16; %代价函数，一般是最小，因为我们是最大化问题，所以要加负号。
        problem_16.grad = @(w16) manifold.egrad2rgrad(w16,-2*R16*w16);
        [w16,wcost,info,options] = steepestdescent(problem_16); % at a random point on the manifold
        %[w,wcost,info,options] = conjugategradient(problem);%共轭梯度下降法
        %[w,wcost,info,options] = barzilaiborwein(problem);  %Barzilai Borwein梯度法
        time_manifold_16(j)       = toc(manifold_tic_16);
        result_manifold_16(j)     = w16'*R16*w16;
      %% convex optimization(SDR)
        sdr_tic = tic;
        count   = 25;
        f_tmp   = 0;
        for p=1:count
            r = (randn(N(j),1)+1i*randn(N(j),1)).*sqrt(1/2);   % (N,1)
            cvx_begin
            variable V(N(j),N(j)) symmetric semidefinite   %变量是一个(N)*(N)的对称半正定矩阵
            maximize( real(trace(R1*V)))
            subject to
            diag(V) == 1;
            cvx_end
            [U,Sigma] = eig(V);
            w         = U*Sigma^(1/2)*r;   % (N*1)
            f         = w'*R1*w;       %随机次数为count次，找到其中最大的f对应的波束赋形向量w和高斯随机向量r
            if f>f_tmp
                f_tmp = max(f,f_tmp);
                r_tmp = r;
                w_tmp = w;     %求解出来的w_tmp为啥比w_的维度要小10的倍数个元素
            end
        end
        theta_opt     = angle(w_tmp);
        w_opt         = exp(1i*theta_opt);
        time_sdr(j)   = toc(sdr_tic);
        result_sdr(j) = w_opt'*R1*w_opt;
        %rank4
        sdr_tic4 = tic;
        f_tmp4   = 0;
        for p=1:count
            r = (randn(N(j),1)+1i*randn(N(j),1));   % (N,1)
            cvx_begin
            variable V(N(j),N(j)) symmetric semidefinite   %变量是一个(N)*(N)的对称半正定矩阵
            maximize( real(trace(R4*V4)))
            subject to
            diag(V4) == 1;
            cvx_end
            [U4,Sigma4] = eig(V4);
            w4         = U4*Sigma4^(1/2)*r;   % (N*1)
            f4         = w4'*R4*w4;       %随机次数为count次，找到其中最大的f对应的波束赋形向量w和高斯随机向量r
            if f4>f_tmp4
                f_tmp4 = max(f4,f_tmp4);
                r_tmp = r;
                w_tmp4 = w4;     %求解出来的w_tmp为啥比w_的维度要小10的倍数个元素
            end
        end
        theta_opt4     = angle(w_tmp4);
        w_opt4         = exp(1i*theta_opt4);
        time_sdr_4(j)   = toc(sdr_tic4);
        result_sdr_4(j) = w_opt4'*R4*w_opt4;
        %rank4
        sdr_tic8 = tic;
        f_tmp8   = 0;
        for p=1:count
            r = (randn(N(j),1)+1i*randn(N(j),1));   % (N,1)
            cvx_begin
            variable V(N(j),N(j)) symmetric semidefinite   %变量是一个(N)*(N)的对称半正定矩阵
            maximize( real(trace(R8*V8)))
            subject to
            diag(V8) == 1;
            cvx_end
            [U8,Sigma8] = eig(V8);
            w8         = U8*Sigma8^(1/2)*r;   % (N*1)
            f8         = w8'*R8*w8;       %随机次数为count次，找到其中最大的f对应的波束赋形向量w和高斯随机向量r
            if f8>f_tmp8
                f_tmp8 = max(f8,f_tmp8);
                r_tmp = r;
                w_tmp8 = w8;     %求解出来的w_tmp为啥比w_的维度要小10的倍数个元素
            end
        end
        theta_opt8     = angle(w_tmp8);
        w_opt8         = exp(1i*theta_opt8);
        time_sdr_8(j)   = toc(sdr_tic8);
        result_sdr_8(j) = w_opt8'*R8*w_opt8;
        %rank4
        sdr_tic16 = tic;
        f_tmp16   = 0;
        for p=1:count
            r = (randn(N(j),1)+1i*randn(N(j),1));   % (N,1)
            cvx_begin
            variable V(N(j),N(j)) symmetric semidefinite   %变量是一个(N)*(N)的对称半正定矩阵
            maximize( real(trace(R16*V16)))
            subject to
            diag(V16) == 1;
            cvx_end
            [U16,Sigma16] = eig(V16);
            w16         = U16*Sigma16^(1/2)*r;   % (N*1)
            f16         = w16'*R16*w16;       %随机次数为count次，找到其中最大的f对应的波束赋形向量w和高斯随机向量r
            if f16>f_tmp16
                f_tmp16 = max(f16,f_tmp16);
                r_tmp = r;
                w_tmp16 = w16;     %求解出来的w_tmp为啥比w_的维度要小10的倍数个元素
            end
        end
        theta_opt16     = angle(w_tmp16);
        w_opt16         = exp(1i*theta_opt16);
        time_sdr_16(j)   = toc(sdr_tic16);
        result_sdr_16(j) = w_opt16'*R16*w_opt16;
    end
    result_manifold_average = result_manifold_average + result_manifold;
    result_verify_average = result_verify_average + result_verify;
    time_manifold_average   = time_manifold_average   + time_manifold;
    time_verify_average      = time_verify_average      + time_verify;
    result_manifold_average_4 = result_manifold_average_4 + result_manifold_4;
    result_verify_average_4 = result_verify_average_4 + result_verify_4;    
    time_manifold_average_4   = time_manifold_average_4   + time_manifold_4;
    time_verify_average_4      = time_verify_average_4      + time_verify_4;
    result_manifold_average_8 = result_manifold_average_8 + result_manifold_8;
    result_verify_average_8 = result_verify_average_8 + result_verify_8;    
    time_manifold_average_8   = time_manifold_average_8   + time_manifold_8;
    time_verify_average_8      = time_verify_average_8      + time_verify_8;
    result_manifold_average_16 = result_manifold_average_16 + result_manifold_16;
    result_verify_average_16 = result_verify_average_16 + result_verify_16;    
    time_manifold_average_16   = time_manifold_average_16   + time_manifold_16;
    time_verify_average_16      = time_verify_average_16      + time_verify_16;

    result_sdr_average = result_sdr_average + result_sdr;
    time_sdr_average = time_sdr_average + time_sdr;

    result_sdr_average_4 = result_sdr_average_4 + result_sdr_4;
    time_sdr_average_4 = time_sdr_average_4 + time_sdr_4;

    result_sdr_average_8 = result_sdr_average_8 + result_sdr_8;
    time_sdr_average_8 = time_sdr_average_8 + time_sdr_8;

    result_sdr_average_16 = result_sdr_average_16 + result_sdr_16;
    time_sdr_average_16 = time_sdr_average_16 + time_sdr_16;
end
result_manifold_average = result_manifold_average/loop;
result_verify_average = result_verify_average/loop;
result_sdr_average = result_sdr_average/loop;
time_manifold_average   = time_manifold_average/loop;
time_verify_average      = time_verify_average/loop;
 time_sdr_average = time_sdr_average/loop;
result_manifold_average_4 = result_manifold_average_4/loop;
result_verify_average_4 = result_verify_average_4/loop;
result_sdr_average_4 = result_sdr_average_4/loop;
time_manifold_average_4   = time_manifold_average_4/loop;
time_verify_average_4      = time_verify_average_4/loop;
time_sdr_average_4 = time_sdr_average_4/loop;
result_manifold_average_8 = result_manifold_average_8/loop;
result_verify_average_8 = result_verify_average_8/loop;
result_sdr_average_8 = result_sdr_average_8/loop;
time_manifold_average_8   = time_manifold_average_8/loop;
time_verify_average_8      = time_verify_average_8/loop;
time_sdr_average_8 = time_sdr_average_8/loop;
result_manifold_average_16 = result_manifold_average_16/loop;
result_verify_average_16 = result_verify_average_16/loop;
result_sdr_average_16 = result_sdr_average_16/loop;
time_manifold_average_16   = time_manifold_average_16/loop;
time_verify_average_16      = time_verify_average_16/loop;
time_sdr_average_16 = time_sdr_average_16/loop;
%% plot
%time
figure(1);
hold on;
plot(N,real(time_manifold_average_16),'-.sk','LineWidth',1);
plot(N,real(time_verify_average_16),'-.^b','LineWidth',1);
plot(N,real(time_sdr_average_16),'-.vb','LineWidth',1);
plot(N,real(time_manifold_average_8),':sk','LineWidth',1);
plot(N,real(time_verify_average_8),':^b','LineWidth',1);
plot(N,real(time_sdr_average_8),':vb','LineWidth',1);
plot(N,real(time_manifold_average_4),'--sk','LineWidth',1);
plot(N,real(time_verify_average_4),'--^b','LineWidth',1);
plot(N,real(time_sdr_average_4),'--vb','LineWidth',1);
plot(N,real(time_manifold_average),'-sk','LineWidth',1);
plot(N,real(time_verify_average),'-^b','LineWidth',1);
plot(N,real(time_sdr_average),'-vb','LineWidth',1);
legend('MO, 16','RA, 16','SDR, 16','MO, 8','RA, 8','SDR, 8','MO, 4','RA, 4','SDR, 4','MO, 1','RA, 1','SDR, 1','Interpreter','latex','FontSize',8);
xlabel('Number of Units in RIS', 'Interpreter', 'latex', 'FontSize', 10);
ylabel('time', 'Interpreter', 'latex', 'FontSize', 10);
grid on;
grid minor;
%dB time
figure(2);
hold on;
plot(N,10*log10(real(time_manifold_average_16)),'-.sr','LineWidth',0.5);
plot(N,10*log10(real(time_verify_average_16)),'-.^b','LineWidth',0.5);
plot(N,10*log10(real(time_sdr_average_16)),'-.vk','LineWidth',0.5);
plot(N,10*log10(real(time_manifold_average_8)),':sr','LineWidth',0.5);
plot(N,10*log10(real(time_verify_average_8)),':^b','LineWidth',0.5);
plot(N,10*log10(real(time_sdr_average_8)),':vk','LineWidth',0.5);
plot(N,10*log10(real(time_manifold_average_4)),'--sr','LineWidth',0.5);
plot(N,10*log10(real(time_verify_average_4)),'--^b','LineWidth',0.5);
plot(N,10*log10(real(time_sdr_average_4)),'--vk','LineWidth',0.5);
plot(N,10*log10(real(time_manifold_average)),'-sr','LineWidth',0.5);
plot(N,10*log10(real(time_verify_average)),'-^b','LineWidth',0.5);
plot(N,10*log10(real(time_sdr_average)),'-vk','LineWidth',0.5);
legend('MO, 16','RA, 16','SDR, 16','MO, 8','RA, 8','SDR, 8','MO, 4','RA, 4','SDR, 4','MO, 1','RA, 1','SDR, 1','Interpreter','latex','FontSize',8,'NumColumns',4);
xlabel('Number of Units in RIS $N$', 'Interpreter', 'latex', 'FontSize', 15);
ylabel('time/dB', 'Interpreter', 'latex', 'FontSize', 15);
grid on;
grid minor;
% gain
figure(3);
hold on;
plot(N,real(result_manifold_average_16),'-.sk','LineWidth',1);
plot(N,real(result_verify_average_16),'-.^b','LineWidth',1);
plot(N,real(result_sdr_average_16),'-.vb','LineWidth',1);
plot(N,real(result_manifold_average_8),':sk','LineWidth',1);
plot(N,real(result_verify_average_8),':^b','LineWidth',1);
plot(N,real(result_sdr_average_8),':vb','LineWidth',1);
plot(N,real(result_manifold_average_4),'--sk','LineWidth',1);
plot(N,real(result_verify_average_4),'--^b','LineWidth',1);
plot(N,real(result_sdr_average_4),'--vb','LineWidth',1);
plot(N,real(result_manifold_average),'-sk','LineWidth',1);
plot(N,real(result_verify_average),'-^b','LineWidth',1);
plot(N,real(result_sdr_average),'-vb','LineWidth',1);
legend('MO, 16','RA, 16','SDR, 16','MO, 8','RA, 8','SDR, 8','MO, 4','RA, 4','SDR, 4','MO, 1','RA, 1','SDR, 1','Interpreter','latex','FontSize',8);
xlabel('Number of Units in RIS', 'Interpreter', 'latex', 'FontSize', 10);
ylabel('gain', 'Interpreter', 'latex', 'FontSize', 10);
grid on;
grid minor;
%gain/dB
figure(4);
hold on;
plot(N,10*log10(real(result_manifold_average_16)),'-.sr','LineWidth',0.5);
plot(N,10*log10(real(result_verify_average_16)),'-.^b','LineWidth',0.5);
plot(N,10*log10(real(result_sdr_average_16)),'-.vk','LineWidth',0.5);
plot(N,10*log10(real(result_manifold_average_8)),':sr','LineWidth',0.5);
plot(N,10*log10(real(result_verify_average_8)),':^b','LineWidth',0.5);
plot(N,10*log10(real(result_sdr_average_8)),':vk','LineWidth',0.5);
plot(N,10*log10(real(result_manifold_average_4)),'--sr','LineWidth',0.5);
plot(N,10*log10(real(result_verify_average_4)),'--^b','LineWidth',0.5);
plot(N,10*log10(real(result_sdr_average_4)),'--vk','LineWidth',0.5);
plot(N,10*log10(real(result_manifold_average)),'-sr','LineWidth',0.5);
plot(N,10*log10(real(result_verify_average)),'-^b','LineWidth',0.5);
plot(N,10*log10(real(result_sdr_average)),'-vk','LineWidth',0.5);
legend('MO, 16','RA, 16','SDR, 16','MO, 8','RA, 8','SDR, 8','MO, 4','RA, 4','SDR, 4','MO, 1','RA, 1','SDR, 1','Interpreter','latex','FontSize',8,'NumColumns',4);
xlabel('Number of Units in RIS $N$', 'Interpreter', 'latex', 'FontSize', 10);
ylabel('gain/dB', 'Interpreter', 'latex', 'FontSize', 10);
grid on;
grid minor;