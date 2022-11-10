% beta 
clc;
clear;
N  = [1:1:500];
stack  = [5 2;5 8;30 2;30 8];

transmit_antenna =2;
loop =1000;
beta_average = zeros(length(N),4);
beta = zeros(length(N),4);
%% algorithm
for k=1:loop
    for j =1:length(N)  
        for rf=1:4
        %% R1 generator, Rician cascaded
        p = stack(rf,2);% transmitter antennas
        n = N(j);
        K_1 = 10;
        K_2 = stack(rf,1);
        
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
        %%
        M     = N(j);
        [V,D] = eig(R1);
        eigen_value=diag(D);
        leading_vector = sqrt(n)*V(:,M);
        %leading_vector = V(:,M);
        abs_leadingvector = abs(leading_vector);
        beta(j,rf)=sum(abs_leadingvector)/N(j);
        fprintf('k=%d,n=%d\n',k,n);
        end
    end
    beta_average(:,:) =beta_average(:,:)+ beta(:,:);
end
beta_average(:,:) = beta_average(:,:)/loop;
figure(1);

plot(N,beta_average(:,1),'-b','LineWidth',2);
hold on;
plot(N,beta_average(:,2),'-k','LineWidth',2);
plot(N,beta_average(:,3),'-r','LineWidth',2);
plot(N,beta_average(:,4),'-g','LineWidth',2);
legend('$K_2=5,n_t=2$','$K_2=5,n_t=8$','$K_2=30,n_t=2$','$K_2=30,n_t=8$','Interpreter', 'latex');
xlabel('$N,K_1=10$','Interpreter', 'latex');
ylabel('$\beta$','Interpreter', 'latex');
