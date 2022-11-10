close all; clear; clc

coeff = 50;
p = 10*coeff;% transmitter antennas n_t
c = 10;%c=n/p
n = round(p*c,0);% N
K_1 = sqrt(10);
K_2 = 100000;

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
%Phi_total = H_1;
% =======================
%SCM = Phi_total*Phi_total';
SCM = Phi_total*Phi_total'/n;
% =======================
eigs_SCM = eig(SCM);
eigs_SCM = sort(eigs_SCM);
edges=linspace((1-sqrt(c))^2-eps,(1+sqrt(c))^2+eps,300);


%a = (1-sqrt(c))^2;
b = (1+sqrt(c))^2/(c*(K_1+1));


isolated_eigs =(1/c+1/K_1);
%isolated_eigs(eig_M<=sqrt(c)) = NaN;


subplot(1,2,1);
subplot(1,2,2);


figure(1);
eigs_SCM_without_zero = eigs_SCM(n-p+1:n);
histogram(eigs_SCM_without_zero,50, 'Normalization', 'pdf', 'EdgeColor', 'white','FaceColor','red');
hold on;
%mu=sqrt( max(edges-a,0).*max(b-edges,0) )/2/pi/c./edges;
%plot(edges,mu,'r', 'Linewidth',2);
%plot(b,zeros(length(b),1),'b^', 'LineWidth', 2,'MarkerSize',10);
plot(isolated_eigs,zeros(length(isolated_eigs),1)+0.05,'bo', 'LineWidth', 2,'MarkerSize',10);
xlabel('Eigenvalues of $\mathbf R/N$, $c=10$, $K_1=\sqrt{10}$', 'Interpreter', 'latex', 'FontSize', 10);
% plot(isolated_eigs,zeros(length(isolated_eigs),1)+0.1,'bo', 'LineWidth', 2,'MarkerSize',10);
% xlabel('Eigenvalues of $\mathbf R/N$, $c=10$, $K_1=10$', 'Interpreter', 'latex', 'FontSize', 10);
ylabel('Empirical Spectrum Distribution', 'Interpreter', 'latex', 'FontSize', 10);
%title('$c=0.5,\ell_{i}=\sqrt {0.6}$', 'Interpreter', 'latex', 'FontSize', 15)
legend('Empirical eigenvalues', '$\hat{\lambda}_1$ given by (18)', 'FontSize', 10, 'Interpreter', 'latex')
%axis([0 max(eigs_SCM)+.5 0 max(mu)*1.1]);
%axis([0 6 0 max(mu)*1.1]);
