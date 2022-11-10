function R=GEM_R(N,rank)
p = rank;% transmitter antennas
n = N;
K_2 = 10 ;
K_1 = 10;
H_w1 = (randn(n,p)+1i*randn(n,p))/sqrt(2);
eig_M = 10;
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
R = Phi_total*Phi_total';