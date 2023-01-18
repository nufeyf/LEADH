function final_B = LEADH(LTrain,GTrain,param)
    % parameters 

    max_iter = param.max_iter;
    alpa1 = param.alpa1; 
    alpa2 = param.alpa2;
    alpa3=param.alpa3;
    beta = param.beta; 
    sigma = param.sigma;
    gama = param.gama;
    nbits = param.nbits;
    k=nbits;
    n = size(LTrain,1);
    dimL = size(LTrain,2);
    E = ones(1, n);
   %% matrix initialization
    % class correlation
   % GTrain_ = NormalizeFea(LTrain')'; % global: l2-norm column normalized label matrix
%     GTrain_1 = NormalizeFea(LTrain')'; % global: l2-norm column normalized label matrix
%     Cg1 = GTrain_1'*GTrain_1; % global class correlation
%     Dg1 = diag(sum(Cg1)); % global degree matrix
    GTrain_ = NormalizeFea(LTrain,0); % global: l2-norm column normalized label matrix
    Cg = GTrain_'*GTrain_; % global class correlation
    Dg = diag(sum(Cg)); % global degree matrix
    %Cg=eye(dimL);
    %Dg=eye(dimL);
    clear GTrain_ GTrain_m_ LTrain_m
    %initization
    %G = randn(n, nbits);
    %B = sign(-1+(1-(-1))*rand(n, nbits));
    %W = sign(-1+(1-(-1))*rand(n, nbits));
    B = sign(randn(n, nbits)); B(B==0) = -1;
    W = sign(randn(n, nbits)); W(W==0) = -1;
    Jb = B-W;
    
    Zp = randn(nbits, nbits);
    [Pr,~,Qr] = svd(Zp);    
    P = Qr * Pr';
    Zu = randn(nbits, nbits);
    [Pz,~,Qz] = svd(Zu);    
    U = Qz * Pz';
    Jp =P-U;
    H=randn(n, nbits);
    for i = 1:max_iter
        fprintf('hash learning iter %3d\n', i);
        tempM=-alpa1*(H'*H*U*B'*B)+2*k*alpa1*(H'*(2*GTrain*(GTrain'*B)-E'*(E*B)))+2*alpa2*(H'*B)+sigma*U-Jp;
        [A1, ~, A2] = svd(tempM);
        P = A2 * A1';
        clear A1 A2
        tempD=alpa3*(LTrain'*LTrain)+gama*eye(dimL);
        D=tempD\(alpa3*(LTrain'*B));
        d1=beta*Dg';
        d2=gama*pinv(H'*H);
        d3=-beta*(pinv(H'*H)*H'*LTrain*Cg');
        F= sylvester(d2,d1,d3);
        %temp1=alpa1*k*(GTrain*(GTrain'*B)*P')+alpa2*(B*P')+beta*(LTrain*Cg'*F');
        temp1=alpa1*k*(2*GTrain*(GTrain'*(B*P'))-E'*(E*(B*P')))+alpa2*(B*P')+beta*(LTrain*(Cg'*F'));
        
        temp2=alpa1*(P*(B'*(B*P')))+alpa2*eye(nbits)+beta*(F*(Dg*F'));
        
        H=temp1/temp2;
        
        tempb=2*alpa1*k*((2*GTrain*(GTrain'*(H*P))-E'*(E*(H*P))))-alpa1*W*(P'*(H'*(H*P)))+2*alpa2*(H*P)+alpa3*(LTrain*D)+(sigma*W-Jb);
        B = sign(tempb);

        tempU=-alpa1*H'*(H*(P*(B'*B)))+sigma*P+Jp;
        [U1, ~, U2] = svd(tempU);
        U= U2 * U1';
        clear U1 U2
        Jp = P-U;
        tempW=-alpa1*B*(P'*(H'*(H*P)))+sigma*B+Jb;
        W=sign(tempW);
        Jb = B-W;

    end

    final_B = sign(B);
    final_B(final_B==0) = -1;

end