function a = F_intercept(FunctionValue)
%�����ά�Ľؾ�

    [N,M] = size(FunctionValue);

    %�ҳ��߽��
    [~,Choosed(1:M)] = min(FunctionValue,[],1);
    L2NormABO = zeros(N,M);
    for i = 1 : M
    	L2NormABO(:,i) = sum(FunctionValue(:,[1:i-1,i+1:M]).^2,2);
    end
    [~,Choosed(M+1:2*M)] = min(L2NormABO,[],1);
    [~,Extreme] = max(FunctionValue(Choosed,:),[],1);
    Extreme = unique(Choosed(Extreme));
    
    %����ؾ�
    if length(Extreme) < M
        a = max(FunctionValue,[],1);
    else
        Hyperplane = FunctionValue(Extreme,:)\ones(M,1);
        a = 1./Hyperplane';
    end
end

