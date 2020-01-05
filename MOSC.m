clear all
clc
matlabpool open
path= 'data\';
DD = dir(fullfile(path,'*.mat'));
for dd = 1: length(DD)
    load(fullfile(path,DD(dd).name));
    fp = fopen(['result\',DD(dd).name,'3.txt'],'wt');
    fea = normalizeData(fea);
    data=[fea gnd];
    [numS, dim]=size(fea);
    if dim>200
        [features,weights] = MI(fea,gnd,12);
        fea1=fea(:,features(1:200));
    else
        fea1=fea;
    end
    cluster_num = length(unique(gnd));
    times=1;
    s1 = 0;
    s2 = 0;
    parfor runs=1:times
        indexnumber = 1;
        disp(runs)
        ObjectiveNum = 2;
        p1 = [99 13  7  5  3  3  3  2  2];
        p2 = [ 0  0  0  0  3  2  1  2  2];
        p1 = p1(ObjectiveNum-1);
        p2 = p2(ObjectiveNum-1);   
        [popsize,W] = F_weight(p1,p2,ObjectiveNum);
        W(W==0) = 0.000001;
        for i = 1 : popsize
            W(i,:) = W(i,:)./norm(W(i,:));
        end
        Dist=pdist2(data(:,1:end-1),data(:,1:end-1));
        [numR,numSS] = size(fea1);
        Population=rand(popsize,numSS+2);
        Sum_pop=sum(Population(:,1:end-2)')';
        repmat(Sum_pop,1,numSS);
        Population1=Population(:,1:end-2)./repmat(Sum_pop,1,numSS);
        FunctionValue = zeros(popsize, ObjectiveNum);
        label_FunctionValue = zeros(popsize,numR);
        for qkt = 1:popsize
            k_number = ceil(Population(qkt,end-1)*10);
            Type = 1;
            sigma = ceil(Population(qkt,end)*20);
            WW=SimGraph_NearestNeighborsNew(Population1(qkt,:), fea1', k_number, Type, sigma);
            [C,L,U] = SpectralClustering(WW, cluster_num,3);%%%
            dtype=1;
            [DB,CH,Dunn,KL,Han,~] = valid_internal_deviation(data(:,1:end-1),C,dtype);
            cp = valid_compactness(data(:,1:end-1), C);
            FunctionValue(qkt,:)=[cp -CH];
            label_FunctionValue(qkt,:)=C;
        end
        Boundary=[1;0];
        Coding='Real';
        Z = min(FunctionValue);
        a = F_intercept(FunctionValue);
        generation = 5;
        result = zeros(generation,2);
        for iter = 1:generation
            iter
%             num = 0;
            for ip = 1:popsize
                %产生子代
                P = 1:popsize;
                kx = randperm(length(P));
                CR=0.8;
                Offspring = F_generator1(Population(ip,:),Population(P(kx(1)),:),Population(P(kx(2)),:),Population(P(kx(3)),:),Population(P(kx(4)),:),Boundary,CR);
%                 Offspring = F_generator(Population(ip,:),Population(P(kx(1)),:),Population(P(kx(2)),:),Boundary,1);
                Offspring1=Offspring(1:end-2)./sum(Offspring(1:end-2));
                k_number = ceil(Offspring(end-1)*10);
                Type = 1;
                sigma = ceil(Offspring(end)*20);
                WW=SimGraph_NearestNeighborsNew(Offspring1(1:end), fea1', k_number, Type, sigma);
                [C,L,U] = SpectralClustering(WW, cluster_num,3);
                label_OffFunValue=C;
                dtype = 1;  
                [DB,CH,Dunn,KL,Han,~] = valid_internal_deviation(data(:,1:end-1),C,dtype);
                cp = valid_compactness(data(:,1:end-1), C);
                OffFunValue=[cp -CH];
                Z = min(Z,OffFunValue);
                if any(sum(FunctionValue<=repmat(OffFunValue,popsize,1),2)==ObjectiveNum)
                    continue;
                end
                for j = randperm(popsize)
                    for i = 1 : ObjectiveNum
                        if a(i) == Z(i)
                            a(i) = Z(i) + 0.001;
                        end
                    end
                    ScaledFun = (FunctionValue(j,:)-Z)./(a-Z);
                    ScaledOffFun = (OffFunValue-Z)./(a-Z);
                    d1_old = sum(W(j,:).*ScaledFun);
                    d2_old = norm(ScaledFun-d1_old*W(j,:));
                    d1_new = sum(W(j,:).*ScaledOffFun);
                    d2_new = norm(ScaledOffFun-d1_new*W(j,:));
                    if d2_new < d2_old || d2_new == d2_old && d1_new < d1_old
%                         num = num + 1;
                        Population(j,:) = Offspring;
                        FunctionValue(j,:) = OffFunValue;
                        label_FunctionValue(j,:)=label_OffFunValue;
                        a = F_intercept(FunctionValue);
						break;
                    end
                end
%                 更新最优理想点
                Z = min(Z,OffFunValue);
            end
            % label_FunctionValue
            rr = zeros(popsize,2);
            for ij = 1:1:popsize
                [~, ~, Rn, NMI] = exMeasure(label_FunctionValue(ij,:)',data(:,end));
                rr(ij,:)=[NMI Rn];
            end
            [~,Ind]=max(rr(:,1));
            result(indexnumber,:) = rr(Ind,:);
            indexnumber = indexnumber + 1;
%             fprintf(fp, 'iter=%d num=%d NMI = %f Rn = %f\n',iter,num,rr(Ind,1),rr(Ind,2));
        end
        [rrmin,Indd]=max(result(:,1));
        result(Indd,:)
        s1 = s1+result(Indd,1);
        s2 = s2+result(Indd,2);
    end
    finalresult(dd,1)=s1/times;
    finalresult(dd,2)=s2/times;
    finalresult
    fprintf(fp, '%5.4f %5.4f\n',finalresult(dd,:));
    clearvars -except finalresult DD path
end
matlabpool close
% dlmwrite(['result.csv' ],finalresult, 'precision', 4, 'newline', 'pc');