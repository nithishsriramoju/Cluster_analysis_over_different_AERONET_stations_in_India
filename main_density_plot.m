clc;clear all;
prd = readtable('predictions_kmeans.xlsx');
dt = readtable('Total_Data_with_dates.xlsx');

FMF = dt.FMF;
SSA = dt.ssa1;
lbl = prd.Var1;

fmf1=[];ssa1=[];
fmf2=[];ssa2=[];
fmf3=[];ssa3=[];
for i=1:length(FMF)
    if lbl(i)==0
        fmf1=[fmf1,FMF(i)];
        ssa1=[ssa1,SSA(i)];
    elseif lbl(i)==1
        fmf2=[fmf2,FMF(i)];
        ssa2=[ssa2,SSA(i)];
    else
        fmf3=[fmf3,FMF(i)];
        ssa3=[ssa3,SSA(i)];
    end
end
box on
figure;
dscatter(ssa1',fmf1');
figure;
dscatter(ssa2',fmf2')
figure;
dscatter(ssa3',fmf3')


