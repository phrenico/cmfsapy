addpath(genpath('../../../benchmark_data/manifold_data/'));
addpath(genpath('./idEstimation/idEstimation'));
addpath(genpath('./progressbar'));

names = ['1', '2', '3' '4' '5' '6' '7' '9' "101" '102' '103' '104' '11' '12' '13'];
m = length(names);
n = 100;
dims = zeros(m, n, 2);
progressbar
for j=1:m
    myFiles = dir(fullfile('../../../benchmark_data/manifold_data/', strcat('M_', names(j),'_*.mat')));
    for i=1:n
        myFiles(i).name
        load(myFiles(i).name);
        dims(j, i, 1) = DANCoFit(x, 'fractal', true, 'modelfile', './idEstimation/idEstimation/DANCo_fits');
        dims(j, i, 2) = DANCoFit(x, 'fractal', false, 'modelfile', './idEstimation/idEstimation/DANCo_fits');
    end
    progressbar(j/m)
end
save_name = '../../danco_matlab_benchmark_res';
save(save_name, 'dims');
