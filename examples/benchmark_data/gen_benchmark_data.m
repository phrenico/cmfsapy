n_realizations = 100
N = 2500
dataset = [1, 2, 3, 4, 5, 6, 7,  9, 10, 10, 10, 10, 11, 12, 13] ;
D = [11, 5, 6, 8, 3, 36, 3, 20, 11, 18, 25, 71, 3, 20, 13];
names = [1, 2, 3, 4, 5, 6, 7,  9, 101, 102, 103, 104, 11, 12, 13]

a = names(1, 2)
my_path = './manifold_data/'

for i=1:15
    for j=1:n_realizations
        x = GenerateManifoldData(dataset(i), D(i), N);
        filename = sprintf('%s%s_%d_%d.mat', my_path,'M', names(i), j);
        save("-v7", filename, "x");
    endfor
endfor