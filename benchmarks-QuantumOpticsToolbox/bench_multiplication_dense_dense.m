
function result = bench_multiplication_dense_dense()
    name = 'multiplication_dense_dense';
    cutoffs = [50:50:1001];
    result = [];
    for N = cutoffs
        op1 = (1.+0.3j)*rand(N, N);
        op2 = (1.+0.3j)*rand(N, N);
        f_ = @() f(op1, op2);
        result = [result, timeit(f_)];
    end
    savebenchmark(name, cutoffs, result)
end

function result = f(op1, op2)
    result = op1*op2;
end
