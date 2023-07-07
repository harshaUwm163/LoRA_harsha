import torch
import math

def check_tff_params_validity(M,m,N,eigVals):
    if (M*m/N < 2) or (math.floor(M*m/N) > M-3):
        print(f'M = {M}, m = {m}, N = {N} do not satisfy the conditions')
        exit()
    for l in eigVals:
        if l < 2:
            print(f'one of the eigen values- {l} is < 2, check for others as well')
            exit()

    return True

def construct_tff_from_eigvals(M,m,N,eigVals):
    check_tff_params_validity(M,m,N,eigVals)

    canonical_basis = torch.eye(N)
    k = 1
    gen_basis = []
    for j in range(0,N):
        while eigVals[j] != 0:
            if eigVals[j] < 2 and eigVals[j] != 1:
                gen_basis.append( math.sqrt(eigVals[j]/2)*canonical_basis[j] + math.sqrt(1 - eigVals[j]/2) * canonical_basis[(j+1) % N] )
                gen_basis.append( math.sqrt(eigVals[j]/2)*canonical_basis[j] - math.sqrt(1 - eigVals[j]/2) * canonical_basis[(j+1) % N] )
                k = k+2
                eigVals[j+1] = eigVals[j+1] - (2 - eigVals)
                eigVals[j] = 0
            else:
                gen_basis.append(canonical_basis[j])
                k = k + 1
                eigVals[j] -= 1
        
    gen_basis = torch.stack(gen_basis, dim=0)
    gen_basis = gen_basis.view(-1, M,N)
    gen_basis = gen_basis.permute(1,0,2)
    return gen_basis
    

def construct_sparse_tffs(M,m,N):
    # the checks are done while constructing the Tffs. 
    eigVals = torch.tensor([m*M/N] * M)
    return construct_tff_from_eigvals(M,m,N, eigVals)

if __name__ == "__main__":
    N = 10
    M = 10
    m = 2
    stffs = construct_sparse_tffs(M,m,N)

    a = 1