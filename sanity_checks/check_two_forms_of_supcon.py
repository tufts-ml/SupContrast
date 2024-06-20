import numpy as np
from scipy.special import logsumexp
import copy

def log_sim(z_s_D, z_p_D, tau=0.1):
    ''' Compute log of similarity function

    $$
    \log \exp \left( z_s \cdot z_p / tau \right)
    $$

    Returns
    -------
    s : float
    '''
    return np.inner(z_s_D, z_p_D) / tau

def L_theirs(i, p, pos_ids_P, neg_ids_N, z_BD):
    ''' Implementation of SupCon loss via Eq 2 of Khosla et al

    See https://arxiv.org/pdf/2004.11362.pdf#page=5

    Uses logsumexp trick for numerical stability

    Returns
    -------
    loss : float
        Value computed at specific S,p indices for current batch
    '''
    
    # Verify the provided indices cover the current batch
    assert np.allclose(
        np.sort(np.concatenate(([i, p], pos_ids_P, neg_ids_N))),
        np.arange(z_BD.shape[0]))

    z_i_D = z_BD[i]
    z_p_D = z_BD[p]
    z_pos_PD = z_BD[pos_ids_P]
    z_neg_ND = z_BD[neg_ids_N]

    log_numer = log_sim(z_i_D, z_p_D)
    denom_list = [log_numer]
    for pp in pos_ids_P:
        denom_list.append(
            log_sim(z_i_D, z_BD[pp]))
    for nn in neg_ids_N:
        denom_list.append(
            log_sim(z_i_D, z_BD[nn]))
    log_denom = logsumexp(denom_list)
    P = 1 + len(pos_ids_P) # divide by number of partners for i
    return 1/float(P) * (log_numer - log_denom)

def L_ours(S, p, pos_ids_P, neg_ids_N, z_BD):
    ''' Implementation of SupCon loss via Eq 3 of SINCERE paper

    Returns
    -------
    loss : float
        Value computed at specific S,p indices for current batch
    '''

    # Verify the provided indices cover the current batch
    assert np.allclose(
        np.sort(np.concatenate(([S, p], pos_ids_P, neg_ids_N))),
        np.arange(z_BD.shape[0]))

    z_s_D = z_BD[S]
    z_p_D = z_BD[p]

    log_numer = log_sim(z_s_D, z_p_D)
    denom_list = [log_numer]
    for pp in pos_ids_P:
        denom_list.append(
            log_sim(z_p_D, z_BD[pp]))
    for nn in neg_ids_N:
        denom_list.append(
            log_sim(z_p_D, z_BD[nn]))
    log_denom = logsumexp(denom_list)

    P = 1 + len(pos_ids_P) # divide by number of partners for S
    return 1/float(P) * (log_numer - log_denom)

if __name__ == '__main__':
    B = 6
    D = 3
    prng = np.random.RandomState(1)

    z_BD = 0.3 * prng.randn(B, D)
    z_BD[:B//2] -= 0.2
    z_BD[B//2:] += 0.2

    y_B = np.zeros(B, dtype=np.int32)
    y_B[B//2:] += 1

    total_ours = 0.0
    total_theirs = 0.0
    for S in range(B):
        neg_ids_N = np.flatnonzero(y_B != y_B[S])

        partner_ids = np.flatnonzero(y_B == y_B[S]).tolist()
        partner_ids.remove(S)
        
        for p in partner_ids:
            other_pos_ids_P = copy.deepcopy(partner_ids)
            other_pos_ids_P.remove(p)
            loss_ours = L_ours(S, p, other_pos_ids_P, neg_ids_N, z_BD)
            loss_theirs = L_theirs(p, S, other_pos_ids_P, neg_ids_N, z_BD)

            total_ours += loss_ours
            total_theirs += loss_theirs

            print("S=%2d p=%2d  | L^ours(S,p) = % .4f  | L^theirs(p, S) =% .4f" % (
                S, p, loss_ours, loss_theirs))

    print("Total sum over all (S,p) pairs:")
    print(" % .4f ours " % total_ours)
    print(" % .4f theirs" % total_theirs)



