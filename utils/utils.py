import numpy as np
from numpy import interp
from numpy.linalg import norm


def get_error_state(norm_obs_n, x_lim, v_lim, eIx_lim, eIb1_lim, args):
    # norm_obs_1 = (ex_norm, eIx_norm, ev_norm, b3, ew12_norm)
    # norm_obs_2 = (b1, eb1_norm, eIb1_norm, eW3_norm)
    if args.framework in ("CTDE","DTDE"):
        norm_obs_1, norm_obs_2 = norm_obs_n[0], norm_obs_n[1]
        ex = norm_obs_1[0:3] * x_lim
        eIx = norm_obs_1[3:6] * eIx_lim
        ev = norm_obs_1[6:9] * v_lim
        eb1 = norm_obs_2[3] * np.pi
        eIb1 = norm_obs_2[4] * eIb1_lim
    # norm_obs = (ex_norm, eIx_norm, ev_norm, R_vec, eb1_norm, eIb1_norm, eW_norm)
    elif args.framework == "SARL":
        ex = norm_obs_n[0][0:3] * x_lim
        eIx = norm_obs_n[0][3:6] * eIx_lim
        ev = norm_obs_n[0][6:9] * v_lim 
        eb1 = norm_obs_n[0][18] * np.pi
        eIb1 = norm_obs_n[0][19] * eIb1_lim

    return ex, eIx, ev, eb1, eIb1


def benchmark_reward_func(ex, ev, eb1, args):    
    reward_eX = -args.Cx * (norm(ex))
    reward_eV = -args.Cv * (norm(ev))
    reward_eb1 = -args.Cb1 * abs(eb1)
    rwd = reward_eX + reward_eV + reward_eb1

    rwd_min = -np.ceil(args.Cx+args.Cv+args.Cb1)
    return interp(rwd, [rwd_min, 0.], [0., 1.]) # linear interpolation [0,1]