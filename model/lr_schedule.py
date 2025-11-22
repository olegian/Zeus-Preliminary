import math

def cosine_lr_schedule(
    t: int, # current iteration number
    alpha_max: float, # max lr in schedule
    alpha_min: float, # min lr in schedule
    T_w: int, # num warmup iterations
    T_c: int, # num annealed iterations
):
    assert T_w <= T_c, "number of warms shuold always be less than num annealed ops"

    if t < T_w:
        return (t / T_w) * alpha_max
    elif t < T_c:
        # spec page 23 for this formula
        return alpha_min + ((1 + math.cos(((t - T_w) / (T_c - T_w)) * math.pi)) * (alpha_max - alpha_min) / 2)
    else:
        return alpha_min
