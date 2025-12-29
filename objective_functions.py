import numpy as np
from wamit_utils import WamitInputGenerator, WamitOutputParser, run_wamit, cal_power

def shape_opt_func(vector:np.ndarray, Hs:float, Tp:float, h:float)->float:
    writer = WamitInputGenerator()
    writer.input_init()
    writer.input_optimize(vector, h)
    run_wamit()
    parser = WamitOutputParser()
    rao = parser.cal_rao()
    pto_matrix = parser.pto_matrix()

    each_power = cal_power(rao, pto_matrix, Hs, Tp)
    total_power = np.sum(each_power)
    
    return total_power, each_power