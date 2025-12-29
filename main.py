import config
from optimizers import DE, PSO, DEPSO
from objective_functions import shape_opt_func

if __name__ == "__main__":
    # PSO(shape_opt_func, config.BOUNDS, config.PARTICLE_SIZE, config.CURRENT_LOC, config.h, config.Hs, config.Tp, config.STEP_SIZE, config.MAX_ITER,
    #    w=config.PSO_PARAMS['w'], c1=config.PSO_PARAMS['c1'], c2=config.PSO_PARAMS['c2'], results_path=config.RESULTS_PATH, iter_path=config.ITER_PATH)
    DE(shape_opt_func, config.BOUNDS, config.PARTICLE_SIZE, config.CURRENT_LOC, config.h, config.Hs, config.Tp, config.STEP_SIZE, config.MAX_ITER,
       F=config.DE_PARAMS['F'], CR=config.DE_PARAMS['CR'], results_path=config.RESULTS_PATH, iter_path=config.ITER_PATH)