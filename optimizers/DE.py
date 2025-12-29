import numpy as np
import time
import inspect
import utils
from wamit_utils import normalize_layout_vector, distance_check

def DE(func, bounds, particle_size:int, loc_name:str, h:float, Hs:float, Tp:float, step_size:float, max_iter=100,
       F=0.7, CR=0.5, results_path='Calculation_results.res', iter_path='Iteration_results.res'):
    
    func_name = inspect.stack()[0].function
    memory_list = []
    bounds_keys = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'd', 'D']
    lower = np.array([bounds[key][0] for key in bounds_keys])
    upper = np.array([bounds[key][1] for key in bounds_keys])
    dimensions = len(bounds_keys)

    pops = []
    while len(pops) < particle_size:
        candidates = utils.generate_on_grid(particle_size, lower, upper, step_size)
        for cand in candidates:
            cand = np.round(cand, 1)
            cand = np.clip(cand, lower, upper)
            if distance_check(cand.tolist(), cand[6]):
                pops.append(cand)
                if len(pops) == particle_size:
                    break
    pops = np.array(pops)

    fitness_list = []
    for idx, vector in enumerate(pops):
        utils.print_start_message(idx, vector)
        start_eval_time = time.time()
        fitness, each_wec_power = func(vector, Hs, Tp, h)
        elapsed = time.time() - start_eval_time
        utils.print_eval_message(elapsed)
        fitness_list.append(fitness)
        memory_list.append(list(vector) + [fitness])
        utils.write_results(list(vector) + [fitness] + list(np.round(each_wec_power, 2)), results_path)

    best_idx = np.argmax(fitness_list)
    gbest = pops[best_idx].copy()
    gbest_fitness = fitness_list[best_idx]

    utils.print_iter_start_message(func_name, loc_name, Hs, Tp, h)

    for iteration in range(1, max_iter + 1):
        print(f'\n Iteration {iteration} Calculate...')
        cnt_memory, iter_start = 0, time.time()
        for i in range(particle_size):
            print("-"*50)
            while True:
                idxs = [idx for idx in range(particle_size) if idx != i]
                a, b, c = pops[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lower, upper)
                mutant = np.round(mutant, 1)
                cross_points = np.random.rand(dimensions) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimensions)] = True
                trial = np.where(cross_points, mutant, pops[i])
                trial = utils.apply_step_size(trial, step_size)
                trial = np.round(trial, 1)
                trial = np.clip(trial, lower, upper)
                if distance_check(trial.tolist(), trial[6]):
                    print(f'Trial Position {i + 1}: {trial}')
                    break

            trial = normalize_layout_vector(trial)
            memory = next((item for item in memory_list if np.allclose(np.array(item[:-1]), trial, atol=1e-4)), None)

            if memory is None:
                start_eval_time = time.time()
                trial_fitness, each_wec_power = func(trial, Hs, Tp, h)
                elapsed = time.time() - start_eval_time
                utils.print_eval_message(elapsed)
                memory_list.append(list(trial) + [trial_fitness])
                utils.write_results(list(trial) + [trial_fitness] + list(np.round(each_wec_power, 2)), results_path)
            else:
                trial_fitness = memory[-1]
                cnt_memory += 1
                print(f" Memory Hit! Fitness: {trial_fitness/1000:.2f} kW")
            
            if trial_fitness > fitness_list[i]:
                pops[i], fitness_list[i] = trial, trial_fitness
                if trial_fitness > gbest_fitness:
                    gbest, gbest_fitness = trial, trial_fitness
                    print(f' Global Best Updated -> {gbest_fitness/1000:.2f} kW')
        
        total_time = np.round((time.time() - iter_start) / 60, 2)
        utils.write_results(list(gbest) + [gbest_fitness], iter_path)
        utils.print_summary_message(iteration, gbest, gbest_fitness, total_time, cnt_memory)
    utils.move_results_file(loc_name)