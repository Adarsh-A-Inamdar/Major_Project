# src/ga_tune.py
from deap import base, creator, tools, algorithms
import random, numpy as np

# search spaces
LR_SPACE = (1e-5, 1e-2)
WD_SPACE = (0.0, 1e-2)
BATCH_OPTS = [16,32,64]
HEAD_OPTS  = [128,256,512]

def quick_train_and_eval(lr, wd, bs, head):
    # Your training and evaluation logic here
    # Return a float between 0 and 1 representing validation accuracy
    val_acc = 0.5  # placeholder
    return val_acc

def eval_cfg(ind):
    lr   = 10**np.interp(ind[0], [0,1], [np.log10(LR_SPACE[0]), np.log10(LR_SPACE[1])])
    wd   = np.interp(ind[1], [0,1], [WD_SPACE[0], WD_SPACE[1]])
    bs   = BATCH_OPTS[int(ind[2]*len(BATCH_OPTS))%len(BATCH_OPTS)]
    head = HEAD_OPTS[int(ind[3]*len(HEAD_OPTS))%len(HEAD_OPTS)]
    # train tiny for 1-2 epochs on a subset → return val accuracy
    val_acc = quick_train_and_eval(lr, wd, bs, head)  # you implement, returns 0..1
    return (val_acc,)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox=base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 4)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_cfg)
toolbox.register("mate", tools.cxBlend, alpha=0.3)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=12)
algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=8, verbose=True)
best = tools.selBest(pop, k=1)[0]
print("Best individual:", best.fitness.values)
