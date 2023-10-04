import random
import operator
import math
import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools, gp
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import MLBenchmarks.regression_datasets_loaders as rdl

# Define protected division function to handle division by zero
def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return left / (right+1e-10)

# Load the Auto MPG dataset (replace 'auto-mpg.csv' with your dataset)
dataset = rdl.load_auto_mpg()
X = dataset['data']
y = dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a DEAP "individual" class for genetic programming
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Define the primitive set
pset = gp.PrimitiveSet("MAIN", arity=7)
pset.addPrimitive(operator.add, arity=2)
pset.addPrimitive(operator.sub, arity=2)
pset.addPrimitive(operator.mul, arity=2)
#pset.addPrimitive(protected_div, arity=2)
#pset.addPrimitive(math.sqrt, arity=1)
pset.addEphemeralConstant("rand_const", lambda: random.uniform(0, 1))
pset.renameArguments(ARG0="cylinders")
pset.renameArguments(ARG1="displacement")
pset.renameArguments(ARG2="horsepower")
pset.renameArguments(ARG3="weight")
pset.renameArguments(ARG4="acceleration")
pset.renameArguments(ARG5="model_year")
pset.renameArguments(ARG6="origin")

# Create the DEAP toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Define GP parameters
population_size = 500
generations = 1000

# Create the population
population = toolbox.population(n=population_size)

# Define the fitness function
def evaluate_individual(individual, X_train, y_train):
    func = toolbox.compile(expr=individual)
    y_pred = [func(*x) for x in X_train]
    mse = mean_squared_error(y_train, y_pred)
    return mse,

# Configure the genetic algorithm
toolbox.register("evaluate", evaluate_individual, X_train=X_train, y_train=y_train)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Run the genetic algorithm
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.3, ngen=generations, verbose=True)

# Get the best individual from the population
best_individual = tools.selBest(population, k=1)[0]

# Compile and evaluate the best individual on the test set
best_func = toolbox.compile(expr=best_individual)
y_pred_test = [best_func(*x) for x in X_test]
mse_test = mean_squared_error(y_test, y_pred_test)
print(f"Mean Squared Error on Test Set: {mse_test:.2f}")
