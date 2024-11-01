import argparse
import torch
import random
import numpy as np
from deap import base, creator, tools
from exp.exp_main import Exp_Main

# Define the GA parameters
POP_SIZE = 20  # Population size
GENS = 10      # Number of generations
CXPB = 0.7     # Crossover probability
MUTPB = 0.2    # Mutation probability

# Define the individual and population
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize MSE
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("d_model", random.choice, [256, 512, 1024])  # Define ranges
toolbox.register("batch_size", random.choice, [16, 32, 64])
toolbox.register("learning_rate", random.uniform, 1e-5, 1e-3)
toolbox.register("freq", random.choice, ['t', 'h'])
toolbox.register("features", random.choice, ['M', 'S', 'MS'])
toolbox.register("n_heads", random.randint, 1, 10)
toolbox.register("e_layers", random.randint, 1, 10)
toolbox.register("d_layers", random.randint, 1, 10)
toolbox.register("moving_avg", random.randint, 10, 40)
toolbox.register("train_epochs", random.randint, 1, 10)
toolbox.register("itr", random.randint, 1, 5)


# Register individual and population creation
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.d_model, toolbox.batch_size,
                  toolbox.learning_rate, toolbox.freq, toolbox.features,
                  toolbox.n_heads, toolbox.e_layers, toolbox.d_layers,
                  toolbox.moving_avg, toolbox.train_epochs, toolbox.itr))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Fitness function
def evaluate(individual):
    args = make_args()
    d_model, batch_size, learning_rate, freq, features, n_heads, e_layers, d_layers, moving_avg, train_epochs, itr = individual
    args.d_model = d_model
    args.batch_size = batch_size
    args.learning_rate = learning_rate
    args.freq = freq
    args.features = features
    args.n_heads = n_heads
    args.e_layers = e_layers
    args.d_layers = d_layers
    args.moving_avg = moving_avg
    args.train_epochs = train_epochs
    args.itr = itr

    if args.freq == 'h':
        args.data = 'ETTh1'
    else:
        args.data = 'ETTm1'

    args.data_path = args.data + '.csv'
    args.model_id = args.data + '_' + str(args.seq_len) + '_' + str(args.pred_len)

    print('Args in experiment:')
    print(args)

    exp = Exp_Main(args)
    # Initialize variables to track average MSE over multiple runs
    total_mse = 0

    # Run training and testing multiple times as per args.itr
    for ii in range(args.itr):
        # Set up the setting string with ii for tracking
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_GAOptimization_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        # Train the model
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        # Test the model and accumulate the MSE
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        mse, mae = exp.test(setting)
        total_mse += mse  # Accumulate the MSE for averaging

    # Calculate the average MSE over all iterations
    avg_mse = total_mse / args.itr
    return avg_mse

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Main GA loop
def main():
    population = toolbox.population(n=POP_SIZE)
    for gen in range(GENS):
        print(f"-- Generation {gen} --")

        # Evaluate individuals
        fitnesses = map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Select and clone the next generation's population
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Update population and evaluate new individuals
        population[:] = offspring

    # Get best individual after all generations
    best_ind = tools.selBest(population, 1)[0]
    print("Best individual is:", best_ind, "with fitness:", best_ind.fitness.values)


def make_args():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='ETTh1_96_24', help='model id')
    parser.add_argument('--model', type=str, required=False, default='Autoformer',
                        help='model name, options: [Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

    # model define
    parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
    parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    return args


if __name__ == "__main__":
    main()
