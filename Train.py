import argparse, pdb
import datasets, torch_model
from utils import Transforms
from torch.utils.data import random_split, DataLoader
import torch
from torch_model import Driver, train



def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='./data/drive_data.csv')
    parser.add_argument('-t', help='train size fraction',    dest='train_size',         type=float, default=0.8)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=64)
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()
    args = vars(args)

    #print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in args.items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    if torch.cuda.is_available():
        print('Using GPU !!!')
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
    else:
        print('Using CPU !!!')
        device = torch.device("cpu")

    # Create Dataset
    drivingData = datasets.Dataset(args['data_dir'], Transforms())

    # Split Dataset
    train_size = int(len(drivingData) * args['train_size'])
    training_set, val_set = random_split(drivingData, [train_size, len(drivingData) - train_size])

    # Create Dataloaders
    loader_parameters = {'batch_size': args['batch_size'],
                         'shuffle': True,
                         'num_workers': 2}

    trainLoader = DataLoader(training_set, **loader_parameters)
    valLoader = DataLoader(val_set, **loader_parameters)

    # Initialize Model
    model = Driver(batch_size=loader_parameters['batch_size'])

    # Start Training
    train(model, device, lr=args['learning_rate'], epochs=args['nb_epoch'], batch_size=loader_parameters['batch_size'],
          trainingLoader=trainLoader, validationLoader=valLoader)

if __name__ == "__main__":
    main()