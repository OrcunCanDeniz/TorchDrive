import argparse
import datasets
from utils import Transforms
from torch.utils.data import random_split, DataLoader
import torch
from torch_model import Driver, train



def main():
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='./data/drive_data.csv')
    parser.add_argument('-m', help='path to model', dest='model', type=str, default=None)
    parser.add_argument('-t', help='train size fraction',    dest='train_size',         type=float, default=0.8)
    parser.add_argument('-e', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
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

    # Set device
    if torch.cuda.is_available():
        print('Using GPU !!!')
        device = torch.device("cuda:0") # choose GPU number 0, as computation device
        torch.backends.cudnn.benchmark = True #CUDNN autotuner to find best algorithm for present hardware
    else:
        print('Using CPU !!!')
        device = torch.device("cpu")# choose CPU as computation device

    # Create Dataset
    drivingData = datasets.Dataset(args['data_dir'], Transforms()) # Create Dataset

    # Split Dataset
    train_size = int(len(drivingData) * args['train_size'])# Calculate training set size
    training_set, val_set = random_split(drivingData, [train_size, len(drivingData) - train_size])#Split Train/Test sets

    #Wrap datasets with Dataloaders
    trainLoader = DataLoader(training_set, batch_size=args["batch_size"], num_workers=3, shuffle=True)
    valLoader = DataLoader(val_set, batch_size=args["batch_size"], num_workers=3, shuffle=False)

    # Initialize Model
    if args['model'] is None:
        # Initiate model
        model = Driver(batch_size=args['batch_size'])
    else:
        #load model
        print('Loading model ...')
        model = torch.load(args['model'])


    # Start Training
    train(model, device, lr=args['learning_rate'], epochs=args['nb_epoch'],
          trainingLoader=trainLoader, validationLoader=valLoader)

if __name__ == "__main__":
    main()