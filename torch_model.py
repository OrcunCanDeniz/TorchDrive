import tqdm, pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import vis_train

class Driver(nn.Module):
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)

    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
    """
    def __init__(self, batch_size):
        super(Driver, self).__init__()

        self.first_linear_size = 1

        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(5,5), stride=(2,2))
        self.conv_2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5,5), stride=(2,2))
        self.conv_3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5,5), stride=(2,2))
        self.conv_4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        self.conv_5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        self.drop = nn.Dropout(p=0.5, inplace=False)

        _ = self._convs(torch.randn((batch_size,3,66,200)))

        self.lin1 = nn.Linear(self.flattened_size,100)
        self.lin2 = nn.Linear(100, 50)
        self.lin3 = nn.Linear(50, 10)
        self.lin4 = nn.Linear(10, 1)

    def _convs(self, image):
        image = image / 127.5 - 1

        conv1 = F.elu(self.conv_1(image), alpha=0.3)
        conv2 = F.elu(self.conv_2(conv1), alpha=0.3)
        conv3 = F.elu(self.conv_3(conv2), alpha=0.3)
        conv4 = F.elu(self.conv_4(conv3), alpha=0.3)
        conv5 = F.elu(self.conv_5(conv4), alpha=0.3)
        drop = self.drop(conv5)
        flat = torch.flatten(drop, start_dim=1, end_dim=3)
        self.flattened_size = flat.shape[1]

        return flat

    def forward(self, image):

        flat = self._convs(image)
        lin1 = F.elu(self.lin1(flat), alpha=0.3)
        lin2 = F.elu(self.lin2(lin1), alpha=0.3)
        lin3 = F.elu(self.lin3(lin2), alpha=0.3)
        lin4 = self.lin4(lin3)

        return lin4.squeeze()


def train(net, device, epochs, lr, trainingLoader, validationLoader):
    model = net.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lossFunc = nn.MSELoss(reduction='sum')

    training_losses = []
    training_epoch_losses = torch.tensor([])
    validation_losses = []
    validation_epoch_losses = torch.tensor([])
    min_loss = torch.tensor(10000000)
    for i in tqdm.tqdm(range(epochs)):
        model.train()
        for img_batch, labels in trainingLoader:
            img_batch, labels = img_batch.to(device).float(), labels.to(device).float()
            img_batch = img_batch.permute(0,3,1,2)
            outputs = model(img_batch)
            loss = lossFunc(outputs, labels)
            training_epoch_losses = torch.cat((loss.cpu().unsqueeze(dim=0), training_epoch_losses), 0)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #print(img_batch.shape)

        training_epoch_loss = training_epoch_losses.mean()
        training_losses.append(training_epoch_loss)
        training_epoch_losses = torch.tensor([])

        model.eval();

        with torch.no_grad():
            for val_img_batch, val_labels in validationLoader:
                val_img_batch, val_labels = val_img_batch.to(device).float(), val_labels.to(device).float()
                val_img_batch = val_img_batch.permute(0,3,1,2)
                outputs = model(val_img_batch)
                val_loss = lossFunc(outputs, val_labels)
                validation_epoch_losses = torch.cat((val_loss.cpu().unsqueeze(dim=0), validation_epoch_losses), 0)

        validation_epoch_loss = validation_epoch_losses.mean()
        validation_losses.append(validation_epoch_loss)
        validation_epoch_losses = torch.tensor([])

        if validation_epoch_loss < min_loss:
            min_loss = training_epoch_loss
            model_name = 'driver' + '_best.pt'
            torch.save(model, model_name)

    vis_train(training_losses=training_losses, validationLosses=validation_losses, epochs=epochs)

