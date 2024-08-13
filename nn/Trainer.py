from nn.Module import Module
from nn.Loss import Loss
from nn.Optimizer import Optimizer
from nn.Dataloader import Dataloader


class Trainer:
    def __init__(self, model: Module, loss: Loss, optimizer: Optimizer) -> None:
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def train(self, dataloader: Dataloader, epochs: int) -> list:
        loss_history = []
        len_dataloader = len(dataloader)
        for epoch in range(epochs):
            for i, (x, y) in enumerate(dataloader):
                self.model.clear_gradients()
                y_pred = self.model(x)
                # print(f"y_pred shape: {y_pred.shape}, y shape: {y.shape}")

                loss = self.loss.forward(y_pred, y)
                loss_history.append(round(loss,4))
                gradient = self.loss.backward(y_pred, y)
                # print("Gradient shape:", gradient.shape)

                self.model.backward(gradient)
                self.optimizer.step(self.model)
                print(f"Epochs:{epoch+1}/{epochs} batch:{i}/{len_dataloader} loss:{round(loss, 2)}")
        return loss_history
