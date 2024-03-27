import torch
from data_preprocesssing import *
from model import *
from train import *
from eval import *




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    data_dir = '../data'
    batch_size = 32
    image_size = 224
    num_epochs = 3
    data_processor = DataPreprocessor(data_dir, batch_size, image_size)
    data_loaders = data_processor.create_data_loaders()

    print("----------Training Start----------------")
    model = train_model(data_loaders, num_epochs, device)
    print("----------Training Completed----------------")
    evaluate_model(model, data_loaders['test'], device)
    torch.save(model.state_dict(), '../pretrained_models/model.pth')
if __name__ == "__main__":
    main()
