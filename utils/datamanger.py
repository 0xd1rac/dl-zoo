import torch 
from torchvision import datasets, transforms

class DataManager:
    @staticmethod 
    def get_mnist_dataloaders(batch_size:int=64, transform=None, download:bool=True, data_dir:str="../data/mnist"):
        
        if not transform:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))  # Scale images to [-1, 1]
            ])
         # Load the training set
        train_dataset = datasets.MNIST(root=data_dir,train=True,download=download,transform=transform)
         # Load the test set
        test_dataset = datasets.MNIST(root=data_dir,train=False,download=download,transform=transform)

        # Train loader
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

        # Test loader
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
        return train_loader, test_loader

    @staticmethod
    def get_cifar10_dataloaders(batch_size:int=64, transform=None, download:bool=True, data_dir:str="../data/cifar10")
        if not transform:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Scale images to [-1, 1]
            ])
        # Load the training set
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=transform)
        # Load the test set
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=transform)
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
