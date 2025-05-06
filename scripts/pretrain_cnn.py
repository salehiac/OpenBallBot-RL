import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import pdb
import pickle
import argparse


class DepthImageDataset(Dataset):
    def __init__(self, image_data_dict):
        """
        image_data_dict expects the strucutre documented in collect_depth_image_paths and load_depth_images
        """
        self.samples = []
        for log_id, episodes in image_data_dict.items():
            for ep_id, images in episodes.items():
                for img in images:
                    self.samples.append((log_id, ep_id, img))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _ , _, img = self.samples[idx]
        img = torch.from_numpy(img).float().reshape(1,img.shape[0],img.shape[1])
        return img/255.0 #we normalize here to avoid a large pickle file 

    def plot_im(self,idx):
        _,_,im=self.samples[idx]
        plt.imshow(im)
        plt.show()

def collect_depth_image_paths(root_dir):
    """
    Collects paths to all depth images under the given root directory.

    Returns a dict:
    {
        'log_<id>': {
            episode_idx: [list of depth image paths]
        },
        ...
    }
    """
    root = Path(root_dir)
    data = {}

    for log_dir in root.glob('log_*'):
        log_id = log_dir.name
        data[log_id] = {}

        for ep_dir in log_dir.glob('rgbd_log_episode_*'):
            try:
                episode_idx = int(ep_dir.name.split('_')[-1])
            except ValueError:
                continue  # Skip malformed dirs

            depth_dir = ep_dir / 'depth'
            if not depth_dir.is_dir():
                continue

            image_paths = sorted(depth_dir.glob('*.*'))  # Adjust extension filter if needed
            data[log_id][episode_idx] = list(image_paths)

    return data

def load_depth_images(data_dict):
    """
    Converts image paths to actual image arrays (numpy).

    Returns a structure with same shape as `data_dict`, but image paths replaced by arrays.
    """
    return {
        log_id: {
            ep_idx: [np.array(Image.open(p)) for p in image_paths]
            for ep_idx, image_paths in episodes.items()
        }
        for log_id, episodes in data_dict.items()
    }


class TinyAutoencoder(nn.Module):
    def __init__(self,H,W,in_c=1,out_sz=16):
        super().__init__()

        F1=16
        F2=16
        self.encoder = nn.Sequential(
                torch.nn.Conv2d(1, F1, kernel_size=3, stride=2, padding=1), #output BxF1xH/2xW/2
                torch.nn.BatchNorm2d(F1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(F1, F2, kernel_size=3, stride=2, padding=1), #output BxF2xH/4xW/4
                torch.nn.BatchNorm2d(F2),
                torch.nn.LeakyReLU(),
                torch.nn.Flatten(),                                       
                torch.nn.Linear(F2*H//4*W//4, out_sz),                                  
                torch.nn.BatchNorm1d(out_sz),
                torch.nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
                nn.Linear(out_sz, F2 * H//4 * W//4),  # out: B x (F2*16*16)
                nn.BatchNorm1d(F2 * H//4 * W//4),
                nn.LeakyReLU(),
                nn.Unflatten(1, (F2, H//4, W//4)),     # out: B x F2 x 16 x 16

                nn.ConvTranspose2d(F2, F1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32
                nn.BatchNorm2d(F1),
                nn.LeakyReLU(),

                nn.ConvTranspose2d(F1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # 64x64
                nn.Sigmoid(),  # assuming input is normalized to [0,1]
                )

    def forward(self, x):
        z = self.encoder(x)
        out=self.decoder(z)
        return out


def train_autoencoder(model, train_loader, val_loader, device, epochs, lr, save_path):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val=float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x in train_loader:
            x = x.to(device)
            out = model(x)
            loss = torch.nn.functional.mse_loss(out, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            counter=0
            for x in val_loader:
                x = x.to(device)
                out = model(x)
                loss = torch.nn.functional.mse_loss(out, x)
                val_loss += loss.item() * x.size(0)

                if epoch%5==0 and counter%100==0:
                    out_1=out[0,0].detach().cpu().numpy()
                    x_1=x[0,0].detach().cpu().numpy()
                    plt.imshow(np.concatenate([out_1,x_1],1))
                    plt.show()
                counter+=1
                #print(counter)


        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.8f}, val_loss={val_loss:.8f}")
        if val_loss<best_val:
            p_sum=sum([param.abs().sum().item() for param in model.encoder.parameters() if param.requires_grad])
            print(f"improved val loss, saving **ENCODER** with p_sum={p_sum}")
            model.encoder.p_sum=p_sum
            best_val=val_loss
            torch.save(model.encoder,f"{save_path}/encoder_epoch_{epoch}")


def main(args):

    if args.data_path[-4:]==".pkl":
        print("loading from pickled dataset...")
        with open(args.data_path,"rb") as fl:
            dataset=pickle.load(fl)
    else:
        print("reading from raw dataset, this can take a while...")

    
        root_directory = args.data_path
        image_paths = collect_depth_image_paths(root_directory)
        image_data = load_depth_images(image_paths)

        dataset = DepthImageDataset(image_data)
    
    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)#I'm also shuffling here for display reasons

    #dataset.plot_im(np.random.randint(len(train_loader)))

    if args.save_dataset_as_pickle:
        with open(f"{args.save_dataset_as_pickle}/dataset.pkl","wb") as fl:
            pickle.dump(dataset,fl)
    
    #print(len(dataset))
    #pdb.set_trace()

    model=TinyAutoencoder(H=64,W=64)
    train_autoencoder(model,
            train_loader,
            val_loader,
            device=torch.device("cuda"),
            epochs=100,
            lr=1e-3,
            save_path=args.save_encoder_to)


if __name__=="__main__":
    _parser = argparse.ArgumentParser(description="Pretrain small encoder")
    _parser.add_argument("--data_path", type=str,required=True, help="Either a pkl file, or a directory that has the structure from the gather_data.py logging script")
    _parser.add_argument("--save_dataset_as_pickle", type=str,default="",help="an optional path where the dataset will be saved as a pkl")
    _parser.add_argument("--save_encoder_to",type=str, required=True, help="")

    _args = _parser.parse_args()
    main(_args)


