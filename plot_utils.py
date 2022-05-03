import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np 
import os
import torch
import cv2
from unet import unet
from PIL import Image
import torchvision.transforms as T
from data_loader import TowelDataset_1
from torch.utils.data import DataLoader
from utils import weights_init, compute_map, compute_iou, compute_auc
from torch.autograd import Variable
import pdb

def plot_test_data(img_num, root, savefig=None):
    rgb_img = np.load(root+"/rgb/"+str(img_num)+".npy")/255.0
    print(rgb_img)
    imgs = {}
    imgs["orig"] = rgb_img[... ,::-1] #Converts BGR to RGB
    mask_dir = root+"/masks"
    maskdirs = os.listdir(mask_dir)
    for i in range(len(maskdirs)):
        img =  convert_mask_to_rgb(np.load(root+"/masks/"+maskdirs[i]+"/"+str(img_num)+".npy"))
        imgs[str(maskdirs[i])] = img
    rows = 1
    cols = len(imgs.keys())
    print(list(imgs.keys()))
    fig = plt.figure()
    ax = []
    for i in range(cols*rows):
        name = list(imgs.keys())[i]
        ax.append( fig.add_subplot(rows, cols, i+1) )
        ax[-1].set_title(name)  # set title
        plt.imshow(imgs[name])
    if(savefig == None):
        plt.show()
    else:
        plt.savefig(savefig, bbox_inches='tight')
    return None
def get_network_output(model, depth_img, normalized=True, thresh=0.5):
    transform = T.Compose([T.ToTensor()])
    depth_img = transform(Image.fromarray(depth_img, mode='F'))
    if not normalized:
        depth_img = normalize(depth_img)
    inp = torch.stack([depth_img])
    model.eval()
    # out = (torch.sigmoid(model(inp)[0])>thresh).float()
    out = torch.sigmoid(model(inp)).detach().float()
    out = out.squeeze(0)
    print('outshape',out.shape)
    print((out>0.5).int())
    return (out>0.5).int()
    
def plot_model(model, img_num, root):
    rgb_img = np.load(root+"/rgb/"+str(img_num)+".npy")/255.0
    imgs = {}
    imgs["orig"] = rgb_img[... ,::-1] #Converts BGR to RGB
    from IPython import embed; embed()
    depth_img = color_depth(np.load(root+"/depth/"+str(img_num)+".npy"))
    imgs["depth"] = depth_img
    mask_dir = root+"/masks"
    maskdirs = os.listdir(mask_dir)
    for i in range(len(maskdirs)):
        img =  convert_mask_to_rgb(np.load(root+"/masks/"+maskdirs[i]+"/"+str(img_num)+".npy"))
        imgs["orig_"+str(maskdirs[i])] = img
    model.eval()

    for i in range(len(maskdirs)):
        img =  convert_mask_to_rgb(np.load(root+"/masks/"+maskdirs[i]+"/"+str(img_num)+".npy"))
        imgs["orig_"+str(maskdirs[i])] = img
    rows = 2
    cols = len(imgs.keys())
    fig = plt.figure()
    plt.imshow(depth_img)
    plt.show()
    # from IPython import embed; embed()
    pass

def convert_mask_to_rgb(img):
    res_img = img 
    res_img = np.stack([res_img, res_img, res_img],axis=-1)    
    return res_img

def color_depth(depth_map):
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.03), cv2.COLORMAP_JET)[... ,::-1] #Converts BGR to RGB
    return depth_colormap

def normalize(img_depth):
    img_depth = img_depth
    min_I = img_depth.min()
    max_I = img_depth.max()
    img_depth[img_depth<=min_I] = min_I
    img_depth = (img_depth - min_I) / (max_I - min_I)
    print(img_depth)
    return img_depth


if(__name__ == "__main__"):
    root = "/home/sashank/deepl_project/data/dataset/test"
    datasize = "" 
    val_data = TowelDataset_1(root_dir="/home/sashank/deepl_project/data/dataset/test/", phase='train',num_masks = 2, use_transform=False, datasize=datasize)
    val_loader = DataLoader(val_data, batch_size=1, num_workers=4)
    #img_num = 250
    # val = plot_test_data(img_num, root, "test.png")
    # plot_model(1, img_num, root)
    # print(val)
    model = unet(n_classes=2, in_channels=1)
    model.load_state_dict(torch.load("models/11_epoch285"))
    use_gpu = 1
    #depth_img = np.load(root+"/depth/"+str(img_num)+".npy").astype(float)
    #out = get_network_output(model, depth_img, normalized=False, thresh=0.5)
    count = 0
    for iter, batch in enumerate(val_loader):
        inputs = batch['X']
        labels = batch['Y']
        outs = model(inputs)
        outs = torch.sigmoid(outs)
        outs = outs.squeeze(0)
        out0 = outs[0].detach().numpy()
        out1 = outs[1].detach().numpy()
        labels = labels.squeeze(0)
        label0  = labels[0].detach().numpy()
        label1 = labels[1].detach().numpy()
       # pdb.set_trace()
        fig = plt.figure()
        plt.imshow(out0)
        plt.savefig('output0.png')

        plt.imshow(out1)
        plt.savefig('output1.png')

        # m1 = np.load(root+"/masks/1/"+str(img_num)+".npy").astype(float)
        # m2 = np.load(root+"/masks/2/"+str(img_num)+".npy").astype(float)

        #pdb.set_trace()
        plt.imshow(label0)
        plt.savefig('m1.png')

        plt.imshow(label1)
        plt.savefig('m2.png')
        rgb = batch['rgb']
        rgb = rgb.squeeze(0)
        
        rgb = rgb.permute(1,2,0)


        fig = plt.figure()

        ax1 = fig.add_subplot(2,3,1)
        ax1.set_title('Input')
        ax1.imshow(rgb.detach().numpy())


        ax2 = fig.add_subplot(2,3,2)
        ax2.set_title('Mask 1')
        ax2.imshow(label0)

        ax3 = fig.add_subplot(2,3,3)
        ax3.set_title('Mask 2')
        ax3.imshow(label1)

        ax4 = fig.add_subplot(2,3,4)
        ax4.set_title('Depth(Model Input)')
        inputs = inputs.squeeze(0)
        inputs = inputs.permute(1,2,0)
        ax4.imshow(inputs.detach().numpy())

        ax5 = fig.add_subplot(2,3,5)
        ax5.set_title('Output Mask 1')
        ax5.imshow(out0)

        ax6 = fig.add_subplot(2,3,6)
        ax6.set_title('Output Mask 2')
        ax6.imshow(out1)

        fig.suptitle('Model Predictions')
        plt.show()

        plt.savefig('model_preds_train.png')

        count = count + 1
        if count > 1:

            break


    #plt.show()
