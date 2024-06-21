#!/usr/bin/env python
# coding: utf-8

# # Osteisarcoma Label Cleaning Multiple Instance Learning

# #### 0. Import Packages

# In[1]:


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from scipy.stats import binom
import cv2 as cv
import numpy as np
from skimage import morphology
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import random
import openslide
import numpy as np
import random
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import cv2 as cv
from skimage import morphology
import matplotlib.pyplot as plt
import wandb
import os
from os.path import basename
import json 
import pandas as pd
from __future__ import print_function, division
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import pdb
from torch.utils.data import Dataset, DataLoader
from glob import glob
from skimage import io, transform
from PIL import Image


# #### 1. Basic Slide information 

# In[2]:


# Path to your WSI file
wsi_path_1 = '/cis/home/jhu104/jhu101/OTS-23-17323 - 2023-06-15 18.31.27.ndpi'
wsi_label_path_1 = "/export/io86/data/jhu101/OTS-23-17323 - 2023-06-15 18.31.27.ndpi.xml"

#Open the WSI file
slide = openslide.OpenSlide(wsi_path_1)
# print(f"Slide Level Count:{slide.level_count}")
# print(f"Slide Dimensions:{slide.dimensions}")
# print(f"Slide level dimensions:{slide.level_dimensions}")


# # 1.Generate Tissue Mask

# In[3]:


def binary_Aaron(img,adjust_otsu, fill_size=50, remove_size=50):
    # 16714505 1.21
    # 16714503 1.25
    #Changing the BGR Channel to Gray
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #Setting a threhold where >threhold =255; <threhold =0. This is for denoising
    otsu_threshold, _ = cv.threshold(gray, 0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    #Apply to our downsampling WSI
    binary = gray<= (otsu_threshold*adjust_otsu)
    #Don't quite understand this part? Why should we do this.
    binary = morphology.remove_small_objects(morphology.remove_small_holes(binary, fill_size),remove_size)
    return binary

    


# In[4]:


def generate_tissue_mask(slide_ob,unit,adjust_otsu):
    """
    Generate tissue mask (downsampled) for WSI
    - Input
        slide_ob: slide object
        unit: downsample scale
        adjust_otsu: adjust the OTSU threshold
    - Return
        mask tissue
    """
    #Get the width and height of WSI
    width,height = slide_ob.dimensions
    #Get the downsampling width and height of WSI
    width_downsample, height_downsample = width//unit, height//unit
    #Get the thumnail with respect to the down-sampling WSI 
    thumbnail = slide_ob.get_thumbnail((width_downsample,width_downsample))
    #Using opencv to resize the thumnial. Aim to convert the image to numpy array
    thumbnail = cv.resize(np.array(thumbnail)[:,:,:3],(width_downsample,height_downsample))
    #Use binary to denoise our image and convert it to numpy array
    mask_tissue = np.array(binary_Aaron(thumbnail, adjust_otsu),dtype=np.uint8) 
    return mask_tissue


# In[5]:


#Generate label mask
def generate_label_mask(slide_ob, annotations, annotation_label_mapping, unit):
    """
    Generate label mask (downsampled) for WSI
    - Input
        slide_ob: slide object
        annotations: {'annotation_key':{'outer': [(x,y),....],'inner':[(x,y),...]}}
        'annotation_label_mapping': {'annotation_key': int}
        unit: downsample scale
    - Return
        Mask: label mask
    """
    width, height = slide_ob.dimensions
    Mask = np.zeros((int(height/unit),int(width/unit)),dtype=float)
    for annotation_key in annotations.keys():
        mask =  Image.new('1', (int(np.round(width/unit)),int(np.round(height/unit))))
        draw = ImageDraw.Draw(mask)
        for contour in annotations[annotation_key]['outer']:
            contour = [(i[0]/unit,i[1]/unit) for i in contour]
            draw.polygon(contour,fill=1,outline=0)
        for contour in annotations[annotation_key]['inner']:
            contour = [(i[0]/unit,i[1]/unit) for i in contour]
            draw.polygon(contour,fill=0,outline=0)
        mask = np.array(mask)
        Mask[mask==1] = annotation_label_mapping[annotation_key]
    return Mask


# In[6]:


adjust_otsu =1
mask_tissue = generate_tissue_mask(slide,256,adjust_otsu)
plt.imshow(mask_tissue)
plt.show()


# # 2. Generate Mask Label

# In[7]:


def generate_label_mask(slide_ob, annotations, annotation_label_mapping, unit):
    """
    Generate label mask (downsampled) for WSI
    - Input
        slide_ob: slide object
        annotations: {'annotation_key':{'outer': [(x,y),....],'inner':[(x,y),...]}}
        'annotation_label_mapping': {'annotation_key': int}
        unit: downsample scale
    - Return
        Mask: label mask
    """
    #Get the width and height of our WSI
    width, height = slide_ob.dimensions
    #Build an empty label mask matrix for storing the labels later
    Mask = np.zeros((int(height/unit),int(width/unit)),dtype=float)
    #Find the labels for each classes: Viable, necrosis and stroma
    for annotation_key in annotations.keys():
      #Create a new image with mode 1(1-bit pixels,black and white)
        mask =  Image.new('1', (int(np.round(width/unit)),int(np.round(height/unit))))
        #Draw the labels for each class
        draw = ImageDraw.Draw(mask)
        #Get the coordinates information for each location inside one class
        for contour in annotations[annotation_key]['outer']:
          #Consider the downsampling factor
            contour = [(i[0]/unit,i[1]/unit) for i in contour]
            #Use polygon to connect the vetex and draw the regions wrt classes
            draw.polygon(contour,fill=1,outline=0)
        for contour in annotations[annotation_key]['inner']:
            contour = [(i[0]/unit,i[1]/unit) for i in contour]
            draw.polygon(contour,fill=0,outline=0)
        #Convert our created image into arrays(each mask representing the labels for each class)
        mask = np.array(mask)
        #Mapping all our class labels to one big matrix, which means the matrix will contain all the labels(In our case,1 for viable,2 for necrosis, 3 for stroma )
        Mask[mask==1] = annotation_label_mapping[annotation_key]
    return Mask


# In[8]:


def read_Aaron_annotations(xml_path):
    root = ET.parse(xml_path)
    Annotations={'viable':{'outer':[], 'inner':[]},
                'necrosis':{'outer':[], 'inner':[]},
                'stroma':{'outer':[], 'inner':[]},
                'type4':{'outer':[], 'inner':[]}}
    for a in root.iter('Annotation'):
        for r in a.iter('Region'):
            Annotation = []
            for v in r.iter('Vertex'):
                Annotation.append((float(v.attrib['X']), float(v.attrib['Y'])))
            if a.attrib['LineColor'] == '16711680' :
                Annotations['viable']['outer'].append(Annotation)
            elif a.attrib['LineColor'] == '255':
                Annotations['necrosis']['outer'].append(Annotation)
            elif a.attrib['LineColor'] == '65280' or a.attrib['LineColor'] == '1376057':
                Annotations['stroma']['outer'].append(Annotation)
            elif a.attrib['LineColor'] == '65535':
                Annotations['type4']['outer'].append(Annotation)
    return Annotations


# In[9]:


#set Annotation mapping as following:
annotation_label_mapping ={
    'stroma':1,
    'viable':2,
    'necrosis':3,
    'type4':4
}
#Read the annotations
Annotations=read_Aaron_annotations(wsi_label_path_1)
# Annotations2=read_Aaron_annotations(wsi_label_path_2)
#Get the label Mask
Mask = generate_label_mask(slide,Annotations,annotation_label_mapping,256)
# Mask2 = generate_label_mask(slide2,Annotations2,annotation_label_mapping,256)
# Mask_dic = {"s1":Mask,"s2":Mask2}
np.sum((np.array(Mask))==2)
# plt.imshow(Mask)


# # 3. Prepare the Dataset

# In[10]:


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from scipy.stats import binom
import cv2 as cv
import numpy as np
from skimage import morphology
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import random


# #### 3.1 Dataset: Mixslide Bags

# In[11]:


class TSbags_random_mixpatch(data_utils.Dataset):
    def __init__(self, slide_obs, masks_tissue, masks_label, level, patch_shape, length_bag_mean, num_bags, transform):
        self.slide_obs = slide_obs
        self.masks_tissue = masks_tissue
        self.masks_label = masks_label
        self.level = level
        self.patch_shape =patch_shape
        self.length_bag_mean = length_bag_mean
        self.num_bags = num_bags
        self.transform = transform
        self.Patch_list, self.Label_patch_list = self._mix_patches()
        self.bags_list, self.labels_list = self._create_bags()  
        
    def _mix_patches(self):
        # to return: [(slide_ID,row,col)]
        Patch_list = []
        Label_patch_list = []
        for slide_ID in self.slide_obs.keys():
            mask_tissue = self.masks_tissue[slide_ID]
            mask_label = self.masks_label[slide_ID]
            for label in range(len(self.num_bags)):
                ROW, COL = np.where((mask_tissue==1)&(mask_label==(label+1)))
                Patch_list.extend([(slide_ID,ROW[i],COL[i]) for i in range(len(ROW))])
                Label_patch_list.extend([(slide_ID, label) for _ in ROW])
        return Patch_list, Label_patch_list
            
    def _create_bags(self):            
        bags_list = []
        labels_list = []
        for slide_ID in self.slide_obs.keys():
            # Retrieve both the label and the global index for patches from the current slide
            patches_for_slide = [(index, label) for index, (s_id, label) in enumerate(self.Label_patch_list) if s_id == slide_ID]
            path_for_oneslide = np.array([label for index, label in patches_for_slide])
            global_indices = [index for index, label in patches_for_slide]

            for label in range(len(self.num_bags)):
                # Find local indices of the patches with the specific label
                local_label_indices = np.where(path_for_oneslide == label)[0]
                # Map local indices to global indices
                label_indices = [global_indices[i] for i in local_label_indices]

                for bag_idx in range(self.num_bags[label]):
                    if len(label_indices) <= self.length_bag_mean*2+self.length_bag_mean/2:
                        pass
                    else:
                        length_bag = binom.rvs(n=int(self.length_bag_mean * 2), p=0.5)
                        # Sample from the global indices
                        selected_indices = random.sample(label_indices, length_bag)
                        bags_list.append(selected_indices)
                        labels_list.append(label)

        return bags_list, labels_list

    def _pack_one_bag(self,indices):
        Bag = []
        for index in indices:
            slide_ID, row_unit, col_unit = self.Patch_list[index]
            factor = self.slide_obs[slide_ID].level_downsamples[self.level]
            unit = int(self.slide_obs[slide_ID].dimensions[0]/self.masks_tissue[slide_ID].shape[1])
            upperLeft_x = int(col_unit * unit + unit/2 - self.patch_shape/2*factor)
            upperLeft_y = int(row_unit * unit + unit/2 - self.patch_shape/2*factor)
            patch = self.slide_obs[slide_ID].read_region((upperLeft_x, upperLeft_y),self.level,(self.patch_shape,self.patch_shape))
            patch = Image.fromarray(np.array(patch)[:,:,:3])
            if self.transform is not None:
                patch = self.transform(patch)
            Bag.append(patch)
        Bag = np.stack(Bag,axis=0)
        return Bag  
    def __len__(self):
        return len(self.bags_list)  
    def __getitem__(self, index):
        indices = self.bags_list[index]
        bag = self._pack_one_bag(indices)
        label = self.labels_list[index]
        return bag, label


# In[12]:


#Mix patches
def create_dataset_mixpatch(slides, tissue_masks, label_masks, num_bags, level, patch_shape,length_bag_mean = 10):
    """
    Generate data loaders
    - Input
        slides: dictionary {'slide_ID':slide_ob}
        tissue_masks: dictionary {'slide_ID':array}
        label_masks: dictionary {'slide_ID':array}
        num_bags:list
        level:int
        patch_shape:int
    - Return
        Dataset
    """
    # Training loaders
    Dataset = TSbags_random_mixpatch(slide_obs = slides,
                                            masks_tissue = tissue_masks, 
                                            masks_label = label_masks, 
                                            level = level, 
                                            patch_shape = patch_shape, 
                                            length_bag_mean = length_bag_mean, 
                                            num_bags = num_bags, 
                                          transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))      
    return Dataset


# # 4. Define the Model

# #### 4.1 LC-MIL model architecture

# In[13]:


class Attention_modern_multi(nn.Module):
    def __init__(self,cnn,focal_loss=False):
        super(Attention_modern_multi,self).__init__()
        #Attention Pooling input dimension
        self.L = 1000
        self.D = 64
        self.K = 1
        self.focal_loss = focal_loss
        #Feature_extracter vgg16 with first two child weights freezed
        self.feature_extractor = cnn
        #Attention Pooling
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K))
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 4)
        )
    def forward(self,x):
        x = x .squeeze(0)

        H = self.feature_extractor(x)

        A = self.attention(H)
        A = torch.transpose(A,1,0)
        #Rescale attention weights between 0 and 1
        A = F.softmax(A,dim=1)
        M = torch.mm(A,H)
        Y_prob = self.classifier(M)
        #Convert Raw logit to probability
        Y_prob = F.softmax(Y_prob,dim=1)

        return Y_prob, A
    def calculate_classification_error(self, X, Y):
        Y_prob,_ = self.forward(X)
        #Choose the class with the max probability
        Y_hat = torch.argmax(Y_prob[0])
        #Calculate the classification accuracy
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat, Y_prob
    def calculate_objective(self, X, Y):
        Y_prob, A = self.forward(X) # 这里的ylabel应该是0，1，2，3，所以dataloader里面的label应该减去1
        #print(f"Y_prob:{Y_prob}")
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        if not self.focal_loss:
            loss = nn.CrossEntropyLoss()
            #print(f"focal_loss; label:{Y}")
            neg_log_likelihood = loss(Y_prob, Y)
        else:
            #print(f"label:{Y}")
            Y_prob_target = Y_prob[0,Y.cpu().data]
            #print(f"else; Y_prob_target:{Y_prob_target}")
            if Y_prob_target.cpu().data.numpy()[0]<0.2:
                gamma = 5
            else:
                gamma = 3
            neg_log_likelihood =-1. *(1-Y_prob_target)**gamma* torch.log(Y_prob_target)
        return neg_log_likelihood, A


# In[14]:


class fully_connected(nn.Module):
	"""docstring for BottleNeck"""
	def __init__(self, model, num_ftrs, num_classes):
		super(fully_connected, self).__init__()
		self.model = model
		self.fc_4 = nn.Linear(num_ftrs,num_classes)

	def forward(self, x):
		x = self.model(x)
		x = torch.flatten(x, 1)
		out_1 = x
		out_3 = self.fc_4(x)
		return  out_3


device = torch.device("cuda:6")
torch.cuda.set_device(device)
model = torchvision.models.densenet121(pretrained=True)
for param in model.parameters():
	param.requires_grad = False
model.features = nn.Sequential(model.features , nn.AdaptiveAvgPool2d(output_size= (1,1)))
num_ftrs = model.classifier.in_features
model_final = fully_connected(model.features, num_ftrs, 30)
model = model.to(device)
model_final = model_final.to(device)
model_final = nn.DataParallel(model_final)
params_to_update = []
criterion = nn.CrossEntropyLoss()

model_final.load_state_dict(torch.load('/cis/home/jhu104/osteosarcoma/Dense_net/KimiaNetPyTorchWeights.pth'))

model_final.module.fc_4 = nn.Linear(num_ftrs, 1000)
for param in model_final.parameters():
	param.requires_grad = False
for param in model_final.module.fc_4.parameters():
    param.requires_grad = True


# In[15]:


def load_vgg16_tune():
    vgg = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
    for param in vgg.features.parameters():
        param.requires_grad = False
    
    for layer_num in range(21,31):
        for param in vgg.features[layer_num].parameters():
            param.requires_grad = True
    
    # for layer_num,child in enumerate(vgg.features.children()):
    #     for param in child.parameters():
            # print(f'Layer {layer_num}, requires_grad: {param.requires_grad}')

    return vgg
    


# #### 4.2 Training Functions

# In[16]:


def train(model, optimizer, Dataloader_train):
    model.train()
    train_loss = 0.
    train_error = 0.
    optimizer.zero_grad()
    for batch_idx, (data, label) in enumerate(Dataloader_train):
        bag_label = label
        data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss, _ = model.calculate_objective(data, bag_label)
        error, _, _ = model.calculate_classification_error(data, bag_label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.data.cpu().numpy()[0]
        train_error += error
        del data
        del bag_label
    train_loss /= len(Dataloader_train)
    train_error /= len(Dataloader_train)
    return model, train_loss, 1-train_error


# In[17]:


def val(model, Dataloader_val):
    val_loss = 0.
    val_error = 0.      
    for batch_idx, (data, label) in enumerate(Dataloader_val):
        bag_label = label
        data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data,requires_grad=False), Variable(bag_label,requires_grad=False)
        loss, _ = model.calculate_objective(data, bag_label)
        error, _, _ = model.calculate_classification_error(data, bag_label)
        val_loss += loss.data.cpu().numpy()[0]
        val_error += error
        del data
        del bag_label
    val_loss /= len(Dataloader_val)
    val_error /= len(Dataloader_val)
    return val_loss, 1-val_error


# In[18]:


def test(model, Dataloader_test):
    test_error = 0. 
    pred_labels = []   
    true_labels = [] 
    for batch_idx, (data, label) in enumerate(Dataloader_test):
        bag_label = label
        data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data,requires_grad=False), Variable(bag_label,requires_grad=False)
        error, pred, _ = model.calculate_classification_error(data, bag_label)
        pred_labels.append(pred.data.cpu().item())
        true_labels.append(bag_label.data.cpu().item())
        test_error += error
        if batch_idx%699 ==0:
            print(f"{batch_idx}th test accuracy:{1-(test_error/(batch_idx+1)):.3f}")
        del data
        del bag_label
    test_error /= len(Dataloader_test)
    return pred_labels, true_labels,1-test_error


# In[19]:


def Train(model, Dataset_train, optimizer, scheduler, validation=False, Dataset_val = None):
    wandb.watch(model, log = "all", log_freq = 10)
    
    n_train = len(Dataset_train)
    split_train = 400
    indices_train = random.sample(list(range(n_train)),k=n_train)
    Train_loss = []
    Train_accuracy = []
    Val_loss = []
    Val_accuracy = []

    if validation:
        n_val = len(Dataset_val)
        indices_val = random.sample(list(range(n_val)),k=400)

    for i in range(n_train//split_train):
        print(f"epoch: {i}")
        Sampler_train = torch.utils.data.sampler.SubsetRandomSampler(indices_train[i*split_train:(i+1)*split_train])
        Dataloader_train = data_utils.DataLoader(Dataset_train, sampler = Sampler_train, batch_size = 1, shuffle = False)
        model.cuda()
        model, train_loss, train_accuracy = train(model, optimizer, Dataloader_train)
        Train_loss.append(train_loss)
        
        Train_accuracy.append(train_accuracy)
        scheduler.step()

        #Log training metric to wandb
        wandb.log({ "train_loss": train_loss, "train_accuracy": train_accuracy})

        print("epoch={}/{}, train loss = {:.3f}, train_accuracy = {:.3f}".format(i, n_train//split_train, train_loss, train_accuracy))
        if validation:
            Sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
            Dataloader_val = data_utils.DataLoader(Dataset_val, sampler = Sampler_val, batch_size = 1, shuffle = False)
            val_loss, val_accuracy = val(model, Dataloader_val)
            Val_loss.append(val_loss)
            Val_accuracy.append(val_accuracy)

            #log validation metric to wandb
            wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy})


           
            print("epoch={}/{}, val loss = {:.3f}, val_accuracy = {:.3f}".format(i, n_train//split_train, val_loss, val_accuracy))
        
        if i%30 ==0:
                torch.save(model.state_dict(), f"/cis/home/jhu104/osteosarcoma/model_experiment_setting_multiple_slides/ml_mixpatch_{i}_epoch.pth")
        
    torch.save(model.state_dict(), f"/cis/home/jhu104/osteosarcoma/model_experiment_setting_multiple_slides/ml_final.pth")
    return model, Train_loss, Train_accuracy, Val_loss, Val_accuracy


# # 5A. Training Using Multiple slides

# #### 5.0 OSTU setting

# In[20]:


#step0
#Read OSTU dictoinary
with open("/cis/home/jhu104/osteosarcoma/OSTU/ostu.dic", 'r') as j:
     adjust_otsu_all1 = json.loads(j.read())

#step1
#Get basename dictionray
data_dic = {}
for key,value in adjust_otsu_all1.items():
     basename = os.path.basename(key)
     data_dic[basename] = value


#step2
#Read training file from zhenzhen
split_df = pd.read_csv("/cis/home/jhu104/osteosarcoma/OSTU/osteosarcoma_experiment_setting.csv")


#step3
#change the dictionary to df
ostu_df = pd.DataFrame(data_dic.items())
ostu_df.rename(columns={0: "image_name", 1: "OSTU"}, inplace=True)


#step4
#merge the table
split_df=split_df.merge(ostu_df,on = ["image_name"])
split_df=split_df.sort_values(by="dataset")


# #### 5.1 Split the dataset

# In[21]:


#Get all the label
label_list = []
directory = '/export/io86/data/jhu101'
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
        label_list.append(filepath)


#Find matched label
def find_matched_label(image,label):
    label_match = []
    for i in range(len(image)):
        appendix = os.path.basename(image[i])
        appendix = appendix+".xml"
        for j in range(len(label)):
            if appendix == os.path.basename(label_list[j]):
                label_match.append(label_list[j])
    return label_match


#Find all image path
random.seed(42)
directory = '/cis/home/jhu104/jhu101'
datalist=[]
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
        datalist.append(filepath)
#randomly shuffle the dataset
random.shuffle(datalist)


#Get train,test and validation data path
def get_data(directory):
    base = '/cis/home/jhu104/jhu101'
    datatrain=[]
    for i in range(len(directory)):
        filepath = os.path.join(base, directory[i])
        datatrain.append(filepath)
    return datatrain
 

#Split into training,validation and test
#train
train_data = get_data(split_df[split_df["dataset"] =="training"]["image_name"].to_list())
train_label = find_matched_label(train_data,label_list)

#val
val_data = get_data(split_df[split_df["dataset"] =="validation"]["image_name"].to_list())
val_label = find_matched_label(val_data,label_list)

#test
test_data = get_data(split_df[split_df["dataset"] =="test"]["image_name"].to_list())
test_label = find_matched_label(test_data,label_list)


# #### 5.2 Training Slide Settings

# In[22]:


#Find settings for slides
def slide_setting(data,label):
    dic = {}
    for i in range(len(data)):
        Annotations=read_Aaron_annotations(label[i])
        slide = openslide.OpenSlide(data[i])
        Mask = generate_label_mask(slide,Annotations,annotation_label_mapping,256)
        # print(f"""{i+1}st slide: viable: {round(np.sum(np.array(Mask)==1)/20)},
        #                         necrosis: {round(np.sum(np.array(Mask)==2)/20)}, 
        #                         stroma: {round(np.sum(np.array(Mask)==3)/20)}, type4: {round(np.sum(np.array(Mask)==4)/20)}""")

        # viable = round(np.sum(np.array(Mask) == 1) / 20)
        # necrosis = round(np.sum(np.array(Mask) == 2) / 10)
        # stroma = round(np.sum(np.array(Mask) == 3) / 20)
        # type4 = round(np.sum(np.array(Mask) == 4)/2 ) 
        viable = 500
        necrosis = 500
        stroma = 500
        type4 = 500
        dic[(data[i])] ={
            'stroma': stroma,
            'viable': viable,
            'necrosis': necrosis,
            'type4': type4,
        }
    return dic

#Get the bag length
Settings = slide_setting(train_data,train_label)

#Insert the OTSU
def insert_otsu(settingslides,adjust_otsu_type):
    for key, adjust_otsu in zip(settingslides.keys(), adjust_otsu_type):
        settingslides[key]['adjust_otsu'] = adjust_otsu
    return settingslides

#Get the train setting

adjust_otsu_train = split_df[split_df["dataset"] =="training"]["OSTU"].to_list()
Settings =insert_otsu(Settings,adjust_otsu_train)


# In[23]:


#Load all training WSIs
unit = 256
level = 2 # 10x magnification
patch_shape = 256
annotation_label_mapping ={
    'stroma':1,
    'viable':2,
    'necrosis':3,
    'type4':4
}
slides_train = {}
num_bags_train = {}
annotations_train = {}
tissue_masks_train = {}
label_masks_train = {}
for i, slide_ID in enumerate(train_data):
    slides_train[slide_ID] = openslide.OpenSlide(slide_ID)
    annotations_train[slide_ID] = read_Aaron_annotations(train_label[i])
    num_bags_train[slide_ID] = [Settings[slide_ID]['stroma'],Settings[slide_ID]['viable'],Settings[slide_ID]['necrosis'],Settings[slide_ID]['type4']]    
    tissue_masks_train[slide_ID] = generate_tissue_mask(slides_train[slide_ID],unit=unit,adjust_otsu=Settings[slide_ID]['adjust_otsu'])
    label_masks_train[slide_ID] = generate_label_mask(slides_train[slide_ID], annotations_train[slide_ID], annotation_label_mapping, unit)

#Find the train bag total length
train_bag_len =[0,0,0,0]
for idx,key in num_bags_train.items():
    train_bag_len[0] += key[0] 
    train_bag_len[1] += key[1] 
    train_bag_len[2] += key[2] 
    train_bag_len[3] += key[3]


# #### 5.2 Validation Slide Settings

# In[24]:


#Find settings for validation slides
def slide_setting(data,label):
    dic = {}
    for i in range(len(data)):
        Annotations=read_Aaron_annotations(label[i])
        slide = openslide.OpenSlide(data[i])
        Mask = generate_label_mask(slide,Annotations,annotation_label_mapping,256)
        # print(f"""{i+1}st slide: viable: {round(np.sum(np.array(Mask)==1)/20)},
        #                         necrosis: {round(np.sum(np.array(Mask)==2)/20)}, 
        #                         stroma: {round(np.sum(np.array(Mask)==3)/20)}, type4: {round(np.sum(np.array(Mask)==4)/20)}""")

        # viable = round(np.sum(np.array(Mask) == 1) / 20)
        # necrosis = round(np.sum(np.array(Mask) == 2) / 20)
        # stroma = round(np.sum(np.array(Mask) == 3) / 20)
        # type4 = round(np.sum(np.array(Mask) == 4)/8 ) 
        viable = 500
        necrosis = 500
        stroma = 500
        type4 = 500
        dic[(data[i])] ={
            'stroma': stroma,
            'viable': viable,
            'necrosis': necrosis,
            'type4': type4,
        }
    return dic


#Get the bag length
Settings_val = slide_setting(val_data,val_label)

#Insert the OTSU
def insert_otsu(settingslides,adjust_otsu_type):
    for key, adjust_otsu in zip(settingslides.keys(), adjust_otsu_type):
        settingslides[key]['adjust_otsu'] = adjust_otsu
    return settingslides


adjust_otsu_val= split_df[split_df["dataset"] =="validation"]["OSTU"].to_list()
Settings_val =insert_otsu(Settings_val,adjust_otsu_val)


# In[25]:


#Load validation WSI
if len(val_data)>0:
    slides_val = {}
    num_bags_val = {}
    annotations_val= {}
    tissue_masks_val = {}
    label_masks_val = {}
    for i, slide_ID in enumerate(val_data):
        slides_val[slide_ID] = openslide.OpenSlide(slide_ID)
        annotations_val[slide_ID] = read_Aaron_annotations(val_label[i])
        num_bags_val[slide_ID] = [Settings_val[slide_ID]['stroma'],Settings_val[slide_ID]['viable'],Settings_val[slide_ID]['necrosis'],Settings_val[slide_ID]['type4']]    
        tissue_masks_val[slide_ID] = generate_tissue_mask(slides_val[slide_ID],unit=unit,adjust_otsu=Settings_val[slide_ID]['adjust_otsu'])
        label_masks_val[slide_ID] = generate_label_mask(slides_val[slide_ID], annotations_val[slide_ID], annotation_label_mapping, unit)

#Find the train bag total length
val_bag_len =[0,0,0,0]
for idx,key in num_bags_val.items():
    val_bag_len[0] += key[0] 
    val_bag_len[1] += key[1] 
    val_bag_len[2] += key[2] 
    val_bag_len[3] += key[3] 


# #### 5.3 Test Slide Settings

# In[26]:


#Find settings for test slides
def slide_setting(data,label):
    dic = {}
    for i in range(len(data)):
        Annotations=read_Aaron_annotations(label[i])
        slide = openslide.OpenSlide(data[i])
        Mask = generate_label_mask(slide,Annotations,annotation_label_mapping,256)
        # print(f"""{i+1}st slide: viable: {round(np.sum(np.array(Mask)==1)/20)},
        #                         necrosis: {round(np.sum(np.array(Mask)==2)/20)}, 
        #                         stroma: {round(np.sum(np.array(Mask)==3)/20)}, type4: {round(np.sum(np.array(Mask)==4)/20)}""")

        # viable = round(np.sum(np.array(Mask) == 1) / 20)
        # necrosis = round(np.sum(np.array(Mask) == 2) / 20)
        # stroma = round(np.sum(np.array(Mask) == 3) / 20)
        # type4 = round(np.sum(np.array(Mask) == 4)/8 ) 
        viable = 500
        necrosis = 500
        stroma = 500
        type4 = 500
        dic[(data[i])] ={
            'stroma': stroma,
            'viable': viable,
            'necrosis': necrosis,
            'type4': type4,
        }
    return dic


#Get the bag length
Settings_test = slide_setting(test_data,test_label)

#Insert the OTSU
def insert_otsu(settingslides,adjust_otsu_type):
    for key, adjust_otsu in zip(settingslides.keys(), adjust_otsu_type):
        settingslides[key]['adjust_otsu'] = adjust_otsu
    return settingslides


adjust_otsu_test= split_df[split_df["dataset"] =="test"]["OSTU"].to_list()
Settings_test =insert_otsu(Settings_test,adjust_otsu_test)


# In[27]:


#Load test WSI
if len(test_data)>0:
    slides_test= {}
    num_bags_test = {}
    annotations_test= {}
    tissue_masks_test = {}
    label_masks_test = {}
    for i, slide_ID in enumerate(test_data):
        slides_test[slide_ID] = openslide.OpenSlide(slide_ID)
        annotations_test[slide_ID] = read_Aaron_annotations(test_label[i])
        num_bags_test[slide_ID] = [Settings_test[slide_ID]['stroma'],Settings_test[slide_ID]['viable'],Settings_test[slide_ID]['necrosis'],Settings_test[slide_ID]['type4']]    
        tissue_masks_test[slide_ID] = generate_tissue_mask(slides_test[slide_ID],unit=unit,adjust_otsu=Settings_test[slide_ID]['adjust_otsu'])
        label_masks_test[slide_ID] = generate_label_mask(slides_test[slide_ID], annotations_test[slide_ID], annotation_label_mapping, unit)

#Find the train bag total length
test_bag_len =[0,0,0,0]
for idx,key in num_bags_test.items():
    test_bag_len[0] += key[0] 
    test_bag_len[1] += key[1] 
    test_bag_len[2] += key[2] 
    test_bag_len[3] += key[3] 


# #### 5.4 Training

# In[ ]:


wandb.init(
    project= "osteosarcoma- ModelTrain",
    name = "Generalization",
    config={
        "epochs":6,
        "batch_size": 1,
        "lr": 0.000001,
    }
)
config = wandb.config
device = torch.device("cuda:0")
torch.cuda.set_device(device)
model = Attention_modern_multi(load_vgg16_tune(),True)
model = model.to(device)
optimizer = optim.Adam(model.parameters(),lr=0.00002, betas=(0.9, 0.999), weight_decay =10e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.2)
train_bag_len = [600,600,600,600]
val_bag_len = [600,600,600,600]
Dataset_train = create_dataset_mixpatch(slides_train,
                                  tissue_masks_train, label_masks_train, 
                                  train_bag_len, level, patch_shape,length_bag_mean = 10)
if len(slides_val)>0:
    Dataset_val = create_dataset_mixpatch(slides_val,
                                  tissue_masks_val, label_masks_val, 
                                  val_bag_len, level, patch_shape,length_bag_mean = 10)
    model, Train_loss, Train_accuracy, Val_loss, Val_accuracy = Train(model, Dataset_train, optimizer, scheduler, validation=True, Dataset_val = Dataset_val)
else:
    model, Train_loss, Train_accuracy, Val_loss, Val_accuracy = Train(model, Dataset_train, optimizer, scheduler, validation=False, Dataset_val = None)

# save model


# # 6. Test data slide level accuracy

# #### 6.1 Construct bag

# In[29]:


class TSbags_random_oneslide(data_utils.Dataset):
    def __init__(self, slide_ob, mask_tissue, mask_label, level, patch_shape, length_bag_mean, num_bags, transform):
        self.slide_ob = slide_ob
        self.mask_tissue = mask_tissue
        self.mask_label = mask_label
        self.level = level
        self.patch_shape =patch_shape
        self.length_bag_mean = length_bag_mean
        self.num_bags = num_bags
        self.transform = transform
        self.unit = int(self.slide_ob.dimensions[0]/mask_tissue.shape[1])
        self.bags_list, self.labels_list = self._create_bags()  
    def _create_bags(self):            
        bags_list = []
        labels_list = []
        for label in range(len(self.num_bags)):         
            ROW, COL = np.where((self.mask_tissue==1)&(self.mask_label==label+1))
            for bag_idx in range(self.num_bags[label]):
                if len(ROW) <=self.length_bag_mean:
                    pass
                else: 
                    length_bag = binom.rvs (n=int(self.length_bag_mean*2), p=0.5)
                    indices = np.random.randint(0,len(ROW),length_bag)
                    bags_list.append((ROW[indices], COL[indices]))
                    labels_list.append(label)
        return bags_list, labels_list
    def _pack_one_bag(self,row_list, col_list):
        Bag = []
        for i in range(len(row_list)):
            row_unit, col_unit = row_list[i], col_list[i]
            factor = self.slide_ob.level_downsamples[self.level]
            upperLeft_x = int(col_unit * self.unit + self.unit/2 - self.patch_shape/2*factor)
            upperLeft_y = int(row_unit * self.unit + self.unit/2 - self.patch_shape/2*factor)
            patch = self.slide_ob.read_region((upperLeft_x, upperLeft_y),self.level,(self.patch_shape,self.patch_shape))
            patch = Image.fromarray(np.array(patch)[:,:,:3])
            if self.transform is not None:
                patch = self.transform(patch)
            Bag.append(patch)
        Bag = np.stack(Bag,axis=0)
        return Bag  
    def __len__(self):
        return len(self.bags_list)  
    def __getitem__(self, index):
        row_list, col_list = self.bags_list[index]
        bag = self._pack_one_bag(row_list, col_list)
        label = self.labels_list[index]
        return bag, label


# #### 6.2 Construct Dataset

# In[30]:


def create_dataset_mixbag(slides, tissue_masks, label_masks, num_bags, level, patch_shape,length_bag_mean = 10):
    """
    Generate data loaders
    - Input
        slides: dictionary {'slide_ID':slide_ob}
        tissue_masks: dictionary {'slide_ID':array}
        label_masks: dictionary {'slide_ID':array}
        num_bags:dict{'slide_ID':list}
        level:int
        patch:int
    - Return
        Dataset
    """
    # Training loaders
    dataset = TSbags_random_oneslide(slide_ob = slides,
                                        mask_tissue = tissue_masks, 
                                        mask_label = label_masks, 
                                        level = level, 
                                        patch_shape = patch_shape, 
                                        length_bag_mean = length_bag_mean, 
                                        num_bags = num_bags, 
                                        transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))      
    return dataset


# #### 6.3 Test Result

# In[ ]:


# load model
path = "/cis/home/jhu104/osteosarcoma/model_experiment_setting_multiple_slides/ml_mixpatch_60_epoch.pth"
device = torch.device("cuda:0")
model = Attention_modern_multi(load_vgg16_tune())
model.load_state_dict(torch.load(path))
model = model.to(device)
test_bag_len_slide = [500,500,500,500]

    # 'stroma':1,
    # 'viable':2,
    # 'necrosis':3,
    # 'type4':4

#Prediction accuracy
def prediction_accuracy(pred,true,test_bag_len_slide):
    listname = ['stroma', 'viable', 'necrosis', 'cartilage']
    label_type = int(len(pred)/test_bag_len_slide[0])
    label_name = np.unique(true)
    #Get classification result
    for i in range(label_type):
        for j in range(len(listname)):
            class_len = int(len(pred)/label_type)
            result = np.sum(np.array(pred[i*class_len:(i+1)*class_len]) ==j)/class_len
            print(f"{listname[label_name[i]]}: classification result for {listname[j]}: {result*100:.2f}%")


#Import test data
for idx,slide_name in enumerate(test_data):
    Dataset_test = create_dataset_mixbag(slides_test[test_data[idx]],
                                    tissue_masks_test[test_data[idx]], label_masks_test[test_data[idx]], 
                                    test_bag_len_slide, level, patch_shape,length_bag_mean = 10)

    print(f"{idx}th slide: {test_data[idx]}")
    #Test accuracy
    test_accuracy_list= []
    Dataloader_test = data_utils.DataLoader(Dataset_test, batch_size = 1, shuffle = False)
    pred_labels_test, true_labels,test_accuracy = test(model, Dataloader_test)
    test_accuracy_list.append(test_accuracy)
    print("test_accuracy = {:.3f}".format(test_accuracy))
    print("Classification Results")
    prediction_accuracy(pred_labels_test,true_labels,test_bag_len_slide)
    print("")


