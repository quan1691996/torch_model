from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time
import os
import pandas  as pd
import matplotlib.pyplot as plt
from util import *

def parse_cfg(config_file):
     file = open(config_file,'r')
     file = file.read().split('\n')
     file =  [line for line in file if len(line)>0 and line[0] != '#']
     file = [line.lstrip().rstrip() for line in file]

     final_list = []
     element_dict = {}
     for line in file:

          if line[0] == '[':
               if len(element_dict) != 0:     # appending the dict stored on previous iteration
                         final_list.append(element_dict)
                         element_dict = {} # again emtying dict
               element_dict['type'] = ''.join([i for i in line if i != '[' and i != ']'])
               
          else:
               val = line.split('=')
               element_dict[val[0].rstrip()] = val[1].lstrip()  #removing spaces on left and right side
          
     final_list.append(element_dict) # appending the values stored for last set
     return final_list

class DummyLayer(nn.Module):
    def __init__(self):
        super(DummyLayer, self).__init__()
        

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def create_model(blocks):
#     blocks = parse_cfg(cfgfile)
    darknet_details = blocks[0]
    channels = 3 
    output_filters = []
    modulelist = nn.ModuleList()
    
    for i,block in enumerate(blocks[1:]):
        seq = nn.Sequential()
        if (block["type"] == "convolutional"):
            activation = block["activation"]
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            strides = int(block["stride"])
            use_bias= False if ("batch_normalize" in block) else True
            pad = (kernel_size - 1) // 2
            
            conv = nn.Conv2d(in_channels=channels, out_channels=filters, kernel_size=kernel_size, 
                             stride=strides, padding=pad, bias = use_bias)
            seq.add_module("conv_{0}".format(i), conv)
            
            if "batch_normalize" in block:
                bn = nn.BatchNorm2d(filters)
                seq.add_module("batch_norm_{0}".format(i), bn)

            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                seq.add_module("leaky_{0}".format(i), activn)
            
        elif (block["type"] == "upsample"):
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
            seq.add_module("upsample_{}".format(i), upsample)
        
        elif (block["type"] == 'route'):
            # start and end is given in format (eg:-1 36 so we will find layer number from it.
            # we will find layer number in negative format
            # so that we can get the number of filters in that layer
            block['layers'] = block['layers'].split(',')
            block['layers'][0] = int(block['layers'][0])
            start = block['layers'][0]
            if len(block['layers']) == 1:               
                filters = output_filters[i + start]
                       
            
            elif len(block['layers']) > 1:
                block['layers'][1] = int(block['layers'][1]) - i 
                end = block['layers'][1]
                filters = output_filters[i + start] + output_filters[i + end]
                  
            
            route = DummyLayer()
            seq.add_module("route_{0}".format(i),route)
                
      
        elif block["type"] == "shortcut":
            from_ = int(block["from"])
            shortcut = DummyLayer()
            seq.add_module("shortcut_{0}".format(i),shortcut)
            
            
        elif block["type"] == "yolo":
            mask = block["mask"].split(",")
            mask = [int(m) for m in mask]
            anchors = block["anchors"].split(",")
            anchors = [(int(anchors[i]), int(anchors[i + 1])) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            block["anchors"] = anchors
            
            detectorLayer = DetectionLayer(anchors)
            seq.add_module("Detection_{0}".format(i),detectorLayer)
                
        modulelist.append(seq)
        output_filters.append(filters)  
        channels = filters
    
    return darknet_details, modulelist

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_model(self.blocks)
        
    def forward(self, x, CUDA=False):
        modules = self.blocks[1:]
        outputs = {}   #We cache the outputs for the route layer
        write = 0     #This is explained a bit later
        for i, module in enumerate(modules):        
            module_type = (module["type"])
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
                outputs[i] = x
                
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                if len(layers) > 1:
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1,map2),1)
       
                outputs[i] = x
                
            elif  module_type == "shortcut":
                from_ = int(module["from"])

                # just adding outputs for residual network
                x = outputs[i-1] + outputs[i+from_]  
                outputs[i] = x
                
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                
                #Get the input dimensions
                inp_dim = int(self.net_info["height"])
                #Get the number of classes
                num_classes = int(module["classes"])
            
                #Transform 
                x = x.data   # get the data at that point
                x = prediction(x,inp_dim,anchors,num_classes,CUDA)
                
                if not write:              #if no collector has been intialised. 
                    detections = x
                    write = 1
                else:       
                    detections = torch.cat((detections, x), 1)

                outputs[i] = outputs[i-1]
                
        try:
            return detections   #return detections if present
        except:
            return 0
     
    def load_weights(self, weightfile):
        
        #Open the weights file
        fp = open(weightfile, "rb")

        #The first 4 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4. IMages seen 
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        
        #The rest of the values are the weights
        # Let's load them up
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                
                conv = model[0]
                
                if (batch_normalize):
                    bn = model[1]
                    
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
                    
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                    
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
                # Note: we dont have bias for conv when batch normalization is there

CUDA = torch.cuda.is_available()
#Set up the neural network
print("Loading network.....")
model = Darknet("./torch_model/YOLO/cfg/yolov3.cfg")
model.load_weights("./torch_model/YOLO/cfg/yolov3.weights")
print("Network successfully loaded")
classes = load_classes("./torch_model/YOLO/cfg/coco.names")
print('Classes loaded')
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

#Set the model in evaluation mode
model.eval()

def write_image(x, batches, results):
    c1 = tuple(x[1:3])
    c2 = tuple(x[3:5])
    img = results[int(x[0])]
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = (0,0,255)
    cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), color, 2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])) ,color, -1)
    cv2.putText(img, label, (int(c1[0]), int(c1[1]) + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img

W=640
H=480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','U','Y','V'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FPS, 30)

batch_size = 1
while True:
    ret, frame = cap.read()

    if ret:
        im_batches, orig_ims, im_dim_list = prep_image_video(frame,inp_dim)
        im_batches = [im_batches] # list of resized images
        orig_ims = [orig_ims] # list of original images
        im_dim_list = [im_dim_list] # dimension list
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
        
            
        if CUDA:
            im_dim_list = im_dim_list.cuda()

            
        # converting image to batches    
        reminder = 0
        if (len(im_dim_list) % batch_size): #if reminder is there, reminder = 1
            reminder = 1

        if batch_size != 1:
            num_batches = 1 // batch_size + reminder            
            im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,len(im_batches))])) 
                        for i in range(num_batches)] 
            
        nms_thesh = 0.5
        i = 0
        write = False
            
        objs = {}    
        for batch in im_batches:
                #load the image 
                start = time.time()
                if CUDA:
                    batch = batch.cuda()       
                #Apply offsets to the result predictions
                #Tranform the predictions as described in the YOLO paper
                #flatten the prediction vector 
                # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes) 
                # Put every proposed box as a row.
                with torch.no_grad():
                    predictions = model(Variable(batch), CUDA)
                
                predictions = write_results(predictions, confidence=0.5, num_classes=80, nms_conf = nms_thesh)
                
                if type(prediction) == int:
                    i += 1
                    continue


                predictions[:,0] += i*batch_size
                        
                if not write:
                    output_cam = predictions
                    write = 1
                else:
                    output_cam = torch.cat((output_cam,predictions))  # concating predictions from each batch
                i += 1
                
                if CUDA:
                    torch.cuda.synchronize()
            
        try:
            output_cam
        except NameError:
            print("No detections were made")
            exit()

            
        #Before we draw the bounding boxes, the predictions contained in our output tensor 
        #are predictions on the padded image, and not the original image. Merely, re-scaling them 
        #to the dimensions of the input image won't work here. We first need to transform the
        #co-ordinates of the boxes to be measured with respect to boundaries of the area on the
        #padded image that contains the original image

        im_dim_list = torch.index_select(im_dim_list, 0, output_cam[:,0].long())
        scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
        output_cam[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
        output_cam[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2       
        output_cam[:,1:5] /= scaling_factor
            
        for i in range(output_cam.shape[0]):
            output_cam[i, [1,3]] = torch.clamp(output_cam[i, [1,3]], 0.0, im_dim_list[i,0])
            output_cam[i, [2,4]] = torch.clamp(output_cam[i, [2,4]], 0.0, im_dim_list[i,1])

        print(orig_ims[0].shape)
        list(map(lambda x: write_image(x, im_batches, orig_ims), output_cam))
        frame = np.array(orig_ims[0], dtype=np.uint8)

        cv2.imshow('usb cam test', frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    if not ret:
        continue

cap.release()
cv2.destroyAllWindows()