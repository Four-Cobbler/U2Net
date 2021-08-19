import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):                                                # 归一化概率图d
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):                         # 保存并输出图片

    predict = pred
    predict = predict.squeeze()                                 # 去除predict中的单一维[1,2,3,4]->[2,3,4]
    predict_np = predict.cpu().data.numpy()                     # 将数据转化为numpy并转存到CPU中（之前位于GPU）

    im = Image.fromarray(predict_np*255).convert('RGB')         # 将array形式转化为图片形式，且转换为RGB模式（未转换前array数组为BGR格式）
    img_name = image_name.split(os.sep)[-1]                     # 将文件路径（image_name）按照分隔符“\”全部切开（-1参数的作用），例：C:\Users\kilok\  ->  C: Users kilok
    image = io.imread(image_name)                               # 按路径读入图片，此处选用io.imread，故读取格式为RGB格式
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)    # 对图像进行缩放处理，采用双线性采样方式缩放

    pb_np = np.array(imo)                                       # 缩放后图片转化为array数组

    aaa = img_name.split(".")                                   # 按.切分路径
    bbb = aaa[0:-1]                                             # 按左起第0个元素直到倒数第一个元素-1进行切片
    imidx = bbb[0]                                              # 第一个切片，如C：
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]                            

    imo.save(d_dir+imidx+'.png')                                # 循环保存图片文件as.png

def main():

    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp



    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')   # 在当前程序路径加入test_data test_images
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep) # 模型预测结果的存放目录，我这里是C:\Users\......\U-2-Net\test_data\u2net_results
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')      # 学习到的模型参数文件所在目录

    img_name_list = glob.glob(image_dir + os.sep + '*')                                         # 测试文件夹下的图片路径列表
    print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )                                                       # 加载数据
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")                                                       # 这两行打印可以去掉
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()                                                                              # 送入显卡处理或送入CPU处理
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):                                 # 同时遍历键与值

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)                                                  # 对输入图片送入模型处理

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)                                                                   # 归一化得到的mask

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,prediction_dir)                                  # 保存mask图到相应目录

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
