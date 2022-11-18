import torch
import torch.nn as nn
#還是改我自己寫一個mian函數的
from tensorboardX import SummaryWriter
from t_optim import ScheduledOptim
import torchvision.transforms as transforms
import torch.optim as optim
import dataset_load as dataset
import torch.utils.data as torchdata
# from model.mobilenetv2 import MobileNetV2
from model.cnn import CNN
from tqdm import tqdm
import model.resnet as model
import os
import numpy as np

np.set_printoptions(threshold=np.inf)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

batch_size = 64
bird_num=100#鸟类种类数
iteration_num=128#迭代次数设为100，即进行100次训练

#加载数据
print('Starting loading data.')
#转化成【-1，1】之间的张量,暂时未用
# transform = transforms.Compose(
#     [
#     transforms.Resize((224, 224)),
#      transforms.Normalize((0.5), (0.5))
#      ])
train_dataset = dataset.Bird_train_Dataset()
val_dataset = dataset.Bird_dev_Dataset()
train_loader = torchdata.DataLoader(train_dataset, batch_size=64,sampler=torchdata.RandomSampler(train_dataset),num_workers=8,drop_last=True)
val_loader = torchdata.DataLoader(val_dataset, batch_size=64, sampler=torchdata.RandomSampler(val_dataset),num_workers=8)

#
# for i, data in tqdm(enumerate(val_loader)):
#     # get the inputs
#     inputs, labels = data
#     print("v_inputs=",inputs)
#     #print("v_labels=",labels)
print('Successfully loading data!')
#定义模型
# net= model.ResNet18().cuda()

net = model.ResNet18().cuda()
#net = CNN().cuda()
writer = SummaryWriter("./runs/ResNet18")
#定义损失器
criterion = nn.CrossEntropyLoss() #分类交叉熵作损失函数
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)#动量SGD做优化器
optimize = ScheduledOptim(optimizer,iteration_num,lr=0.01)

if torch.cuda.is_available():
    # device = torch.device("cuda:0")
    # net.to(device)
    print('The GPU is available.')
    net = net.cuda()
    net = torch.nn.DataParallel(net,device_ids=[0,1,2,3])
    criterion= criterion.cuda()
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    # net = nn.DataParallel(net)
# net.to(device)

#训练网络
best_corrects=0

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

print('Starting training.')
for epoch in range(iteration_num):  # loop over the dataset multiple times
    print('epoch:', epoch)
    net.train()
    running_loss = 0.0
    sum_loss =0.0
    train_acc = 0.0
    totall = 0
    n = 0
    maxxs =0
    for i,data in tqdm(enumerate(train_loader)):

        # get the inputs
        inputs, labels = data
        #print("labelslen=",len(labels))
        # inputs = transform(inputs)
        #print("inputs",inputs)
        # inputs = inputs.type(torch.Tensor)
        inputs = inputs.cuda()
        labels = labels.cuda()
        labels = labels.view(labels.size(0))
        # print(labels.size())
        # zero the parameter gradients
        inputs,label_a,label_b,lam=mixup_data(inputs,labels)

        net.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # print(labels)
        #loss = criterion(outputs, labels)
        loss=mixup_criterion(criterion,outputs,label_a,label_b,lam)
        #print("loss",loss,"loss_item",loss.item())
        # pred = outputs.cpu().detach().numpy()
        # print(pred)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        sum_loss +=running_loss
        if i % 20 == 199:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))

            running_loss = 0.
        if n<i:
            n=i

        _,prediction = torch.max(outputs.data,1)
        # print("prediction",prediction)
        train_acc +=(prediction==labels).sum().float()
        # print("train_acc=",train_acc)
        totall+=labels.size(0)
        # print("total=",totall)
    print ("*****************ACCURACY:train_acc     {%.4f}**********************"%(train_acc/totall))
    writer.add_scalar("train/Acc",train_acc/totall,epoch)
    writer.add_scalar("train/Loss",sum_loss/n,epoch)
    if (epoch+1) % 10 == 0 and epoch < 30:
        optimize.update_lr(epoch=epoch)
    elif (epoch+1) % 10 ==0 and epoch < 100:
        optimize.update_lr(epoch=epoch)
    writer.add_scalar("train/LR",optimize._optimizer.param_groups[0]['lr'],epoch)
    print("LR:",optimize._optimizer.param_groups[0]['lr'])


    print('Finished Training')

    #每个epoch验证集测试

    correct = 0
    total = 0
    corr_num=[]
    val_loss=[]
    net.eval()
    with torch.no_grad():
        for (images, labels) in val_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
                labels = labels.view(labels.size(0))
            outputs = net(images)
            loss = criterion(outputs, labels)
            # print("loss",loss)
            _, predicted = torch.max(outputs.data, 1)
            # print("val_predict",predicted.size())
            total += labels.size(0)
            correct += (predicted == labels).sum().float()
            print("val_total",total)
            corr_num.append(correct)
            val_loss.append(loss.item())
            print("val_correct",len(corr_num))
        print('Accuracy of the network on the 100 test data: %d %%' % (100 *(correct / total) ))

    #若正确的数量多，即准确率高，则保存模型.pt文件
    if correct > best_corrects:
        print('Saving new best model at epoch ' + str(epoch) + ' (' + str(len(corr_num)/val_dataset.__len__()) + '/' + str(
            val_dataset.__len__()) + ')')
        torch.save(net, 'best_model_'+"CNN8"+str(epoch)+'_acc_'+str(100*(correct / total))+'.pt')
        #best_corrects = len(corr_num)
        best_corrects=correct
    # if (epoch+1)%5==0:
    #     torch.save(net,"the_{}".format(epoch+1)+"_valacc_"+str(len(corr_num))+'.pt')

# #利用模型预测测试集的内容
# test_dataset = dataset.Bird_test_Dataset()
# test_loader = torchdata.DataLoader(test_dataset, batch_size=batch_size, sampler=torchdata.RandomSampler(test_dataset))
# # 创建一个一模一样的模型
# model=model.VGG19()
# # 加载预训练模型的参数
# model.load_state_dict(torch.load('best_model{0}_acc_{1}.pt'.format(str(epoch), str(sum(corr_num)))))
# flag=0#用来计数音频
# for data in test_loader:
#     if torch.cuda.is_available():
#         images = data.cuda()
#     outputs = model(images)
#     _, predicted = torch.max(outputs.data, 1)
#     #写入csv文件
#     number= str(flag).rjust(4, '0')
#     results = dict(zip('test'+number+'.wav', predicted))
#     with open('results.csv', 'w', newline='')as f:
#         writer = csv.DictWriter(f)
#         writer.writerow(results)
#     flag+=1