#coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 100)
    parser.add_argument("--keep_step", type=int, default = 25000)
    parser.add_argument("--decay_step", type=int, default = 25000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

##定义卷积核
def default_conv(in_channels,out_channels,kernel_size,bias=True):
    return nn.Conv2d(in_channels,out_channels,
                     kernel_size,padding=kernel_size//2,   #保持尺寸
                     bias=bias)
##定义ReLU
def default_relu():
    return nn.ReLU(inplace=True)
## reshape
def get_feature(x):
    return x.reshape(x.size()[0],128,16,16)

class Discriminator(nn.Module):
    def __init__(self, conv=default_conv, relu=default_relu):
        super(Discriminator, self).__init__()
        main = [conv(3, 32, 3),
                relu(),
                conv(32, 64, 3),
                relu(),
                conv(64, 128, 3),
                relu(),
                conv(128, 256, 3),
                relu()]
        self.main = nn.Sequential(*main)
        self.fc = nn.Linear(256 * 192 * 256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x:(batchsize,3,192,256)
        x = self.main(x)  # (b,256,192,256)
        x = x.view(-1, 256 * 192 * 256)  # (b,256*192*256)
        x = self.fc(x)  # (b,1)
        return x

    def name(self):
        return 'Discriminator'

def train_gmm(opt, train_loader, model, board):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c =  inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
            
        grid, theta = model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')        #grid_sample双线性采样，结果模拟扭曲衣物
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

        visuals = [ [im_h, shape, im_pose], 
                   [c, warped_cloth, im_c], 
                   [warped_grid, (warped_cloth+im)*0.5, im]]
        
        loss = criterionL1(warped_cloth, im_c)    #模拟扭曲衣服和实际穿上去有扭曲的衣服热图的L1损失
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))


def train_tom(opt, train_loader, model, board):
    model.cuda()
    model.train()
    discriminator = Discriminator()     #判别器初始化
    discriminator.cuda()
    ## 读入图片数据,分batch
    print('===> Data preparing...')
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    ## 判别值
    target_real = Variable(torch.ones(opt.batch_size, 1))
    target_false = Variable(torch.zeros(opt.batch_size, 1))
    one_const = Variable(torch.ones(opt.batch_size, 1))

    target_real = target_real.cuda()
    target_false = target_false.cuda()
    one_const = one_const.cuda()

    ## 优化器
    optim_generator = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    ## 误差函数
    #        content_criterion = nn.MSELoss()
    adversarial_criterion = nn.BCEWithLogitsLoss()
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()
    mean_dis_loss = 0.0
    mean_gen_con_loss = 0.0
    gen_total_loss = 0.0
    mean_gen_total_loss = 0.0
    gen_loss=0.0
    gen_L1_loss=0.0
    gen_VGG_loss=0.0
    gen_Mask_loss=0.0
    ## 训 练 开 始

    for step in range(opt.keep_step + opt.decay_step):

            iter_start_time = time.time()
            inputs = train_loader.next_batch()

            im = inputs['image'].cuda()
            im_pose = inputs['pose_image']
            im_h = inputs['head']
            shape = inputs['shape']

            agnostic = inputs['agnostic'].cuda()
            c = inputs['cloth'].cuda()
            cm = inputs['cloth_mask'].cuda()
        
            outputs = model(torch.cat([agnostic, c],1))             #输出是人衣合体
            p_rendered, m_composite = torch.split(outputs, 3,1)     #按照dim=1这个维度去分，每大块包含3个小块
            p_rendered = F.tanh(p_rendered)
            m_composite = F.sigmoid(m_composite)
            p_tryon = c * m_composite+ p_rendered * (1 - m_composite)   # 看不懂

            visuals = [ [im_h, shape, im_pose],
                   [c, cm*2-1, m_composite*2-1], 
                   [p_rendered, p_tryon, im]]

            ## 1.固定G,训练判别器D
            discriminator.zero_grad()
            im = im.cuda()
            p_tryon = p_tryon.cuda()

            dis_loss1 = adversarial_criterion(discriminator(im), target_real)
            dis_loss2 = adversarial_criterion(discriminator(p_tryon.detach()), target_false)  ##注意经过G的网络再进入D网络之前要detach()之后再进入
            dis_loss = 0.5 * (dis_loss1 + dis_loss2)
            #                print('epoch:%d--%d,判别器loss:%.6f'%(epoch,i,dis_loss))
            dis_loss.backward()
            optim_discriminator.step()
            mean_dis_loss += dis_loss

            ## 2.固定D,训练生成器G
            model.zero_grad()
            gen_loss = adversarial_criterion(discriminator(p_tryon), one_const)
            gen_L1_loss = criterionL1(p_tryon, im)  ##固定D更新G
            gen_VGG_loss = criterionVGG(p_tryon, im)
            gen_Mask_loss = criterionMask(p_tryon, im)
            gen_total_loss = 0.25*(gen_loss+gen_L1_loss+gen_VGG_loss+gen_Mask_loss)
            #                print('epoch:%d--%d,生成器loss:%.6f'%(epoch,i,gen_total_loss))
            gen_total_loss.backward()
            optim_generator.step()

            mean_gen_total_loss += gen_total_loss

            if (step + 1) % opt.display_count == 0:
                board_add_images(board, 'combine', visuals, step + 1)
                board.add_scalar('gen_loss', gen_total_loss, step + 1)
                board.add_scalar('dcm_L1', dis_loss, step + 1)
                t = time.time() - iter_start_time
                print('step: %8d, time: %.3f, gen_loss: %.4f, dcm_loss: %.4f'
                      % (step + 1, t, gen_total_loss,dis_loss),flush=True)

            if (step + 1) % opt.save_count == 0:
                save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step + 1)))


def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))
   
    # create dataset 
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))
   
    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth'))
    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_tom(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'tom_final.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
        
  
    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
