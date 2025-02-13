"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.S3DISDataLoader import S3DISDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
from pytorch3d import ops
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint_train', type=int, default=1024, help='Point Number when training')
    parser.add_argument('--npoint_eval', type=int, default=1024, help='Point Number when evaluate')
    parser.add_argument('--nanchor', type=int, default=128, help='Anchor Number [default: 128]')
    parser.add_argument('--anchor_type', type=str, default='fixed', help='Anchor type [fiexd or learn]')
    parser.add_argument('--dist_type', type=str, default='cdist', help='cdist or eudist')
    parser.add_argument('--nsegment', type=int, default=1, help='Seperate cdist calculate GPU load')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')

    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/s3dis/stanford_indoor3d/'
    NUM_CLASSES = 13
    NUM_POINT_TRAIN = args.npoint_train
    NUM_POINT_EVAL = args.npoint_eval
    BATCH_SIZE = args.batch_size

    print("start loading training data ...")
    TRAIN_DATASET = S3DISDataset(split='train', data_root=root, num_point=NUM_POINT_TRAIN, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None, bs=BATCH_SIZE)
    print("start loading test data ...")
    TEST_DATASET = S3DISDataset(split='test', data_root=root, num_point=NUM_POINT_EVAL, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None, bs=BATCH_SIZE)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=10,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=10,
                                                 pin_memory=True, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''ANCHORS SETTING'''
    NUM_ANCHORS = args.nanchor
    NUM_SEG = args.nsegment
    try:
        anchors = np.load(str(checkpoints_dir) + '/anchors.npy')
        log_string('Use original anchors')
    except:
        anchors_size = (NUM_ANCHORS, 3)
        # np.random.seed = 49
        anchors = np.random.random(anchors_size)
        np.save(str(checkpoints_dir) + '/anchors.npy', anchors)
        log_string('Generate anchors')

    anchors = torch.Tensor(anchors)
    
    # anchors = anchors.repeat(BATCH_SIZE, 1, 1)
    
    if args.anchor_type == 'learn':
        anchors = anchors.float().cuda()
        anchors.requires_grad = True
        # anchors = torch.nn.parameter.Parameter(anchors, True)  # add anchor as a learnable param
        # anchors = torch.Tensor(anchors)
    else:
        anchors = anchors.float().cuda()
    
    anchors_B = anchors.unsqueeze(0)
    if NUM_SEG == 1:
        anchors_B = anchors_B.expand(BATCH_SIZE, NUM_ANCHORS, 3)
    else:
        assert(BATCH_SIZE%NUM_SEG == 0)
        anchors_B = anchors_B.expand(BATCH_SIZE//NUM_SEG, NUM_ANCHORS, 3)

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))
    
    # TODO:修改传入数据的shape
    classifier = MODEL.get_model(NUM_CLASSES, NUM_ANCHORS + 9).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/model.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])

        # start_epoch = checkpoint['epoch']
        # best_iou = checkpoint['best_iou']
        start_epoch = 0
        best_iou = 0

        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        best_iou = 0
        classifier = classifier.apply(weights_init)

    """Leraning method"""
    # 1. anchor + cdist learn together
    if args.anchor_type == 'learn':
        opt_params = [{"params":classifier.parameters()},{"params":anchors}]
    else:
        opt_params = classifier.parameters()

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            # [classifier.parameters(), anchors],
            # [{"params":classifier.parameters()},{"params":anchors}],
            opt_params,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(opt_params, lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum
    def convex_dist(points1, points2, pcd, M=64):
        '''
        Args:
            points1 - BxSx3 
            points2 - BxSx3
            pcd - BxNx3
        Rets:
            cdist - BxS
        '''

        # print(points1.shape[1], points2.shape[1])
        assert(points1.shape[1]==points2.shape[1])
        B, S, N = points1.shape[0], points1.shape[1], pcd.shape[1]
        dist = torch.norm(points2 - points1, dim=-1, keepdim=True) #BxSx1
        n = (points2 - points1) / dist #BxSx3
        positions = (torch.arange(0, M) / (M-1))[None, None, :, None].expand(B, S, -1, 1).to(points1.device) #BxSxMx1
        dd = dist[:, :, None, :].expand(-1, -1, M, -1) #BxSxMx1
        nn = n[:, :, None, :].expand(-1, -1, M, -1) #BxSxMx3
        pp1 = points1[:, :, None, :].expand(-1, -1, M, -1) #BxSxMx3
        sampled_points = pp1 + nn * dd * positions #BxSxMx3
        pp = pcd[:, None, :, :].expand(-1, S, -1, -1) #BxSxNx3
        nn_dist, _, _= ops.knn_points(sampled_points.reshape(B*S, M, 3), pp.reshape(B*S, N, 3))
        nn_dist = nn_dist.view(B, S, M, 1)
        cdist = torch.sqrt(torch.amax(nn_dist.squeeze(-1), dim=-1)) #BxS
        return cdist
    def nn_convex_dist(points1, points2, pcd, M=100):
        """
        Args:
            points1 - BxS_1x3
            points2 - BxS_2x3
            pcd - BxNx3

        Rets:
            dists_1 - BxS_1
            dists_2 - BxS_2
        """
        # TODO: WATCH GPU INCREASE
        B, S1, S2 = points1.shape[0], points1.shape[1], points2.shape[1]
        pp1 = points1[:, :, None, :].expand(-1, -1, S2, -1).reshape(B, -1 ,3) #BxS1S2x3
        pp2 = points2[:, None, :, :].expand(-1, S1, -1, -1).reshape(B, -1 ,3) #BxS1S2x3
        # TODO: 只对pcd进行downsample
        cdists = convex_dist(pp1, pp2, pcd, M).view(B, S1, -1) #BxS1xS2
        # dists_1, dists_2 = cdists.amin(dim=-1), cdists.amin(dim=-1)
        return cdists

    def eu_dist(points1, pcd):
        """
        Args:
            points1 - BxSx3
            pcd - BxNx3

        Rets:
            dists_1 - BxNxS
        """
        points1_eu = points1.unsqueeze(1)
        points1_eu = points1_eu.permute(0, 3, 1, 2)
        pcd_eu = pcd.unsqueeze(2)
        pcd_eu = pcd_eu.permute(0, 3, 1, 2)
        edist = torch.pairwise_distance(pcd_eu, points1_eu, p=2)
        return edist

    def norm_dist(points1, pcd):
        """
        calculate normal vectors
        use open3d
        calculate on whole pcd, dataloader
        return: BxNx3
        """


    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0

    # 在训练最开始之前实例化一个GradScaler对象
    scaler = GradScaler()
    
    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, (points, target, coord_max_xyz) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            # test
            # if i >= 10:
            #     break

            optimizer.zero_grad()
            with autocast():
                points = points.data.numpy()
                # 方法一 不将点云移动中心计算cdist
                points_norm = points[:, :, 9:12]
                points = points[:, :, :9]
                # 方法二 将点云移动中心计算cdist
                # points_norm = points[:, :, 9:12]
                # TODO: 暂时取消rotation
                # points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
                points = torch.Tensor(points)
                points_norm = torch.Tensor(points_norm)
                points, target = points.float().cuda(), target.long().cuda()
                points_norm = points_norm.float().cuda()
                # 分布计算cdist
                
                # TODO: CHCECK GRAD
                # 整体计算cdist
                # TODO: 对pcd3 降采样256传入. idex抽帧
                
                if args.dist_type == 'cdist':
                    if NUM_SEG == 1:
                        # 整体计算cdist
                        c_distance = nn_convex_dist(points_norm, anchors_B, points_norm, M=64) # B*N*num_anchors
                    else:
                        # 分块计算cdist
                        # B = points_norm.shape[0]
                        assert(BATCH_SIZE%NUM_SEG == 0)
                        seg_len = BATCH_SIZE//NUM_SEG
                        for i in range(NUM_SEG):
                            start = i*seg_len
                            end = start + seg_len
                            c_distance_seg = nn_convex_dist(points_norm[start:end, :, :], anchors_B, points_norm[start:end, :, :], M=64) # B*N*num_anchors
                            if i == 0:
                                c_distance = c_distance_seg
                                continue
                            c_distance = torch.cat((c_distance, c_distance_seg), 0)
                    points = torch.cat((points, c_distance), 2) # B*N*(3 + num_anchors)
                elif args.dist_type == 'eudist':
                    eu_distance = eu_dist(anchors_B, points_norm)
                    points = torch.cat((points, eu_distance), 2) # B*N*(3 + num_anchors)
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)

            # no autocast
            # Scales loss. 为了梯度放大.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT_TRAIN)
            loss_sum += loss
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))


        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target, coord_max_xyz) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                # test
                # if i >= 10:
                #     break
                with autocast():
                    points = points.data.numpy()
                    # 方法一 不将点云移动中心计算cdist
                    points_norm = points[:, :, 9:12]
                    points = points[:, :, :9]
                    # 方法二 将点云移动中心计算cdist
                    # points_norm = points[:, :, 9:12]
                    # TODO: 暂时取消rotation
                    # points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
                    points = torch.Tensor(points)
                    points_norm = torch.Tensor(points_norm)
                    points, target = points.float().cuda(), target.long().cuda()
                    points_norm = points_norm.float().cuda()
                    # anchors_B = anchors.unsqueeze(0)
                    # anchors_B = anchors_B.expand(BATCH_SIZE, NUM_ANCHORS, 3)

                    if args.dist_type == 'cdist':
                        if NUM_SEG == 1:
                            #整体计算cdist
                            c_distance = nn_convex_dist(points_norm, anchors_B, points_norm, M=64) # B*N*num_anchors
                        else:
                            # 分块计算cdist
                            B = points_norm.shape[0]
                            assert(B%NUM_SEG == 0)
                            seg_len = B//NUM_SEG
                            for i in range(NUM_SEG):
                                start = i*seg_len
                                end = start + seg_len
                                c_distance_seg = nn_convex_dist(points_norm[start:end, :, :], anchors_B, points_norm[start:end, :, :], M=64) # B*N*num_anchors
                                if i == 0:
                                    c_distance = c_distance_seg
                                    continue
                                c_distance = torch.cat((c_distance, c_distance_seg), 0)
                        points = torch.cat((points, c_distance), 2) # B*N*(3 + num_anchors)
                    elif args.dist_type == 'eudist':
                        eu_distance = eu_dist(anchors_B, points_norm)
                        points = torch.cat((points, eu_distance), 2) # B*N*(3 + num_anchors)
                    points = points.transpose(2, 1)

                    seg_pred, trans_feat = classifier(points)
                    pred_val = seg_pred.contiguous().cpu().data.numpy()
                    seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                    batch_label = target.cpu().data.numpy()
                    target = target.view(-1, 1)[:, 0]
                    loss = criterion(seg_pred, target, trans_feat, weights)
                    loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT_EVAL)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))

            # save model 
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': mIoU
            }
            if args.anchor_type == 'learn':
                anchors_np = anchors.cpu().numpy()
                np.save(str(checkpoints_dir) + '/anchors.npy', anchors_np)
                # 为了检测anchor是否变化
                # np.save(str(checkpoints_dir) + '/anchors' + str(epoch) + '.npy', anchors_np)
            torch.save(state, savepath)
            log_string('Saving model....')
            
            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model' + str(epoch) + '_' +str(mIoU)[:5] +'.pth'
                if args.anchor_type == 'learn':
                    np.save(str(checkpoints_dir) + '/best_anchor' + str(epoch) + '.npy', anchors_np)
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1


if __name__ == '__main__':
    os.chdir(sys.path[0])
    args = parse_args()
    main(args)
