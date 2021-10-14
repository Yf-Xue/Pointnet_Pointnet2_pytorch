import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset


class S3DISDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=5000, test_area=5, block_size=1.0, sample_rate=1.0, transform=None, bs=16):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        self.batch_size = bs
        self.anchors = [] # 随机观察点
        self.global_max = 0.0

        # if anchors_type == 1:  # case 1: anchors are fixed 128
        #     anchors_size = (128, 3)
        #     # np.random.seed = 49
        #     self.anchors = np.random.random(anchors_size)
        # # TODO: FIX THESE TWO PART
        # elif anchors_type == 2:  # case 2: anchors are learnable
            
        # elif anchors_type == 3:  # case 3: anchors are proposed??
            
            
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(13)

        # TODO: here need to be further modified
        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyzrgbl, N*7
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(14))
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        self.room_coord_max_xyz = np.amax(self.room_coord_max, axis=1)
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print(self.labelweights)
        # sample_prob = num_point_all / np.sum(num_point_all)
        # num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        # room_idxs = []
        # for index in range(len(rooms_split)):
        #     room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        room_idxs = range(len(rooms_split))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]

        # 随机降采样
        if self.num_point > N_points:
            selected_point_idxs = np.random.choice(N_points, self.num_point, replace=True)
        else:
            selected_point_idxs = np.random.choice(N_points, self.num_point, replace=False)
        # normalize
        selected_points = points[selected_point_idxs, :3]  # num_point * 3: xyz
        current_points = np.zeros((self.num_point, 3))
        # room_coord_max_global = np.amax(self.room_coord_max[room_idx], axis=1)
        current_points[:, 0] = selected_points[:, 0] / self.room_coord_max_xyz[room_idx]
        current_points[:, 1] = selected_points[:, 1] / self.room_coord_max_xyz[room_idx]
        current_points[:, 2] = selected_points[:, 2] / self.room_coord_max_xyz[room_idx]      
        current_labels = labels[selected_point_idxs]

        # points + convex_feature
        # current_cdist = convex_dist(current_points, self.anchors, self.batch_size, M=100)
        # current_points = np.concatenate((current_points, current_cdist), axis=-1)
        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)

class ScannetDatasetWholeScene():
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area=5, stride=0.5, block_size=1.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.scene_points_num = []
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is -1]
        else:
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is not -1]
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(data[:, 6])
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
        self.room_coord_max_xyz = np.amax(self.room_coord_max, axis=1)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(13)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(14))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:,:6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        # grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        # grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]),  np.array([])
        N_points = points.shape[0]

        # 随机降采样
        if self.num_point > N_points:
            selected_point_idxs = np.random.choice(N_points, self.num_point, replace=True) # 随机降采样
        else:
            selected_point_idxs = np.random.choice(N_points, self.num_point, replace=False)
        # normalize
        selected_points = points[selected_point_idxs, :3]  # num_point * 3: xyz
        normlized_xyz = np.zeros((point_size, 3))
        coord_max_xyz = np.amax(coord_max, axis=1)
        normlized_xyz[:, 0] = selected_points[:, 0] / coord_max_xyz
        normlized_xyz[:, 1] = selected_points[:, 1] / coord_max_xyz
        normlized_xyz[:, 2] = selected_points[:, 2] / coord_max_xyz
        label_room = labels[selected_point_idxs].astype(int)
        sample_weight = self.labelweights[label_room]

        data_room = normlized_xyz.reshape((-1, self.block_points, normlized_xyz.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = selected_point_idxs.reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)

if __name__ == '__main__':
    data_root = '../data/s3dis/stanford_indoor3d/'
    num_point, test_area, block_size, sample_rate = 20000, 5, 1.0, 1

    point_data = S3DISDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    point_data_len = point_data.__len__()
    pick_list = np.random.choice(point_data_len, 5, replace=False)
    label_color = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255],[0, 255, 255], [255, 0, 255], [255, 255, 0], [255, 255, 255],
                           [100, 149, 237], [46, 139, 87], [189, 183, 107], [160, 82, 45], [250, 128, 114], [130, 130, 130]])

    for i in pick_list:
        pick = point_data.__getitem__(i)
        pick_point = pick[0]
        pick_label = pick[1]
        
        for j in range(pick_label.shape[0]):
            if j == 0:
                color_label = label_color[int(pick_label[j])]
                # color_label.reshape((1,3))
                continue
            cur_color = label_color[int(pick_label[j])]
            color_label = np.vstack((color_label,cur_color))
        pick_data = np.hstack((pick_point, color_label))
        np.savetxt('../data/s3dis/visual_data/20k/' + str(i) + '.txt', pick_data)
        


    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=2, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()