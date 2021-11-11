import os

class Config():
    def __init__(self) -> None:
        # model config
        self.depth = 34
        self.input_size = 512
        self.num_classes = 80
        self.dataset_format = 'coco'  # ['coco', 'voc']

        # train config
        self.epoches = 200
        self.batch_size = 64
        self.lr = 1e-4

        # path config
        self.train_data_root = '/home/workspace/chencheng/code/dataset/MSCOCO/train2014'
        self.train_label_path = '/home/workspace/chencheng/code/dataset/MSCOCO/annotations/instances_train2014.json'
        self.test_data_root = '/home/workspace/chencheng/code/dataset/MSCOCO/val2014/'
        self.test_label_path = '/home/workspace/chencheng/code/dataset/MSCOCO/annotations/instances_val2014.json'
        self.voc_root = '/home/workspace/chencheng/code/dataset/VOC2007'

        self.log_path = 'log'
        self.ckpt_path = 'ckpt/resnet34_coco'

        self.load_from = '/home/workspace/chencheng/code/SimpleCenternet/ckpt/resnet34_coco/epoch_1_loss_0.24.pth'
        self._check_root()
    
    def _check_root(self):
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)