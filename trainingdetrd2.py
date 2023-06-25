# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper   # the default mapper
from detectron2.data import detection_utils as utils
from detectron2.data import build_detection_train_loader
setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2 
from cv2 import imshow

import os
from detectron2.utils.visualizer import ColorMode
# import some common detectron2 utilities
from detectron2.engine import DefaultTrainer

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import random
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode


#change for custom dataset
from detectron2.data.datasets import register_coco_instances
#example of registering a custom dataset
#register_coco_instances("fruits_nuts", {}, "./data/trainval.json", "./data/images")
#from https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html example
#reference to 
#register_coco_instances("my_dataset", {}, "json_annotation.json", "path/to/image/dir")

#register_coco_instances("fruits_nuts", {}, r"C:\Users\phil\Documents\GitHub\testdata\trainval.json", r"C:\Users\phil\Documents\GitHub\testdata\images")


from detectron2.data import DatasetCatalog

# later, to access the data:
#ata: List[Dict] = DatasetCatalog.get("my_dataset")

#print(dataset_dicts)


#isualise - doesnt work for some reason
'''
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    print(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=fruits_nuts_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow(vis.get_image()[:, :, ::-1])
'''

import torch 
print(torch.cuda.is_available())

from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
register_coco_instances("fruits_nuts", {}, r"C:\Users\phil\detectron2\datasets\bridgedefect\defect_set-1.json", r"C:\Users\phil\detectron2\datasets\bridgedefect\images")
dataset_dicts = DatasetCatalog.get("fruits_nuts")
fruits_nuts_metadata = MetadataCatalog.get("fruits_nuts")

dataset_dicts = DatasetCatalog.get("fruits_nuts")
'''
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=fruits_nuts_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow(out.get_image()[:, :, ::-1])
'''

#to convert model, follow the following;

def detr_train():
    
    from d2.detr import add_detr_config
    from d2.train_net import Trainer
    #https://huggingface.co/transformers/v4.9.2/model_doc/detr.html#transformers.DetrConfig
    cfg = get_cfg()
    add_detr_config(cfg)
    cfg.merge_from_file("d2/configs/detr_256_6_6_torchvision.yaml")

    cfg.MODEL.WEIGHTS = "converted_model.pth"

    cfg.DATASETS.TRAIN = ("fruits_nuts",)
    cfg.DATASETS.TEST = ()

    cfg.OUTPUT_DIR = './outputs/'

    cfg.MODEL.WEIGHTS = "converted_model.pth"
    #cfg.MODEL.DETR.HIDDEN_DIM = 256
    #cfg.MODEL.DETR.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.DETR.NUM_CLASSES = 20 #should be the number of classes + 1 (21) https://towardsdatascience.com/training-detr-on-your-own-dataset-bcee0be05522
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = 100  # example only has one class (ballon)

    # from detectron2.engine import DefaultTrainer
    

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = DefaultTrainer(cfg) 
    trainer = Trainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()



    #tensorboard --logdir="C:\Users\phil\Documents\GitHub\detr\outputs"


'''
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

evaluator = COCOEvaluator("custom_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "custom_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
'''


def filter_predictions_from_outputs(outputs,
                                    threshold=0.1,
                                    verbose=True):

  predictions = outputs["instances"].to("cpu")

  if verbose:
    print(list(predictions.get_fields()))

  # Reference: https://github.com/facebookresearch/detectron2/blob/7f06f5383421b847d299b8edf480a71e2af66e63/detectron2/structures/instances.py#L27
  #
  #   Indexing: ``instances[indices]`` will apply the indexing on all the fields
  #   and returns a new :class:`Instances`.
  #   Typically, ``indices`` is a integer vector of indices,
  #   or a binary mask of length ``num_instances``

  indices = [i
            for (i, s) in enumerate(predictions.scores)
            if s >= threshold
            ]

  filtered_predictions = predictions[indices]

  return filtered_predictions

def detr_predict():
    from d2.detr import add_detr_config

    cfg = get_cfg()

    add_detr_config(cfg)
    cfg.merge_from_file("d2/configs/detr_256_6_6_torchvision.yaml")

    cfg.DATASETS.TRAIN = ("custom_train",)
    cfg.DATASETS.TEST = ("custom_val",)

    cfg.OUTPUT_DIR = './outputs/'

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.DETR.NUM_CLASSES = 1

    predictor = DefaultPredictor(cfg)

    dataset_name = cfg.DATASETS.TRAIN[0]


    threshold = 0.8

    
    for d in random.sample(dataset_dicts, 3):    
        im = cv2.imread(d["file_name"])    
        outputs = predictor(im)

        filtered_predictions = filter_predictions_from_outputs(outputs,
                                                            threshold=threshold)
        
        v = Visualizer(im[:, :, ::-1],
                    metadata=fruits_nuts_metadata, 
                    scale=0.5, 
        )
        out = v.draw_instance_predictions(filtered_predictions)
        img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)
        #cv2.imshow(out.get_image()[:, :, ::-1])
        plt.imshow(img)
        plt.show()
    return









def train():
    #DatasetCatalog.register("my_dataset", dataset_dicts)
    #print(dataset_dicts)



    cfg.DATASETS.TRAIN = ("fruits_nuts",)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 4
    persistent_workers=True

    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_C4_3x/137849393/model_final_f97cb7.pkl"  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.025
    cfg.SOLVER.MAX_ITER = (
        10000
    )  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        750
    )  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20  # 3 classes (date, fig, hazelnut)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8 

    os.makedirs(r"C:\Users\phil\Documents\GitHub\output", exist_ok=True)

    trainer = MyTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    return cfg

def predict():
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 1
    #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (248)  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20  # 3 classes (date, fig, hazelnut)



    cfg.DATASETS.TRAIN = ("fruits_nuts",)
    
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    print('output path:',cfg.OUTPUT_DIR) 
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    

    
    cfg.MODEL.WEIGHTS = r"C:\Users\phil\Documents\GitHub\detectrontest\output\model_final.pth"
    print('weights:',cfg.MODEL.WEIGHTS) 
    cfg.DATASETS.TEST = ("fruits_nuts", )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES 
    
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import Visualizer, ColorMode
    import matplotlib.pyplot as plt
    #import cv2.cv2 as cv2
    

    folder_path = r"C:\Users\phil\detectron2\datasets\bridgedefect\testing"
    output_path = r"C:\Users\phil\detectron2\datasets\bridgedefect\testing\output"
    for filename in os.listdir(folder_path):
    
        test_data = [{'file_name': filename,
                  'image_id': 2}]
        print('output test data:',test_data)
        predictor = DefaultPredictor(cfg)
        im = cv2.imread(folder_path + '/' + test_data[0]["file_name"])

        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                    metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                    scale=0.5,
                    instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        print(out)
        img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)
        suffix = '_processed'
        print(folder_path + r'\output' + test_data[0]["file_name"])
        #plt.imshow(img)
        #plt.show()
        cv2.imwrite(output_path + test_data[0]["file_name"], img)



class MyPredictor(DefaultPredictor):

    @classmethod
    def build_Predict_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[
            T.Resize((800, 800)),
            ],
            )
        return build_detection_train_loader(cfg, mapper=mapper)
        
    #tensorboard --logdir="C:\Users\phil\Documents\GitHub\detectrontest\output"
class MyTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[
            T.Resize((800, 800)),
            T.RandomRotation(angle= 90, sample_style="range"),
            T.RandomBrightness(0.8, 1.2),
            T.RandomSaturation(0.8, 1.2),
            T.RandomContrast(0.8, 1.2),
            T.RandomLighting(1),
            T.RandomFlip(prob=0.5),
            T.RandomCrop("absolute", (640, 640))
            ],
            instance_mask_format="polygon"
            )
        return build_detection_train_loader(cfg, mapper=mapper)


def main():
    #detr_predict()
    detr_train()
    #predict()
    return

if __name__ == '__main__':
    main()

