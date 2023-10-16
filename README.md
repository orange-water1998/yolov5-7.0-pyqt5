



YOLOv5 🚀 是世界上最受欢迎的视觉AI，代表 <a href="https://ultralytics.com">Ultralytics</a> 对未来视觉人工智能方法的开源研究，结合数千小时研发过程中积累的经验教训和最佳实践。


我们希望这里的资源将帮助您充分利用YOLOv5。请浏览YOLOv5 <a href="https://docs.ultralytics.com/yolov5">Docs</a> for details, raise an issue on <a href="https://github.com/ultralytics/yolov5/issues/new/choose">GitHub</a> for support, and join our <a href="https://ultralytics.com/discord">Discord</a> community for questions and discussions!

To request an Enterprise License please complete the form at [Ultralytics Licensing](https://ultralytics.com/license).






YOLOv5 has been designed to be super easy to get started and simple to learn. We prioritize real-world results.

<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/155040763-93c22a27-347c-4e3c-847a-8094621d3f4e.png"></p>
<details>
  <summary>YOLOv5-P5 640 Figure</summary>

<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/155040757-ce0934a3-06a6-43dc-a979-2edbbd69ea0e.png"></p>
</details>
<details>
  <summary>Figure Notes</summary>

- **COCO AP val** denotes mAP@0.5:0.95 metric measured on the 5000-image [COCO val2017](http://cocodataset.org) dataset over various inference sizes from 256 to 1536.
- **GPU Speed** measures average inference time per image on [COCO val2017](http://cocodataset.org) dataset using a [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) V100 instance at batch-size 32.
- **EfficientDet** data from [google/automl](https://github.com/google/automl) at batch size 8.
- **Reproduce** by `python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n6.pt yolov5s6.pt yolov5m6.pt yolov5l6.pt yolov5x6.pt`

</details>

### Pretrained Checkpoints

| Model                                                                                           | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | Speed<br><sup>CPU b1<br>(ms) | Speed<br><sup>V100 b1<br>(ms) | Speed<br><sup>V100 b32<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>@640 (B) |
| ----------------------------------------------------------------------------------------------- | --------------------- | -------------------- | ----------------- | ---------------------------- | ----------------------------- | ------------------------------ | ------------------ | ---------------------- |
| [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt)              | 640                   | 28.0                 | 45.7              | **45**                       | **6.3**                       | **0.6**                        | **1.9**            | **4.5**                |
| [YOLOv5s](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt)              | 640                   | 37.4                 | 56.8              | 98                           | 6.4                           | 0.9                            | 7.2                | 16.5                   |
| [YOLOv5m](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt)              | 640                   | 45.4                 | 64.1              | 224                          | 8.2                           | 1.7                            | 21.2               | 49.0                   |
| [YOLOv5l](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt)              | 640                   | 49.0                 | 67.3              | 430                          | 10.1                          | 2.7                            | 46.5               | 109.1                  |
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt)              | 640                   | 50.7                 | 68.9              | 766                          | 12.1                          | 4.8                            | 86.7               | 205.7                  |
|                                                                                                 |                       |                      |                   |                              |                               |                                |                    |                        |
| [YOLOv5n6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n6.pt)            | 1280                  | 36.0                 | 54.4              | 153                          | 8.1                           | 2.1                            | 3.2                | 4.6                    |
| [YOLOv5s6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s6.pt)            | 1280                  | 44.8                 | 63.7              | 385                          | 8.2                           | 3.6                            | 12.6               | 16.8                   |
| [YOLOv5m6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m6.pt)            | 1280                  | 51.3                 | 69.3              | 887                          | 11.1                          | 6.8                            | 35.7               | 50.0                   |
| [YOLOv5l6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l6.pt)            | 1280                  | 53.7                 | 71.3              | 1784                         | 15.8                          | 10.5                           | 76.8               | 111.4                  |
| [YOLOv5x6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x6.pt)<br>+ [TTA] | 1280<br>1536          | 55.0<br>**55.8**     | 72.7<br>**72.7**  | 3136<br>-                    | 26.2<br>-                     | 19.4<br>-                      | 140.7<br>-         | 209.8<br>-             |

<details>
  <summary>Table Notes</summary>

- All checkpoints are trained to 300 epochs with default settings. Nano and Small models use [hyp.scratch-low.yaml](https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-low.yaml) hyps, all others use [hyp.scratch-high.yaml](https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-high.yaml).
- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](http://cocodataset.org) dataset.<br>Reproduce by `python val.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`
- **Speed** averaged over COCO val images using a [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) instance. NMS times (~1 ms/img) not included.<br>Reproduce by `python val.py --data coco.yaml --img 640 --task speed --batch 1`
- **TTA** [Test Time Augmentation](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation) includes reflection and scale augmentations.<br>Reproduce by `python val.py --data coco.yaml --img 1536 --iou 0.7 --augment`

</details>

## <div align="center">Segmentation</div>

Our new YOLOv5 [release v7.0](https://github.com/ultralytics/yolov5/releases/v7.0) instance segmentation models are the fastest and most accurate in the world, beating all current [SOTA benchmarks](https://paperswithcode.com/sota/real-time-instance-segmentation-on-mscoco). We've made them super simple to train, validate and deploy. See full details in our [Release Notes](https://github.com/ultralytics/yolov5/releases/v7.0) and visit our [YOLOv5 Segmentation Colab Notebook](https://github.com/ultralytics/yolov5/blob/master/segment/tutorial.ipynb) for quickstart tutorials.

<details>
  <summary>Segmentation Checkpoints</summary>

<div align="center">
<a align="center" href="https://ultralytics.com/yolov5" target="_blank">
<img width="800" src="https://user-images.githubusercontent.com/61612323/204180385-84f3aca9-a5e9-43d8-a617-dda7ca12e54a.png"></a>
</div>

We trained YOLOv5 segmentations models on COCO for 300 epochs at image size 640 using A100 GPUs. We exported all models to ONNX FP32 for CPU speed tests and to TensorRT FP16 for GPU speed tests. We ran all speed tests on Google [Colab Pro](https://colab.research.google.com/signup) notebooks for easy reproducibility.

| Model                                                                                      | size<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | Train time<br><sup>300 epochs<br>A100 (hours) | Speed<br><sup>ONNX CPU<br>(ms) | Speed<br><sup>TRT A100<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>@640 (B) |
| ------------------------------------------------------------------------------------------ | --------------------- | -------------------- | --------------------- | --------------------------------------------- | ------------------------------ | ------------------------------ | ------------------ | ---------------------- |
| [YOLOv5n-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-seg.pt) | 640                   | 27.6                 | 23.4                  | 80:17                                         | **62.7**                       | **1.2**                        | **2.0**            | **7.1**                |
| [YOLOv5s-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-seg.pt) | 640                   | 37.6                 | 31.7                  | 88:16                                         | 173.3                          | 1.4                            | 7.6                | 26.4                   |
| [YOLOv5m-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m-seg.pt) | 640                   | 45.0                 | 37.1                  | 108:36                                        | 427.0                          | 2.2                            | 22.0               | 70.8                   |
| [YOLOv5l-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l-seg.pt) | 640                   | 49.0                 | 39.9                  | 66:43 (2x)                                    | 857.4                          | 2.9                            | 47.9               | 147.7                  |
| [YOLOv5x-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x-seg.pt) | 640                   | **50.7**             | **41.4**              | 62:56 (3x)                                    | 1579.2                         | 4.5                            | 88.8               | 265.7                  |

- All checkpoints are trained to 300 epochs with SGD optimizer with `lr0=0.01` and `weight_decay=5e-5` at image size 640 and all default settings.<br>Runs logged to https://wandb.ai/glenn-jocher/YOLOv5_v70_official
- **Accuracy** values are for single-model single-scale on COCO dataset.<br>Reproduce by `python segment/val.py --data coco.yaml --weights yolov5s-seg.pt`
- **Speed** averaged over 100 inference images using a [Colab Pro](https://colab.research.google.com/signup) A100 High-RAM instance. Values indicate inference speed only (NMS adds about 1ms per image). <br>Reproduce by `python segment/val.py --data coco.yaml --weights yolov5s-seg.pt --batch 1`
- **Export** to ONNX at FP32 and TensorRT at FP16 done with `export.py`. <br>Reproduce by `python export.py --weights yolov5s-seg.pt --include engine --device 0 --half`

</details>

<details>
  <summary>Segmentation Usage Examples &nbsp;<a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/segment/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a></summary>

### Train

YOLOv5 segmentation training supports auto-download COCO128-seg segmentation dataset with `--data coco128-seg.yaml` argument and manual download of COCO-segments dataset with `bash data/scripts/get_coco.sh --train --val --segments` and then `python train.py --data coco.yaml`.

```bash
# Single-GPU
python segment/train.py --data coco128-seg.yaml --weights yolov5s-seg.pt --img 640

# Multi-GPU DDP
python -m torch.distributed.run --nproc_per_node 4 --master_port 1 segment/train.py --data coco128-seg.yaml --weights yolov5s-seg.pt --img 640 --device 0,1,2,3
```

### Val

Validate YOLOv5s-seg mask mAP on COCO dataset:

```bash
bash data/scripts/get_coco.sh --val --segments  # download COCO val segments split (780MB, 5000 images)
python segment/val.py --weights yolov5s-seg.pt --data coco.yaml --img 640  # validate
```

### Predict

Use pretrained YOLOv5m-seg.pt to predict bus.jpg:

```bash
python segment/predict.py --weights yolov5m-seg.pt --source data/images/bus.jpg
```

```python
model = torch.hub.load(
    "ultralytics/yolov5", "custom", "yolov5m-seg.pt"
)  # load from PyTorch Hub (WARNING: inference not yet supported)
```

| ![zidane](https://user-images.githubusercontent.com/26833433/203113421-decef4c4-183d-4a0a-a6c2-6435b33bc5d3.jpg) | ![bus](https://user-images.githubusercontent.com/26833433/203113416-11fe0025-69f7-4874-a0a6-65d0bfe2999a.jpg) |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |

### Export

Export YOLOv5s-seg model to ONNX and TensorRT:

```bash
python export.py --weights yolov5s-seg.pt --include onnx engine --img 640 --device 0
```

</details>
