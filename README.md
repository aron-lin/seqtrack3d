# SeqTrack3d: Exploring Sequence Information for Robust 3D Point Cloud Tracking 

Welcome to SeqTrack3d! 🚀 This repository houses the magic behind our simple yet powerful multi-frame point cloud target tracker that leverages an attention mechanism to enhance tracking robustness.

## 🛠 Environment Setup

Before we get started, let's set up our environment to ensure everything runs smoothly.

### PyTorch
Our project runs on PyTorch, and we recommend using version 2.0 or above. Follow the official guide to [install PyTorch](https://pytorch.org/get-started/previous-versions/).

If you're a fan of Docker, you can streamline the setup by using this 🐋 Docker image: [pytorch/pytorch:2.0.1](https://hub.docker.com/r/pytorch/pytorch/tags?page=&page_size=&ordering=&name=2.0.1), which saves you the trouble of manually installing PyTorch.

### Other Dependencies
To install the rest of the dependencies, simply clone our repo and run the installation command:

```
git clone https://github.com/aron-lin/seqtrack3d.git
cd seqtrack3d
pip install -r requirements.txt
```

### Dataset Preparation

Ready to feed your model with data? We utilize the nuscenes and waymo datasets. Follow the setup instructions on [Open3DSOT's GitHub](https://github.com/Ghostish/Open3DSOT?tab=readme-ov-file#setup) to get started.

## 🏋️ Training

Getting your SeqTrack3D model up and training is straightforward. Follow these steps to initiate training with your desired configuration. Open your terminal or command line interface and navigate to the project directory. Then, execute the following command:

```bash
python main.py --cfg cfgs/seqtrack3d_nuscenes.yaml --batch_size 4 --epoch 20 --seed 42 --tag "Enter_your_custom_tag_here_for_this_training_session"
```

Jumpstart your projects with our pretrained models available in the `pretrainedmodel` folder. It's all set for integrating our model into your applications!

## 🧪 Testing

To test your SeqTrack3D model with a pre-trained checkpoint, simply run the following command in your terminal or command line interface. Ensure you're in the project directory before executing the command:

```bash
python main.py --cfg cfgs/seqtrack3d_nuscenes.yaml --checkpoint pretrainedmodel/seqtrack_nuscenes_car_succ_62_prec_71.ckpt --test
```

## 📈 Viewing Results 

Check out the `output` folder in the root directory for training logs and testing results. Each experiment is neatly organized by the training/testing start time, dataset, and tag.

## 🙏 Acknowledgment

Special thanks to:

1. [Open3DSOT](https://github.com/Ghostish/Open3DSOT) for their fantastic tracking framework.
2. The team behind [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch) for their clear attention mechanism implementation.
3. [DETR](https://github.com/facebookresearch/detr) by Facebook Research for the influential ideas that have deeply benefited our project.

Each of these contributions has been pivotal in shaping our work. We're incredibly grateful for the community's shared knowledge and innovation.

## 📄 License 

Our project is open-sourced under the MIT license. Feel free to explore, modify, and share your innovations with the world.