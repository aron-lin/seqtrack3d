""" 
baseModel.py
Created by zenn at 2021/5/9 14:40
Modified by Aron Lin at Jun 6 17:39:22 CST 2023
"""

import torch
import torch.nn as nn
from easydict import EasyDict
import pytorch_lightning as pl
from datasets import points_utils
from utils.metrics import TorchSuccess, TorchPrecision, AverageMeter, TorchRuntime, TorchNumFrames
from utils.metrics import estimateOverlap, estimateAccuracy
from utils.waymo_metrics import estimateWaymoOverlap # only for waymo IOU
import torch.nn.functional as F
import numpy as np
from nuscenes.utils import geometry_utils

from datasets.misc_utils import get_history_frame_ids_and_masks,get_last_n_bounding_boxes
from datasets.misc_utils import generate_timestamp_prev_list

import time

class BaseModelMF(pl.LightningModule):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            config = EasyDict(kwargs)
        self.config = config
        self.train_dataloader_length = kwargs.get('train_dataloader_length', None)

        # testing metrics
        self.prec = TorchPrecision()
        self.success = TorchSuccess()
        self.runtime = TorchRuntime()

        self.prec_step = TorchPrecision()
        self.success_step = TorchSuccess()

        self.n_frames = TorchNumFrames()


    def configure_optimizers(self):
        if self.config.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=self.config.wd)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_step,
                                                    gamma=self.config.lr_decay_rate)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.config.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd,
                                         betas=(0.5, 0.999), eps=1e-06)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_step,
                                                    gamma=self.config.lr_decay_rate)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.config.optimizer.lower() == 'adamonecycle':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd,
                                         betas=(0.5, 0.999), eps=1e-06)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.max_lr,
                epochs=self.config.epoch,
                steps_per_epoch=self.train_dataloader_length)
            # The single-cycle learning rate needs to be explicitly updated step by step
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}} 
        else:
            raise ValueError("Invalid optimizer. Please choose from 'sgd', 'adam', or 'adamonecycle'.")



    def compute_loss(self, data, output):
        raise NotImplementedError

    def build_input_dict(self, sequence, frame_id, results_bbs, **kwargs):
        raise NotImplementedError

    def evaluate_one_sample(self, data_dict, ref_box):
        end_points = self(data_dict)

        estimation_box = end_points['aux_estimation_boxes']
        estimation_box_cpu = estimation_box.squeeze(0).detach().cpu().numpy()

        valid_mask = end_points['valid_mask'].squeeze(0).detach().cpu().numpy()

        if len(estimation_box.shape) == 3:
            best_box_idx = estimation_box_cpu[:, 4].argmax()
            estimation_box_cpu = estimation_box_cpu[best_box_idx, 0:4]

        candidate_box = points_utils.getOffsetBB(ref_box, estimation_box_cpu, degrees=self.config.degrees,
                                                 use_z=self.config.use_z,
                                                 limit_box=self.config.limit_box)

        return candidate_box,valid_mask

    def evaluate_one_sequence(self, sequence):
        """
        :param sequence: a sequence of annos {"pc": pc, "3d_bbox": bb, 'meta': anno}
        :return:
        """
        ious = []
        distances = []

        results_bbs = []
        for frame_id in range(len(sequence)):  # tracklet
            if frame_id == 0:
                # the first frame
                this_bb = sequence[frame_id]["3d_bbox"]
                prev_bb = sequence[frame_id]["3d_bbox"]
                results_bbs.append(this_bb)
                new_refboxs = [prev_bb] # Update in special cases
            else:
                this_bb = sequence[frame_id]["3d_bbox"]

                # construct input dict
                data_dict, ref_bb = self.build_input_dict(sequence, frame_id, results_bbs)
                # run the tracker
                if torch.sum(data_dict['points'][:,:,:3]) == 0:
                    results_bbs.append(ref_bb)
                    print("Empty pointcloud!")
                    new_refboxs = [ref_bb]
                else:
                    candidate_box,*_ = self.evaluate_one_sample(data_dict, ref_box=ref_bb)
                    results_bbs.append(candidate_box)

            
            this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                           up_axis=self.config.up_axis)

            this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                             up_axis=self.config.up_axis)
            ious.append(this_overlap)
            distances.append(this_accuracy)

        return ious, distances, results_bbs

    def validation_step(self, batch, batch_idx):
        sequence = batch[0]  # unwrap the batch with batch size = 1
        start_time = time.time()
        ious, distances, *_ = self.evaluate_one_sequence(sequence)
        end_time = time.time()
        runtime = end_time-start_time
        n_frames = len(sequence)

        self.success(torch.tensor(ious, device=self.device))
        self.prec(torch.tensor(distances, device=self.device))
        self.success_step(torch.tensor(ious, device=self.device))
        self.prec_step(torch.tensor(distances, device=self.device))

        self.log('success/test', self.success, on_epoch=True)
        self.log('precision/test', self.prec, on_epoch=True)

        self.log('success/test_step', self.success_step, on_step=True, on_epoch=False)
        self.log('precision/test_step', self.prec_step, on_step=True, on_epoch=False)

        self.runtime(torch.tensor(runtime, device=self.device),
                     torch.tensor(n_frames, device=self.device))

        self.success_step.reset()
        self.prec_step.reset()

    def on_validation_epoch_end(self):
        self.logger.experiment.add_scalars('metrics/test',
                                    {'success': self.success.compute(),
                                        'precision': self.prec.compute(),},
                                    global_step=self.global_step)

        self.logger.experiment.add_scalars('runtime',
                                       {'runtime':1.0/self.runtime.compute()},
                                       global_step=self.global_step)


    def test_step(self, batch, batch_idx):
        sequence = batch[0]  # unwrap the batch with batch size = 1
        start_time = time.time()
        ious, distances, result_bbs, *_= self.evaluate_one_sequence(sequence)
        end_time = time.time()
        runtime = end_time-start_time
        n_frames = len(sequence)

        
        self.success(torch.tensor(ious, device=self.device))
        self.prec(torch.tensor(distances, device=self.device))

        self.log('success/test', self.success,  on_epoch=True) 
        self.log('precision/test', self.prec,  on_epoch=True) 
        self.success_step(torch.tensor(ious, device=self.device))
        self.prec_step(torch.tensor(distances, device=self.device))
        self.n_frames(torch.tensor(n_frames, device=self.device))

        self.log('success/test_step', self.success_step, on_step=True, on_epoch=False)
        self.log('precision/test_step', self.prec_step, on_step=True, on_epoch=False)

        self.success_step.reset()
        self.prec_step.reset()


        self.runtime(torch.tensor(runtime, device=self.device),
                     torch.tensor(n_frames, device=self.device))
        self.logger.experiment.add_scalars('FPS', {'fps': 1.0/self.runtime.compute()}, global_step=batch_idx)

        return result_bbs

    def on_test_epoch_end(self):
        self.logger.experiment.add_scalars('metrics/test/current',
                                    {'success': self.success.compute(),
                                        'precision': self.prec.compute()},
                                    global_step=self.global_step)

        self.logger.experiment.add_scalars('metrics/fps',
                                    {'runtime':1.0/self.runtime.compute(),},
                                    global_step=self.global_step)
        self.logger.experiment.add_scalars('frames',
                                    {'frame':self.n_frames.compute(),},
                                    global_step=self.global_step)

class MotionBaseModelMF(BaseModelMF):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()

    def build_input_dict(self, sequence, frame_id, results_bbs): # Note: There may be cases of input with empty point clouds
        assert frame_id > 0, "no need to construct an input_dict at frame 0"

        prev_frame_ids, valid_mask = get_history_frame_ids_and_masks(frame_id,self.hist_num)
        prev_frames = [sequence[id] for id in prev_frame_ids]
        this_frame = sequence[frame_id]
        this_pc = this_frame['pc']
        bbox_size = this_frame['3d_bbox'].wlh
        prev_pcs = [frame['pc'] for frame in prev_frames]
        ref_boxs = get_last_n_bounding_boxes(results_bbs,valid_mask)
        num_hist = len(valid_mask)

        prev_frame_pcs = []
        for i, prev_pc in enumerate(prev_pcs):
            prev_frame_pc = points_utils.generate_subwindow_with_aroundboxs(prev_pc, ref_boxs[i], ref_boxs[0],
                                                        scale=self.config.bb_scale,
                                                        offset=self.config.bb_offset)
            prev_frame_pcs.append(prev_frame_pc)

        this_frame_pc = points_utils.generate_subwindow_with_aroundboxs(this_pc, ref_boxs[0], ref_boxs[0],
                                                        scale=self.config.bb_scale,
                                                        offset=self.config.bb_offset)

        # canonical_box = points_utils.transform_box(ref_boxs[0], ref_boxs[0])
        ref_boxs = [
            points_utils.transform_box(ref_box, ref_boxs[0]) for ref_box in ref_boxs
        ]

        prev_points_list = [points_utils.regularize_pc(prev_frame_pc.points.T, self.config.point_sample_size)[0] for prev_frame_pc in prev_frame_pcs] #采样到特定数量,这里的策略是在已有的点里面重复随机选，直到达到特定数量

        this_points, idx_this = points_utils.regularize_pc(this_frame_pc.points.T,
                                                           self.config.point_sample_size,
                                                           seed=1) 
        seg_mask_prev_list = [geometry_utils.points_in_box(ref_box, prev_points.T[:3,:], 1.25).astype(float) for ref_box,prev_points in zip(ref_boxs,prev_points_list)]#应当只考虑xyz特征

        # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
        # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
        if frame_id != 1:
            for seg_mask_prev in seg_mask_prev_list:
                # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
                # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
                seg_mask_prev[seg_mask_prev == 0] = 0.2
                seg_mask_prev[seg_mask_prev == 1] = 0.8
        seg_mask_this = np.full(seg_mask_prev_list[0].shape, fill_value=0.5)

        timestamp_prev_list = generate_timestamp_prev_list(valid_mask,self.config.point_sample_size)
        timestamp_this = np.full((self.config.point_sample_size, 1), fill_value=0.1)
        prev_points_list = [
        np.concatenate([prev_points, timestamp_prev, seg_mask_prev[:, None]],
                       axis=-1)
        for prev_points, timestamp_prev, seg_mask_prev in zip(
            prev_points_list, timestamp_prev_list, seg_mask_prev_list)
        ]

        this_points = np.concatenate([this_points, timestamp_this, seg_mask_this[:, None]], axis=-1)

        stack_points_list = prev_points_list + [this_points]
        stack_points = np.concatenate(stack_points_list, axis=0)

        ref_box_thetas = [
            ref_box.orientation.degrees * ref_box.orientation.axis[-1]
            if self.config.degrees else ref_box.orientation.radians *
            ref_box.orientation.axis[-1] for ref_box in ref_boxs
        ]
        ref_box_list = [
            np.append(ref_box.center,
                      theta).astype('float32') for ref_box, theta in zip(
                          ref_boxs, ref_box_thetas)
        ]
        ref_boxs_np = np.stack(ref_box_list, axis=0)

        data_dict = {"points": torch.tensor(stack_points[None, :], device=self.device, dtype=torch.float32), 
                     "ref_boxs":torch.tensor(ref_boxs_np[None, :], device=self.device, dtype=torch.float32), 
                     "valid_mask":torch.tensor(valid_mask, device=self.device, dtype=torch.float32).unsqueeze(0), 
                     "bbox_size":torch.tensor(bbox_size[None, :],device=self.device, dtype=torch.float32),
                     }

        if getattr(self.config, 'box_aware', False):
            stack_points_split = np.split(stack_points, num_hist + 1, axis=0)
            hist_points_list = stack_points_split[:num_hist] 
            candidate_bc_prev_list= [
                points_utils.get_point_to_box_distance(hist_points[:, :3], ref_box)
                for hist_points, ref_box in zip(hist_points_list, ref_boxs)
            ]
            candidate_bc_this = np.zeros_like(candidate_bc_prev_list[0])
            candidate_bc_prev_list = candidate_bc_prev_list + [candidate_bc_this]
            candidate_bc = np.concatenate(candidate_bc_prev_list, axis=0)

            data_dict.update({'candidate_bc': points_utils.np_to_torch_tensor(candidate_bc.astype('float32'),
                                                                              device=self.device)})
        return data_dict, results_bbs[-1]