from datasets import sampler, \
                    nuscenes_lidar_mf,  \
                    waymo_data_mf
                    



def get_dataset(config, type='train', **kwargs):
    if config.dataset == 'nuscenes_mf':
        data = nuscenes_lidar_mf.NuScenesMFDataset(path=config.path,
                                             split=kwargs.get('split', 'train_track'),
                                             category_name=config.category_name,
                                             version=config.version,
                                             key_frame_only=True if type != 'test' else config.key_frame_only,
                                             # can only use keyframes for training
                                             preloading=config.preloading,
                                             preload_offset=config.preload_offset if type != 'test' else -1,
                                             min_points=1 if kwargs.get('split', 'train_track') in
                                                             [config.val_split, config.test_split] else -1,
                                            hist_num = config.hist_num)
    elif config.dataset == 'waymo_mf':
        data = waymo_data_mf.WaymoDataset(path=config.path,
                                       split=kwargs.get('split', 'train'),
                                       category_name=config.category_name,
                                       preloading=config.preloading,
                                       preload_offset=config.preload_offset,
                                       tiny=config.tiny,
                                       hist_num = config.hist_num)
    else:
        data = None
  
    if type.lower() == 'train_motion_mf':
        return sampler.MotionTrackingSamplerMF(dataset=data,
                                             config=config)
    
    else:
        return sampler.TestTrackingSampler(dataset=data, config=config)
