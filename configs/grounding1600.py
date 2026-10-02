_base_ = '/dev/shm/aicomp_grounding/vendor/mmdetection/configs/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco.py'
classes = ('person','boat','animal','seat','sign','bicycle','car','ball','light','garbage can','uav','tricycle')
model = dict(language_model=dict(name='/dev/shm/aicomp_grounding/bert'), backbone=dict(frozen_stages=2, with_cp=True), bbox_head=dict(num_classes=12), test_cfg=dict(max_per_img=100))
load_from = '/dev/shm/aicomp_grounding/grounding_swinb.pth'
train_pipeline = [dict(type='LoadImageFromFile'),dict(type='LoadAnnotations',with_bbox=True),dict(type='RandomFlip',prob=.5),dict(type='RandomChoice',transforms=[[dict(type='RandomChoiceResize',scales=[(480,1280),(640,1280),(736,1280),(800,1280)],keep_ratio=True)],[dict(type='RandomChoiceResize',scales=[(400,1600),(500,1600)],keep_ratio=True),dict(type='RandomCrop',crop_type='absolute_range',crop_size=(384,600),allow_negative_crop=True),dict(type='RandomChoiceResize',scales=[(640,1280),(800,1280)],keep_ratio=True)]]),dict(type='PackDetInputs',meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor','flip','flip_direction','text','custom_entities'))]
test_pipeline = [dict(type='LoadImageFromFile'),dict(type='FixScaleResize',scale=(800,1280),keep_ratio=True),dict(type='LoadAnnotations',with_bbox=True),dict(type='PackDetInputs',meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor','text','custom_entities'))]
train_dataloader = dict(batch_size=2,num_workers=2,persistent_workers=True,dataset=dict(data_root='/home/fbohan/AIC/data/train/',ann_file='/home/fbohan/AIC/data/annotations/train1600.json',data_prefix=dict(img=''),metainfo=dict(classes=classes),filter_cfg=dict(filter_empty_gt=False),pipeline=train_pipeline,return_classes=True))
val_dataloader = dict(batch_size=1,num_workers=2,persistent_workers=True,dataset=dict(data_root='/home/fbohan/AIC/data/train/',ann_file='/home/fbohan/AIC/data/annotations/val400.json',data_prefix=dict(img=''),metainfo=dict(classes=classes),pipeline=test_pipeline,return_classes=True))
test_dataloader=val_dataloader
val_evaluator=dict(type='GroundingCocoMetric',ann_file='/home/fbohan/AIC/data/annotations/val400.json',metric='bbox',classwise=True,proposal_nums=(1,10,100),outfile_prefix='/home/fbohan/AIC/checkpoints/gpu6_storage/grounding/validation')
test_evaluator=val_evaluator
optim_wrapper=dict(_delete_=True,type='OptimWrapper',accumulative_counts=4,optimizer=dict(type='AdamW',lr=5e-5,weight_decay=1e-4),clip_grad=dict(max_norm=.1,norm_type=2),paramwise_cfg=dict(custom_keys={'backbone':dict(lr_mult=.1),'language_model':dict(lr_mult=0),'absolute_pos_embed':dict(decay_mult=0)}))
train_cfg=dict(type='EpochBasedTrainLoop',max_epochs=12,val_interval=1)
param_scheduler=[dict(type='LinearLR',start_factor=.01,by_epoch=False,begin=0,end=200),dict(type='CosineAnnealingLR',eta_min=5e-7,by_epoch=True,begin=0,end=12,T_max=12)]
default_hooks=dict(checkpoint=dict(type='CheckpointHook',interval=1,max_keep_ckpts=1,save_best='coco/bbox_mAP',save_optimizer=True,save_param_scheduler=True),logger=dict(type='LoggerHook',interval=8))
work_dir='/home/fbohan/AIC/checkpoints/gpu6_storage/grounding/grounding1600'
randomness=dict(seed=20261002,deterministic=False)
auto_scale_lr=dict(enable=False,base_batch_size=8)
