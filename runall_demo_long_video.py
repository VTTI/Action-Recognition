# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import random
from collections import deque
from operator import itemgetter
import os
import os.path as osp
import yaml
import sys

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import collate, scatter

from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
THICKNESS = 1
LINETYPE = 1

EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]


def show_results_video_csv(result_queue,
                       text_info,
                       thr,
                       msg,
                       frame,
                       video_writer,
                       ind,
                       csv_file,
                       label_color=(255, 255, 255),
                       msg_color=(128, 128, 128)):
    if len(result_queue) != 0:
        text_info = {}
        results = result_queue.popleft()
        for i, result in enumerate(results):
            selected_label, score = result
            if score < thr:
                break
            csv_file.write(str(ind) + ","+str(i) + ","+ selected_label+ ","+str(round(score, 2)) + "\n")
            location = (0, 40 + i * 20)
            text = selected_label + ': ' + str(round(score, 2))
            text_info[location] = text
            cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                        label_color, THICKNESS, LINETYPE)
    elif len(text_info):
        for location, text in text_info.items():
            cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                        label_color, THICKNESS, LINETYPE)
    else:
        cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, msg_color,
                    THICKNESS, LINETYPE)
    video_writer.write(frame)
    return text_info


def show_results(model, data, label, input_vid, config):
    frame_queue = deque(maxlen=config['sample_length'])
    result_queue = deque(maxlen=1)

    cap = cv2.VideoCapture(input_vid)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    msg = 'Preparing action recognition ...'
    text_info = {}
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = (frame_width, frame_height)

    output_dir = '/vtti/scratch/asonth/PoseML_data6sec_for_mmaction2/predictions_slowfast'

    ind = 0
    output_vid = osp.join(output_dir, osp.splitext(osp.basename(input_vid))[0]+'_slowfast.mp4')
    video_writer = cv2.VideoWriter(output_vid, fourcc, fps, frame_size)
    prog_bar = mmcv.ProgressBar(num_frames)
    backup_frames = []

    output_csv = osp.join(output_dir, osp.splitext(osp.basename(input_vid))[0]+'_slowfast.csv')
    csv_file = open(output_csv, "w+")
    csv_file.write("frame_no,detection,label,confidence,x_min,y_min,x_max,y_max\n")

    while ind < num_frames:
        ind += 1
        prog_bar.update()
        ret, frame = cap.read()
        if frame is None:
            # drop it when encounting None
            continue
        backup_frames.append(np.array(frame)[:, :, ::-1])
        if ind == config['sample_length']:
            # provide a quick show at the beginning
            frame_queue.extend(backup_frames)
            backup_frames = []
        elif ((len(backup_frames) == config['inputStep']
               and ind > config['sample_length']) or ind == num_frames):
            # pick a frame from the backup
            # when the backup is full or reach the last frame
            chosen_frame = random.choice(backup_frames)
            backup_frames = []
            frame_queue.append(chosen_frame)

        ret, scores = inference(model, data, config, frame_queue)

        if ret:
            num_selected_labels = min(len(label), 5)
            scores_tuples = tuple(zip(label, scores))
            scores_sorted = sorted(
                scores_tuples, key=itemgetter(1), reverse=True)
            results = scores_sorted[:num_selected_labels]
            result_queue.append(results)

        
        text_info = show_results_video_csv(result_queue, text_info,
                                        config['threshold'], msg, frame,
                                        video_writer, ind, csv_file)        


    cap.release()
    cv2.destroyAllWindows()    


def inference(model, data, config, frame_queue):
    if len(frame_queue) != config['sample_length']:
        # Do no inference when there is no enough frames
        return False, None

    cur_windows = list(np.array(frame_queue))
    if data['img_shape'] is None:
        data['img_shape'] = frame_queue[0].shape[:2]

    cur_data = data.copy()
    cur_data['imgs'] = cur_windows
    cur_data = config['test_pipeline'](cur_data)
    cur_data = collate([cur_data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        cur_data = scatter(cur_data, [torch.device('cuda:2')])[0]
    with torch.no_grad():
        scores = model(return_loss=False, **cur_data)[0]

    if config['stride'] > 0:
        pred_stride = int(config['sample_length'] * config['stride'])
        for _ in range(pred_stride):
            frame_queue.popleft()

    # for case ``args.stride=0``
    # deque will automatically popleft one element

    return True, scores


def main():
    with open('poseml_long_video.yaml', 'r') as stream:
        config = yaml.load(stream)

    cfg = Config.fromfile(config['configFile'])
    config['cfg_options'] = {}
    cfg.merge_from_dict(config['cfg_options'])

    # # ----- Line added for size mismatch error wih TANet -----
    # # use default config with tanet (configs/recognition/tanet/tanet_r50_dense_1x1x8_100e_kinetics400_rgb.py)
    # cfg.model.cls_head.num_classes = 10
    # # --------------------------------------------------------

    model = init_recognizer(cfg, config['checkpoint'], device=torch.device('cuda:2'))
    data = dict(img_shape=None, modality='RGB', label=-1)
    with open(config['label'], 'r') as f:
        label = [line.strip() for line in f]

    # prepare test pipeline from non-camera pipeline
    cfg = model.cfg
    sample_length = 0
    pipeline = cfg.data.test.pipeline
    pipeline_ = pipeline.copy()
    for step in pipeline:
        if 'SampleFrames' in step['type']:
            sample_length = step['clip_len'] * step['num_clips']
            data['num_clips'] = step['num_clips']
            data['clip_len'] = step['clip_len']
            pipeline_.remove(step)
        if step['type'] in EXCLUED_STEPS:
            # remove step to decode frames
            pipeline_.remove(step)
    test_pipeline = Compose(pipeline_)

    assert sample_length > 0
    config['sample_length'] = sample_length
    config['test_pipeline'] = test_pipeline

    input_vid = sys.argv[1]

    show_results(model, data, label, input_vid, config)


if __name__ == '__main__':
    main()