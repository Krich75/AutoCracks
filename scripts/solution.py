import numpy as np
import scipy as sp
import pandas as pd
import itertools
import enum
import sys
import os

import scipy.interpolate

import skimage
import skimage.morphology
import skimage.filters

from sklearn.externals import joblib


class Type(enum.IntEnum):
    BACKGROUND = 0
    BONE = 1
    JOINT = 2


def find_bones(pixel_map):
    node_types = {}
    bone_graph = {}
    bone_map = np.zeros(pixel_map.shape, dtype=np.uint32)
    last_bone_idx = 10
    visited = set()
    
    def create_new_bone(type_):
        nonlocal last_bone_idx
        blob_index = last_bone_idx
        last_bone_idx += 1
        node_types[blob_index] = type_
        return blob_index
    
    def add_edge(from_index, to_index):
        bone_graph.setdefault(from_index, set()).add(to_index)
        bone_graph.setdefault(to_index, set()).add(from_index)
    
    def flood(y, x, expected_type, index):
        bone_map[y, x] = index
        for dy in (-1, 0, 1):
            new_y = y + dy
            if new_y < 0 or new_y >= pixel_map.shape[0]:
                continue
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                new_x = x + dx
                if new_x < 0 or new_x >= pixel_map.shape[1]:
                    continue
                if bone_map[new_y, new_x] != 0:
                    continue
                new_type = pixel_map[new_y, new_x]
                if new_type == Type.BACKGROUND:
                    continue
                if new_type == expected_type:
                    new_index = index
                else:
                    new_index = create_new_bone(new_type)
                    add_edge(index, new_index)
                flood(new_y, new_x, new_type, new_index)
                
    for pixel_y, pixel_x in np.transpose(pixel_map.nonzero()):
        if bone_map[pixel_y, pixel_x] != 0:
            continue

        bone_type = pixel_map[pixel_y, pixel_x]
        bone_index = create_new_bone(bone_type)
        flood(pixel_y, pixel_x, bone_type, bone_index)
        
    return bone_map, node_types, bone_graph
    

def process_joint(joint_idx, graph, map_,
                  max_bone_points=10,
                  min_angle=3*np.pi/4,
                  debug=False):
    bones = graph.get(joint_idx, set())
    bone_points = {}
    is_joint = (map_ == joint_idx)
    visited = set()
    bone_vectors = {}
    for bone in bones:
        for y, x in np.transpose(np.nonzero(map_ == bone)):
            near_joint = np.sum(is_joint[y-1:y+2, x-1:x+2])
            if near_joint:
                if debug:
                    print('Start bone', bone, 'from', y, x)
                curr_pos = curr_y, curr_x = y, x
                visited.add(curr_pos)
                bone_points[bone] = [curr_pos]
                queue = [curr_pos]
                for curr_step in range(1, max_bone_points):
                    new_queue = []
                    for curr_pos in queue:
                        for dy, dx in [
                            (-1, 0),
                            (1, 0),
                            (0, -1),
                            (0, 1),
                            (-1, -1),
                            (-1, 1),
                            (1, -1),
                            (1, 1),
                        ]:
                            curr_y, curr_x = curr_pos
                            new_pos = new_y, new_x = curr_y + dy, curr_x + dx
#                             print(curr_pos, '->',  new_pos)
                            if not (
                                0 <= new_y < map_.shape[0]
                                and 0 <= new_x < map_.shape[1]
                            ):
#                                 print('Invalid')
                                continue
                            if new_pos in visited:
#                                 print('Already visited')
                                continue
                            if map_[new_y, new_x] != bone:
#                                 print('Wrong bone',  map_[new_y, new_x], 'Expected', bone)
                                continue
#                             print('Append', new_pos)
                            new_queue.append(new_pos)
                            visited.add(new_pos)
#                             bone_points.setdefault(bone, []).append(curr_pos)
                    if not new_queue:
                        break
                    queue = new_queue
                if curr_step < 3:
                    break
                bone_points[bone].extend(queue)
                if debug:
                    print('Bone', bone, 'points', bone_points[bone])
                vector = np.array([
                    bone_points[bone][0][1] - bone_points[bone][-1][1],
                    bone_points[bone][0][0] - bone_points[bone][-1][0],
                ])
                vector_norm = np.linalg.norm(vector)
                if vector_norm == 0:
                    vector_norm = 1
                vector = vector.astype(float) / vector_norm
                bone_vectors[bone] = vector
                break
    if debug:
        for idx, points in bone_points.items():
            arrow(points[-1][1],
                  points[-1][0],
                  -points[-1][1] + points[0][1],
                  -points[-1][0] + points[0][0],
                  color='r',
                  width=0.5)
        print(bone_vectors)
    bones_to_merge = set(bone_vectors)
    while True:
        max_angle = None
        merged_bones = None
        for first_bone, second_bone in itertools.combinations(bones_to_merge, 2):
            angle = np.arccos(bone_vectors[first_bone] @ bone_vectors[second_bone])
            if debug:
                print(first_bone, second_bone, angle)
            if (max_angle is None or angle > max_angle) and angle > min_angle:
                merged_bones = first_bone, second_bone
                max_angle = angle
        if max_angle is None:
            break
        else:
            yield merged_bones
            for b in merged_bones:
                bones_to_merge.remove(b)


class DSU:
    def __init__(self):
        self.roots = {}
    
    def __getitem__(self, item):
        return self.get_root(item)
    
    def get_root(self, idx):
        root = self.roots.setdefault(idx, idx)
        if root != idx:
            root = self.get_root(root)
            self.roots[idx] = root
        return root

    def merge(self, first_idx, second_idx):
        if np.random.rand() < 0.5:
            self._merge(first_idx, second_idx)
        else:
            self._merge(second_idx, first_idx)
    
    def _merge(self, child_idx, parent_idx):
        self.roots[self.get_root(child_idx)] = self.get_root(parent_idx)


def extract_segments(
    image,
    skeleton_blur_std=1,
    joint_std=1.5,
    joint_thereshold=0.003,
    min_angle=3*np.pi/4,
    max_bone_joint_points=10,
    debug=False,
):
    raw_image_binarized = (image > 0)
    if skeleton_blur_std:
        blured_image = skimage.filters.gaussian(raw_image_binarized, skeleton_blur_std)
    else:
        blured_image = raw_image_binarized
    binarized_image = blured_image > skimage.filters.thresholding.threshold_otsu(blured_image)
    skeleton = skimage.morphology.skeletonize(binarized_image)
       
    corners = np.zeros(skeleton.shape, dtype=np.uint8)
    corners += skeleton
    corners[1:, :] += skeleton[:-1, :]
    corners[:-1, :] += skeleton[1:, :]
    corners[:, 1:] += skeleton[:, :-1]
    corners[:, :-1] += skeleton[:, 1:]

    corners[1:, 1:] += skeleton[:-1, :-1]
    corners[:-1, 1:] += skeleton[1:, :-1]
    corners[1:, :-1] += skeleton[:-1, 1:]
    corners[:-1, :-1] += skeleton[1:, 1:]
    
    corners = (skeleton & (corners > 3))
    
    skeleton_with_joints = np.zeros(skeleton.shape)
    skeleton_with_joints[skeleton] = Type.BONE
    skeleton_with_joints[
        skeleton & (skimage.filters.gaussian(corners, joint_std) > joint_thereshold)
    ] = Type.JOINT
    
    bone_map, node_types, bone_graph = find_bones(skeleton_with_joints)
    
    crack_idx = DSU()
    
    if debug:
        imshow(skeleton)
    
    for idx, t in node_types.items():
        if t == Type.JOINT:
            for i, j in process_joint(idx, bone_graph, bone_map,
                                      min_angle=min_angle,
                                      max_bone_points=max_bone_joint_points,
                                      debug=debug):
                crack_idx.merge(i, j)
    
    merged_skeleton = bone_map.copy()
    for y, x in np.transpose(merged_skeleton.nonzero()):
        merged_skeleton[y, x] = crack_idx[bone_map[y, x]]
    
    bones_only = merged_skeleton.copy()
    for idx, v in node_types.items():
        if v == Type.JOINT:
            bones_only[bones_only == idx] = 0

    layer_mask = raw_image_binarized.copy().astype(np.uint32)
    layer_mask_nonzero = scipy.interpolate.NearestNDInterpolator(
        np.transpose(bones_only.nonzero()), bones_only[bones_only != 0],
    )(np.transpose(raw_image_binarized.nonzero()))
    layer_mask[layer_mask != 0] = layer_mask_nonzero
    
    if debug:
        for i, t in node_types.items():
            if t == 2:
                y, x = np.mean(np.where(bone_map == i), axis=1).astype(int)
                text(x, y, str(i), color='r', size=20)
        show()
    
    return layer_mask


def soft_propogate_labels(previous_layer, next_level, last_label,
                          fill_ratio=0.9,
                          interpolate_ratio=0.5,
                          n_interpolation_options=2):
    result = next_level.copy()
    labels = np.unique(next_level)
    for label in labels:
        if label == 0:
            continue
        label_mask = next_level == label
        possible_true_labels_ratio = (
            np.bincount(previous_layer[label_mask])
            / label_mask.sum()
        )
        best_match = np.argmax(possible_true_labels_ratio)
        match_ratio = possible_true_labels_ratio[best_match]
        if best_match != 0 and match_ratio >= fill_ratio:
            true_label = best_match
            result[next_level == label] = true_label
        elif possible_true_labels_ratio[0] < interpolate_ratio:
            next_layer_label_mask = next_level == label
            options = np.argsort(possible_true_labels_ratio[1:])[-n_interpolation_options:] + 1
            previous_layer_label_mask = (
                next_layer_label_mask
                & (np.isin(previous_layer, options))
            )
            previous_layer_labels = previous_layer[previous_layer_label_mask]
            next_layer_mask_nonzero = scipy.interpolate.NearestNDInterpolator(
                np.transpose(previous_layer_label_mask.nonzero()),
                previous_layer_labels,
            )(np.transpose(next_layer_label_mask.nonzero()))
            result[next_level == label] = next_layer_mask_nonzero
        else:
            last_label += 1
            true_label = last_label
            result[next_level == label] = true_label
    return result, last_label


def rle_encode(data):
    raw = data.ravel()
    diffs = raw[1:] != raw[:-1]
    ends = np.where(diffs)[0]
    ends = np.hstack([ends, raw.shape[0] - 1])
    lens = np.hstack([
        ends[0] + 1,
        ends[1:] - ends[:-1],
    ])
    starts = np.hstack([0, np.cumsum(lens)[:-1]])
    labels = raw[starts]
    return np.transpose([labels, lens])


def rle_decode(data, shape):
    labels = data.T[0]
    len = data.T[1]
    ends = np.cumsum(len)
    starts = ends - len
    result = np.zeros(np.sum(len), dtype=np.uint16)
    for start, end, label in zip(starts, ends, labels):
        result[start:end] = label
    return result.reshape(shape)


def get_labels(crack_probability, n_init_layers):
    sys.setrecursionlimit(1000000)
    np.random.seed(10)
    
    extracted_layers = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(extract_segments)(crack_probability[i])
        for i in range(crack_probability.shape[0])
    )
    
    helper_mask = extract_segments(
        skimage.morphology.opening(
            np.mean(crack_probability[:n_init_layers], axis=0) > 0
        ),
        0,
        joint_std=2,
        min_angle=np.pi/2,
        joint_thereshold=0.002,
    )

    max_label = np.max(helper_mask)
    fixed_layers = [helper_mask]
    for i in range(0, len(extracted_layers)):
        layer_fixed, max_label = soft_propogate_labels(
            fixed_layers[i],
            extracted_layers[i],
            max_label,
            fill_ratio=0.75,
            interpolate_ratio=0.4, 
            n_interpolation_options=3,
        )
        fixed_layers.append(layer_fixed)
    return np.array(fixed_layers[1:])


def run():
    import argparse
    parser = argparse.ArgumentParser(
        description='Script for crack segmentation',
        epilog='Author: amorgun https://github.com/amorgun',
    )
    parser.add_argument(
        '-s', '--shape',
        dest='shape',
        help='shape of the probability data in (z,y,x) order. Used only with .csv input files. Example: "333,480,640"'
    )
    parser.add_argument(
        'probability_file',
        help='location of the probability file input in .npy. format.'
    )
    parser.add_argument(
        '-t',
        default=50,
        type=int,
        dest='n_init_layers',
        help='number of layers used for building the initial labels.'
    )
    parser.add_argument(
        '-z',
        default=None,
        dest='output_npy',
        help='location of the output in .npy format.'
    )
    parser.add_argument(
        'output_file',
        help='location of the output in .csv format.'
    )
    args = parser.parse_args()
    if args.probability_file.endswith('.csv'):
        input_shape = tuple(int(i) for i in args.shape.split(','))
        crack_probability = rle_decode(
            np.loadtxt(args.probability_file, int, delimiter=','),
            input_shape,
        )
    else:
        crack_probability = np.load(args.probability_file)
    labels = get_labels(crack_probability, args.n_init_layers)
    rle_encoded = rle_encode(labels)
    np.savetxt(args.output_file,  rle_encoded, '%d', ',')
    if args.output_npy:
        np.save(args.output_npy, labels)


if __name__ == '__main__':
    run()
