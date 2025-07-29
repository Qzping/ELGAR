# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2021.02.12

from typing import List, Dict
from icecream import ic

can_display = True

try:
    from psbody.mesh import Mesh

    from body_visualizer.mesh.psbody_mesh_cube import points_to_cubes
    from body_visualizer.mesh.psbody_mesh_sphere import points_to_spheres
    from body_visualizer.tools.mesh_tools import rotateXYZ
    from body_visualizer.tools.vis_tools import colors
    from psbody.mesh import MeshViewers

except Exception as e:
    print(e)
    print('psbody.mesh based visualization could not be started. skipping ...')
    can_display = False

from torch import nn
import torch

from human_body_prior.tools.model_loader import load_model

import numpy as np

from human_body_prior.tools.omni_tools import copy2cpu as c2c

from human_body_prior.tools.omni_tools import log2file

from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.omni_tools import flatten_list
from torch.optim.lr_scheduler import StepLR


def unify_body_mappings(dataset='openpose25',with_hands_id = False):
    """Unify different joint definations.

    Output unified defination:
        ['Nose'(0),
        'RShoulder'(1), 'RElbow'(2), 'RWrist'(3),
        'LShoulder'(4), 'LElbow'(5), 'LWrist'(6),
        'RHip'(7), 'RKnee'(8), 'RAnkle'(9),
        'LHip'(10), 'LKnee'(11), 'LAnkle'(12),
        'REye'(13), 'LEye'(14),
        'REar'(15), 'LEar'(16),
        'RBigToe'(17), 'RSmallToe'(18), 'RHeel'(19),
        'LBigToe'(20), 'LSmallToe'(21), 'LHeel'(22),
        for wrist rot fitting
        'RIndex1'(23), 'RMiddle1'(24), 'RRing1'(25), 'RPinky1'(26),
        'LIndex1'(27), 'LMiddle1'(28), 'LRing1'(29), 'LPinky1'(30),
        ]

    Args:
    dataset: `openpose25`, `coco`(17) and `smpl`.
    Returns:
    a list of indexs that maps the joints to a unified definition.
    """
    if dataset == 'smplx':
        # note SMPL needs to be "left-right flipped" to be consistent
        # with others
        mappings = np.array([
                                55,
                                18, 20,
                                19, 21,  # shoulders excluded
                                1, 4, 7,
                                2, 5, 8,
                                57, 56,
                                59, 58,
                                60, 61, 62,
                                63, 64, 65,
                                0
                            ], dtype=np.int32)
        if with_hands_id:
            mappings_hands_id = np.array([
                                            25, 26, 27, 67,
                                            28, 29, 30, 68,
                                            31, 32, 33, 70,
                                            34, 35, 36, 69,
                                            37, 38, 39, 66,

                                            40, 41, 42, 72,
                                            43, 44, 45, 73, 
                                            46, 47, 48, 75,
                                            49, 50, 51, 74,
                                            52, 53, 54, 71,
                                        ], dtype=np.int32)

            mappings = np.hstack((mappings,mappings_hands_id), dtype=np.int32)
        return mappings
    
    elif dataset == 'coco':
        mappings = np.array([
                                0,
                                7, 9,
                                8, 10,  # shoulders excluded
                                11, 13, 15,
                                12, 14, 16,
                                1, 2,
                                3, 4,
                                17, 18, 19,
                                20, 21, 22,
                                -1
                            ], dtype=np.int32)
        if with_hands_id:
            mappings_hands_id = np.array([
                                            92, 93, 94, 95,
                                            96, 97, 98, 99,
                                            100, 101, 102, 103,
                                            104, 105, 106, 107,
                                            108, 109, 110, 111,

                                            113, 114, 115, 116,
                                            117, 118, 119, 120, 
                                            121, 122, 123, 124,
                                            125, 126, 127, 128,
                                            129, 130, 131, 132,
                                        ], dtype=np.int32)

            mappings = np.hstack((mappings,mappings_hands_id), dtype=np.int32)
        return mappings
    else:
        raise ValueError(f'{dataset} is not supported')


def visualize(points, bm_f, mvs, kpts_colors, verbosity=2, logger=None):
    from human_body_prior.tools.omni_tools import log2file

    if logger is None: logger = log2file()

    def view(opt_objs, body_v, virtual_markers, opt_it):
        if verbosity <= 0: return
        opt_objs_cpu = {k: c2c(v) for k, v in opt_objs.items()}

        total_loss = np.sum([np.sum(v) for k, v in opt_objs_cpu.items()])
        message = 'it {} -- [total loss = {:.2e}] - {}'.format(opt_it, total_loss, ' | '.join(
            ['%s = %2.2e' % (k, np.sum(v)) for k, v in opt_objs_cpu.items()]))
        logger(message)
        if verbosity > 1 and can_display:
            bs = body_v.shape[0]
            np.random.seed(100)
            frame_ids = list(range(bs)) if bs <= len(mvs) else np.random.choice(bs, size=len(mvs),
                                                                                replace=False).tolist()
            if bs > len(mvs): message += ' -- [frame_ids: {}]'.format(frame_ids)
            for dispId, fId in enumerate(
                    frame_ids):  # check for the number of frames in mvs and show a randomly picked number of frames in body if there is more to show than row*cols available
                new_body_v = rotateXYZ(body_v[fId], [-90, 0, 0])

                orig_mrk_mesh = points_to_spheres(rotateXYZ(c2c(points[fId]), [-90, 0, 0]), radius=0.01,
                                                  point_color=kpts_colors)
                virtual_markers_mesh = points_to_cubes(rotateXYZ(virtual_markers[fId], [-90, 0, 0]), radius=0.01,
                                                       point_color=kpts_colors)
                new_body_mesh = Mesh(new_body_v, bm_f, vc=colors['grey'])

                # linev = rotateXYZ(np.hstack((c2c(points[fId]), virtual_markers[fId])).reshape((-1, 3)), [-90,0,0])
                # linee = np.arange(len(linev)).reshape((-1, 2))
                # ll = Lines(v=linev, e=linee)
                # ll.vc = (ll.v * 0. + 1) * np.array([0.00, 0.00, 1.00])
                # mvs[dispId].set_dynamic_lines([ll])

                # orig_mrk_mesh = points_to_spheres(data_pc, radius=0.01, vc=colors['blue'])
                mvs[dispId].set_dynamic_meshes([orig_mrk_mesh, virtual_markers_mesh])
                mvs[dispId].set_static_meshes([new_body_mesh])

            mvs[0].set_titlebar(message)
            # if out_dir is not None: mv.save_snapshot(os.path.join(out_dir, '%05d_it_%.5d.png' %(frame_id, opt_it)))

    return view

class AdamInClosure():
    def __init__(self, var_list, lr, max_iter=100, tolerance_change=1e-5):
        self.optimizer = torch.optim.Adam(var_list, lr)
        self.max_iter = max_iter
        self.tolerance_change = tolerance_change

    def step(self, closure):
        prev_loss = None
        for it in range(self.max_iter):
            loss = closure()
            self.optimizer.step()
            if prev_loss is None:
                prev_loss = loss
                continue
            if torch.isnan(loss):
                # breakpoint()
                break
            if abs(loss - prev_loss) < self.tolerance_change:
                print('abs(loss - prev_loss) <  self.tolerance_change')
                break

    def zero_grad(self):
        self.optimizer.zero_grad()

class SGDInClosure():
    def __init__(self, var_dict, base_lr, max_iter=100, tolerance_change=1e-5):
        self.optimizer = torch.optim.SGD(params=list(var_dict.values()), lr=base_lr)
        self.var_dict = var_dict
        self.max_iter = max_iter
        self.base_lr = base_lr
        self.tolerance_change = tolerance_change

    def get_optimizer(self, var_dict, step, base_lr):
        """Setup opimizer with a warm up learning rate."""
        if step < 100:
            optimizer = torch.optim.SGD([
                {'params': [var_dict['trans']], 'lr': base_lr * 0.1},
                {'params': [var_dict['scaling']], 'lr': base_lr * 0.1},
                {'params': [var_dict['root_orient']], 'lr': base_lr * 0.0001},
                {'params': [var_dict['poZ_body']], 'lr': base_lr * 0},
                {'params': [var_dict['betas']], 'lr': 0.0},
            ])

        elif step < 500:
            optimizer = torch.optim.SGD([
                {'params': [var_dict['trans']], 'lr': base_lr * 10},
                {'params': [var_dict['scaling']], 'lr': base_lr * 10},
                {'params': [var_dict['root_orient']], 'lr': base_lr * 0.01},
                {'params': [var_dict['poZ_body']], 'lr': base_lr * 0},
                {'params': [var_dict['betas']], 'lr': 0.0},
            ])

        else:
            optimizer = torch.optim.SGD([
                {'params': [var_dict['trans']], 'lr': base_lr * 10},
                {'params': [var_dict['scaling']], 'lr': base_lr * 10},
                {'params': [var_dict['root_orient']], 'lr': base_lr * 0.001},
                {'params': [var_dict['poZ_body']], 'lr': base_lr * 0.0025},
                {'params': [var_dict['betas']], 'lr': 0.005},
            ])
        return optimizer

    def step(self, closure):
        prev_loss = None
        for it in range(self.max_iter):
            loss = closure()
            self.optimizer = self.get_optimizer(var_dict=self.var_dict, step=it, base_lr=self.base_lr)
            self.optimizer.step()
            if prev_loss is None:
                prev_loss = loss
                continue
            if torch.isnan(loss):
                # breakpoint()
                break
            # if abs(loss - prev_loss) < self.tolerance_change:
            #     print('loss:', loss)
            #     print('abs(loss - prev_loss) <  self.tolerance_change')
            #     break

    def zero_grad(self):
        self.optimizer.zero_grad()


def ik_fit(optimizer,  source_kpts_model, static_vars, vp_model, extra_params={}, on_step=None, gstep=0):
    data_loss = extra_params.get('data_loss', torch.nn.MSELoss(reduction='mean'))

    def fit(weights, free_vars):
        fit.gstep += 1
        optimizer.zero_grad()

        if free_vars['use_vposer']:
            free_vars['pose_body'] = vp_model.decode(free_vars['poZ_body'])['pose_body'].contiguous().view(-1, 63)
        else:
            if 'pose_body_without_wrist' in list(static_vars.keys()):
                free_vars['pose_body'] = torch.cat((static_vars['pose_body_without_wrist'][:,:-3],free_vars['poZ_body'],static_vars['pose_body_without_wrist'][:,-3:]),axis = 1)
            else:
                free_vars['pose_body'] = free_vars['poZ_body']

        nonan_mask = torch.isnan(free_vars['poZ_body']).sum(-1) == 0

        opt_objs = {}
        body_vars = {**free_vars, **static_vars}
        res = source_kpts_model(body_vars)

        if 'scaling' in list(free_vars.keys()):
            source_kpts_mapped = res['source_kpts'][:, unify_body_mappings(dataset='smplx'), :].float()
            target_kpts_mapped = static_vars['target_kpts'][:, unify_body_mappings(dataset='coco'), :].float()

            if fit.gstep == 1:
                links = np.array([[0, 13],
                                [0, 14],
                                [13, 15],
                                [14, 16],
                                [1, 2],
                                [2, 3],
                                [4, 5],
                                [5, 6],
                                [1, 4],
                                [7, 8],
                                [8, 9],
                                [10, 11],
                                [11, 12],
                                [7, 10]])
                source_kpts_mapped_detached = source_kpts_mapped.detach().cpu().numpy()
                target_kpts_mapped_detached = target_kpts_mapped.detach().cpu().numpy()
                source_diffs = source_kpts_mapped_detached[:, links[:, 0], :] - source_kpts_mapped_detached[:, links[:, 1], :]
                target_diffs = target_kpts_mapped_detached[:, links[:, 0], :] - target_kpts_mapped_detached[:, links[:, 1], :]
                source_distances = np.linalg.norm(source_diffs, axis=2).sum(axis=1)
                target_distances = np.linalg.norm(target_diffs, axis=2).sum(axis=1)

                scales =  target_distances / source_distances
                scales = torch.from_numpy(scales).reshape(-1, 1).to(target_kpts_mapped.device)
                
                with torch.no_grad():
                    free_vars['scaling'] *= scales

            source_kpts_mapped = free_vars['scaling'][:, np.newaxis] * source_kpts_mapped
        else:
            source_kpts_mapped = res['source_kpts'][:, unify_body_mappings(dataset='smplx',with_hands_id=True), :].float()
            target_kpts_mapped = static_vars['target_kpts'][:, unify_body_mappings(dataset='coco',with_hands_id=True), :].float()

            source_kpts_mapped = static_vars['scaling'][:, np.newaxis] * source_kpts_mapped
        

        if 'betas' in list(static_vars.keys()):
            if 'pose_body_without_wrist' in list(static_vars.keys()):
                source_kpts_mapped_without_only_finger = source_kpts_mapped[:,23:43,:]
                target_kpts_mapped_without_only_finger = target_kpts_mapped[:,23:43,:]
                opt_objs['data'] = data_loss(source_kpts_mapped_without_only_finger, target_kpts_mapped_without_only_finger)

            else:
                source_kpts_mapped_without_lh_elbow = torch.cat((source_kpts_mapped[:,:2,:],source_kpts_mapped[:,3,:].unsqueeze(1)*100,source_kpts_mapped[:,4:5,:],source_kpts_mapped[:,6,:].unsqueeze(1)*100,source_kpts_mapped[:,6:,:]),axis=1)
                target_kpts_mapped_without_lh_elbow  = torch.cat((target_kpts_mapped[:,:2,:],target_kpts_mapped[:,3,:].unsqueeze(1)*100,target_kpts_mapped[:,4:5,:],target_kpts_mapped[:,6,:].unsqueeze(1)*100,target_kpts_mapped[:,6:,:]),axis=1)

                opt_objs['data'] = data_loss(source_kpts_mapped_without_lh_elbow, target_kpts_mapped_without_lh_elbow)

            
        else:
            opt_objs['data'] = data_loss(source_kpts_mapped, target_kpts_mapped)
        # source_vel = source_kpts_mapped[1:]-source_kpts_mapped[:-1]
        # target_vel = target_kpts_mapped[1:]-target_kpts_mapped[:-1]
        opt_objs['vel'] = data_loss(source_kpts_mapped[1:]-source_kpts_mapped[:-1],target_kpts_mapped[1:]-target_kpts_mapped[:-1])

        # regularization term
        if 'betas' in list(free_vars.keys()):
            opt_objs['betas'] = torch.pow(free_vars['betas'][nonan_mask], 2).sum()
        
        if 'pose_body_without_wrist' not in list(static_vars.keys()):
            opt_objs['poZ_body'] = torch.pow(free_vars['poZ_body'][nonan_mask], 2).sum()

        opt_objs = {k: opt_objs[k] * v for k, v in weights.items() if k in opt_objs.keys()}
        loss_total = torch.sum(torch.stack(list(opt_objs.values())))

        if fit.gstep == 0:
            ic(loss_total)
        # breakpoint()

        loss_total.backward()

        if on_step is not None:
            on_step(opt_objs, c2c(res['body'].v), c2c(res['source_kpts']), fit.gstep)

        fit.free_vars = {k: v for k, v in free_vars.items()}  # if k in IK_Engine.fields_to_optimize}
        # fit.nonan_mask = nonan_mask
        fit.final_loss = loss_total
        fit.static_vars = static_vars
        return loss_total

    fit.gstep = gstep
    fit.final_loss = None
    fit.free_vars = {}
    
    # fit.nonan_mask = None
    return fit


class IK_Engine(nn.Module):

    def __init__(self,
                 vposer_expr_dir: str,
                 data_loss,
                 optimizer_args: dict = {'type': 'ADAM'},
                 stepwise_weights: List[Dict] = [{'data': 10., 'poZ_body': .01, 'betas': .5}],
                 display_rc: tuple = (2, 1),
                 verbosity: int = 1,
                 num_betas: int = 10,
                 logger=None,
                 ):
        '''

        :param vposer_expr_dir: The vposer directory that holds the settings and model snapshot
        :param data_loss: should be a pytorch callable (source, target) that returns the accumulated loss
        :param optimizer_args: arguments for optimizers
        :param stepwise_weights: list of dictionaries. each list element defines weights for one full step of optimization
                                 if a weight value is left out, its respective object item will be removed as well. imagine optimizing without data term!
        :param display_rc: number of row and columns in case verbosity > 1
        :param verbosity: 0: silent, 1: text, 2: text/visual. running 2 over ssh would need extra work
        :param logger: an instance of human_body_prior.tools.omni_tools.log2file
        '''

        super(IK_Engine, self).__init__()

        assert isinstance(stepwise_weights, list), ValueError('stepwise_weights should be a list of dictionaries.')
        assert np.all(['data' in l for l in stepwise_weights]), ValueError(
            'The term data should be available in every weight of anealed optimization step: {}'.format(
                stepwise_weights))

        # self.data_loss = torch.nn.SmoothL1Loss(reduction='mean') if data_loss is None else data_loss
        self.data_loss = torch.nn.MSELoss(reduction='mean') if data_loss is None else data_loss
        self.num_betas = num_betas
        self.stepwise_weights = stepwise_weights
        self.verbosity = verbosity
        self.optimizer_args = optimizer_args

        self.logger = log2file() if logger is None else logger

        if verbosity > 1 and can_display:
            mvs = MeshViewers(display_rc, keepalive=True)
            self.mvs = flatten_list(mvs)
            self.mvs[0].set_background_color(colors['white'])
        else:
            self.mvs = None

        self.vp_model, _ = load_model(vposer_expr_dir,
                                      model_code=VPoser,
                                      remove_words_in_model_weights='vp_model.',
                                      disable_grad=True)

    def forward(self, source_kpts, target_kpts, initial_body_params={}):
        '''
        source_kpts is a function that given body parameters computes source key points that should match target key points
        Try to reconstruct the bps signature by optimizing the body_poZ
        '''

        bs = target_kpts.shape[0] 
        coco_pelvis = torch.mean(target_kpts[:, 11:13, :], dim=1)
        coco_pelvis = coco_pelvis.unsqueeze(1)
        target_kpts = torch.cat((target_kpts, coco_pelvis), dim=1)

        on_step = visualize(target_kpts,
                            kpts_colors=source_kpts.kpts_colors,
                            bm_f=source_kpts.bm_f,
                            mvs=self.mvs,
                            verbosity=self.verbosity,
                            logger=self.logger)

        comp_device = target_kpts.device
        
        static_vars = {
            'target_kpts': target_kpts
        }

        for k,v in list(initial_body_params.items()):
            if k in ['used_finger_idx', 'use_vposer',]:
                static_vars[k] = v
                continue
            elif k == 'pose_hand':# (1,15*3*2)
                static_vars[k] = v.reshape(bs,15*3*2)
                continue
            elif k != 'pose_body':
                static_vars[k] = torch.ones_like(v)*(v.mean(axis=0))
        
        if list(initial_body_params.keys()) == []:
            # bool
            initial_body_params['use_vposer'] = torch.ones(1, device=comp_device, dtype=torch.float,
                                                             requires_grad=False)

        if 'pose_body' not in initial_body_params:
            initial_body_params['pose_body'] = torch.zeros([bs, 63], device=comp_device, dtype=torch.float,
                                                           requires_grad=False)
        if 'trans' not in initial_body_params:
            initial_body_params['trans'] = torch.zeros([bs, 3], device=comp_device, dtype=torch.float,
                                                       requires_grad=False)
        if 'betas' not in initial_body_params:
            initial_body_params['betas'] = torch.zeros([bs, self.num_betas], device=comp_device, dtype=torch.float,
                                                       requires_grad=False)
        if 'root_orient' not in initial_body_params:
            initial_body_params['root_orient'] = torch.zeros([bs, 3], device=comp_device, dtype=torch.float,
                                                             requires_grad=False)
        # Add scaling
        if 'scaling' not in initial_body_params:
            initial_body_params['scaling'] = torch.ones([bs, 1], device=comp_device, dtype=torch.float,
                                                             requires_grad=False)


        if initial_body_params['use_vposer']:
            initial_body_params['poZ_body'] = self.vp_model.encode(initial_body_params['pose_body']).mean
        else:
            initial_body_params['poZ_body'] = initial_body_params['pose_body']

        if len(static_vars) == 1:
            free_vars = {k: torch.nn.Parameter(v.detach(), requires_grad=True) for k, v in initial_body_params.items() if
                        k in ['betas','trans', 'poZ_body', 'root_orient','scaling','use_vposer']}
        else:
            
            free_vars = {k: torch.nn.Parameter(v.detach(), requires_grad=True) for k, v in initial_body_params.items() if
                        k in ['poZ_body', 'use_vposer']}

        if self.optimizer_args['type'].upper() == 'LBFGS':
            optimizer = torch.optim.LBFGS(list(free_vars.values()),
                                        lr=self.optimizer_args.get('lr', 1),
                                        max_iter=self.optimizer_args.get('max_iter', 100),
                                        tolerance_change=self.optimizer_args.get('tolerance_change', 1e-5),
                                        max_eval=self.optimizer_args.get('max_eval', None),
                                        history_size=self.optimizer_args.get('history_size', 100),
                                        line_search_fn='strong_wolfe')
            
        elif self.optimizer_args['type'].upper() == 'ADAM':
            optimizer = AdamInClosure(list(free_vars.values()),
                                      lr=self.optimizer_args.get('lr', 1e-3),
                                      max_iter=self.optimizer_args.get('max_iter', 100),
                                      tolerance_change=self.optimizer_args.get('tolerance_change', 1e-5),
                                      )

        elif self.optimizer_args['type'].upper() == 'SGD':
            optimizer = SGDInClosure( var_dict=free_vars,
                                      base_lr=self.optimizer_args.get('lr', 1e-3),
                                      max_iter=self.optimizer_args.get('max_iter', 100),
                                      tolerance_change=self.optimizer_args.get('tolerance_change', 1e-5),
                                      )
        else:
            raise ValueError('optimizer_type not recognized.')

        gstep = 0
        closure = ik_fit(optimizer,
                         source_kpts_model=source_kpts,
                         static_vars=static_vars,
                         vp_model=self.vp_model,
                         extra_params={'data_loss': self.data_loss},
                         on_step=on_step,
                         gstep=gstep)

        for wts in self.stepwise_weights:
            optimizer.step(lambda: closure(wts, free_vars))
            free_vars = closure.free_vars
        
        results = {'free_vars':closure.free_vars,'static_vars':static_vars}
        return results #closure.free_vars  # , closure.nonan_mask
