import torch
import torchvision
from net import Net, weights_init_kaiming
from data import load_images_names_in_data_set, get_bb_of_gt_from_pascal_xml_annotation
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
from PIL import Image, ImageDraw, ImageFont

from repvgg_in import deploy_model
from utils import cal_iou, reward_func, reward_func_gfirs
import matplotlib.pyplot as plt
import cv2
import time
import math
import datetime
import random
import sys
from optim_alg_python.multi_agent_lr_optimizer import SimpleLROptimizer

# Hyperparameters
BATCH_SIZE = 128
LR = 1e-6
GAMMA = 0.9
MEMORY_CAPACITY = 1000
Q_NETWORK_ITERATION = 100
epochs = 1
NUM_ACTIONS = 9
his_actions = 10
subscale = 0.2
NUM_STATES = 5 * 5 * 1280 + his_actions * NUM_ACTIONS
path_voc = "home/data/VOCdevkit/VOC2012/"
USE_GFIRS = True
ALPHA_MAX = 1.0
BETA = 1.5
LAMBDA_PHYSICS = 0.3
LAMBDA_EFFICIENCY = 0.05

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(1)


class HilbertStateEncoder(nn.Module):
    """Lightweight state encoder that maps states to Hilbert space"""
    def __init__(self, state_dim=NUM_STATES, embed_dim=256):
        super(HilbertStateEncoder, self).__init__()
        self.encoder = nn.Linear(state_dim, embed_dim)
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
    def forward(self, state):
        """Map state to Hilbert space
        Args:
            state: State tensor (batch_size, state_dim) or (state_dim,)
        Returns:
            embedding: Hilbert space representation (batch_size, embed_dim) or (embed_dim,)
        """
        return self.encoder(state)


class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)  # for all transitions

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # store transition in self.data
        self.update(tree_idx, p)  # add p to the tree
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1  # left kid of the parent node
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # kid node is out of the tree, so parent is the leaf node
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]


class Memory(object):  # stored as (s, a, r, s_) in SumTree
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max of p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty(
            (n, 1))
        pri_seg = self.tree.total_p / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max=1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculation ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors.data, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class HilbertMemory(Memory):
    """Enhanced memory buffer guided by Hilbert distance"""
    def __init__(self, capacity, state_dim=NUM_STATES, embed_dim=256, 
                 diversity_weight=0.1, max_cache=500, device='cuda'):
        super(HilbertMemory, self).__init__(capacity)
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.diversity_weight = diversity_weight
        self.device = device
        self.max_cache = max_cache
        
        self.state_encoder = HilbertStateEncoder(state_dim, embed_dim).to(device)
        self.encoded_states_cache = []
        self.cache_counter = 0
        
    def hilbert_distance(self, state1_embed, state2_embed):
        """Compute L2 distance in Hilbert space"""
        return torch.norm(state1_embed - state2_embed, p=2).item()
    
    def compute_diversity_score(self, new_state):
        """Compute diversity score of new state (minimum distance to stored states)
        Args:
            new_state: numpy array with shape (state_dim,)
        Returns:
            diversity_score: float representing diversity
        """
        if len(self.encoded_states_cache) == 0:
            return 0.0
        
        with torch.no_grad():
            new_state_tensor = torch.FloatTensor(new_state).unsqueeze(0).to(self.device)
            new_embed = self.state_encoder(new_state_tensor)
            
            sample_size = min(50, len(self.encoded_states_cache))
            sampled_embeds = random.sample(self.encoded_states_cache, sample_size)
            
            distances = [self.hilbert_distance(new_embed, embed) for embed in sampled_embeds]
            
            return min(distances) if distances else 0.0
    
    def store(self, transition):
        """Store transition with priority combining TD error and Hilbert distance diversity"""
        state = transition[:self.state_dim]
        diversity_score = self.compute_diversity_score(state)
        
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        
        enhanced_priority = max_p * (1.0 + self.diversity_weight * diversity_score)
        enhanced_priority = np.clip(enhanced_priority, 0, self.abs_err_upper)
        
        self.tree.add(enhanced_priority, transition)
        
        if len(self.encoded_states_cache) < self.max_cache:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                state_embed = self.state_encoder(state_tensor)
                self.encoded_states_cache.append(state_embed)
        else:
            replace_idx = random.randint(0, self.max_cache - 1)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                state_embed = self.state_encoder(state_tensor)
                self.encoded_states_cache[replace_idx] = state_embed
        
        self.cache_counter += 1


class DQN:
    """Deep Q-Network with Hilbert-Kernel enhancement (HK-DQN)"""

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.eval_net, self.target_net = Net(), Net()
        self.eval_net.apply(weights_init_kaiming)
        self.target_net.apply(weights_init_kaiming)
        self.eval_net.to(self.device)
        self.target_net.to(self.device)
        self.learn_step_counter = 0
        self.train_loss = []
        self.learn_step = []
        self.memory_counter = 0
        self.memory = HilbertMemory(
            capacity=MEMORY_CAPACITY,
            state_dim=NUM_STATES,
            embed_dim=256,
            diversity_weight=0.1,
            max_cache=500,
            device=self.device
        )
        self.optimizer = torch.optim.Adam(params=self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.miou_history = []
        self.precision50_history = []
        self.lr_optimizer = SimpleLROptimizer(
            initial_lr=LR,
            lb=1e-7,
            ub=1e-3,
            update_frequency=100
        )
        self.lr_history = []

    def choose_action(self, state, EPISILO):
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)
        if np.random.uniform() <= EPISILO:
            action = np.random.randint(0, NUM_ACTIONS)
        else:
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].cpu().item()
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        self.memory.store(transition)
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        tree_idx, batch_memory, ISWeights = self.memory.sample(BATCH_SIZE)

        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES]).to(self.device)
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES + 1].astype(int)).to(self.device)
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES + 1:NUM_STATES + 2]).to(self.device)
        batch_next_state = torch.FloatTensor(batch_memory[:, NUM_STATES + 2:2 * NUM_STATES + 2]).to(self.device)

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_eval4next = self.eval_net(batch_next_state).detach()

        max_act4next = q_eval4next.max(1)[1]
        max_act4next1 = max_act4next.view(BATCH_SIZE, 1)
        selected_q_next = q_next.gather(1, max_act4next1)
        q_target_unterminated = batch_reward + GAMMA * selected_q_next.view(BATCH_SIZE, 1)
        q_target = torch.where(batch_action != 10, q_target_unterminated, batch_reward)

        self.abs_errors = torch.sum(torch.abs(q_target - q_eval), dim=1)
        loss = torch.mean(torch.mean(torch.Tensor(ISWeights).to(self.device) * (q_target - q_eval) ** 2, dim=1))
        self.memory.batch_update(tree_idx, self.abs_errors.cpu())

        loss_value = loss.cpu().detach().item()
        self.train_loss.append(loss_value)
        current_lr = self.lr_optimizer.step(loss_value)
        self.lr_history.append(current_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.learn_step_counter % 100 == 0:
            status = self.lr_optimizer.get_status()
            if status['coordinator_status']['phase']:
                print(f"  [LR Optimization] Step {self.learn_step_counter}: LR={current_lr:.2e}, "
                      f"Phase={status['coordinator_status']['phase']}, "
                      f"Explore/Exploit={status['coordinator_status']['w_explore']:.2f}/{status['coordinator_status']['w_exploit']:.2f}")

    def save(self):

        Eval_Net_PATH = 'home/pth/eval_net_aeroplane.pth'
        torch.save(self.eval_net, Eval_Net_PATH)
        Target_Net_PATH = 'home/pth/target_net_aeroplane.pth'
        torch.save(self.target_net, Target_Net_PATH)
        print('Complete')

    def load(self):
        self.eval_net = torch.load('home/pth/eval_net_aeroplane.pth')


def init_process(image, transform=None):
    if transform:
        image = transform(image)
    return image.unsqueeze(0)


def inter_process(image, bbx, transform=None):
    (left, upper, right, lower) = (bbx[0], bbx[2], bbx[1], bbx[3])
    image_crop = image.crop((left, upper, right, lower))
    if transform:
        image_crop = transform(image_crop)
    return image_crop.unsqueeze(0)


def update_bbx(bbx, action):
    """Update bounding box based on action"""
    new_bbx = np.zeros(4)
    if action == 0:
        new_bbx[0] = bbx[0] - (bbx[1] - bbx[0]) * subscale
        new_bbx[1] = bbx[1] - (bbx[1] - bbx[0]) * subscale
        new_bbx[2] = bbx[2]
        new_bbx[3] = bbx[3]
    elif action == 1:
        new_bbx[0] = bbx[0] + (bbx[1] - bbx[0]) * subscale
        new_bbx[1] = bbx[1] + (bbx[1] - bbx[0]) * subscale
        new_bbx[2] = bbx[2]
        new_bbx[3] = bbx[3]
    elif action == 2:
        new_bbx[0] = bbx[0]
        new_bbx[1] = bbx[1]
        new_bbx[2] = bbx[2] - (bbx[3] - bbx[2]) * subscale
        new_bbx[3] = bbx[3] - (bbx[3] - bbx[2]) * subscale
    elif action == 3:
        new_bbx[0] = bbx[0]
        new_bbx[1] = bbx[1]
        new_bbx[2] = bbx[2] + (bbx[3] - bbx[2]) * subscale
        new_bbx[3] = bbx[3] + (bbx[3] - bbx[2]) * subscale
    elif action == 4:
        new_bbx[0] = bbx[0]
        new_bbx[1] = bbx[1]
        new_bbx[2] = bbx[2] + (bbx[3] - bbx[2]) * subscale * 1 / 2
        new_bbx[3] = bbx[3] - (bbx[3] - bbx[2]) * subscale * 1 / 2
    elif action == 5:
        new_bbx[0] = bbx[0] + (bbx[1] - bbx[0]) * subscale * 1 / 2
        new_bbx[1] = bbx[1] - (bbx[1] - bbx[0]) * subscale * 1 / 2
        new_bbx[2] = bbx[2]
        new_bbx[3] = bbx[3]
    elif action == 6:
        new_bbx[0] = bbx[0] + (bbx[1] - bbx[0]) * subscale * 1 / 2
        new_bbx[1] = bbx[1] - (bbx[1] - bbx[0]) * subscale * 1 / 2
        new_bbx[2] = bbx[2] + (bbx[3] - bbx[2]) * subscale * 1 / 2
        new_bbx[3] = bbx[3] - (bbx[3] - bbx[2]) * subscale * 1 / 2
    elif action == 7:
        new_bbx[0] = bbx[0] - (bbx[1] - bbx[0]) * subscale * 1 / 2
        new_bbx[1] = bbx[1] + (bbx[1] - bbx[0]) * subscale * 1 / 2
        new_bbx[2] = bbx[2] - (bbx[3] - bbx[2]) * subscale * 1 / 2
        new_bbx[3] = bbx[3] + (bbx[3] - bbx[2]) * subscale * 1 / 2
    elif action == 8:
        new_bbx = bbx
    return new_bbx


def main(args):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.use_gpu) else "cpu")
    image_names = np.array(load_images_names_in_data_set('aeroplane', path_voc))

    deploy_model.gap = nn.Sequential(nn.AdaptiveAvgPool2d(5))
    deploy_model.linear = nn.Sequential()
    feature_exactrator = deploy_model.to(device)

    single_plane_image_names = []
    single_plane_image_gts = []
    dqn = DQN(device)
    EPISILO = args.EPISILO

    for image_name in image_names:
        annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc)
        if len(annotation) > 1:
            continue
        single_plane_image_names.append(image_name)
        single_plane_image_gts.append(annotation[0][1:])

    trans = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    for i in range(epochs):
        ep_reward = 0
        iou_sum = 0
        precision50_count = 0
        total_samples = 0
        for index, image_name in enumerate(single_plane_image_names):
            image_path = os.path.join(path_voc + "JPEGImages", image_name + ".jpg")
            image_original = Image.open(image_path)
            width, height = image_original.size
            bbx_gt = single_plane_image_gts[index]

            image = init_process(image_original, trans).to(device)

            bbx = [0, width, 0, height]
            prev_bbx = None
            history_action = np.zeros(his_actions * NUM_ACTIONS)
            with torch.no_grad():
                vector = feature_exactrator(image).cpu().detach().numpy().reshape(5 * 5 * 1280)
            state = np.concatenate([history_action, vector])
            step = 0
            while step < 50:
                iou = cal_iou(bbx, bbx_gt)
                if iou >= 0.9:
                    action = 8
                else:
                    action = dqn.choose_action(state, EPISILO)

                new_bbx = update_bbx(bbx, action)
                if USE_GFIRS:
                    reward = reward_func_gfirs(bbx, new_bbx, bbx_gt, action, step, prev_bbx, 
                                               ALPHA_MAX, BETA, LAMBDA_PHYSICS, LAMBDA_EFFICIENCY)
                else:
                    reward = reward_func(bbx, new_bbx, bbx_gt, action, step)
                action_vec = np.zeros(NUM_ACTIONS)
                action_vec[action] = 1.0
                history_action = np.concatenate([history_action[NUM_ACTIONS:], action_vec])

                with torch.no_grad():
                    vector = feature_exactrator(
                        inter_process(image_original, new_bbx, trans).to(device)).cpu().detach().numpy().reshape(
                        5 * 5 * 1280)
                next_state = np.concatenate([history_action, vector])

                dqn.store_transition(state, action, reward, next_state)

                ep_reward += reward

                if dqn.memory_counter >= MEMORY_CAPACITY:
                    dqn.learn()

                if action == 8 or step == 49:
                    current_iou = cal_iou(bbx, bbx_gt)
                    iou_sum += current_iou
                    if current_iou > 0.5:
                        precision50_count += 1
                    total_samples += 1
                    
                    if i == epochs - 1:
                        draw = ImageDraw.Draw(image_original)
                        draw.rectangle([bbx_gt[0], bbx_gt[2], bbx_gt[1], bbx_gt[3]], outline='green', width=2)
                        draw.rectangle([bbx[0], bbx[2], bbx[1], bbx[3]], outline='red', width=2)
                        font = ImageFont.truetype('home/fonts/Droid-Sans.ttf', 30)
                        draw.text([bbx[0], bbx[2]], str(round(iou, 2)), fill=(255, 0, 0), font=font)

                        image_original.save(
                            'home/visualization/visualization_aeroplane_train' + "/" + image_name + ".jpg")
                    break

                prev_bbx = bbx
                state = next_state
                bbx = new_bbx
                step += 1

        if EPISILO > 0.1:
            EPISILO -= 0.1
            
        epoch_miou = iou_sum / total_samples if total_samples > 0 else 0
        epoch_precision50 = precision50_count / total_samples if total_samples > 0 else 0
        dqn.miou_history.append(epoch_miou)
        dqn.precision50_history.append(epoch_precision50)
        
        now = datetime.datetime.now()
        print("episode: {} , this epoch reward is {}, miou is {}, precision50 is {}, date:{}".format(
            i, round(ep_reward, 3), round(epoch_miou, 3), round(epoch_precision50, 3), now))


    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.plot(np.arange(len(dqn.train_loss)), dqn.train_loss)
    plt.xlabel('learn_step')
    plt.ylabel('train_loss')
    plt.title('Training Loss')
    plt.subplot(1, 4, 2)
    plt.plot(np.arange(len(dqn.lr_history)), dqn.lr_history)
    plt.xlabel('learn_step')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.title('Dynamic Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 4, 3)
    plt.plot(np.arange(len(dqn.miou_history)), dqn.miou_history)
    plt.xlabel('Episode')
    plt.ylabel('mIoU')
    plt.title('Mean IoU')
    plt.subplot(1, 4, 4)
    plt.plot(np.arange(len(dqn.precision50_history)), dqn.precision50_history)
    plt.xlabel('Episode')
    plt.ylabel('Precision@0.5')
    plt.title('Precision@0.5')
    
    plt.tight_layout()
    plt.savefig("home/visualization/loss/aeroplane_train_metrics.png", dpi=300)
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(dqn.train_loss)), dqn.train_loss)
    plt.xlabel('learn_step')
    plt.ylabel('train_loss')
    plt.savefig("home/visualization/loss/aeroplane_train.png", dpi=300)

    dqn.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hierarchical Object Detection with Deep Reinforcement Learning')
    parser.add_argument('--gpu-devices', default='1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--use_gpu', default=True, action='store_true')
    parser.add_argument('--EPISILO', type=int, default=0.9)

    main(parser.parse_args())
