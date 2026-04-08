import argparse
import os
import random
import copy  # 用于复制教师模型

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

import adapterlib
# 移除了 ControlNet，保留其他必要的模块
from adapterlib import (BinaryDiceLoss, FocalLoss, PQAdapter,
                        TextualAdapter, VisualAdapter)
from dataset import Dataset, PromptDataset
from tools import get_logger, get_transform, normalize, setup_seed


def train(args):
    img_size = args.image_size
    features_list = args.features_list
    save_path = args.save_path
    dataset_name = args.dataset
    batch_size = args.batch_size
    k_shots = args.k_shots
    seed = args.seed
    vl_reduction = args.vl_reduction
    pq_mid_dim = args.pq_mid_dim
    pq_context = args.pq_context

    mode = 'train'

    log_file = f'{dataset_name}_{seed}seed_{k_shots}shot_{mode}_log.txt'
    logger = get_logger(args.save_path, log_file)

    logger.info('\n')
    logger.info(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ====================== Model Initialization  ======================

    if args.pretrained_model == 'ViT-L/14@336px':
        model, _ = adapterlib.load(args.pretrained_model, device=device)
        DPAM_layer = 20
        patch_size = 14
        input_dim = 768
        model.visual.DAPM_replace(DPAM_layer=DPAM_layer)
    elif args.pretrained_model == 'VITB16_PLUS_240':
        model, _ = adapterlib.load(args.pretrained_model, device=device)
        DPAM_layer = 10
        patch_size = 16
        input_dim = 640
        model.visual.DAPM_replace(DPAM_layer=DPAM_layer)
    elif args.pretrained_model == 'ViT-L-14-CLIPA-336':
        model, _ = adapterlib.load(args.pretrained_model, device=device)
        DPAM_layer = 20
        patch_size = 14
        input_dim = 768
        model.visual.DAPM_replace(DPAM_layer=DPAM_layer)
    else:
        raise ValueError(f"Unsupported pretrained_model: {args.pretrained_model}")

    # ====================== 关键新增 1：准备教师模型 ======================
    # 在给学生模型插入/激活 Adapter 之前，复制一份纯净的教师模型并完全冻结
    teacher_model = copy.deepcopy(model)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # ====================== 关键新增 2：激活内部轻量级 Adapter ======================
    # 冻结基座 CLIP 的所有原生参数
    for param in model.parameters():
        param.requires_grad = False

    # 仅激活视觉分支最后 6 层的内部并联 Adapter
    visual_transformer = model.visual.transformer
    num_layers = len(visual_transformer.resblocks)
    adapter_layers = 6
    for i in range(num_layers - adapter_layers, num_layers):
        block = visual_transformer.resblocks[i]
        block.use_adapter = True  # 开启我们在 clip.py 里加的开关
        for param in block.adapter.parameters():
            param.requires_grad = True  # 赋予梯度

    # ====================== Init Adapters ======================
    textual_learner = TextualAdapter(model.to("cpu"), img_size, args.n_ctx)
    visual_learner = VisualAdapter(img_size, patch_size, input_dim=input_dim, reduction=vl_reduction)
    pq_learner = PQAdapter(img_size, patch_size, context=pq_context, input_dim=input_dim, mid_dim=pq_mid_dim,
                           layers_num=len(features_list))

    model.to(device)
    teacher_model.to(device)
    textual_learner.to(device)
    visual_learner.to(device)
    pq_learner.to(device)

    model.train()  # 注意：学生模型基座由于包含可训练的 Adapter，需开启 train 模式
    textual_learner.train()
    visual_learner.train()
    pq_learner.train()

    # BatchNorm 在 batch_size=1 且处于 train 模式时会直接报错。
    # 这里选择将 BN 层置为 eval，仅避免崩溃；BN 的 affine 参数仍可参与训练。
    if batch_size == 1:
        for m in visual_learner.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()

    if args.checkpoint_path:
        logger.info('\n' + "loading checkpoint from: " + args.checkpoint_path)
        checkpoint_adapter = torch.load(args.checkpoint_path, map_location=device)
        if "textual_learner" in checkpoint_adapter:
            textual_learner.load_state_dict(checkpoint_adapter["textual_learner"], strict=False)
        if "visual_learner" in checkpoint_adapter:
            visual_learner.load_state_dict(checkpoint_adapter["visual_learner"], strict=False)
        if "pq_learner" in checkpoint_adapter:
            pq_learner.load_state_dict(checkpoint_adapter["pq_learner"], strict=False)
        if "internal_adapters" in checkpoint_adapter:
            # 加载内部 Adapter 权重
            model.load_state_dict(checkpoint_adapter["internal_adapters"], strict=False)

    textual_learner_parameters = sum(p.numel() for p in textual_learner.parameters() if p.requires_grad)
    visual_learner_parameters = sum(p.numel() for p in visual_learner.parameters() if p.requires_grad)
    pq_learner_parameters = sum(p.numel() for p in pq_learner.parameters() if p.requires_grad)
    internal_adapter_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    learned_parameters = textual_learner_parameters + visual_learner_parameters + pq_learner_parameters + internal_adapter_parameters
    fixed_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"textual_learner params:{(textual_learner_parameters):.0f}",
          f"visual_learner params:{(visual_learner_parameters) / 1e+6:.1f}M",
          f"pq_learner params:{(pq_learner_parameters) / 1e+6:.1f}M",
          f"internal_adapter params:{(internal_adapter_parameters) / 1e+6:.1f}M",
          f"learned all parameters:{(learned_parameters) / 1e+6:.1f}M",
          f"fixed params:{(fixed_parameters) / 1e+6:.1f}M",
          f"all params:{(learned_parameters + fixed_parameters) / 1e+6:.1f}M"
          )

    # ====================== Optimizer and Loss  ======================
    # 将内部的轻量级 Adapter 参数也加入优化器
    optimizer_params = list(textual_learner.parameters()) + list(visual_learner.parameters()) + list(
        pq_learner.parameters())
    optimizer_params += [p for p in model.parameters() if p.requires_grad]

    # 避免重复参数导致 optimizer 警告/未来版本报错
    seen_param_ids = set()
    optimizer_params_uniq = []
    for p in optimizer_params:
        pid = id(p)
        if pid in seen_param_ids:
            continue
        seen_param_ids.add(pid)
        optimizer_params_uniq.append(p)
    optimizer_params = optimizer_params_uniq

    optimizer = torch.optim.Adam(
        optimizer_params,
        lr=args.learning_rate,
        betas=(0.5, 0.999)
    )

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    # ====================== Data  ======================
    preprocess, target_transform = get_transform(image_size=args.image_size)
    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, \
                         dataset_name=dataset_name, k_shots=k_shots, save_dir=save_path, mode='train', seed=seed)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    obj_list = train_data.obj_list

    # ====================== forward and backward ======================
    textual_learner.prepare_static_text_feature(model)
    for epoch in tqdm(range(args.epoch)):
        local_loss_list = []
        global_loss_list = []
        kd_loss_list = []

        for items in tqdm(train_data_loader):
            prompt_image = items['prompt_img'].to(device)  # B*s*c*h*w
            b, s, c, h, w = prompt_image.shape
            prompt_image = prompt_image.reshape(-1, c, h, w)

            image = items['img'].to(device)
            label = items['anomaly']

            gt = items['img_mask'].squeeze().to(device)
            if gt.dim() == 2:
                gt = gt.unsqueeze(0)  # 修复batch为1时降维错误
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            # ====================== 关键修改 3：特征提取逻辑分离 ======================
            # 1. 教师模型提取纯净特征 (不计算梯度)
            with torch.no_grad():
                teacher_query_feats, teacher_query_patch_feats = teacher_model.encode_image(image, args.features_list,
                                                                                            DPAM_layer=DPAM_layer)
                prompt_feats, prompt_patch_feats = teacher_model.encode_image(prompt_image, args.features_list,
                                                                              DPAM_layer=DPAM_layer)

                prompt_feats = prompt_feats.reshape(b, s, -1)
                for idx in range(len(args.features_list)):
                    prompt_patch_feats[idx] = rearrange(prompt_patch_feats[idx], '(b s) l d -> b s l d', b=b, s=s)

            # 2. 学生模型提取特征 (必须有梯度，因为内部有可训练的 Adapter)
            query_feats, query_patch_feats = model.encode_image(image, args.features_list, DPAM_layer=DPAM_layer)

            # 3. 计算特征蒸馏 Loss (KD Loss)
            loss_kd = 0
            for t_feat, s_feat in zip(teacher_query_patch_feats, query_patch_feats):
                loss_kd += F.mse_loss(s_feat, t_feat)

            local_loss = 0
            global_loss = 0

            # ====================== visual_adapter ======================
            if args.visual_learner:
                static_text_features = textual_learner.static_text_features
                # 移除了 ControlNet 后，直接将原特征送入 VisualAdapter
                global_logit, local_score = visual_learner(query_feats, query_patch_feats, static_text_features)

                global_loss += F.cross_entropy(global_logit, label.long().to(device))

                local_loss += loss_focal(local_score, gt)
                local_loss += loss_dice(local_score[:, 1, :, :], gt)
                local_loss += loss_dice(local_score[:, 0, :, :], 1 - gt)

            # ====================== textual_adapter ======================
            if args.textual_learner:
                learned_prompts, tokenized_prompts = textual_learner()
                learned_text_features = model.encode_text(learned_prompts, tokenized_prompts).float()  # [2, 768]
                global_logit, local_score = textual_learner.compute_global_local_score(query_feats, query_patch_feats,
                                                                                       learned_text_features)

                global_loss += F.cross_entropy(global_logit, label.long().to(device))

                local_loss += loss_focal(local_score, gt)
                local_loss += loss_dice(local_score[:, 1, :, :], gt)
                local_loss += loss_dice(local_score[:, 0, :, :], 1 - gt)

            # ====================== pq_adapter ======================
            if args.pq_learner:
                global_logit, local_score_list, align_score_list = pq_learner(query_feats, query_patch_feats,
                                                                              prompt_feats, prompt_patch_feats)

                for i in range(len(global_logit)):
                    global_loss += F.cross_entropy(global_logit[i], label.long().to(device))

                for i in range(len(local_score_list)):
                    local_loss += loss_focal(local_score_list[i], gt)
                    local_loss += loss_dice(local_score_list[i][:, 1, :, :], gt)
                    local_loss += loss_dice(local_score_list[i][:, 0, :, :], 1 - gt)

            optimizer.zero_grad()

            # ====================== 关键修改 4：融合蒸馏 Loss ======================
            total_loss = local_loss + global_loss + args.lambda_kd * loss_kd
            total_loss.backward()

            optimizer.step()

            global_loss_list.append(global_loss.item() if isinstance(global_loss, torch.Tensor) else 0)
            local_loss_list.append(local_loss.item() if isinstance(local_loss, torch.Tensor) else 0)
            kd_loss_list.append(loss_kd.item())

        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], global_loss:{:.4f}, local_loss:{:.4f}, kd_loss:{:.4f}'.format(
                epoch + 1, args.epoch, np.mean(global_loss_list), np.mean(local_loss_list), np.mean(kd_loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')

            # 只保存基座里“可训练参数”的 state_dict 条目（内部 Adapter）
            trainable_param_names = {name for name, p in model.named_parameters() if p.requires_grad}
            model_sd = model.state_dict()
            internal_adapters_sd = {k: v for k, v in model_sd.items() if k in trainable_param_names}

            save_state = {
                "textual_learner": textual_learner.state_dict(),
                "visual_learner": visual_learner.state_dict(),
                "pq_learner": pq_learner.state_dict(),
                # 保存植入在基座里的 Adapter 权重
                "internal_adapters": internal_adapters_sd,
            }
            torch.save(save_state, ckp_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("adapter", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default=None, help="path to checkpoint for resuming")
    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")
    parser.add_argument("--pretrained_model", type=str, default='ViT-L/14@336px', help="pre-trained model name")
    parser.add_argument("--n_ctx", type=int, default=12, help="the textual prompt length of textual learner")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    # 注意：这里的学习率已经调小，以适配微调，防止零初始化被瞬间打破
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    parser.add_argument("--k_shots", type=int, default=1, help="how many normal samples")
    parser.add_argument("--visual_learner", action="store_true", help="Enable visual adapter")
    parser.add_argument("--textual_learner", action="store_true", help="Enable textual adapter")
    parser.add_argument("--pq_learner", action="store_true", help="Enable prompt-query adapter")
    parser.add_argument("--vl_reduction", type=int, default=4, help="the reduction number of visual learner")
    parser.add_argument("--pq_mid_dim", type=int, default=128, help="the number of the first hidden layer in pqadapter")
    parser.add_argument("--pq_context", action="store_true", help="Enable context feature")

    # === 新增超参数：特征蒸馏的权重系数 ===
    parser.add_argument("--lambda_kd", type=float, default=1.0, help="Weight for Knowledge Distillation loss")

    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)