"""Training script for adapter on custom dataset."""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

import adapterlib
from adapterlib import (BinaryDiceLoss, FocalLoss, PQAdapter, TextualAdapter,
                          VisualAdapter)
from dataset import Dataset
from tools import get_logger, get_transform, setup_seed


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
        raise ValueError(f"Unsupported pretrained model: {args.pretrained_model}")

    # ====================== Init Adapters ======================
    textual_learner = TextualAdapter(model.to("cpu"), img_size, args.n_ctx)
    visual_learner = VisualAdapter(img_size, patch_size, input_dim=input_dim, reduction=vl_reduction)
    pq_learner = PQAdapter(img_size, patch_size, context=pq_context, input_dim=input_dim, mid_dim=pq_mid_dim, layers_num=len(features_list))

    model.to(device)
    textual_learner.to(device)
    visual_learner.to(device)
    pq_learner.to(device)

    model.eval()
    textual_learner.train()
    visual_learner.train()
    pq_learner.train()

    # ====================== Load Checkpoint (if provided) ======================
    start_epoch = 0
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        logger.info('\n' + "Loading checkpoint from: " + args.checkpoint_path)
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        # 加载模型权重
        if "textual_learner" in checkpoint:
            textual_learner.load_state_dict(checkpoint["textual_learner"])
            logger.info("Loaded textual_learner weights")
        if "visual_learner" in checkpoint:
            visual_learner.load_state_dict(checkpoint["visual_learner"])
            logger.info("Loaded visual_learner weights")
        if "pq_learner" in checkpoint:
            pq_learner.load_state_dict(checkpoint["pq_learner"])
            logger.info("Loaded pq_learner weights")
        
        # 如果checkpoint中保存了epoch信息，可以从中恢复
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
            logger.info(f"Resuming from epoch {start_epoch + 1}")
        elif "last_epoch" in checkpoint:
            start_epoch = checkpoint["last_epoch"]
            logger.info(f"Resuming from epoch {start_epoch + 1}")
    elif args.checkpoint_path:
        logger.warning(f"Checkpoint path provided but file not found: {args.checkpoint_path}")
        logger.warning("Starting training from scratch")

    textual_learner_parameters = sum(p.numel() for p in textual_learner.parameters())
    visual_learner_parameters = sum(p.numel() for p in visual_learner.parameters())
    pq_learner_parameters = sum(p.numel() for p in pq_learner.parameters())

    learned_parameters = textual_learner_parameters + visual_learner_parameters + pq_learner_parameters
    fixed_parameters = sum(p.numel() for p in model.parameters())

    logger.info(f"textual_learner params: {textual_learner_parameters:.0f}")
    logger.info(f"visual_learner params: {visual_learner_parameters/1e+6:.1f}M")
    logger.info(f"pq_learner params: {pq_learner_parameters/1e+6:.1f}M")
    logger.info(f"learned all parameters: {learned_parameters/1e+6:.1f}M")
    logger.info(f"fixed params: {fixed_parameters/1e+6:.1f}M")
    logger.info(f"all params: {(learned_parameters+fixed_parameters)/1e+6:.1f}M")

    # ====================== Optimizer and Loss  ======================
    optimizer = torch.optim.Adam(
        list(textual_learner.parameters()) + list(visual_learner.parameters()) + list(pq_learner.parameters()),
        lr=args.learning_rate,
        betas=(0.5, 0.999)
    )

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    # ====================== Data  ======================
    preprocess, target_transform = get_transform(image_size=args.image_size)
    train_data = Dataset(
        root=args.train_data_path, 
        transform=preprocess, 
        target_transform=target_transform,
        dataset_name=dataset_name, 
        k_shots=k_shots, 
        save_dir=save_path, 
        mode='train', 
        seed=seed
    )
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    obj_list = train_data.obj_list

    logger.info(f"Training with {len(train_data)} samples from {len(obj_list)} classes")
    logger.info(f"Classes: {obj_list}")

    # ====================== forward and backward ======================
    textual_learner.prepare_static_text_feature(model)
    
    for epoch in tqdm(range(start_epoch, args.epoch), desc="Epochs"):
        local_loss_list = []
        global_loss_list = []

        for items in tqdm(train_data_loader, desc=f"Epoch {epoch+1}"):
            prompt_image = items['prompt_img'].to(device)  # B*s*c*h*w
            b, s, c, h, w = prompt_image.shape
            prompt_image = prompt_image.reshape(-1, c, h, w)

            image = items['img'].to(device)
            label = items['anomaly']

            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            with torch.no_grad():
                query_feats, query_patch_feats = model.encode_image(image, args.features_list, DPAM_layer=DPAM_layer)
                prompt_feats, prompt_patch_feats = model.encode_image(prompt_image, args.features_list, DPAM_layer=DPAM_layer)

                prompt_feats = prompt_feats.reshape(b, s, -1)
                for idx in range(len(args.features_list)):
                    prompt_patch_feats[idx] = rearrange(prompt_patch_feats[idx], '(b s) l d -> b s l d', b=b, s=s)

            local_loss = 0
            global_loss = 0
            
            # ====================== visual_adapter ======================
            if args.visual_learner:
                static_text_features = textual_learner.static_text_features
                global_logit, local_score = visual_learner(query_feats, query_patch_feats, static_text_features)

                global_loss += F.cross_entropy(global_logit, label.long().cuda())

                local_loss += loss_focal(local_score, gt)
                local_loss += loss_dice(local_score[:, 1, :, :], gt)
                local_loss += loss_dice(local_score[:, 0, :, :], 1-gt)

            # ====================== textual_adapter ======================
            if args.textual_learner:
                learned_prompts, tokenized_prompts = textual_learner()
                learned_text_features = model.encode_text(learned_prompts, tokenized_prompts).float()
                global_logit, local_score = textual_learner.compute_global_local_score(query_feats, query_patch_feats, learned_text_features)

                global_loss += F.cross_entropy(global_logit, label.long().cuda())

                local_loss += loss_focal(local_score, gt)
                local_loss += loss_dice(local_score[:, 1, :, :], gt)
                local_loss += loss_dice(local_score[:, 0, :, :], 1-gt)

            # ====================== pq_adapter ======================
            if args.pq_learner:
                global_logit, local_score_list, align_score_list = pq_learner(query_feats, query_patch_feats, prompt_feats, prompt_patch_feats)

                for i in range(len(global_logit)):
                    global_loss += F.cross_entropy(global_logit[i], label.long().cuda())

                for i in range(len(local_score_list)):
                    local_loss += loss_focal(local_score_list[i], gt)
                    local_loss += loss_dice(local_score_list[i][:, 1, :, :], gt)
                    local_loss += loss_dice(local_score_list[i][:, 0, :, :], 1-gt)

            optimizer.zero_grad()
            (local_loss + global_loss).backward()
            optimizer.step()
            global_loss_list.append(global_loss.item())
            local_loss_list.append(local_loss.item())

        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], global_loss:{:.4f}, local_loss:{:.4f}'.format(
                epoch + 1, args.epoch, np.mean(global_loss_list), np.mean(local_loss_list)
            ))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({
                "textual_learner": textual_learner.state_dict(),
                "visual_learner": visual_learner.state_dict(),
                "pq_learner": pq_learner.state_dict(),
                "epoch": epoch + 1,  # 保存当前epoch，便于后续恢复训练
            }, ckp_path)
            logger.info(f"Model saved to {ckp_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("adapter Custom Dataset Training", add_help=True)
    parser.add_argument("--train_data_path", type=str, required=True, help="path to custom dataset root")
    parser.add_argument("--save_path", type=str, required=True, help='path to save checkpoints and logs')
    parser.add_argument("--dataset", type=str, default='custom_dataset', help="dataset name")
    parser.add_argument("--pretrained_model", type=str, default='ViT-L/14@336px', help="pre-trained model name")
    parser.add_argument("--n_ctx", type=int, default=12, help="the textual prompt length of textual learner")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    parser.add_argument("--k_shots", type=int, default=1, help="how many normal samples for few-shot")
    parser.add_argument("--visual_learner", action="store_true", help="Enable visual adapter")
    parser.add_argument("--textual_learner", action="store_true", help="Enable textual adapter")
    parser.add_argument("--pq_learner", action="store_true", help="Enable prompt-query adapter")
    parser.add_argument("--vl_reduction", type=int, default=4, help="the reduction number of visual learner")
    parser.add_argument("--pq_mid_dim", type=int, default=128, help="the number of the first hidden layer in pqadapter")
    parser.add_argument("--pq_context", action="store_true", help="Enable context feature")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="path to checkpoint for resuming training (optional)")
    
    args = parser.parse_args()
    
    # 确保保存目录存在
    os.makedirs(args.save_path, exist_ok=True)
    
    setup_seed(args.seed)
    train(args)







