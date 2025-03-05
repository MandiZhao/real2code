import os
from os.path import join
from glob import glob
from natsort import natsorted
import wandb
import json
from tqdm import tqdm 
import numpy as np 
from torch.utils.data import DataLoader, WeightedRandomSampler
import argparse
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.modeling.mask_decoder import MLP as MaskDecoderMLP
from torch.optim import Adam
from monai.losses import DiceCELoss, DiceLoss, FocalLoss
from PIL import Image
import torch 
import torch.nn as nn 
from copy import deepcopy
import torchvision 
from torchvision.transforms.functional import crop as tvf_crop
from torchvision.transforms.functional import resize as tvf_resize
from torchvision.transforms import Compose, ColorJitter, ToTensor, Normalize, RandomCrop
from datetime import datetime
import torch.nn.functional as F
from einops import rearrange, repeat
from part_segmentation.sam_datasets import SamH5Dataset
from torch.nn import DataParallel

CKPT_PATH="/home/mandi/sam_vit_h_4b8939.pth"

def extend_instance(obj, mixin):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(
        base_cls_name, (mixin, base_cls), {}
    ) 

class PointsQueryMixin(nn.Module): 
    def forward(self, batch, device, original_size=(480, 480)): 
        rgb = batch["image"] 
        rgb = self.preprocess(rgb)
        # breakpoint()
        image_embeddings = self.image_encoder(rgb)
        bs = rgb.shape[0]
        all_outputs = []
        for b in range(bs): 
            points = (batch['point_coords'][b], batch['point_labels'][b]) # shape: (num_masks, num_points=1, 2), (num_masks, num_points=1)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(points=points, boxes=None, masks=None)
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings[b:b+1], # shape 1,3,1024,1024
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
        
            pred = self.postprocess_masks(
                        low_res_masks,
                        input_size=(1024, 1024),
                        original_size=original_size,
                    ) # shape (num_mask, 1, 480, 480)
            # pred = pred > model.mask_threshold 
            outputs = {
                    "pred": pred,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            all_outputs.append(outputs)
        # concat all outputs
        stack_outputs = dict()
        for k in all_outputs[0].keys():
            stack_outputs[k] = torch.stack([o[k] for o in all_outputs], dim=0).squeeze(2)
        return stack_outputs

def get_image_transform(model_img_size, jitter=True, random_crop=True, center_crop=False, pad=False):
    
    resize_fn = ResizeLongestSide(model_img_size) 
    if center_crop:
        print("Center cropping the image to be 1440 before resizing!")
        centercrop = torchvision.transforms.CenterCrop(1440)
    if pad:
        print("Padding the image to be 1920 before resizing!") 

    jitter_fn = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1) 

    def transforms(rgb: np.ndarray, masks: np.ndarray):
        original_size = rgb.shape[:2]
        assert rgb.shape[:2] == masks.shape[1:3], "rgb and masks must have the same shape" 
        rgb = torch.as_tensor(rgb).permute(2, 0, 1).contiguous()
        if jitter:
            rgb = jitter_fn(rgb/255) 
            rgb *= 255
            rgb = rgb.clamp(0, 255)
        original_masks = masks.copy()
        # print(masks[(masks != 0) & (masks != 1)])
        masks = [torch.as_tensor(mask, dtype=torch.float) for mask in masks]
        if random_crop:
            crop_scale = np.random.uniform(0.85, 1.0)
            crop_size = (int(crop_scale * original_size[0]), int(crop_scale * original_size[1]))
            i, j, h, w = RandomCrop.get_params(
                rgb, output_size=crop_size,
            )
            rgb = tvf_crop(rgb, i, j, h, w)
            # scale back to original size 
            rgb = tvf_resize(rgb, original_size, antialias=True)  
            masks = [
                tvf_crop(mask, i, j, h, w) for mask in masks
            ] 
            # print('after resizing', torch.stack(masks)[(masks != 0) & (masks != 1)])
            masks = [
                tvf_resize(mask.unsqueeze(0), original_size, antialias=True).squeeze(0) for mask in masks
            ]
        masks = torch.stack(masks, dim=0) 
        masks = (masks > 0).to(torch.float32) # make sure it's binary! after resizing the values are messy
        if center_crop:
            rgb = centercrop(rgb)
        if pad:
            padding = (0, 240) if original_size[0] == 1440 else (240, 0)
            padder = torchvision.transforms.Pad(padding, fill=0)
            rgb = padder(rgb)
            assert rgb.shape[2] == rgb.shape[1] == 1920, f"Expected 1920, got {rgb.shape}"
        rgb = resize_fn.apply_image_torch(rgb.unsqueeze(0)).squeeze(0)
        
        return rgb, masks, original_size

    return transforms

def forward_sam(model, batch, device, original_size=(480, 480)): 
    rgb = batch["image"].to(device) 
    rgb = model.preprocess(rgb)
    image_embeddings = model.image_encoder(rgb)
    points = None  
    sparse_embeddings, dense_embeddings = model.prompt_encoder(points=points, boxes=None, masks=None)
    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
    )
    pred = model.postprocess_masks(
                low_res_masks,
                input_size=(1024, 1024),
                original_size=original_size,
            )
    # pred = pred > model.mask_threshold

    outputs = {
            "pred": pred, # B, 9, original_size
            "iou_predictions": iou_predictions,
            "low_res_logits": low_res_masks,
        }
    return outputs

def forward_sam_points(model, batch, device, original_size=(480, 480)):
    """ NOTE: mask_decoder and prompt_encoder assumes only one image input, and take in BxNx... shape, where B is number of masks, N is number of points per mask """
    rgb = batch["image"].to(device) 
    if isinstance(model, DataParallel):
        model = model.module
    rgb = model.preprocess(rgb)
    # breakpoint()
    image_embeddings = model.image_encoder(rgb)
    bs = rgb.shape[0]
    all_outputs = []
    for b in range(bs): 
        points = (batch['point_coords'][b].to(device), batch['point_labels'][b].to(device)) # shape: (num_masks, num_points=1, 2), (num_masks, num_points=1)
        sparse_embeddings, dense_embeddings = model.prompt_encoder(points=points, boxes=None, masks=None)
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embeddings[b:b+1], # shape 1,3,1024,1024
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
    
        pred = model.postprocess_masks(
                    low_res_masks,
                    input_size=(1024, 1024),
                    original_size=original_size,
                ) # shape (num_mask, 1, 480, 480) 
        outputs = {
                "pred": pred,
                "iou_predictions": iou_predictions,
                "low_res_logits": low_res_masks,
            }
        all_outputs.append(outputs)
    # concat all outputs
    stack_outputs = dict()
    for k in all_outputs[0].keys():
        stack_outputs[k] = torch.stack([o[k] for o in all_outputs], dim=0).squeeze(2)
    return stack_outputs

def get_wandb_table(batch, preds, mask_threshold, softmax=False):
    num_masks = preds.shape[1]
    columns = ["rgb"]
    for i in range(num_masks):
        columns.extend([f"pred_{i}", f"label_{i}"])
    table = wandb.Table(columns=columns)
    batch_size = preds.shape[0]
    labels = batch["masks"]
    if softmax:
        preds = preds.softmax(axis=1)
    for i in range(batch_size):
        rgb = batch["image"][i].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        pred = preds[i].detach().cpu().numpy()  
        pred = (pred > mask_threshold).astype(np.uint8) 
        label = labels[i].detach().cpu().numpy().astype(np.uint8)
        row = [wandb.Image(rgb)]
        
        for j in range(num_masks):
            row.extend([wandb.Image(pred[j]), wandb.Image(label[j])])
        table.add_data(*row)
    return table

def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon) 
    # batch_iou = batch_iou.unsqueeze(1)
    return batch_iou

def compute_iou(preds: torch.Tensor, labels: torch.Tensor):
    """ """
    batch_size = preds.shape[0]
    ious = []
    for i in range(batch_size):
        pred = preds[i]
        label = labels[i] # shape (num_masks, h, w)
        iou = calc_iou(pred, label) # shape (num_masks)
        ious.append(iou)
    ious = torch.stack(ious, dim=0)
    return ious

def get_loss_fn(fc_weight=20, min_loss=False):
    dc_loss_fn = DiceCELoss(sigmoid=True, squared_pred=True)
    fc_loss_fn = FocalLoss()
    if min_loss:
        dc_loss_fn = DiceLoss(sigmoid=True, squared_pred=True, reduction="none")
        fc_loss_fn = FocalLoss(reduction="none")
        # NOTE: dc loss seems fine but focal loss assigns higher loss to first mask
    def compute_loss(mask_preds, iou_preds, labels):
        num_masks = labels.shape[1]
        if min_loss:
            # only backprop the lowest loss for each prediction
            # assume mask_preds and labels are shape (B, num_masks, H, W)
            dc_losses, fc_losses, iou_losses = [], [], []
            bsize = mask_preds.shape[0]
            num_masks = labels.shape[1]
            for b in range(bsize):
                b_preds = mask_preds[b] # shape (num_masks, H, W) 
                for pred in b_preds: 
                    # compute this prediction to all labels
                    pred = pred.unsqueeze(0).repeat(num_masks, 1, 1)
                    for loss_arr, loss_fn in zip([dc_losses, fc_losses], [dc_loss_fn, fc_loss_fn]):
                        loss = loss_fn(pred, labels[b])
                        # take min
                        loss = loss.flatten(1).mean(dim=1)
                        loss_arr.append(loss.min())
                    # compute iou loss
                    gt_iou = calc_iou(pred, labels[b])
                    iou_loss = F.mse_loss(iou_preds[b], gt_iou, reduction="none").min()       
                    iou_losses.append(iou_loss)                                 
            # breakpoint()
            dc_loss = torch.stack(dc_losses).mean()
            fc_loss = torch.stack(fc_losses).mean()
            iou_loss = torch.stack(iou_losses).mean() 
        else:
            fc_loss = fc_loss_fn(mask_preds, labels)
            dc_loss = dc_loss_fn(mask_preds, labels)
            gt_iou = compute_iou(mask_preds, labels) 
            iou_loss = F.mse_loss(iou_preds, gt_iou, reduction="sum")/num_masks

        total_loss = fc_weight * fc_loss + dc_loss + iou_loss

        return total_loss, fc_loss, dc_loss, iou_loss
    return compute_loss

def eval_model(args, model, val_loader, loss_fn, device, forward_fn, epoch=0, original_size=(480, 480)):
    # eval
    model.eval()
    val_total_loss, val_fc_loss, val_dc_loss = 0, 0, 0
    val_iou_loss = 0
    for val_step, val_batch in tqdm(
        enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch}"
    ):
        with torch.no_grad():
            if args.use_dp:
                outputs = model(val_batch, device=device, original_size=original_size)
            else:
                outputs = forward_fn(
                    model, val_batch, device=device, original_size=original_size)
            preds = outputs["pred"] 
            labels = val_batch["masks"].to(device).detach() 
            total_loss, fc_loss, dc_loss, iou_loss = loss_fn(
                preds, outputs["iou_predictions"], labels)
            val_total_loss += total_loss
            val_fc_loss += fc_loss
            val_dc_loss += dc_loss
            val_iou_loss += iou_loss
        if val_step % args.log_freq == 0:
            print(f"epoch: {epoch} | val_loss: {val_total_loss / (val_step+1)}")
        
    if args.wandb:
        mask_threshold = model.module.mask_threshold if isinstance(model, DataParallel) else model.mask_threshold
        table = get_wandb_table(val_batch, preds, mask_threshold)
        wandb.log(
            {
                "val_total_loss": val_total_loss / val_step,
                "val_focal_loss": val_fc_loss / val_step,
                "val_dice_loss": val_dc_loss / val_step,
                "val_iou_loss": val_iou_loss / val_step,
                "val_epoch": epoch,
                f"Pred_Val_epoch{epoch}": table,
                
            }
        )
    return

def save_model_decoder(model, save_name):
    if isinstance(model, DataParallel):
        model = model.module
    torch.save(model.mask_decoder.state_dict(), save_name)

def main(args): 
    # try dataset & loader
    # dataset_cls = MobilityDataset
    forward_fn = forward_sam
    dataset_kwargs = dict(
        root_dir=args.data_dir, 
        rebuttal_objects_only=args.rebuttal_objects,
    )
    forward_fn = forward_sam_points
    dataset_kwargs["prompts_per_mask"] = args.prompts_per_mask
    dataset_cls = SamH5Dataset
    original_size = (512, 512)
    dataset_kwargs['point_mode'] = True
    dataset_kwargs['grid_size'] = args.grid_size
    dataset_kwargs['max_background_masks'] = args.max_bg
    dataset_kwargs['use_cache'] = args.cache
    # if args.cache:
    #     assert args.num_data_workers == 0, "num_data_workers must be 0 when using cache"
    #     print('Using cache and repeated rgb for training dataset only')
    dataset = dataset_cls(
        **dataset_kwargs,
        transform=get_image_transform(1024, jitter=True, random_crop=True), 
    )
    sampler = None
    if args.weight_sample:
        # use weighted random sampler
        weights = dataset.get_obj_weights()
        assert len(weights) == len(dataset), f"weights {len(weights)} must be same length as dataset {len(dataset)}"
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    loader = DataLoader(
        dataset, sampler=sampler, batch_size=args.batch_size, 
        shuffle=(False if args.weight_sample else True), num_workers=args.num_data_workers)
    
    # item = next(iter(loader))
    # save imgs for debug:
    # imgs = item["image"]
    # for i, img in enumerate(imgs):
    #     save_img = Image.fromarray(img.permute(1, 2, 0).numpy().astype(np.uint8))
    #     save_img.save(f'jittered_{i}.png')  
    val_dataset_kwargs = dataset_kwargs.copy()  
    val_dataset_kwargs["use_cache"] = False # don't use cache for val
    val_dataset_kwargs["loop_id"] = 0 #only use the first loop for val
    val_dataset = dataset_cls(
        **val_dataset_kwargs, is_train=False, transform=get_image_transform(1024, jitter=False, random_crop=False)
        )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_data_workers)


    run_name = f"{args.run_name}_pointsTrue_lr{args.lr}_bs{args.batch_size}_ac{args.grad_accum_steps}"
    run_name += f"_{datetime.now().strftime('%m-%d_%H-%M')}"
    run_dir = join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=False)
    with open(join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f)

    ckpt = None if args.skip_load else CKPT_PATH
    model = sam_model_registry[args.sam_type](checkpoint=ckpt)
    # freeze img encoder
    for name, param in model.named_parameters():
        if name.startswith("image_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    print("Model loaded, try reset head") 

    extend_instance(model, PointsQueryMixin)
    trainable = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Trainable params: {trainable}")
    frozen = sum([p.numel() for p in model.parameters() if not p.requires_grad])
    print(f"Frozen params: {frozen}")
    # breakpoint()
    learnable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(learnable_params, lr=args.lr) 

    if args.load_run != "":
        if args.load_epoch == -1:
            load_fname = natsorted(glob(join(args.output_dir, args.load_run, "ckpt_step_*.pth")))[-1]
        else:
            load_fname = join(args.output_dir, args.load_run, f"ckpt_epoch_{args.load_epoch}.pth")
        assert os.path.exists(load_fname), f"File not found: {load_fname}"
        loaded_decoder = torch.load(load_fname, map_location="cpu")
        model.mask_decoder.load_state_dict(loaded_decoder)
        optimizer_fname = load_fname.replace("ckpt", "optim")
        if os.path.exists(optimizer_fname):
            optimizer_state = torch.load(optimizer_fname, map_location="cpu") 
            optimizer.load_state_dict(optimizer_state)
            print(f"Loaded optimizer state from: {optimizer_fname}")
        print(f"Loaded ckpt from: {load_fname}")

    device = torch.device('cuda:0')

    if args.use_dp:
        model = DataParallel(model, device_ids=list(range(args.dp_devices)))
    model.to(device)
    

    if args.wandb:
        run = wandb.init(project="real2code", entity="mandi", group="sam", job_type="train")
        run.name = run_name
        wandb.config.update(vars(args))
    total_step = 0 
    loss_fn = get_loss_fn(fc_weight=args.fc_weight, min_loss=args.min_loss)
    for epoch in range(args.epochs):
        if epoch > 0:
            eval_model(args, model, val_loader, loss_fn, device=device, epoch=epoch, forward_fn=forward_fn, original_size=original_size)
        model.train()
        for step, batch in tqdm(
            enumerate(loader), total=len(loader), desc=f"Epoch {epoch}"
        ):
            if args.use_dp:
                outputs = model(batch, device=device, original_size=original_size)
            else:
                outputs = forward_fn(model, batch, device=device, original_size=original_size)
            preds = outputs["pred"]
            # preds = torch.as_tensor(preds, dtype=torch.float32, requires_grad=True)
            labels = batch["masks"].to(device) #.detach()
            iou_preds = outputs["iou_predictions"]
            total_loss, fc_loss, dc_loss, iou_loss = loss_fn(preds, iou_preds, labels)
            total_loss /= args.grad_accum_steps
            total_loss.backward()
            if (step+1) % args.grad_accum_steps == 0: 
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                 
            total_step += 1
            if total_step % args.log_freq == 0:
                print(f"epoch: {epoch} | step: {step} focal: {fc_loss} | dice: {dc_loss} | iou: {iou_loss} | total: {total_loss}")
                if args.wandb:
                    wandb.log(
                        {
                            "focal_loss": fc_loss,
                            "dice_loss": dc_loss,
                            "iou_loss": iou_loss,
                            "total_loss": total_loss,
                            "total_step": total_step, 
                        }
                    )
            if args.wandb and total_step % args.vis_freq == 0:
                # visualize predictions
                mask_threshold = model.module.mask_threshold if isinstance(model, DataParallel) else model.mask_threshold
                table = get_wandb_table(batch, preds, mask_threshold)
                wandb.log({
                    f"Pred_Train/step_{total_step}": table,
                    "vis_step": total_step,
                    })
            if total_step % args.save_freq == 0:
                save_fname = f"ckpt_step_{total_step}.pth"
                # save only the decoders
                save_model_decoder(model, join(run_dir, save_fname))
                optimizer_state = optimizer.state_dict()
                save_optim_fname = f"optim_step_{total_step}.pth"
                torch.save(optimizer_state, join(run_dir, save_optim_fname))
       
        save_fname  = f"ckpt_epoch_{epoch}.pth"
        save_model_decoder(model, join(run_dir, save_fname))
        optimizer_state = optimizer.state_dict()
        save_optim_fname = f"optim_epoch_{epoch}.pth"

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/local/real/mandi/real2code_dataset_v0")
    parser.add_argument("--output_dir", type=str, default="/store/real/mandi/sam_models")    
    parser.add_argument("--load_run", type=str, default="")
    parser.add_argument("--load_epoch", type=int, default=-1)
    parser.add_argument("--sam_type", default="default", type=str)
    parser.add_argument("--skip_load","-sl", action="store_true") 
    parser.add_argument("--run_name", "-rn", default="sam_tune", type=str)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", "-bs", default=24, type=int)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--grad_accum_steps", "-ac", default=12, type=int)
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--vis_freq", default=2000, type=int)
    parser.add_argument("--save_freq", default=500, type=int)
    parser.add_argument("--fc_weight", default=1, type=int) 
    parser.add_argument("--prompts_per_mask", type=int, default=16)
    parser.add_argument("--num_data_workers", type=int, default=0)
    parser.add_argument("--blender", default=True, action="store_true")
    parser.add_argument("--min_loss", action="store_true")
    parser.add_argument("--grid_size", type=int, default=1)
    parser.add_argument("--max_bg", type=int, default=3)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--weight_sample", action="store_true")
    parser.add_argument("--use_dp", action="store_true")
    parser.add_argument("--dp_devices", '-dp', type=int, default=2)
    parser.add_argument("--rebuttal_objects", action="store_true")
    args = parser.parse_args()
    main(args)
