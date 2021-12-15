import time
import torch
import torch.nn as nn
import src.utils as utils


def train_fixbi(args, loaders, optimizers, schedulers, models_sd, models_td, sp_params, losses, epoch,met_file):
    print("Epoch: [{}/{}]".format(epoch, args.epochs))
    start = time.time()
    if epoch < args.bim_start:
        src_train_loader, tgt_train_loader = loaders[0], loaders[1]
    else:
        src_train_loader, tgt_train_loader = loaders[2], loaders[3]
    optimizer_sd, optimizer_td = optimizers[0], optimizers[1]
    lr_ratio = optimizers[2]
    scheduler_sd, scheduler_td = schedulers[0], schedulers[1]
    sp_param_sd, sp_param_td = sp_params[0], sp_params[1]
    ce, mse = losses[0], losses[1]

    utils.set_model_mode('train', models=models_sd)
    utils.set_model_mode('train', models=models_td)

    models_sd = nn.Sequential(*models_sd)
    models_td = nn.Sequential(*models_td)

    fixmix_sd_loss, fixmix_td_loss = 0, 0
    bim_sd_loss, bim_td_loss = 0, 0 
    tsp_sd_loss, tsp_td_loss = 0, 0
    tcr_loss = 0

    for step, (src_data, tgt_data) in enumerate(zip(src_train_loader, tgt_train_loader)):
        src_imgs, src_labels = src_data
        tgt_imgs, tgt_labels = tgt_data
        src_imgs, src_labels = src_imgs.cuda(non_blocking=True), src_labels.cuda(non_blocking=True)
        tgt_imgs, tgt_labels = tgt_imgs.cuda(non_blocking=True), tgt_labels.cuda(non_blocking=True)

        x_sd, x_td = models_sd(tgt_imgs), models_td(tgt_imgs)

        pseudo_sd, top_prob_sd, threshold_sd = utils.get_target_preds(args.th, x_sd)
        pseudo_td, top_prob_td, threshold_td = utils.get_target_preds(args.th, x_td)

        f_sd_loss = utils.get_fixmix_loss(models_sd, src_imgs, tgt_imgs, src_labels, pseudo_sd, args.lam_sd)
        f_td_loss = utils.get_fixmix_loss(models_td, src_imgs, tgt_imgs, src_labels, pseudo_td, args.lam_td)

        total_loss = f_sd_loss + f_td_loss

        fixmix_sd_loss += f_sd_loss
        fixmix_td_loss += f_td_loss

        # Bidirectional Matching
        if epoch >= args.bim_start:
            bim_mask_sd = torch.ge(top_prob_sd, threshold_sd)
            bim_mask_sd = torch.nonzero(bim_mask_sd).squeeze()

            bim_mask_td = torch.ge(top_prob_td, threshold_td)
            bim_mask_td = torch.nonzero(bim_mask_td).squeeze()

            if bim_mask_sd.dim() > 0 and bim_mask_td.dim() > 0:
                if bim_mask_sd.numel() > 0 and bim_mask_td.numel() > 0:
                    bim_mask = min(bim_mask_sd.size(0), bim_mask_td.size(0))
                    b_sd_loss = ce(x_sd[bim_mask_td[:bim_mask]], pseudo_td[bim_mask_td[:bim_mask]].cuda().detach())
                    b_td_loss = ce(x_td[bim_mask_sd[:bim_mask]], pseudo_sd[bim_mask_sd[:bim_mask]].cuda().detach())

                    total_loss += (b_sd_loss + b_td_loss)*lr_ratio

                    bim_sd_loss += b_sd_loss
                    bim_td_loss += b_td_loss

        # Self-penalization
        if epoch < args.bim_start and epoch >= args.sp_start:
            sp_mask_sd = torch.lt(top_prob_sd, threshold_sd)
            sp_mask_sd = torch.nonzero(sp_mask_sd).squeeze()

            sp_mask_td = torch.lt(top_prob_sd, threshold_td)
            sp_mask_td = torch.nonzero(sp_mask_td).squeeze()

            if sp_mask_sd.dim() > 0 and sp_mask_td.dim() > 0:
                if sp_mask_sd.numel() > 0 and sp_mask_td.numel() > 0:
                    sp_mask = min(sp_mask_sd.size(0), sp_mask_td.size(0))
                    sp_sd_loss = utils.get_sp_loss(x_sd[sp_mask_sd[:sp_mask]], pseudo_sd[sp_mask_sd[:sp_mask]], sp_param_sd)
                    sp_td_loss = utils.get_sp_loss(x_td[sp_mask_td[:sp_mask]], pseudo_td[sp_mask_td[:sp_mask]], sp_param_td)

                    total_loss += sp_sd_loss + sp_td_loss

                    tsp_sd_loss += sp_sd_loss
                    tsp_td_loss += sp_td_loss

        # Consistency Regularization
        if epoch >= args.cr_start:
            mixed_cr = 0.5 * src_imgs + 0.5 * tgt_imgs
            out_sd, out_td = models_sd(mixed_cr), models_td(mixed_cr)
            cr_loss = mse(out_sd, out_td)
            total_loss += cr_loss*lr_ratio
            tcr_loss += cr_loss

        optimizer_sd.zero_grad()
        optimizer_td.zero_grad()
        total_loss.backward()
        optimizer_sd.step()
        optimizer_td.step()
    
    scheduler_sd.step()
    scheduler_td.step()

    print('Fixed MixUp Loss (SDM): {:.4f}'.format(fixmix_sd_loss.item()/step))
    print('Fixed MixUp Loss (TDM): {:.4f}'.format(fixmix_td_loss.item()/step))

    e="Epoch: [{}/{}]\n".format(epoch, args.epochs)
    s1='Fixed MixUp Loss (SDM): {:.4f}\n'.format(fixmix_sd_loss.item()/step)
    s2='Fixed MixUp Loss (TDM): {:.4f}\n'.format(fixmix_td_loss.item()/step)
    file1 = open(met_file, 'a')
    file1.write(e)
    file1.write(s1)
    file1.write(s2)
    
    if tsp_sd_loss+tsp_td_loss > 0:
        print('Penalization Loss (SDM): {:.4f}', tsp_sd_loss/step)
        print('Penalization Loss (TDM): {:.4f}', tsp_td_loss/step)
        p1='Penalization Loss (SDM): {:.4f}\n'.format(tsp_sd_loss/step)
        p2='Penalization Loss (TDM): {:.4f}\n'.format(tsp_td_loss/step)
        file1.write(p1)
        file1.write(p2)
    if epoch >= args.bim_start:
        print('Bidirectional Loss (SDM): {:.4f}'.format(bim_sd_loss.item()/step))
        print('Bidirectional Loss (TDM): {:.4f}'.format(bim_td_loss.item()/step))
        b1='Bidirectional Loss (SDM): {:.4f}\n'.format(bim_sd_loss.item()/step)
        b2='Bidirectional Loss (TDM): {:.4f}\n'.format(bim_td_loss.item()/step)
        file1.write(b1)
        file1.write(b2)

    if epoch >= args.cr_start:
        print('Consistency Loss: {:.4f}', tcr_loss.item()/step)
        
        c='Consistency Loss: {:.4f}\n'.format(tcr_loss.item()/step)
        file1.write(c)

    print("Train time: {:.2f}".format(time.time() - start))
    file1.close()


def pretrain(args, loader, optimizer, scheduler, models_sd, criterion, epoch, met_file):
    print("Pre-Train Epoch: [{}/{}]".format(epoch, args.pretrain_epochs))
    start = time.time()

    utils.set_model_mode('train', models=models_sd)

    models_sd = nn.Sequential(*models_sd)

    for step, data in enumerate(loader):
        src_imgs, src_labels = data
        src_imgs, src_labels = src_imgs.cuda(non_blocking=True), src_labels.cuda(non_blocking=True)

        x_sd = models_sd(src_imgs)

        total_loss = criterion(x_sd, src_labels.detach())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
    scheduler.step()

    print('Pre-Training Loss (SDM): {:.4f}'.format(total_loss.item()/step))

    e="Epoch: [{}/{}]\n".format(epoch, args.epochs)
    s1='Pre-Training Loss (SDM): {:.4f}\n'.format(total_loss.item()/step)
    file1 = open(met_file, 'a')
    file1.write(e)
    file1.write(s1)
    
    print("Train time: {:.2f}".format(time.time() - start))
    file1.close()