import torch
import wandb
from lfdtn.LFT import compact_LFT, compact_iLFT
from asset.utils import getPhaseDiff, getPhaseAdd, clmp, listToTensor, showSeq, dimension, li, ui, show_phase_diff, \
    logMultiVideo, wandblog,dmsg,getAvgDiff,manyListToTensor,colorChannels,randColors
from lfdtn.window_func import  get_2D_Gaussian,get_ACGW
from torch.functional import F

def stitch_back(INP,config):
    ## Stitch back!
    if config.channel_multiplier>1:
        chs = INP.shape[1]//config.channel_multiplier
        shapeNS = list(INP.shape)
        shapeNS[1] = shapeNS[1]//config.channel_multiplier
        CHRES=torch.zeros(shapeNS).to(INP.device)
        for ci in range(config.channel_multiplier):
            CHRES = CHRES + INP[:,ci::config.channel_multiplier]
        return clmp(CHRES, False)
    else:
        return INP

def separate(INP,History,T,E,window,model,PD_model,useModel,config):
    if config.channel_multiplier>1:
        if useModel:
            if not config.PD_model_enable:
                if config.channel_multiplier_history>0:
                    return model(History[:,-(config.channel_multiplier_history+1):].squeeze(2))
                else:
                    return model(INP)
                
            T2 = PD_model(T,E)
            MEs, _ = compact_iLFT(T2, window, None, config,is_inp_complex=False,move_window_according_T=False,channel=config.input_channels)
            return model(torch.cat([INP,MEs],dim=1))

        else:
            return torch.cat([INP]*config.channel_multiplier,dim=1)/config.channel_multiplier
    else:
        return INP
 

def predict(dataSet,GT, window,config,MRef_Out,M_transform,segModel,multiplyModel,PD_model,loclConM,phase='train', log_vis=True):
    eps = 1e-8
    angAfterNorm = torch.tensor(0., requires_grad=True)
    channelVar = torch.tensor(0., requires_grad=True)
    T=None
    Energy= None
    BS,SQ,CS,HH,WW = dataSet.shape
    if log_vis:
        T_hist = []
        T_ref_hist = []
    ###seed
    for i in range(1,config.sequence_seed):
        if config.firstSeg:
            SP = segModel(separate(dataSet[:,i-1],None,T,Energy,window,multiplyModel,PD_model,False,config))
            SN = segModel(separate(dataSet[:,i],dataSet[:,:i],T,Energy,window,multiplyModel,PD_model,i==config.sequence_seed-1,config))
        else:
            SP = separate(dataSet[:,i-1],None,T,Energy,window,multiplyModel,PD_model,False,config)
            SN = separate(dataSet[:,i],dataSet[:,:i],T,Energy,window,multiplyModel,PD_model,i==config.sequence_seed-1,config)
        S_curr = SN #torch.Size([BS, CS, HH, WW])

        if  i==1:
            PResPrePure = [SP,SN]
            SPR = stitch_back(SP,config)
            SNR = stitch_back(SN,config)
            if config.firstSeg:
                PRes = [SPR,SNR]
            else:
                PResPre = [SPR,SNR]
                PRes = [segModel(SPR),segModel(SNR)]
                
            if log_vis:
                WTs = [torch.zeros_like(SN), torch.zeros_like(SN)]
        else:
            PResPrePure.append(SN)
            SNR = stitch_back(SN,config)
            if config.firstSeg:
                PRes.append(SNR)
            else:
                PResPre.append(SNR)
                PRes.append(segModel(SNR))

            if log_vis:
                WTs.append(torch.zeros_like(SN))

        # dmsg('SN.shape')#torch.Size([BS, CS, HH, WW])
        lLFT,lCOM = compact_LFT(SN+ eps, window, config,loclConM)
        bLFT,bCOM = compact_LFT(SP+ eps, window, config,loclConM)
        T,Energy = getPhaseDiff(lLFT,bLFT \
                        ,config.use_energy)


        # dmsg('T.shape')#torch.Size([BS,L_y*L_x*CS, N_p, N_p, 2])
        T, visB, visA, _ = M_transform(T,Energy, lCOM,
                                            first = True if i==config.start_T_index else False,
                                            throwAway = True if i<config.start_T_index else False)

        # dmsg('T.shape')#torch.Size([BS,L_y*L_x*CS, N_p, N_p, 2])
        if log_vis:
            T_hist.append(visB)
            T_ref_hist.append(visA)



    ###future frame prediction loop
    for i in range(config.sequence_length - config.sequence_seed):

        # dmsg('S_curr.shape','T.shape')# S_curr = torch.Size([BS, CS, HH, WW]) ! T = torch.Size([BS*CS, L_y*L_x, N_p, N_p, 2])
        ###get LFT of current frame
        S_fft,lCOM = compact_LFT(S_curr + eps, window, config,loclConM)
        # dmsg('S_fft.shape')#S_fft = torch.Size([BS,L_y*L_x*CS, N_p, N_p, 2])
        # dmsg('T.shape')#torch.Size([BS, L_y*L_x*CS, N_p, N_p, 2])

        #ET -> extended T
        # ET = T.reshape_as(S_fft)
        ET = T
        # dmsg('ET.shape')#torch.Size([BS,L_y*L_x*CS, N_p, N_p, 2])

        ###apply transform
        NS_fft = getPhaseAdd(S_fft, ET)
        # dmsg('NS_fft.shape')#([BS,L_y*L_x*CS, N_p, N_p, 2])
        ###reconstruction, also get OLA denominator
        NS_, WT = compact_iLFT(NS_fft, window, ET, config,channel=S_curr.shape[1])
        PResPrePure.append(NS_)
        if config.refine_output:
            if config.PD_model_enable and config.PD_model_in_refine_output:
                MEs, _ = compact_iLFT(PD_model(ET,Energy), window, None, config,is_inp_complex=False,move_window_according_T=False,channel=S_curr.shape[1])
            
            if config.refine_output_share_weight:
                inputCR = NS_.view(-1,1,HH,WW)
                if config.PD_model_enable and config.PD_model_in_refine_output:
                    MEs = MEs.view(-1,1,HH,WW)
                    inputCR=torch.cat([inputCR,MEs],dim=1)
                NS_ = MRef_Out(inputCR).view(BS,-1,HH,WW)
            else:
                inputCR = NS_
                if config.PD_model_enable and config.PD_model_in_refine_output:
                    inputCR=torch.cat([inputCR,MEs],dim=1)
                NS_ = MRef_Out(inputCR)
            NS=NS_
        else:
            ###clamp to [0,1]
            NS = clmp(NS_, False)
        ###collect new frame and window overlap
        
        NSR = stitch_back(NS,config)

        if config.firstSeg:
            PRes.append(NSR)
        else:
            PResPre.append(NSR)
            PRes.append(segModel(NSR))


        if log_vis:
            WTs.append(WT)
        ###prepare for next frame prediction loop
        ###update T
        if config.enable_autoregressive_prediction:
            lLFT,lCOM = compact_LFT(NS + eps, window, config,loclConM)
            T,Energy = getPhaseDiff(lLFT, S_fft,config.use_energy)
        else:
            Energy = None

        ###apply transform model
        T, visB, visA, angAfterNormTmp = M_transform(T,Energy, lCOM,
                                    first = False,
                                    throwAway =False)
        angAfterNorm = angAfterNorm + angAfterNormTmp
        if log_vis:
            T_hist.append(visB)
            T_ref_hist.append(visA)


        S_curr = NS

    PRes = listToTensor(PRes)
    PResPrePure = listToTensor(PResPrePure)
    if not config.firstSeg:
        PResPre = listToTensor(PResPre)

    if (torch.isnan(PRes).any()):
        # return False,False
        print(torch.pow((dataSet[:, :, 2:] - PRes[:, :, 2:]).cpu(), 2).mean().item())
        print(torch.isnan(dataSet - PRes).any())
        raise Exception("NAN Exception")



    if log_vis:
        with torch.no_grad():
            WTs = listToTensor(WTs)
            vis_image_string = phase+" predictions"
            vis_anim_string = phase+' animations'
            header = phase+': '
            setting = {'oneD': (dimension == 1), 'revert': True, 'dpi': 2.4, 'show': False,"vmin":0,"vmax":1}
            L2loss = torch.pow((GT - PRes), 2) #L1loss[li:ui]
            L1loss = 0.5 * (PRes.clamp(0, 1) - GT.clamp(0, 1)) + 0.5

            PResShow = colorChannels(PRes,2,False)
            PResPrePureShow = colorChannels(PResPrePure,2,False)
            if config.predictSegmentedGT:
                PResPreShow = PResShow
            else:
                if not config.firstSeg:
                    PResPreShow = colorChannels(PResPre,2,True)
                else:
                    PResPreShow = PResShow
            GTShow = colorChannels(GT,2,False)
            dataSetShow = colorChannels(dataSet,2,True)
            L1lossShow = colorChannels(L1loss,2,True)
            L2lossShow = colorChannels(L2loss,2,True)
            WTsShow = colorChannels(WTs,2,True)
            # dmsg('PRes.shape','dataSet.shape','WTs.shape')#PRes = torch.Size([1, 10, 17, 129, 129]) ! dataSet = torch.Size([1, 10, 17, 129, 129]) ! WTs = torch.Size([1, 10, 17, 129, 129])
            pic = showSeq(False, -1, "PResPrePure,PResPre,PRes,GT,dataSet,L1loss,L2loss,WTs", [PResPrePureShow[li:ui].detach().cpu(),PResPreShow[li:ui].detach().cpu(),PResShow[li:ui].detach().cpu(),
                        GTShow[li:ui].cpu(),dataSetShow[li:ui].cpu(), L1lossShow[li:ui].detach().cpu(),L2lossShow[li:ui], WTsShow[li:ui].clamp(0, 1).detach().cpu()],
                          **setting)
            ####get linear shift encoded in phase diffs
            # dmsg('xx.shape', 'yy.shape', 'visX.shape', 'visY.shape')


            show_phase_diff(pd_list=T_hist, gt=1-PResPreShow, config=config, title=header + "VF from Phase Diffs",clear_motion=False)
            
            
            show_phase_diff(pd_list=T_ref_hist, gt=1-PResPreShow, config=config, title=header + "VF after M_Transform",clear_motion=False)

            # print('logging stuff')
            wandblog({vis_image_string: pic},commit=False)

            pt = PResShow[li:ui].detach().cpu()
            ptpre = PResPreShow[li:ui].detach().cpu()
            gt = GTShow[li:ui].cpu()
            dt = dataSetShow[li:ui].cpu()
            logMultiVideo('Prediction, GT, Diff', [ptpre,pt, gt,dt, 0.5 * (pt.clamp(0, 1) - gt.clamp(0, 1)) + 0.5],config.sequence_seed,
                          vis_anim_string=vis_anim_string)


    if config.channel_multiplier>1:
        channelVar = PResPrePure.mean(dim=[0,1,2,3]).var()
    angAfterNorm = angAfterNorm / float(config.sequence_length - config.sequence_seed)
    return PRes, angAfterNorm,channelVar

