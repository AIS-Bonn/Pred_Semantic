import torch
import torch.nn as nn
from asset.utils import getAvgDiff, createPhaseDiff, justshift, get_delta,positionalencoding2d,dmsg,DenseNetLikeModel
from lfdtn.custom_fft import custom_fft


class cellTransportRefine(nn.Module):
    def __init__(self, config):
        super(cellTransportRefine, self).__init__()
        self.res_y = config.res_y_constrained
        self.res_x = config.res_x_constrained
        self.y_stride = config.stride
        self.x_stride = config.stride
        self.pS = config.window_padding_constrained
        self.N = config.window_size
        self.N_prime = self.N + 2 * self.pS

        self.padSizeAdjust = int(self.x_stride * ((self.N_prime - 1) // self.x_stride))
        self.L_y = (self.res_y + 2 * self.padSizeAdjust - self.N_prime) // self.y_stride + 1
        self.L_x = (self.res_x + 2 * self.padSizeAdjust - self.N_prime) // self.x_stride + 1

        self.mode = config.trans_mode  # 'Fc', 'Conv', 'IFc', 'IConv'
        self.Recpt = config.recpt_IConv
        self.prev_x_c = config.concat_depth
        self.untilIndex = config.untilIndex  # Can be 'None'
        self.fcMiddle = config.fcMiddle * 10
       
        self.denseNet = True
        self.pose_enc_level = 4
        moreC = self.pose_enc_level if config.pos_encoding else 0

        inpDim = 4 if config.use_variance else 2
        if config.useCOM:
            inpDim+=2+config.localContentC

        self.dimentionMultiplier =config.input_channels
        self.hiddenFC = 6*self.dimentionMultiplier
        self.nonLin = eval('torch.nn.'+config.tr_non_lin+"()")

        if self.mode == 'Conv':
            if config.shareChannelWeight:
                self.cnn = DenseNetLikeModel( inputC=(inpDim * (self.prev_x_c + 1))+moreC,
                            outputC=2, layerCount=len(config.tran_filter_sizes), hiddenF=config.tran_hidden_unit,
                                filterS=config.tran_filter_sizes, nonlin = config.tr_non_lin,lastNonLin=False)
            else:
                self.cnn = DenseNetLikeModel( inputC=(inpDim *self.dimentionMultiplier * (self.prev_x_c + 1))+moreC,
                            outputC=2*self.dimentionMultiplier, layerCount=len(config.tran_filter_sizes), hiddenF=config.tran_hidden_unit,
                                filterS=config.tran_filter_sizes, nonlin = config.tr_non_lin,lastNonLin=False)

        self.prev_x = []
        self.config = config
        self.pos_encoding = None
        dmsg('self.L_x','self.L_y','self.N_prime')

    def reparameterize(self, mu, log_var,first):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        # if first:
        self.eps = torch.randn_like(std) # `randn_like` as we need the same size
        return self.eps.mul(std).add(mu)# sampling

    # @profile
    def forward(self, x,energy,COM, first,throwAway):
        # dmsg('COM.shape')#torch.Size([BS, L_y*L_x*CS, 2])
        BS=x.shape[0]
        # dmsg('x.shape')#torch.Size([BS, L_y*L_x*CS, N_p, N_p, 2])
        xV = x.view(-1, self.N_prime, self.N_prime, 2)
        

        if energy is not None:
            # dmsg('energy.shape')#torch.Size([BS, L_y*L_x*CS, N_p, N_p, 1])
            energy = energy.view(-1, self.N_prime, self.N_prime, 1)

        #dmsg('xV.shape')#torch.Size([BS*L_y*L_x*CS, N_p, N_p, 2])
        #dmsg('energy.shape')#torch.Size([BS*L_y*L_x*CS, N_p, N_p, 1])

        # energy = None
        tmpX,varianceX = getAvgDiff(rot=xV,energy=energy, step=1, axis=0, untilIndex=self.untilIndex,variance = self.config.use_variance)
        tmpY,varianceY = getAvgDiff(rot=xV,energy=energy, step=1, axis=1, untilIndex=self.untilIndex,variance = self.config.use_variance)

        # dmsg('tmpX.shape','tmpY.shape')#tmpX = torch.Size([BS*L_y*L_x*CS, 2]) ! tmpY = torch.Size([BS*L_y*L_x*CS, 2])
        tmpX = (torch.atan2(tmpX[:, 1], tmpX[:, 0] + 0.0000001) / (torch.pi * 2 / self.N_prime)).unsqueeze(-1)
        tmpY = (torch.atan2(tmpY[:, 1], tmpY[:, 0] + 0.0000001) / (torch.pi * 2 / self.N_prime)).unsqueeze(-1)

        # dmsg('tmpX.shape','tmpY.shape')#tmpX = torch.Size([BS*L_y*L_x*CS, 1]) ! tmpY = torch.Size([BS*L_y*L_x*CS, 1])
        tmpX = tmpX.view(-1, self.L_y, self.L_x,self.dimentionMultiplier).permute(0,3,1,2).contiguous()
        tmpY = tmpY.view(-1, self.L_y, self.L_x,self.dimentionMultiplier).permute(0,3,1,2).contiguous()

        if COM is not None:
            COMX = COM[:,:,0].view(-1,self.L_y,self.L_x,self.dimentionMultiplier).permute(0,3,1,2).contiguous()
            COMY = COM[:,:,1].view(-1,self.L_y,self.L_x,self.dimentionMultiplier).permute(0,3,1,2).contiguous()
            CONT =[]
            for ccc in range(self.config.localContentC):
                CONT.append(COM[:,:,ccc+2].view(-1,self.L_y,self.L_x,self.dimentionMultiplier).permute(0,3,1,2).contiguous())

        if self.config.use_variance:
            varianceX = varianceX.view(-1, self.L_y, self.L_x,self.dimentionMultiplier).permute(0,3,1,2).contiguous()
            varianceY = varianceY.view(-1, self.L_y, self.L_x,self.dimentionMultiplier).permute(0,3,1,2).contiguous()

        # dmsg('tmpX.shape','tmpY.shape')#tmpX = torch.Size([BS,CS,L_y,L_x]) ! tmpY = torch.Size([BS,CS,L_y,L_x])

        # tmpxToRet = tmpX.mean(dim=1)*self.dimentionMultiplier/2.
        # tmpyToRet = tmpY.mean(dim=1)*self.dimentionMultiplier/2.
        # tmpxToRet,_ = tmpX.max(dim=1)
        # tmpyToRet,_ = tmpY.max(dim=1)
            


        # tmpxToRet = tmpX[:,self.showIdx]
        # tmpyToRet = tmpY[:,self.showIdx]
        # dmsg('tmpxToRet.shape','tmpyToRet.shape')#torch.Size([BS,L_y,L_x]) ! torch.Size([BS,L_y,L_x])
        angBefore = [-tmpX.permute(0,2,3,1),
                     tmpY.permute(0,2,3,1)]  # tmpX is actually -y, tmpY is actually x

        if throwAway:
            return x, angBefore, angBefore, 0

        if "Conv" in self.mode:
            if self.config.shareChannelWeight:
                tmpX =  tmpX.view(-1,1,self.L_y, self.L_x)
                tmpY =  tmpY.view(-1,1,self.L_y, self.L_x)
                if self.config.use_variance:
                    varianceX =  varianceX.view(-1,1,self.L_y, self.L_x)
                    varianceY =  varianceY.view(-1,1,self.L_y, self.L_x)
                if self.config.useCOM:
                    COMX =  COMX.view(-1,1,self.L_y, self.L_x)
                    COMY =  COMY.view(-1,1,self.L_y, self.L_x)
                    for ccc in range(self.config.localContentC):
                        CONT[ccc] =CONT[ccc].view(-1,1,self.L_y, self.L_x)



            if self.config.use_variance and self.config.useCOM:
                lInp = torch.cat([tmpX, tmpY,varianceX,varianceY,COMX,COMY]+CONT, dim=1)
            elif self.config.use_variance:
                lInp = torch.cat((tmpX, tmpY,varianceX,varianceY), dim=1)
            elif self.config.useCOM:
                lInp = torch.cat([tmpX, tmpY,COMX,COMY]+CONT, dim=1)           
            else:
                lInp = torch.cat((tmpX, tmpY), dim=1)
            
        # dmsg('lInp.shape')
        if first:
            self.prev_x = [0.1 * torch.ones_like(lInp) for i in range(self.prev_x_c)]
        else:
            self.prev_x.pop(0)

        self.prev_x.append(lInp)  # TODO: think if we want the gradient for prev, or not!
        
        lInp = torch.cat(self.prev_x, dim=1)
        # dmsg('lInp.shape')#torch.Size([BS, [CS or 1]*[4 or 2]*Prev, L_y,L_x])
        angAfter = None
        logvar=None
        if self.mode == 'Conv':

            if self.config.pos_encoding:
                if self.pos_encoding is None:
                    self.pos_encoding = positionalencoding2d(self.pose_enc_level, lInp.shape[2], lInp.shape[3]).unsqueeze(
                        0).to(lInp.device).detach()
                # dmsg('self.pos_encoding.shape')#torch.Size([1, 4, L_y,L_x])
                lInp = torch.cat(
                    (self.pos_encoding.expand(lInp.shape[0], self.pose_enc_level, lInp.shape[2], lInp.shape[3]),lInp), dim=1)

            # dmsg('lInp.shape')#torch.Size([BS or [BS*CS], inputC, L_y,L_x])

            lInp = self.cnn(lInp)
                

            # dmsg('lInp.shape')#torch.Size([BS, CS*2, L_y,L_x])
            lInp = lInp.view(-1,self.dimentionMultiplier,2,self.L_y, self.L_x)
            # dmsg('lInp.shape')#torch.Size([BS, CS,2, L_y,L_x])



            # dmsg('lInp.shape')#torch.Size([BS,CS,2, L_y,L_x])
            lInp = lInp.permute(0,3,4,1,2).contiguous()
            # dmsg('lInp.shape')#torch.Size([BS, L_y,L_x,CS,2])
            tmpX = lInp[:, :,:,:,0]
            tmpY = lInp[:, :,:,:,1]
            # dmsg('tmpX.shape','tmpY.shape')#tmpX = torch.Size([BS,L_y,L_x,CS]) ! tmpY = torch.Size([BS,L_y,L_x,CS])
            # tmpxToRet = tmpX.mean(dim=3)*self.dimentionMultiplier/2.
            # tmpyToRet = tmpY.mean(dim=3)*self.dimentionMultiplier/2.
            # tmpxToRet,_ = tmpX.max(dim=3)
            # tmpyToRet,_ = tmpY.max(dim=3)
            
            # tmpxToRet = tmpX[:,:,:,self.showIdx]
            # tmpyToRet = tmpY[:,:,:,self.showIdx]
            # dmsg('tmpxToRet.shape','tmpyToRet.shape')#torch.Size([BS,L_y,L_x]) ! torch.Size([BS,L_y,L_x])

            angAfter = [-tmpX,
                            tmpY]  # tmpX is actually -y, tmpY is actually x
            angAfterNorm = (tmpX.abs().mean() + tmpY.abs().mean())/2.

            tmpX = tmpX * (torch.pi * 2 / self.N_prime)
            tmpY = tmpY * (torch.pi * 2 / self.N_prime)
            # dmsg('xV.shape')#torch.Size([BS*L_y*L_x*CS,N_p,N_p,2])
            # dmsg('x.shape')#torch.Size([BS,L_y*L_x*CS,N_p,N_p,2])
            # dmsg('tmpX.shape','tmpY.shape')#tmpX = torch.Size([BS,L_y,L_x,CS]) ! tmpY = torch.Size([BS,L_y,L_x,CS])
            ret = createPhaseDiff(tmpX.view(-1), tmpY.view(-1), xV.shape)
            # dmsg('ret.shape')#torch.Size([BS*L_y*L_x*CS,N_p,N_p,2])
            ret= ret.view_as(x)
            # dmsg('ret.shape')#torch.Size([BS,L_y*L_x*CS,N_p,N_p,2])

        else:
            raise NotImplemented

        return ret, angBefore, angAfter, angAfterNorm

class phaseDiffModel(nn.Module):
    def __init__(self, config):
        super(phaseDiffModel, self,).__init__()
        self.res_y = config.res_y_constrained
        self.res_x = config.res_x_constrained
        self.y_stride = config.stride
        self.x_stride = config.stride
        self.pS = config.window_padding_constrained
        self.N = config.window_size
        self.N_prime = self.N + 2 * self.pS


        self.padSizeAdjust = int(self.x_stride * ((self.N_prime - 1) // self.x_stride))
        self.L_y = (self.res_y + 2 * self.padSizeAdjust - self.N_prime) // self.y_stride + 1
        self.L_x = (self.res_x + 2 * self.padSizeAdjust - self.N_prime) // self.x_stride + 1

        self.untilIndex = config.untilIndex  # Can be 'None'
       
        self.denseNet = True

        if config.PD_model_enable:
            inpDim = 3 if config.use_variance else 1
            inpDim+= 2 if config.PD_model_use_direction else 0
            self.cnn = DenseNetLikeModel( inputC=(inpDim),
                            outputC=1, layerCount=len(config.PD_model_filter_sizes), hiddenF=config.PD_model_hidden_unit,
                                filterS=config.PD_model_filter_sizes, nonlin = config.PD_model_non_lin,lastNonLin=False)
        self.prev_x = []
        self.config = config
        self.pos_encoding = None
    def forward(self, x,energy):
        eps = 1e-8
        xV = x.view(-1, x.shape[2], x.shape[3], 2)
        if energy is not None:
            energy = energy.view(-1, x.shape[2], x.shape[3], 1)
        # energy = None
        tmpX,varianceX = getAvgDiff(rot=xV,energy=energy, step=1, axis=0, untilIndex=self.untilIndex,variance = self.config.use_variance)
        tmpY,varianceY = getAvgDiff(rot=xV,energy=energy, step=1, axis=1, untilIndex=self.untilIndex,variance = self.config.use_variance)

        tmpX = (torch.atan2(tmpX[:, 1], tmpX[:, 0] + 0.0000001) / (torch.pi * 2 / self.N_prime)).unsqueeze(-1)
        tmpY = (torch.atan2(tmpY[:, 1], tmpY[:, 0] + 0.0000001) / (torch.pi * 2 / self.N_prime)).unsqueeze(-1)
        mag = torch.sqrt((tmpX*tmpX)+(tmpY*tmpY)+eps)

        if self.config.PD_model_enable:
            if self.config.use_variance:
                if self.config.PD_model_use_direction:
                    lInp = torch.cat((tmpX,tmpY,varianceX,varianceY,mag), dim=1).view(-1, self.L_x, self.L_y, 5)
                else:
                    lInp = torch.cat((varianceX,varianceY,mag), dim=1).view(-1, self.L_x, self.L_y, 3)
            else:
                if self.config.PD_model_use_direction:
                    lInp = torch.cat((tmpX,tmpY,mag), dim=1).view(-1, self.L_x, self.L_y, 3)
                else:
                    lInp = mag.view(-1, self.L_x, self.L_y, 1)

            lInp = lInp.permute(0, 3, 1, 2).contiguous()
            lInp = self.cnn(lInp)
            lInp = lInp.permute(0, 2, 3, 1).contiguous()
        else:
            lInp=mag

        tmp2  = lInp.reshape(x.shape[0],x.shape[1],1,1)
        return tmp2.expand(x.shape[0],x.shape[1],x.shape[2],x.shape[3])
