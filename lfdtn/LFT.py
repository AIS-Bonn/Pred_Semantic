from torch.functional import F
import torch.nn as nn
from asset.utils import getPhaseAdd,dmsg
from lfdtn.custom_fft import custom_ifft, custom_fft
import torch
from icecream import ic


def fold(T, res_y: int, res_x: int, y_stride: int, x_stride: int, cell_size: int, pad_size: int):
    fold_params = dict(kernel_size=(cell_size, cell_size), padding=(pad_size, pad_size), stride=(y_stride, x_stride))
    return nn.functional.fold(T, output_size=(res_y, res_x), **fold_params)


def calcCOM(inp,config,loclConM=None, eps=1e-8):
    if config.trimCOMWindow:
        inp=inp[:,:,config.max_result_speed:-config.max_result_speed,config.max_result_speed:-config.max_result_speed]

    xs = inp.size(-2)
    ys = inp.size(-1)

    denom = torch.sum(inp, dim=(-2, -1))
    xMesh, yMesh = torch.meshgrid(torch.arange(0, xs), torch.arange(0, ys))
    xMesh = xMesh.to(inp.device)
    yMesh = yMesh.to(inp.device)
    COMX = torch.sum(inp * xMesh, dim=(-2, -1)) / (denom + eps)
    COMY = torch.sum(inp * yMesh, dim=(-2, -1)) / (denom + eps)
    rarr = [COMX/xs, COMY/ys]

    if config.localContentC>0 and loclConM is not None:
        cont = loclConM(inp.reshape(-1,xs*ys)).view(inp.shape[0],inp.shape[1],-1)
        for i in range(config.localContentC):
            rarr.append(cont[:,:,i])
    return torch.stack( rarr,dim=-1)

def extract_local_windows(batch, windowSize: int, y_stride: int = 1, x_stride: int = 1, padding: int = 0):
    BS,CS,HH,WW = batch.shape
    # dmsg('batch.shape')#torch.Size([BS, CS, HH, WW])   
    windowSizeAdjust = windowSize + 2 * padding

    ###this is the image padding, currently assumes x_stride == y_stride
    padSize = int(x_stride * ((windowSizeAdjust - 1) // x_stride))

    fold_params = dict(kernel_size=(windowSizeAdjust, windowSizeAdjust), padding=(padSize, padSize),
                       stride=(y_stride, x_stride))
    # print('image padded2 shape:', imgPadded.shape)
    result = nn.functional.unfold(batch, **fold_params)
    # dmsg('result.shape')#torch.Size([BS,  CS*N_p*N_p,L_y*L_x])
    result = result.view(BS,CS,windowSizeAdjust*windowSizeAdjust,-1)
    # dmsg('result.shape')#torch.Size([BS,CS, N_p*N_p,L_y*L_x])
    result = result.permute(0,3,1,2)
    # dmsg('result.shape')#torch.Size([BS,  L_y*L_x,CS,N_p*N_p])
    # print('result new fold shape:', result.shape)
    return result.reshape(BS, -1, windowSizeAdjust, windowSizeAdjust) #DONOT remove this contiguous, otherwise you get warning:https://github.com/pytorch/pytorch/issues/42300


def LFT(batch, window,config, y_stride: int = 1, x_stride: int = 1, padding: int = 1,useCOM=False,loclConM=None):
    # dmsg('batch.shape')#torch.Size([BS, CS, res_y,res_x])
    windowBatch = extract_local_windows(batch, window.shape[0], y_stride=y_stride, x_stride=x_stride, padding=padding)
    # dmsg('windowBatch.shape')#torch.Size([BS, L_y*L_x*CS, N_p,N_p])
    COM=None
    if useCOM:
        COM =  calcCOM(windowBatch,config,loclConM)
        # dmsg('COM.shape')#torch.Size([BS, L_y*L_x*CS,2])
    windowPadded = F.pad(window, (padding, padding, padding, padding))
    # dmsg('windowPadded.shape')#torch.Size([N_p,N_p])
    localImageWindowsSmoothedPadded = windowBatch * windowPadded


    return custom_fft(localImageWindowsSmoothedPadded),COM


def iLFT(stft2D_result, window, T, res_y: int, res_x: int, y_stride: int = 1, x_stride: int = 1, padding: int = 1,
         eps: float = 1e-8,is_inp_complex=True,move_window_according_T=True,channels=1):
    # dmsg("stft2D_result.shape")#torch.Size([BS, L_y*L_x*CS, N_p, N_p, 2])
    # dmsg("window.shape")#torch.Size([window_size*window_size])
    BS= stft2D_result.shape[0]
    cellSize = window.shape[0]

    ###this is the image padding, currently assumes x_stride == y_stride
    padSize = int(x_stride * ((cellSize - 1) // x_stride))

    cellSizeAdjust = cellSize + 2 * padding
    padSizeAdjust = int(x_stride * ((cellSizeAdjust - 1) // x_stride))

    ###the number of extracted cells along x and y axis
    ###this should be general enough to hold for different padSize
    num_windows_y = (res_y + 2 * padSizeAdjust - cellSizeAdjust) // y_stride + 1
    num_windows_x = (res_x + 2 * padSizeAdjust - cellSizeAdjust) // x_stride + 1
    num_windows_total = num_windows_y * num_windows_x

    if is_inp_complex:
        ifft_result = custom_ifft(stft2D_result)
    else:
        ifft_result = stft2D_result.clone()



    # dmsg("ifft_result.shape")#torch.Size([BS, L_y*L_x,CS, N_p, N_p])
    ifft_result = ifft_result.view((BS, num_windows_y, num_windows_x,channels, cellSizeAdjust, cellSizeAdjust))

    window_big = F.pad(window, (padding, padding, padding, padding), value=0.0)
    window_big = window_big.expand(BS, num_windows_total,channels, -1, -1)

    if move_window_according_T:
        window_big_Complex = custom_fft(window_big)
        window_big_Complex = getPhaseAdd(window_big_Complex, T.view_as(window_big_Complex))
        window_big = custom_ifft(window_big_Complex)

    window_big = window_big.view(BS, num_windows_y, num_windows_x,channels, window_big.shape[3], window_big.shape[4])

    ifft_result *= window_big

    ifft_result = ifft_result.reshape(BS, -1, ifft_result.shape[4] * ifft_result.shape[5]*channels).permute(0, 2, 1)
    test = fold(ifft_result, \
                res_y=res_y, res_x=res_x, y_stride=y_stride, x_stride=x_stride, cell_size=cellSizeAdjust,
                pad_size=padSizeAdjust)

    window_big = (window_big ** 2).reshape(BS, -1, window_big.shape[4] * window_big.shape[5]*channels).permute(0, 2, 1)
    windowTracker = fold(window_big, \
                         res_y=res_y, res_x=res_x, y_stride=y_stride, x_stride=x_stride, cell_size=cellSizeAdjust,
                         pad_size=padSizeAdjust)
                        

    windowTracker += eps
    weighted_result = test / windowTracker
    return weighted_result, windowTracker

def compact_LFT(batch, window, config,loclConM=None):
    return LFT(batch, window,config, x_stride=config.stride, y_stride=config.stride, padding=config.window_padding_constrained,useCOM=config.useCOM,loclConM=loclConM)

def compact_iLFT(LFT_result, window, T, config,is_inp_complex=True,move_window_according_T=True,channel=1):
    return iLFT(LFT_result, window, T, res_y=config.res_y_constrained, res_x=config.res_x_constrained, y_stride=config.stride,\
                    x_stride=config.stride, padding=config.window_padding_constrained,is_inp_complex=is_inp_complex,move_window_according_T=move_window_according_T,channels=channel)