import sys
import os
import wandb
import kornia
import time
import ast
import random as rand
import numpy as np
from lfdtn.dataloaders import get_data_loaders
from lfdtn.train_step import predict
from lfdtn.transform_models import cellTransportRefine,phaseDiffModel
from lfdtn.window_func import get_pascal_window, get_ACGW, get_2D_Gaussian
from asset.utils import generate_name, dmsg, CombinedLossDiscounted,wandblog,DenseNetLikeModel,niceL2S,setIfAugmentData,UNet,LocationAwareConv2d
from past.builtins import execfile
from colorama import Fore
from tqdm import tqdm
import torch
import click


execfile('lfdtn/helpers.py')  # This is instead of 'from asset.helpers import *', to have loadModels and saveModels
# access global variable.

torch.utils.backcompat.broadcast_warning.enabled = True

print("Python Version:", sys.version)
print("PyTorch Version:", torch.__version__)
print("Cuda Version:", torch.version.cuda)
print("CUDNN Version:", torch.backends.cudnn.version())

# This will make some functions faster It will automatically choose between: 
# https://github.com/pytorch/pytorch/blob/1848cad10802db9fa0aa066d9de195958120d863/aten/src/ATen/native/cudnn/Conv
# .cpp#L486-L494 
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# os.environ['WANDB_MODE'] = 'dryrun' #uncomment to prevent wandb logging - 'offline mode'

# https://pytorch.org/docs/stable/notes/randomness.html Reproducibility is not guaranteed! But i tested on BigCuda5 
# and it was reproducible (at least with correct configuration) 
deterministic = True
worker_init_fn = None
if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    randomSeed = 123
    torch.manual_seed(randomSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(randomSeed)
    rand.seed(randomSeed)
    np.random.seed(randomSeed)


    def worker_init_fn(worker_id):
        worker_id = worker_id + randomSeed
        torch.manual_seed(worker_id)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(worker_id)
        rand.seed(worker_id)
        np.random.seed(worker_id)

hyperparameter_defaults = dict(
    dryrun=False,
    inference=False,
    load_model='',
    load_seg='',
    load_pred='',
    limitDS=1.,
    epochs=500,
    batch_size=8,
    sequence_length=10,
    sequence_seed=5,
    max_result_speed=6,
    stride=4,  # 2 to the power of this number ##https://www.calculatorsoup.com/calculators/math/commonfactors.php
    window_size=15,
    window_type='ConfGaussian',
    lg_sigma_bias=0.1729,
    optimizer='AdamW',
    gain_update_lr=1,
    refine_lr = 0.002,
    refine_wd= 0.000001,
    refine_layer_cnt=4,
    refine_layer_cnt_a=6,
    refine_hidden_unit=16,
    refine_filter_size=3,
    ref_non_lin = 'PReLU',
    M_transform_lr=0.001,
    M_transform_wd=0.000001,
    trans_mode='Conv',
    tran_hidden_unit=16,
    tran_filters="13333",
    recpt_IConv=5,
    fcMiddle=10,  # Multiplied by 10
    untilIndex=12,
    concat_depth=4,
    tr_non_lin='PReLU',
    angleNormGain=0.00005,  # 0.0001
    chanVarGain=0,  # 0.0001

    ds_subjects=[1, 5, 6, 7, 8, 9, 11], # [1, 5, 6, 7, 8, 9, 11]
    ds_sequences=[25,26,27,28,29], # list(range(0, 30)) #["Direction", "Direction", "Discuss", "Discuss", "Eating", "Eating", "Greet", "Greet", "Phone", "Phone", "Pose", "Pose", "Purchase", "Purchase", "Sitting", "Sitting", "Sitting Down", "Sitting Down", "Smoke", "Smoke", "Photo", "Photo", "Wait", "Wait", "Walk", "Walk", "Walk Dog", "Walk Dog", "Walk Together", "Walk Together"]#
    ds_cameras = [1], #[0, 1, 2, 3]
    ds_joints =  ['Head','Root','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle'],#['Nose','Head','Neck','Belly','Root','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle'],
    
    # data_key='Skeleton',
    # data_key='planets_3',
    data_key='MMNIST',
    usehigh=False,
    digitCount=2,
    res_x=65,
    res_y=65,
    comp_fair_x = -1,
    comp_fair_y = -1,
    max_loss_tol_general=0.2,
    max_loss_tol_index = 2,
    max_num_param_tol=40000,
    pos_encoding=True,
    use_variance = True,
    use_energy = True,
    lr_scheduler = 'ReduceLROnPlateau',
    patience = 5,
    oneCycleMaxLRGain= 10,
    start_T_index = 2,
    tqdm=False,
    kill_no_improve = 3,
    validate_every=1,
    allways_refine= False,
    num_workers=12,
    hybridGain=0.7,
    augment_data=True,
    gainLR=1,
    excludeLoad="",
    refine_output=True,
    refine_output_share_weight=True,
    loss="L2all",
    test_batch_size=1,
    enable_autoregressive_prediction=False,
    update_T_just_by_FG = False,
    cmd="",
    input_channels=10,
    shareChannelWeight=True,
    firstSeg = False,
    predictSegmentedGT = False,
    predictSignal = False,
    blob = True,
    stUpB=0,
    save_ds_and_exit=False,
    load_ds_from_saved=True,
    train_mode="End2End",# JustSeg, JustPredSignal, JustPredBlob 
    channel_multiplier = 1,
    channel_multiplier_history = 0,
    PD_model_enable = False,
    PD_model_use_direction = True,
    number_of_classes=10,
    locAware=False,
    useCOM=False,
    localContentC=0,
    trimCOMWindow=True,
    PD_model_in_refine_output=False,
)



try:
    print("WANDB_CONFIG_PATHS = ", os.environ["WANDB_CONFIG_PATHS"])
except:
    pass

def mytqdm(x):
    return x

for a in sys.argv:
    if '--dryrun=True' in a:
        os.environ["WANDB_MODE"] = "dryrun"
    if ('--configs' in a and "=" in a) or '.yml' in a:
        try:
            try:
                v = a
                _, v = a[2:].split("=")
            except:
                pass
            if os.path.exists(v):
                v = str(os.getcwd()) + "/" + v
                os.environ["WANDB_CONFIG_PATHS"] = v
                print("Load configs from ", v)
        except Exception as e:
            print(e)
            pass

# print(sys.argv,os.environ,hyperparameter_defaults)
wandb.init(config=hyperparameter_defaults, project="ESANN2021") 
for k in wandb.config.keys():
    if '_constrained' in str(k):
        del wandb.config._items[k]


def myType(val):
    try:
        val = ast.literal_eval(val)
    except ValueError:
        pass
    return val


for a in sys.argv:
    if '--cmd=' in a[:6]:
        wandb.config.update({'cmd': str(a[6:])}, allow_val_change=True)
        continue
    if '--' in a[:2] and "=" in a:
        try:
            k, v = a[2:].split("=")
            v = myType(v)
            # if k in wandb.config.keys():
            #     v = type(wandb.config._items[k])(v)
            wandb.config.update({k: v}, allow_val_change=True)
        except Exception as e:
            pass
config = wandb.config
wandb.save('asset/*')
wandb.save('lfdtn/*')


if config.test_batch_size==-1:
    config.update({'test_batch_size': config.batch_size}, allow_val_change=True)


if config.data_key=='SoccerRollingBall':
    config.update({'number_of_classes': 6}, allow_val_change=True)
    config.update({'res_x': 129}, allow_val_change=True)
    config.update({'res_y': 129}, allow_val_change=True)
    config.update({'refine_layer_cnt': 16}, allow_val_change=True)
    config.update({'useCOM': True}, allow_val_change=True)
    config.update({'PD_model_enable': True}, allow_val_change=True)
    config.update({'PD_model_in_refine_output': True}, allow_val_change=True)
    config.update({'trimCOMWindow': True}, allow_val_change=True)
    config.update({'localContentC': 4}, allow_val_change=True)
    config.update({'window_size': 25}, allow_val_change=True)
    config.update({'stride': 8}, allow_val_change=True)



fromIDXTest = config.sequence_seed
if config.train_mode=="JustSeg":
    config.update({'sequence_seed': config.sequence_length}, allow_val_change=True)
    config.update({'loss': 'L2PredAll'}, allow_val_change=True)
    config.update({'firstSeg': True}, allow_val_change=True)
    config.update({'predictSegmentedGT': False}, allow_val_change=True)
    config.update({'predictSignal': False}, allow_val_change=True)
    config.update({'blob': True}, allow_val_change=True)
    config.update({'stUpB': 0}, allow_val_change=True)
    config.update({'input_channels': config.number_of_classes}, allow_val_change=True)
    config.update({'shareChannelWeight': True}, allow_val_change=True)
    fromIDXTest = 0
elif config.train_mode=="JustPredBlob":
    config.update({'loss': 'L2PredIncremental'}, allow_val_change=True)
    config.update({'predictSegmentedGT': True}, allow_val_change=True)
    config.update({'predictSignal': False}, allow_val_change=True)
    config.update({'blob': True}, allow_val_change=True)
    config.update({'firstSeg': True}, allow_val_change=True)
    config.update({'input_channels': config.number_of_classes}, allow_val_change=True)
    config.update({'shareChannelWeight': True}, allow_val_change=True)
    config.update({'stUpB': config.sequence_seed}, allow_val_change=True)
elif config.train_mode=="JustPredSignal":
    config.update({'loss': 'L2PredIncremental'}, allow_val_change=True)
    config.update({'input_channels': 1}, allow_val_change=True)
    config.update({'firstSeg': False}, allow_val_change=True)
    config.update({'predictSegmentedGT': False}, allow_val_change=True)
    config.update({'predictSignal': True}, allow_val_change=True)
    config.update({'shareChannelWeight': True}, allow_val_change=True)
    config.update({'stUpB': config.sequence_seed}, allow_val_change=True)
elif config.train_mode=="JustPredDigit2":
    config.update({'loss': 'L2Pred'}, allow_val_change=True)
    config.update({'input_channels': 1}, allow_val_change=True)
    config.update({'firstSeg': False}, allow_val_change=True)
    config.update({'predictSegmentedGT': False}, allow_val_change=True)
    config.update({'predictSignal': True}, allow_val_change=True)
    config.update({'shareChannelWeight': True}, allow_val_change=True)
    config.update({'stUpB': config.sequence_seed}, allow_val_change=True)
    config.update({'channel_multiplier': config.digitCount}, allow_val_change=True)    
    config.update({'channel_multiplier_history': 1}, allow_val_change=True)
else:
    if config.firstSeg:
        config.update({'input_channels': config.number_of_classes}, allow_val_change=True)
    else:
        config.update({'input_channels': 1}, allow_val_change=True)
    

if config.channel_multiplier>1:
    config.update({'input_channels': config.input_channels*config.channel_multiplier}, allow_val_change=True)



config.update({'PD_model_filters': config.tran_filters}, allow_val_change=True)
config.update({'PD_model_non_lin': config.tr_non_lin}, allow_val_change=True)
config.update({'PD_model_hidden_unit': config.tran_hidden_unit}, allow_val_change=True)
config.update({'PD_model_lr': config.M_transform_lr}, allow_val_change=True)
config.update({'PD_model_wd': config.M_transform_wd}, allow_val_change=True)



config.tran_filter_sizes = [int(i) for i in list(str(config.tran_filters))]
config.PD_model_filter_sizes =[int(i) for i in list(str(config.PD_model_filters))]



for k in config.keys():
    if "_lr" in str(k) and config.gainLR!=1:
        print("update",k)
        v = config[k]
        wandb.config.update({k: v*config.gainLR}, allow_val_change=True)

config.model_name_constrained = generate_name()

###all other config parameters are in some way constrained, and marked as such by their names!
###these need to be set depending on the dataset
# config.res_y_constrained,config.res_x_constrained = list(trainloader)[0].shape[2:]
config.res_x_constrained = config.res_x
config.res_y_constrained = config.res_y

###these must be calculated depending on other parameters
config.window_padding_constrained = config.max_result_speed

config.image_pad_size_old_constrained = int(config.stride * ((config.window_size - 1) // config.stride))
config.num_windows_y_old_constrained = (
                                                   config.res_y_constrained + 2 * config.image_pad_size_old_constrained - config.window_size) // config.stride + 1
config.num_windows_x_old_constrained = (
                                                   config.res_x_constrained + 2 * config.image_pad_size_old_constrained - config.window_size) // config.stride + 1
config.num_windows_total_old_constrained = config.num_windows_x_old_constrained * config.num_windows_y_old_constrained

config.image_pad_size_constrained = int(
    config.stride * (((config.window_size + 2 * config.window_padding_constrained) - 1) // config.stride))
config.num_windows_y_constrained = (
                                               config.res_y_constrained + 2 * config.image_pad_size_constrained - config.window_size - 2 * config.window_padding_constrained) // config.stride + 1
config.num_windows_x_constrained = (
                                               config.res_x_constrained + 2 * config.image_pad_size_constrained - config.window_size - 2 * config.window_padding_constrained) // config.stride + 1
config.num_windows_total_constrained = config.num_windows_x_constrained * config.num_windows_y_constrained

if ((config.res_x_constrained - 1) % config.stride != 0) or ((config.res_y_constrained - 1) % config.stride != 0):
    print("Not compatible stride", config.res_x_constrained, config.stride)
    sys.exit(0)

###select torch device
avDev = torch.device("cpu")
cuda_devices = list()
if torch.cuda.is_available():
    cuda_devices = [0]
    avDev = torch.device("cuda:" + str(cuda_devices[0]))
    if (len(cuda_devices) > 0):
        torch.cuda.set_device(cuda_devices[0])
print("avDev:", avDev)
dmsg('os.environ["CUDA_VISIBLE_DEVICES"]')
if config.tqdm:
    mytqdm=tqdm

with torch.no_grad():
    LG_Sigma = torch.zeros(1).to(avDev) + config.lg_sigma_bias
    if config.window_type == 'Pascal':
        window = get_pascal_window(config.window_size).to(avDev)
    elif config.window_type == 'ConfGaussian':
        window = get_ACGW(windowSize=config.window_size, sigma=LG_Sigma).detach()
    else:
        window = get_2D_Gaussian(resolution=config.window_size, sigma=LG_Sigma * config.window_size)[0, 0, :, :]

    unit_test_args = dict(res_y=config.res_y_constrained, res_x=config.res_x_constrained, H=config.stride, avDev=avDev,
                          pS=config.window_padding_constrained, bS=config.batch_size)
    LFT_unit_test(window, **unit_test_args)
    del (window)
    del (LG_Sigma)

inference_phase = config.inference
is_sweep = wandb.run.sweep_id is not None

print("config:{")
pretty(config._items,hyperparameter_defaults)
print("}")
critBCE = torch.nn.BCELoss()
critL1 = torch.nn.L1Loss()
critL2 = torch.nn.MSELoss()
critSSIM = kornia.losses.SSIMLoss(window_size=9, reduction='mean')
critHybrid = CombinedLossDiscounted()


MRef_Out = None
M_transform = None

LG_Sigma = torch.tensor(config.lg_sigma_bias, requires_grad=True, device=avDev)  ##TODO

paramN = []
minLR = []
wD = []

chennelC = 1 if config.refine_output_share_weight else config.input_channels

if config.refine_output:
    inpCC = chennelC
    if config.PD_model_enable and config.PD_model_in_refine_output:
        inpCC*=2
    MRef_Out = UNet(inpCC,chennelC,full=False,hd=config.refine_layer_cnt,pad=1,locAware=config.locAware,w=config.res_x,h=config.res_y).to(avDev)
    # MRef_Out = DenseNetLikeModel(inputC=chennelC,outputC=chennelC,layerCount=config.refine_layer_cnt, hiddenF=config.refine_hidden_unit,
    #                             filterS=config.refine_filter_size,nonlin=config.ref_non_lin,lastNonLin=False).to(avDev)
    paramN.append('MRef_Out')
    minLR.append(config.refine_lr)
    wD.append(config.refine_wd)


# segModel = torch.nn.Sequential(
#                                DenseNetLikeModel(inputC=1,outputC=10,layerCount=config.refine_layer_cnt, hiddenF=config.refine_hidden_unit,
#                                 filterS=config.refine_filter_size,nonlin=config.ref_non_lin,lastNonLin=False,initW=False).to(avDev)
#                               )
PD_model=None
if config.channel_multiplier>1:
    inpch = 1+config.channel_multiplier_history
    if config.PD_model_enable:
        PD_model = phaseDiffModel(config).to(avDev)
        inpch +=config.input_channels

    multiplyModel = torch.nn.Sequential(
                            UNet(inpch,config.channel_multiplier,full=True,hd=8,pad=1).to(avDev)
                            )
else:
    multiplyModel = torch.nn.Identity()
    if config.PD_model_enable:
        if config.PD_model_in_refine_output:
            PD_model = phaseDiffModel(config).to(avDev)
        else:
            PD_model = torch.nn.Identity()





if config.predictSignal:
    segModel = torch.nn.Identity()
else:
    if config.predictSegmentedGT:
        segModel = torch.nn.Identity()
    else:
        segModel = torch.nn.Sequential(
                                    UNet(1*config.channel_multiplier,config.number_of_classes,full=True,hd=16,pad=1).to(avDev)
                                    )

M_transform = cellTransportRefine(config).to(avDev)
paramN.append('M_transform')
minLR.append(config.M_transform_lr)
wD.append(config.M_transform_wd)
paramN.append('segModel')
minLR.append(config.refine_lr)
wD.append(config.refine_wd)
paramN.append('multiplyModel')
minLR.append(config.refine_lr)
wD.append(config.refine_wd)
if config.PD_model_enable:
    paramN.append('PD_model')
    minLR.append(config.PD_model_lr)
    wD.append(config.PD_model_wd)

loclConM=None
if config.useCOM and config.localContentC>0:
    inputSize = config.window_size
    if not config.trimCOMWindow:
        inputSize+=(2*config.max_result_speed)
    loclConM = torch.nn.Sequential(torch.nn.Linear(inputSize*inputSize,4*config.localContentC),
                                    torch.nn.PReLU(),
                                    torch.nn.Linear(config.localContentC*4,config.localContentC)
                                    ).to(avDev)
    paramN.append('loclConM')
    minLR.append(config.PD_model_lr)
    wD.append(config.PD_model_wd)


if len(config.load_seg)>3:
    loadModels(config.load_seg,"M_transform,MRef_Out,pred_model,multiplyModel,PD_model,loclConM")
if len(config.load_pred)>3:
    loadModels(config.load_pred,"segModel")
loadModels(config.load_model,config.excludeLoad)

paramList = []

max_lrs = []
for i, p in enumerate(paramN):
    par = eval(p)
    paramList.append({'params': [par] if type(par) is torch.Tensor else par.parameters(),
                      'lr': minLR[i],
                      'weight_decay': wD[i],
                      'name': p})
    max_lrs.append(minLR[i]*config.oneCycleMaxLRGain)

optimizer = eval('torch.optim.'+config.optimizer+'(paramList)')


numParam = 0
for par in optimizer.param_groups:
    numParam += sum(l.numel() for l in par["params"] if l.requires_grad)

config.parameter_number_constrained = numParam
wandblog({"numParam": numParam})


for par in optimizer.param_groups:
    print(Fore.CYAN, par["name"], sum(l.numel() for l in par["params"] if l.requires_grad), Fore.RESET)
    for l in par["params"]:
        if l.requires_grad:
            print(Fore.MAGENTA, l.shape, "  =>", l.numel(), Fore.RESET)

print("Number of trainable params: ", Fore.RED + str(numParam) + Fore.RESET)
if is_sweep and numParam > config.max_num_param_tol:
    wandblog({"cstate": 'High Param', 'sweep_metric': 1.1},commit=True)
    print(Fore.RED, "TOO high #Params ", numParam, " > ", config.max_num_param_tol, Fore.RESET)
    sys.exit(0)

trainloader, validloader, testloader = get_data_loaders(config,key=config.data_key,
                                            size=(config.res_x_constrained, config.res_y_constrained),
                                            batch_size=config.batch_size,test_batch_size=config.test_batch_size, num_workers=config.num_workers, limit=config.limitDS,
                                            sequence_length=config.sequence_length)

if len(config.cmd)>1:
    exec(config.cmd)

print(Fore.MAGENTA,'Trainloader:',len(trainloader),'Validloader:',len(validloader),'Testloader:',len(testloader),Fore.RESET)

startOptimFromIndex = 0
lGains = [i * 1.2 for i in range(config.sequence_length + 1, startOptimFromIndex + 1, -1)]
# lGains = lGains[::-1]
lGains[0]=0.4
lGains[1]=0.8
lGains = [i / sum(lGains) for i in lGains]

print(lGains)

li = 0
ui = 1
t = 0  # Reset the step counter
bestFullL2Loss = 1e25
SHOWINTER = False

print(Fore.MAGENTA + ("Sweep!" if is_sweep else "Normal Run") + Fore.RESET)
if inference_phase:
    print(Fore.CYAN + "Inference Phase!" + Fore.RESET)
    bs = config.batch_size
    inferenceRes = []
    paintEvery = 1 if config.data_key=="SoccerRollingBall" else 3
    paintOncePerEpoch = False
    runs = 1
else:
    print(Fore.GREEN + "Training Phase!" + Fore.RESET)
    bs = config.batch_size
    paintEvery = None
    paintOncePerEpoch = True
    runs = config.epochs if (is_sweep or config.lr_scheduler == 'OneCycleLR' )else 100000000

torch.set_grad_enabled(not inference_phase)

if not inference_phase:
    if config.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=config.patience, threshold=0.0001,
                                                        cooldown=0, verbose=True, min_lr=0.000001)
    elif config.lr_scheduler == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lrs, total_steps=len(trainloader)*config.epochs)

    else:
        class dummyOpt():
            def step(self,inp=None):
                pass
            def get_last_lr(self):
                return [0.0]
            def get_lr(self):
                return [0.0]
        scheduler=dummyOpt()


last_improved=0
while t < runs:
    # torch.autograd.set_detect_anomaly(True)
    with torch.no_grad():
        ###setting window determines the LFT tapering window!
        if config.window_type == 'Pascal':
            window = get_pascal_window(config.window_size).to(avDev)
        elif config.window_type == 'ConfGaussian':
            window = get_ACGW(windowSize=config.window_size, sigma=LG_Sigma).detach()
        else:
            window = get_2D_Gaussian(resolution=config.window_size, sigma=LG_Sigma * config.window_size)[0, 0, :, :]
    # t starts at zero #https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232
    if t == 1 and torch.backends.cudnn.benchmark == True:
        torch.cuda.empty_cache()
    wandbLog = {}

    if not inference_phase:
        start_time = time.time()
        phase = 'Training'
        print('Going through ',phase,' set at epoch', t + 1, '...')
        train_c, train_ll,angAfterNorm_ll,angAfterNorm_t,chanVar_t,chanVar_ll = (0,0,0,0,0,0)

        setEvalTrain(True)
        setIfAugmentData(config.augment_data)
        if paintOncePerEpoch:
            paintEvery = rand.randint(1,len(trainloader))

        for mini_batch in mytqdm(trainloader):
            optimizer.zero_grad()
            train_c += 1
            if config.data_key=="MMNIST" or config.data_key=='SoccerRollingBall':
                if config.predictSignal:
                    data =mini_batch["GT"].to(avDev)
                    dataT =data
                else:
                    if not config.predictSegmentedGT:
                        data =mini_batch["GT"].to(avDev)
                        dataT =mini_batch["FG"][:,:,:,0,:,:].to(avDev)
                    else:
                        dataT =mini_batch["FG"][:,:,:,0,:,:].to(avDev)
                        data = dataT
            else:
                data =mini_batch.to(avDev)
                dataT = data
            

            if paintOncePerEpoch:
                show_images = train_c == paintEvery
            else:
                show_images = True if train_c % paintEvery == 0 else False

            show_images = show_images and not config.dryrun

            pred_frames, angAfterNorm,chanVar= predict(data,dataT, window, config,MRef_Out, M_transform,segModel,multiplyModel,PD_model,loclConM,phase=phase, log_vis=show_images)

            if pred_frames is not False:
                netOut = pred_frames[:, startOptimFromIndex:]
                target = dataT[:, startOptimFromIndex:]
                chanVar_t +=chanVar.item()
                chanVar_loss = chanVar*config.chanVarGain
                angAfterNorm_t +=angAfterNorm.item()
                angAfterNorm_loss = angAfterNorm * config.angleNormGain

                if config.loss=="Hybrid":
                    ll = 0
                    l = torch.pow((target - netOut), 2)
                    for i in range(config.sequence_length - startOptimFromIndex):
                        ll += l[:, i, :, :, :].mean() * lGains[i]
                    
                    ll = (1-config.hybridGain)*ll + (config.hybridGain)*critSSIM(netOut.view(-1,1,netOut.shape[3],netOut.shape[4]),
                        target.view(-1,1,target.shape[3],target.shape[4]))
                elif config.loss=="L2Pred":
                    ll = critL2(pred_frames[:,config.sequence_seed:], target[:,config.sequence_seed:])
                elif config.loss=="L2PredAll":
                    ll = critL2(pred_frames[:,0:], target[:,0:])
                else:
                    upB = min(config.sequence_length,(t//2)+1+config.stUpB)
                    lowB = 0
                    # print(lowB,upB)
                    ll = critL2(pred_frames[:,lowB:upB], target[:,lowB:upB])
                    # ll = critBCE(pred_frames[:,lowB:upB], target[:,lowB:upB])



                ll = ll + angAfterNorm_loss +chanVar_loss
                ll.backward()
                    

                optimizer.step()
                if not inference_phase and config.lr_scheduler == 'OneCycleLR':
                    scheduler.step()

                train_ll += ll.item()
                angAfterNorm_ll += angAfterNorm_loss.item()
                chanVar_ll += chanVar_loss.item()
            else:
                print(Fore.RED + "NAN found!" + Fore.RESET)
                raise BaseException("NAN error!")

            wandblog(wandbLog, commit=(not paintOncePerEpoch and show_images))#Messy steps in wandb

        wandbLog["trainLoss"] = train_ll / train_c
        wandbLog["angAfterNormLoss"] = angAfterNorm_ll / train_c

        wandbLog["angAfterNorm"] = angAfterNorm_t / train_c
        wandbLog["chanVarLoss"] = chanVar_ll / train_c
        wandbLog["chanVar"] = chanVar_t / train_c

        
        print('...done! ',Fore.LIGHTYELLOW_EX+"Took: {:.2f}".format(time.time() - start_time)+" Sec"+Fore.RESET)

    if t%config.validate_every>0 and not inference_phase and not is_sweep:
        print(Fore.LIGHTYELLOW_EX," ==> Skip validation",(config.validate_every-(t%config.validate_every)),"!...",Fore.RESET)
        wandblog(wandbLog, commit=True)
        t=t+1
        continue

    start_time = time.time()
    tPhase = ('Validation' if not inference_phase else 'Testing')
    tloader =  validloader if not inference_phase else testloader
    print('Going through ',tPhase,' set...')


    with torch.no_grad():
        setEvalTrain(False)
        setIfAugmentData(False)
        valid_c, bceFull, bceFullMin, L1FullNet, L2FullNet, ssimFull, ssimHybrid = (0, 0, 0, 0, 0, 0, 0)
        if paintOncePerEpoch:
            paintEvery = rand.randint(1, len(tloader))
        for mini_batch in mytqdm(tloader):
            if config.predictSignal:
                data =mini_batch["GT"].to(avDev)
                dataT =data
            else:
                if not config.predictSegmentedGT:
                    data =mini_batch["GT"].to(avDev)
                    dataT =mini_batch["FG"][:,:,:,0,:,:].to(avDev)
                else:
                    dataT =mini_batch["FG"][:,:,:,0,:,:].to(avDev)
                    data = dataT

            if paintOncePerEpoch:
                show_images = valid_c == paintEvery
            else:
                show_images = True if valid_c % paintEvery == 0 else False
            show_images = show_images and not config.dryrun
            pred_frames, _,_= predict(data,dataT, window, config,MRef_Out,M_transform,segModel,multiplyModel,PD_model,loclConM, phase=tPhase, log_vis=show_images)
            
            netOut = pred_frames[:,  fromIDXTest:,:,:config.comp_fair_y,:config.comp_fair_x].clamp(0,1)
            target = dataT[:,  fromIDXTest:,:,:config.comp_fair_y,:config.comp_fair_x].clamp(0,1)
            valid_c += 1
            bceFull += critBCE(netOut, target)
            bceFullMin += critBCE(target, target)
            L1FullNet += critL1(netOut, target)
            L2FullNet += critL2(netOut, target)
            netOutSSIM = netOut.reshape(-1, 1, netOut.shape[3], netOut.shape[4])
            targetSSIM = target.reshape(-1, 1, target.shape[3], target.shape[4])
            ssimFull += critSSIM(netOutSSIM,targetSSIM)
            ssimHybrid += critHybrid(netOutSSIM,targetSSIM)

            wandblog(wandbLog, commit=(not paintOncePerEpoch and show_images))  # Messy steps in wandb


    if not inference_phase and config.lr_scheduler == 'ReduceLROnPlateau':
        scheduler.step(L2FullNet.item() / valid_c)
    wandbLog["hybridSSIMLoss"] = ssimHybrid.item() / valid_c
    wandbLog["L1FullLoss"] = L1FullNet.item() / valid_c
    wandbLog["L2FullLoss"] = L2FullNet.item() / valid_c
    wandbLog["bceFullMin"] = bceFullMin.item() / valid_c
    wandbLog["bceFull"] = bceFull.item() / valid_c
    wandbLog["SSIMFull"] = ssimFull.item() / valid_c

    paramGain = 1 if (
                config.parameter_number_constrained < config.max_num_param_tol / 3.) else config.parameter_number_constrained / (
                config.max_num_param_tol / 3.)
    wandbLog["paramGain"] = paramGain
    wandbLog["sweep_metric"] = wandbLog["L2FullLoss"] * paramGain

    if wandbLog["L2FullLoss"] < bestFullL2Loss and not inference_phase:
        last_improved = t
        if not config.dryrun:
            nameM = config.data_key[:6]+"_"+"{:.4f}".format(wandbLog["L2FullLoss"]).replace(".","_")+"_"+wandb.run.project+"_"+wandb.run.name+"_"+config.model_name_constrained.replace("-","_")
        else:
            nameM = config.data_key[:6]+"_"+"{:.4f}".format(wandbLog["L2FullLoss"]).replace(".","_")+"_"+config.model_name_constrained.replace("-","_")
        nameM = nameM.replace("-","_").replace(".","_")
        print('Model improved:',Fore.GREEN+str(wandbLog["L2FullLoss"])+Fore.RESET,' Model saved! :', nameM)
        bestFullL2Loss = wandbLog["L2FullLoss"]
        saveModels(nameM)
        cFile=wandb.run._settings._sync_dir+'/files/config.yaml'
        if os.path.exists(cFile):
            open('savedModels/'+nameM+ ".yml", 'wb').write(open(cFile, 'rb').read())

    if is_sweep and config.kill_no_improve>=0 and (t-last_improved)>config.kill_no_improve:
        print(Fore.RED, "No improvement!", (t-last_improved),t,last_improved, Fore.RESET)
        wandblog(wandbLog,commit=True)
        sys.exit(0)

    if t>=config.max_loss_tol_index and is_sweep and wandbLog['sweep_metric'] > config.max_loss_tol_general:
        wandbLog["cstate"]= 'High Loss'
        print(Fore.RED, "Loss too high!", wandbLog['sweep_metric'], is_sweep, Fore.RESET)
        wandblog(wandbLog,commit=True)
        sys.exit(0)

    if inference_phase:
        inferenceRes.append(
            [bceFullMin.item() / valid_c, bceFull.item() / valid_c, L1FullNet.item() / valid_c
            , L2FullNet.item() / valid_c, ssimFull.item() / valid_c])


    t = t + 1
    url = "DRY"
    if not config.dryrun:
        url=click.style(wandb.run.get_url().replace('https://',""), underline=True, fg="blue")
    print('...done! ',Fore.LIGHTYELLOW_EX+"Took: {:.2f}".format(time.time() - start_time)+" Sec"+Fore.RESET,
        'Run:',url)
    wandblog(wandbLog, commit=True)

    if (inference_phase and t >= runs):
        inferenceRes = np.array(inferenceRes)
        print("BCEMin=", inferenceRes[:, 0].mean(), " BCE=", inferenceRes[:, 1].mean(), " L1=",
              inferenceRes[:, 2].mean(), " L2=", inferenceRes[:, 3].mean(), " SSIM=", inferenceRes[:, 4].mean())
        break
wandblog({"cstate": 'Done'},commit=True)
print("Run completed!")
