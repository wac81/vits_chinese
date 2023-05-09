import matplotlib.pyplot as plt

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write

# device = torch.device("cpu")
# model.to(device)
def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


hps = utils.get_hparams_from_file("./configs/woman_csmsc.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint("./logs/woman_csmsc/G_100000.pth", net_g, None) 


stn_tst = get_text("第一，南京不是发展的不行，是大家对他期望很高，", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()

    # x_tst = stn_tst.cpu().unsqueeze(0)
    # x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()

    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

    
from  scipy.io import wavfile 
sampling_rate = 24000
wavfile.write('abc1.wav', sampling_rate, audio)
# ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))
