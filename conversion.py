import IPython.display as ipd
import torch.nn as nn
import numpy as np
import parselmouth
import torch
import math

from mel_processing import spectrogram_torch
import utils


class VoiceConverter(nn.Module):
    def __init__(self, device: int = -1):
        super(VoiceConverter, self).__init__()
            
        if device == -1:
            self.device = torch.device('cpu')
        elif device > -1:
            self.device = torch.device(f'cuda:{device}')
        else:
            raise NotImplementedError
            
    @staticmethod
    def voice_conversion(hps, net_g, y_src, sid_src, sid_tgt, 
                         y_ref=None, f0_scale=1.0, uv_threshold=1.0, 
                         debug=False, device=-1):
        assert y_src.dim() == 3
        if y_ref is not None:
            assert y_ref.dim() == 3

        if device == -1:
            device = torch.device("cpu")
        elif device > -1:
            device = torch.device(f"cuda:{device}")
        else:
            device = torch.device("cpu")

        spec_src = spectrogram_torch(
            y=y_src.squeeze(1), 
            n_fft=hps.data.filter_length, 
            hop_size=hps.data.hop_length, 
            win_size=hps.data.win_length, 
            sampling_rate=hps.data.sampling_rate, 
            center=False
        )
        spec_src_length = torch.LongTensor([spec_src.size(-1)])

        g_src = net_g.emb_g(sid_src).unsqueeze(-1)
        g_tgt = net_g.emb_g(sid_tgt).unsqueeze(-1)

        z, x_q, m_q, logs_q, spec_mask = net_g.enc_q(
            spec_src, spec_src_length, g=g_src)
        z_p = net_g.flow(z, spec_mask, g=g_src)
        z_hat = net_g.flow(z_p, spec_mask, g=g_tgt, reverse=True)

        logf0_src = utils.calculate_f0(
            input=y_src.view(-1), 
            fs=hps.data.sampling_rate, 
            f0min=hps.data.fmin, 
            f0max=hps.data.fmax, 
            use_continuous_f0=False, 
            use_log_f0=True
        ).view(1,1,-1)[...,:spec_src_length.max().item()]

        if y_ref is not None:
            f0_src_median = VoiceConverter.get_pitch_median(
                y_src.view(-1).numpy(), sampling_rate=hps.data.sampling_rate)
            f0_ref_median = VoiceConverter.get_pitch_median(
                y_ref.view(-1).numpy(), sampling_rate=hps.data.sampling_rate)
            f0_shift = f0_ref_median / f0_src_median
        else:
            f0_shift = 1
        
        f0_src = torch.exp(utils.convert_to_inf(logf0_src, uv_threshold)) * f0_shift * f0_scale
        f0_embed = net_g.emb_f0(torch.bucketize(f0_src, net_g.f0_bins).squeeze(1)).transpose(1,2)

        o_hat = net_g.dec(x=torch.cat([z_hat, f0_embed], dim=1) * spec_mask, g=g_tgt)
        if debug:
            print(f"f0_shift: '{f0_shift}' | f0_scale: '{f0_scale}' | f0_ref_median: '{f0_ref_median}' | f0_src_median: '{f0_src_median}'")

        return o_hat, spec_mask, (z, z_p, z_hat, x_q), f0_src

    
    @staticmethod
    def get_pitch_median(y, sampling_rate: int = 22050) -> float:
        sound = VoiceConverter.wav_to_sound(y, sampling_rate)
        pitch = parselmouth.praat.call(sound, "To Pitch", 0, 70, 550)
        pitch_median_temp = parselmouth.praat.call(pitch, "Get quantile", 0.0, 0.0, 0.5, "Hertz")
        if not math.isnan(pitch_median_temp):
            pitch_median = pitch_median_temp
        else:
            print(e, "nan value exist in 'pitch_media_temp'")
            raise ValueError
        return pitch_median


    @staticmethod
    def wav_to_sound(y, sampling_rate: int = 22050) -> parselmouth.Sound:
        r""" load wav file to parselmouth Sound file
        # __init__(self: parselmouth.Sound, other: parselmouth.Sound) -> None \
        # __init__(self: parselmouth.Sound, values: numpy.ndarray[numpy.float64], sampling_frequency: Positive[float] = 44100.0, start_time: float = 0.0) -> None \
        # __init__(self: parselmouth.Sound, file_path: str) -> None
        returns:
            sound: parselmouth.Sound
        """
        if isinstance(y, parselmouth.Sound):
            sound = y
        elif isinstance(y, torch.FloatTensor):
            if y.dim() == 2:
                y = y.squeeze(0)
            sound = parselmouth.Sound(np.asarray(y.detach().cpu().numpy()), sampling_frequency=sampling_rate)
        elif isinstance(y, np.ndarray):
            sound = parselmouth.Sound(y, sampling_frequency=sampling_rate)
        elif isinstance(y, list):
            sound = parselmouth.Sound(np.asarray(y), sampling_frequency=sampling_rate)
        else:
            raise NotImplementedError
        return sound
    
    @staticmethod
    def load_wav_parselmouth(audiopath: str, visualize: bool = False):
        sound = parselmouth.Sound(audiopath)
        if visualize:
            import IPython.display as ipd
            display(ipd.Audio(data=sound.values, rate=sound.sampling_frequency, normalize=False))
        return sound

    @staticmethod
    def pitch_manipulation(sound: parselmouth.Sound, pitch_ratio: float = 1.0, visualize: bool = False):
        manipulation = parselmouth.praat.call(sound, "To Manipulation", 0.01, 75, 600)
        pitch_tier = parselmouth.praat.call(manipulation, "Extract pitch tier")
        parselmouth.praat.call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, pitch_ratio)
        parselmouth.praat.call([pitch_tier, manipulation], "Replace pitch tier")
        sound = parselmouth.praat.call(manipulation, "Get resynthesis (overlap-add)")
        if visualize:
            import IPython.display as ipd
            display(ipd.Audio(data=sound.values, rate=sound.sampling_frequency, normalize=False))
        return sound

