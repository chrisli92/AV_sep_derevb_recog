# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

import logging
import math
from argparse import Namespace
from torch_complex.tensor import ComplexTensor

import numpy
import torch

# from espnet.nets.asr_interface import ASRInterface
from espnet.online_feature_av_training.asr_interface import ASRInterface
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import ErrorCalculator, end_detect
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD, Reporter
from espnet.nets.pytorch_backend.nets_utils import (
    get_subsample,
    make_non_pad_mask,
    th_accuracy,
)
from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.argument import (  # noqa: H301
    add_arguments_transformer_common,
)
from espnet.nets.pytorch_backend.transformer.attention import (  # noqa: H301
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask, target_mask
# from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.online_feature_av_training.plot import PlotAttentionReport
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.utils.fill_missing_args import fill_missing_args
from espnet.online_feature_av_training.feature_transform import feature_transform_for
from espnet.online_feature_av_training.conv_stft import STFT



class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group = add_arguments_transformer_common(group)

        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def get_total_subsampling_factor(self):
        """Get total subsampling factor."""
        return self.encoder.conv_subsampling_factor * int(numpy.prod(self.subsample))

    def __init__(self, aidim, vidim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
        # import pdb; pdb.set_trace()
        # fill missing arguments for compatibility
        args = fill_missing_args(args, self.add_arguments)

        # online feature extraction
        ## initialize stft
        ### the same with offline fbank feature extraction
        self.FRAME_LEN_FBANK = 400
        self.FRAME_HOP_FBANK = 160
        self.NUM_FFT = 512
        self.window = "povey"
        self.stft = STFT(frame_len=self.FRAME_LEN_FBANK, frame_hop=self.FRAME_HOP_FBANK, num_fft=self.NUM_FFT, window=self.window)
        self.feature_transform = feature_transform_for(args)

        # import pdb; pdb.set_trace()
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate

        self.adim = args.adim  # used for CTC (equal to d_model)
        self.mtlalpha = args.mtlalpha

        if args.mtlalpha > 0.0:
            self.ctc = CTC(
                odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
            )
        else:
            self.ctc = None

        self.intermediate_ctc_weight = args.intermediate_ctc_weight
        self.intermediate_ctc_layers = None
        if args.intermediate_ctc_layer != "":
            self.intermediate_ctc_layers = [
                int(i) for i in args.intermediate_ctc_layer.split(",")
            ]
        
        ############## visual setting #################
        self.use_pca = args.use_pca
        self.vidim = args.vidim
        if self.use_pca:
            print(f"vidim: {self.vidim}")
            self.fc = torch.nn.Linear(512, self.vidim)
        self.encoder_dim = aidim+self.vidim
        
        self.encoder = Encoder(
            idim=self.encoder_dim, # 83
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type, # rel_selfattn
            attention_dim=args.adim, # 256
            attention_heads=args.aheads, # 4
            conv_wshare=args.wshare, # 4
            conv_kernel_length=args.ldconv_encoder_kernel_length, # 21_23_25_27_29_31_33_35_37_39_41_43
            conv_usebias=args.ldconv_usebias, # False
            linear_units=args.eunits, # 2048
            num_blocks=args.elayers, # 12 
            input_layer=args.transformer_input_layer, # conv2d
            dropout_rate=args.dropout_rate, # 0.1
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate, # 0.0
            stochastic_depth_rate=args.stochastic_depth_rate, # 0.0
            intermediate_layers=self.intermediate_ctc_layers, # None
            ctc_softmax=self.ctc.softmax if args.self_conditioning else None, # None
            conditioning_layer_dim=odim, # 502
        )
        if args.mtlalpha < 1:
            self.decoder = Decoder(
                odim=odim, # 502
                selfattention_layer_type=args.transformer_decoder_selfattn_layer_type, # selfattn
                attention_dim=args.adim, # 256
                attention_heads=args.aheads, # 4
                conv_wshare=args.wshare, # 4
                conv_kernel_length=args.ldconv_decoder_kernel_length, # 11_13_15_17_19_21
                conv_usebias=args.ldconv_usebias, # False
                linear_units=args.dunits, # 2048
                num_blocks=args.dlayers, # 6
                dropout_rate=args.dropout_rate, # 0.1
                positional_dropout_rate=args.dropout_rate, # 0.1
                self_attention_dropout_rate=args.transformer_attn_dropout_rate, # 0.0
                src_attention_dropout_rate=args.transformer_attn_dropout_rate, # 0.0
            )
            self.criterion = LabelSmoothingLoss(
                odim, # 502
                ignore_id, # -1
                args.lsm_weight, # 0.1
                args.transformer_length_normalized_loss, # 0
            )
        else:
            self.decoder = None
            self.criterion = None
        self.blank = 0
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="asr", arch="transformer")
        self.reporter = Reporter()

        self.reset_parameters(args)

        if args.report_cer or args.report_wer:
            self.error_calculator = ErrorCalculator(
                args.char_list,
                args.sym_space,
                args.sym_blank,
                args.report_cer,
                args.report_wer,
            )
        else:
            self.error_calculator = None
        self.rnnlm = None

    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        initialize(self, args.transformer_init)

    def forward(self, axs_pad, vxs_pad, ilens, ys_pad):
        """E2E forward.

        # :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor axs_pad: batch of padded wav source sequences (B, C=1, t)
        :param torch.Tensor vxs_pad: batch of padded source sequences (B, vTmax, vidim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # [0.0] transform wav to fbank feature 
        # -1. WAV -> STFT
        B, C, t = axs_pad.unsqueeze(1).shape
        # (B, C, t) -> (BxC, t)
        all_s = axs_pad.view(-1, t)
        # -> (BxC, F, T)
        mag, phase = self.stft(all_s)
        # xs_complex = torch.stft(all_s, 512, 160, 400, torch.hann_window(400, device=all_s.device), return_complex=True)
        _, F, T = phase.shape
        # -> (B, C, F, T)
        phase = phase.view(B, C, F, T)
        mag = mag.view(B, C, F, T)
        imag = mag * torch.sin(phase)
        real = mag * torch.cos(phase)
        # (B, C=1, F=257, T)
        axs_pad = ComplexTensor(real, imag)
        
        # 0. STFT -> Fbank
        # (B, C=1, F=257, T) -> (B, F=257, T) -> (B, T, F=257)
        axs_pad = axs_pad.squeeze().permute(0, 2, 1)
        # import pdb; pdb.set_trace() 
        # recorrect ilens
        ilens = ilens + 3  ## stft padded 256 zeros each side.
        if ilens[0] > axs_pad.shape[1]:
            ilens = ilens - (ilens[0] - axs_pad.shape[1])
        # -> xs_pad: (B, aTmax, aidim)
        # print(f"ilens: {ilens}, xs_pad.shape:{axs_pad.shape}")
        axs_pad, ilens = self.feature_transform(axs_pad, ilens, None)

        # [0.1] Visual pca and concat
        # (B, aTmax, aidim) -> (B, aidim, aTmax)
        axs_pad = axs_pad.permute(0, 2, 1)
        if self.use_pca:
            # (B, vTmax, vidim: 512) -> (B, vTmax, vidim': 80)
            vxs_pad = self.fc(vxs_pad)
         # (B, vTmax, vidim') -> (B, vidim', vTmax)
        vxs_pad = vxs_pad.permute(0, 2, 1)
        # interpolate visual feature by audio feature T
        vxs_pad = torch.nn.functional.interpolate(vxs_pad, size=axs_pad.size(2))
        # concat audio and video feature: (B, aidim, aTmax) +  (B, vidim, aTmax) -> (B, aidim+vidim, aTmax) -> (B, aTmax, aidim+vidim)
        xs_pad = torch.cat((axs_pad, vxs_pad), dim=-2).permute(0, 2, 1)
        
        # import pdb; pdb.set_trace()
        # 1. forward encoder
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        if self.intermediate_ctc_layers:
            hs_pad, hs_mask, hs_intermediates = self.encoder(xs_pad, src_mask)
        else:
            hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        self.hs_pad = hs_pad

        # 2. forward decoder
        if self.decoder is not None:
            ys_in_pad, ys_out_pad = add_sos_eos(
                ys_pad, self.sos, self.eos, self.ignore_id
            )
            ys_mask = target_mask(ys_in_pad, self.ignore_id)
            pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
            self.pred_pad = pred_pad

            # 3. compute attention loss
            loss_att = self.criterion(pred_pad, ys_out_pad)
            self.acc = th_accuracy(
                pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
            )
        else:
            loss_att = None
            self.acc = None

        # TODO(karita) show predicted text
        # TODO(karita) calculate these stats
        cer_ctc = None
        loss_intermediate_ctc = 0.0
        if self.mtlalpha == 0.0:
            loss_ctc = None
        else:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
            if not self.training and self.error_calculator is not None:
                ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
                cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
            # for visualization
            if not self.training:
                self.ctc.softmax(hs_pad)

            if self.intermediate_ctc_weight > 0 and self.intermediate_ctc_layers:
                for hs_intermediate in hs_intermediates:
                    # assuming hs_intermediates and hs_pad has same length / padding
                    loss_inter = self.ctc(
                        hs_intermediate.view(batch_size, -1, self.adim), hs_len, ys_pad
                    )
                    loss_intermediate_ctc += loss_inter

                loss_intermediate_ctc /= len(self.intermediate_ctc_layers)

        # 5. compute cer/wer
        if self.training or self.error_calculator is None or self.decoder is None:
            cer, wer = None, None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        # copied from e2e_asr
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = loss_ctc
            if self.intermediate_ctc_weight > 0:
                self.loss = (
                    1 - self.intermediate_ctc_weight
                ) * loss_ctc + self.intermediate_ctc_weight * loss_intermediate_ctc
            loss_att_data = None
            loss_ctc_data = float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            if self.intermediate_ctc_weight > 0:
                self.loss = (
                    (1 - alpha - self.intermediate_ctc_weight) * loss_att
                    + alpha * loss_ctc
                    + self.intermediate_ctc_weight * loss_intermediate_ctc
                )
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
        return self.loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, afeat, vfeat):
        """Encode acoustic features.

        # :param ndarray x: source acoustic feature (T, D)
        :param ndarray x: source acoustic feature (t,)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        # [0] transform wav to fbank
        # (t,) -> (1, t) -> (1, 1, t)
        afeat = torch.as_tensor(afeat).unsqueeze(0).unsqueeze(0)
        
         # -1. WAV -> STFT
        B, C, t = afeat.shape
        # (B, C, t) -> (BxC, t)
        all_s = afeat.view(-1, t)
        # -> (BxC, F, T)
        mag, phase = self.stft(all_s)
        _, F, T = phase.shape
        # -> (B, C, F, T)
        phase = phase.view(B, C, F, T)
        mag = mag.view(B, C, F, T)
        imag = mag * torch.sin(phase)
        real = mag * torch.cos(phase)
        # (B, C=1, F=257, T)
        afeat = ComplexTensor(real, imag)
        
        # 0. STFT -> Fbank
        # (B, C=1, F=257, T) -> (B, F=257, T) -> (B, T, F=257)
        afeat = afeat.squeeze(1).permute(0, 2, 1)
        # -> xs_pad: (B, aTmax, aidim)
        afeat, ilens = self.feature_transform(afeat, [afeat.shape[1]], None)
        
        afeat = torch.as_tensor(afeat)
        vfeat = torch.as_tensor(vfeat)
        
        if afeat.dim() == 2:
            afeat = afeat.unsqueeze(0)
        if vfeat.dim() == 2:
            vfeat = vfeat.unsqueeze(0)
        
        # visual pca and concat with audio
        # (B, aTmax, aidim) -> (B, aidim, aTmax)
        afeat = afeat.permute(0, 2, 1)
        # import pdb; pdb.set_trace()
        if self.use_pca:
            # (B, vTmax, vidim: 512) -> (B, vTmax, vidim': 80)
            vfeat = self.fc(vfeat)
        # (B, vTmax, vidim') -> (B, vidim, vTmax)
        vfeat = vfeat.permute(0, 2, 1)
        # interpolate visual feature by audio feature T
        vfeat = torch.nn.functional.interpolate(vfeat, size=afeat.size(2))
        # concat audio and video feature: (B, aidim, aTmax) +  (B, vidim, aTmax) -> (B, aidim+vidim, aTmax) -> (B, aTmax, aidim+vidim)
        x = torch.cat((afeat, vfeat), dim=-2).permute(0, 2, 1)
        
        enc_output, *_ = self.encoder(x, None)
        
        return enc_output.squeeze(0)

    def recognize(self, afeat, vfeat, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        # enc_output = self.encode(x).unsqueeze(0)
        enc_output = self.encode(afeat, vfeat).unsqueeze(0)
        if self.mtlalpha == 1.0:
            recog_args.ctc_weight = 1.0
            logging.info("Set to pure CTC decoding mode.")

        if self.mtlalpha > 0 and recog_args.ctc_weight == 1.0:
            from itertools import groupby

            lpz = self.ctc.argmax(enc_output)
            collapsed_indices = [x[0] for x in groupby(lpz[0])]
            hyp = [x for x in filter(lambda x: x != self.blank, collapsed_indices)]
            nbest_hyps = [{"score": 0.0, "yseq": [self.sos] + hyp}]
            if recog_args.beam_size > 1:
                raise NotImplementedError("Pure CTC beam search is not implemented.")
            # TODO(hirofumi0810): Implement beam search
            return nbest_hyps
        elif self.mtlalpha > 0 and recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(enc_output)
            lpz = lpz.squeeze(0)
        else:
            lpz = None

        h = enc_output.squeeze(0)

        logging.info("input lengths: " + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None}
        else:
            hyp = {"score": 0.0, "yseq": [y]}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, numpy)
            hyp["ctc_state_prev"] = ctc_prefix_score.initial_state()
            hyp["ctc_score_prev"] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        import six

        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug("position " + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp["yseq"][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp["yseq"]).unsqueeze(0)
                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(
                            self.decoder.forward_one_step, (ys, ys_mask, enc_output)
                        )
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                else:
                    local_att_scores = self.decoder.forward_one_step(
                        ys, ys_mask, enc_output
                    )[0]

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                    local_scores = (
                        local_att_scores + recog_args.lm_weight * local_lm_scores
                    )
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1
                    )
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp["yseq"], local_best_ids[0], hyp["ctc_state_prev"]
                    )
                    local_scores = (1.0 - ctc_weight) * local_att_scores[
                        :, local_best_ids[0]
                    ] + ctc_weight * torch.from_numpy(
                        ctc_scores - hyp["ctc_score_prev"]
                    )
                    if rnnlm:
                        local_scores += (
                            recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                        )
                    local_best_scores, joint_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp["rnnlm_prev"] = rnnlm_state
                    if lpz is not None:
                        new_hyp["ctc_state_prev"] = ctc_states[joint_best_ids[0, j]]
                        new_hyp["ctc_score_prev"] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypothes: " + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    "best hypo: "
                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
                )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("adding <eos> in the last position in the loop")
                for hyp in hyps:
                    hyp["yseq"].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp["score"] += recog_args.lm_weight * rnnlm.final(
                                hyp["rnnlm_prev"]
                            )
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info("end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
            : min(len(ended_hyps), recog_args.nbest)
        ]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize(x, recog_args, char_list, rnnlm)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        return nbest_hyps

    def calculate_all_attentions(self, axs_pad, vxs_pad, ilens, ys_pad):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights (B, H, Lmax, Tmax)
        :rtype: float ndarray
        """
        self.eval()
        with torch.no_grad():
            # self.forward(xs_pad, ilens, ys_pad)
            self.forward(axs_pad, vxs_pad, ilens, ys_pad)
        ret = dict()
        for name, m in self.named_modules():
            if (
                isinstance(m, MultiHeadedAttention)
                or isinstance(m, DynamicConvolution)
                or isinstance(m, RelPositionMultiHeadedAttention)
            ):
                ret[name] = m.attn.cpu().numpy()
            if isinstance(m, DynamicConvolution2D):
                ret[name + "_time"] = m.attn_t.cpu().numpy()
                ret[name + "_freq"] = m.attn_f.cpu().numpy()
        self.train()
        return ret

    def calculate_all_ctc_probs(self, axs_pad, vxs_pad, ilens, ys_pad):
        """E2E CTC probability calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: CTC probability (B, Tmax, vocab)
        :rtype: float ndarray
        """
        ret = None
        if self.mtlalpha == 0:
            return ret

        self.eval()
        with torch.no_grad():
            # self.forward(xs_pad, ilens, ys_pad)
            self.forward(axs_pad, vxs_pad, ilens, ys_pad)
        for name, m in self.named_modules():
            if isinstance(m, CTC) and m.probs is not None:
                ret = m.probs.cpu().numpy()
        self.train()
        return ret
