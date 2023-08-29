from torch import nn
from torch.nn import functional as F

from .attention_recognition_head import AttentionRecognitionHead
from ..loss.sequenceCrossEntropyLoss import SequenceCrossEntropyLoss
from .tps_spatial_transformer import TPSSpatialTransformer
from .stn_head import STNHead
from .resnet_aster import *


class ModelBuilder(nn.Module):
    '''
    PARAMETERS:
            rec_num_classes: Number of output classes for recognition.
            sDim: Dimension of the decoder's hidden states.
            attDim: Dimension of the attention mechanism.
            max_len_labels: Maximum length of recognition labels.
            eos: End-of-sequence symbol used during decoding.
            tps_inputsize, tps_outputsize, tps_margins: Parameters for the Spatial Transformer Network.
            with_lstm: Boolean flag to indicate whether to use LSTM in the encoder.
            n_group: Number of groups for group normalization in the encoder.
            num_control_points: Number of control points for the STN.
            stn_activation: Activation function used in the STN.
            STN_ON: Boolean flag to indicate whether to use the STN for image rectification.
            beam_width: Width of the beam search used during decoding.
    '''

    def __init__(self, rec_num_classes, sDim=512, attDim=512, max_len_labels=12, eos='EOS', tps_inputsize=[32, 64],
           tps_outputsize=[32, 100], tps_margins=[0.05, 0.05], with_lstm=True, n_group=1,
           num_control_points=20, stn_activation='none', STN_ON=True, beam_width=5):
        super(ModelBuilder, self).__init__()

        # Initialize parameters
        self.eos = eos
        self.STN_ON = STN_ON
        self.tps_inputsize = tps_inputsize
        self.beam_width = beam_width

        # Create the encoder
        self.encoder = ResNet_ASTER(with_lstm=with_lstm, n_group=n_group)

        # Create the decoder with attention-based recognition head
        self.decoder = AttentionRecognitionHead(
                                        num_classes=rec_num_classes,
                                        in_planes=self.encoder.out_planes,
                                        sDim=sDim,
                                        attDim=attDim,
                                        max_len_labels=max_len_labels)

        # Create the recognition loss function
        self.rec_crit = SequenceCrossEntropyLoss()

        # If STN is enabled, create the Spatial Transformer Network and STN head
        if self.STN_ON:
            self.tps = TPSSpatialTransformer(
                    output_image_size=tuple(tps_outputsize),
                    num_control_points=num_control_points,
                    margins=tuple(tps_margins))
            self.stn_head = STNHead(
                    in_planes=3,
                    num_ctrlpoints=num_control_points,
                    activation=stn_activation)

    def forward(self, x):
        # Initialize output dict
        return_dict = {'losses' : dict(), 'output' : dict()}
        # Extract input
        images, labels, labels_len = x

        # rectification
        if self.STN_ON:
            # input images are downsampled before being fed into stn_head.
            stn_input = F.interpolate(images, self.tps_inputsize, mode='bilinear', align_corners=True)
            ctrl_points = self.stn_head(stn_input)[1]
            images = self.tps(images, ctrl_points)[0]
            if not self.training:
                # save for visualization
                return_dict['output']['ctrl_points'] = ctrl_points
                return_dict['output']['rectified_images'] = images

        encoder_feats = self.encoder(images)
        encoder_feats = encoder_feats.contiguous()

        if self.training:
            rec_pred = self.decoder([encoder_feats, labels, labels_len])
            loss_rec = self.rec_crit(rec_pred, labels, labels_len)
            return_dict['losses']['loss_rec'] = loss_rec
        else:
            rec_pred, rec_pred_scores = self.decoder.beam_search(encoder_feats, self.beam_width, self.eos)
            rec_pred_ = self.decoder([encoder_feats, labels, labels_len])
            loss_rec = self.rec_crit(rec_pred_, labels, labels_len)
            return_dict['losses']['loss_rec'] = loss_rec
            return_dict['output']['pred_rec'] = rec_pred
            return_dict['output']['pred_rec_score'] = rec_pred_scores

        for k, v in return_dict['losses'].items():
            return_dict['losses'][k] = v.unsqueeze(0)

        return return_dict
