from tensorflow.python.keras.models import *
from stn import spatial_transformer_network as stn_transformer

from test_dwt import *


class StegaStampEncoder(Layer):
    def __init__(self, height, width):
        super(StegaStampEncoder, self).__init__()
        self.secret_dense = Dense(7500, activation='relu', kernel_initializer='he_normal')
        # self.embed = DWT_IDWT_Block_test1(channels=3, times=3)
        # self.dwt = DWT_2D_multilevel(level=4)
        # self.idwt = IDWT_2D_multilevel(level=4)
        self.dwt = DWT_2D(wavename='haar')
        self.idwt = IDWT_2D(wavename='haar')
        self.dwt1 = DWT_2D_simple(wavename='haar')
        self.idwt1 = IDWT_UpSample(wavename='haar')

        self.conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2 = DWT_Block_1(32, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv3 = DWT_Block_1(64, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv5 = IDWT_Block_1(64, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv6 = IDWT_Block_1(32, activation='relu', padding='same', kernel_initializer='he_normal')
        self.residual = Conv2D(3, 1, activation=None, padding='same', kernel_initializer='he_normal')

        self.conv1_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2_1 = DWT_Block_1(32, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv3_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv4_1 = IDWT_Block_1(32, activation='relu', padding='same', kernel_initializer='he_normal')

        self.residual1_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.residual1_2 = Conv2D(3, 1, activation=None, padding='same', kernel_initializer='he_normal')

        self.residual2_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.residual2_2 = Conv2D(3, 1, activation=None, padding='same', kernel_initializer='he_normal')

        self.residual3_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.residual3_2 = Conv2D(3, 1, activation=None, padding='same', kernel_initializer='he_normal')

    def call(self, inputs):
        secret, image = inputs
        secret = secret - .5
        image = image - .5

        secret = self.secret_dense(secret)
        secret = Reshape((50, 50, 3))(secret)
        secret_enlarged = UpSampling2D(size=(4, 4))(secret)
        secret_enlarged1 = UpSampling2D(size=(2, 2))(secret)

        im_LL, im_LH, im_HL, im_HH = self.dwt(image)
        inputs = concatenate([im_LL, secret_enlarged], axis=3)
        conv1 = self.conv1(inputs)
        conv2, detail2 = self.conv2(conv1)
        conv3, detail3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        merge5 = concatenate([conv3, conv4], axis=3)
        conv5 = self.conv5(merge5, detail3)
        merge6 = concatenate([conv2, conv5], axis=3)
        conv6 = self.conv6(merge6, detail2)
        merge7 = concatenate([conv1, conv6, inputs], axis=3)
        residual_ll = self.residual(merge7)

        LH = self.dwt1(im_LH)
        HL = self.dwt1(im_HL)
        HH = self.dwt1(im_HH)
        residual_input = concatenate([LH, HL, HH], axis=3)
        inputs = concatenate([residual_input, secret_enlarged1], axis=3)
        conv1 = self.conv1_1(inputs)
        conv2, detail2 = self.conv2_1(conv1)
        conv3 = self.conv3_1(conv2)
        merge3 = concatenate([conv3, conv2], axis=3)
        conv4 = self.conv4_1(merge3, detail2)
        merge5 = concatenate([conv1, conv4, inputs], axis=3)

        residual_1 = self.residual1_1(merge5)
        residual_1 = self.residual1_2(residual_1)
        residual_lh = self.idwt1(residual_1)

        residual_2 = self.residual2_1(merge5)
        residual_2 = self.residual2_2(residual_2)
        residual_hl = self.idwt1(residual_2)

        residual_3 = self.residual3_1(merge5)
        residual_3 = self.residual3_2(residual_3)
        residual_hh = self.idwt1(residual_3)

        return self.idwt(residual_ll, residual_lh, residual_hl, residual_hh)

