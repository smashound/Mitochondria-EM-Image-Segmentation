import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras import backend as keras
from models.modelbase import ModelBase
from evaluate import dice_coef



def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

class Unet(ModelBase):
	def __init__(self,input_size):
		super(Unet, self).__init__()
		self.input_size = input_size
		self.build_model()
		self.model_checkpoint = ModelCheckpoint('saved_model/unet.hdf5',monitor='val_loss',verbose=1,save_best_only=True)

	def build_model(self):
		inputs = Input(self.input_size+(1,))
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		conv1 = BatchNormalization()(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		conv2 = BatchNormalization()(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		conv3 = BatchNormalization()(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		conv4 = BatchNormalization()(conv4)
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		conv5 = BatchNormalization()(conv5)
		drop5 = Dropout(0.5)(conv5)

		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		merge6 = concatenate([drop4,up6], axis = 3)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
		conv6 = BatchNormalization()(conv6)

		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		merge7 = concatenate([conv3,up7], axis = 3)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
		conv7 = BatchNormalization()(conv7)

		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		merge8 = concatenate([conv2,up8], axis = 3)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
		conv8 = BatchNormalization()(conv8)

		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		merge9 = concatenate([conv1,up9], axis = 3)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = BatchNormalization()(conv9)
		conv10 = Conv2D(1, 1, activation = 'sigmoid',name='preds')(conv9)

		model = Model(input = inputs, output = conv10)

		model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coef_loss, metrics = [dice_coef])
		self.model = model



class Attention_unet(ModelBase):
	def __init__(self,input_size):
		super(Attention_unet, self).__init__()

		self.input_size = input_size
		self.num_seg_class=1
		self.model_checkpoint = ModelCheckpoint('saved_model/attention_unet.hdf5',monitor='val_loss',verbose=1,save_best_only=True)
		self.build_model()

	def expend_as(self,tensor, rep):
		my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
		return my_repeat

	def AttnGatingBlock(self,x, g, inter_shape):
		shape_x = K.int_shape(x)  # 32
		shape_g = K.int_shape(g)  # 16

		theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
		shape_theta_x = K.int_shape(theta_x)

		phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
		upsample_g = Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)  # 16

		concat_xg = add([upsample_g, theta_x])
		act_xg = Activation('relu')(concat_xg)
		psi = Conv2D(1, (1, 1), padding='same')(act_xg)
		sigmoid_xg = Activation('sigmoid')(psi)
		shape_sigmoid = K.int_shape(sigmoid_xg)
		upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

		upsample_psi = self.expend_as(upsample_psi, shape_x[3])
		y = multiply([upsample_psi, x])
		result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
		result_bn = BatchNormalization()(result)
		return result_bn

	def UnetGatingSignal(self,input, is_batchnorm=False):
		shape = K.int_shape(input)
		x = Conv2D(shape[3] * 2, (1, 1), strides=(1, 1), padding="same")(input)
		if is_batchnorm:
			x = BatchNormalization()(x)
		x = Activation('relu')(x)
		return x

	def UnetConv2D(self,input, outdim, is_batchnorm=False):
		x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(input)
		if is_batchnorm:
			x =BatchNormalization()(x)
		x = Activation('relu')(x)

		x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
		if is_batchnorm:
			x = BatchNormalization()(x)
		x = Activation('relu')(x)
		return x


	def build_model(self):
		inputs = Input(self.input_size+(1,))
		conv = Conv2D(16, (3, 3), padding='same')(inputs)  # 'valid'
		conv = LeakyReLU(alpha=0.3)(conv)

		conv1 = self.UnetConv2D(conv, 32,is_batchnorm=True)  # 32 128
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = self.UnetConv2D(pool1, 32,is_batchnorm=True)  # 32 64
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = self.UnetConv2D(pool2, 64,is_batchnorm=True)  # 64 32
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = self.UnetConv2D(pool3, 64,is_batchnorm=True)  # 64 16
		pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		center = self.UnetConv2D(pool4, 128,is_batchnorm=True)  # 128 8

		gating = self.UnetGatingSignal(center, is_batchnorm=True)
		attn_1 = self.AttnGatingBlock(conv4, gating, 128)
		up1 = concatenate([Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',activation="relu")(center), attn_1], axis=3)

		gating = self.UnetGatingSignal(up1, is_batchnorm=True)
		attn_2 = self.AttnGatingBlock(conv3, gating, 64)
		up2 = concatenate([Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',activation="relu")(up1), attn_2], axis=3)

		gating = self.UnetGatingSignal(up2, is_batchnorm=True)
		attn_3 = self.AttnGatingBlock(conv2, gating, 32)
		up3 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu")(up2), attn_3], axis=3)

		up4 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu")(up3), conv1], axis=3)


		conv8 = Conv2D(self.num_seg_class + 1, (1, 1), activation='relu', padding='same')(up4)
		act =  Conv2D(1, (1,1), activation = 'sigmoid',name='preds')(conv8)

		model = Model(inputs=inputs, outputs=act)
		model.compile(optimizer='adam', loss = dice_coef_loss, metrics = [dice_coef])
		self.model = model

from keras.engine import Layer, InputSpec
from keras import initializers as initializations
class Scale(Layer):
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = K.variable(self.gamma_init(shape), name='{}_gamma'.format(self.name))
        self.beta = K.variable(self.beta_init(shape), name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Dense_unet(ModelBase):
	def __init__(self,input_size):
		super(Dense_unet, self).__init__()
		self.concat_axis=3
		self.input_size = input_size
		self.model_checkpoint = ModelCheckpoint('saved_model/dense_unet.hdf5',monitor='val_loss',verbose=1,save_best_only=True)
		self.build_model()

	def build_model(self):
		box=[]
		nb_filter = 16
		eps = 1.1e-5
		nb_dense_block=4
		nb_layers = [2,4,8,16]
		growth_rate=48
		dropout_rate=0.5
		weight_decay=1e-4
		compression = 1.0

		inputs = Input(self.input_size+(1,))
		x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(inputs)
		x = Conv2D(nb_filter, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
		x = BatchNormalization(epsilon=eps, axis=self.concat_axis, name='conv1_bn')(x)
		x = Scale(axis=self.concat_axis, name='conv1_scale')(x)
		x = Activation('relu', name='relu1')(x)
		box.append(x)
		x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
		x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

		# Add dense blocks
		for block_idx in range(nb_dense_block - 1):
			stage = block_idx+2
			x, nb_filter = self.dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
			box.append(x)
			# Add transition_block
			x = self.transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
			nb_filter = int(nb_filter * compression)

		final_stage = stage + 1
		x, nb_filter = self.dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

		x = BatchNormalization(epsilon=eps, axis=self.concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
		x = Scale(axis=self.concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
		x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
		box.append(x)

		up0 = UpSampling2D(size=(1,1))(x)
		line0 = Conv2D(1456, (3, 3), padding="same", kernel_initializer="normal", name="line0")(box[-1])
		up0_sum = add([line0, up0])
		conv_up0 = Conv2D(688, (3, 3), padding="same", kernel_initializer="normal", name = "conv_up0")(up0_sum)
		bn_up0 = BatchNormalization(name = "bn_up0")(conv_up0)
		ac_up0 = Activation('relu', name='ac_up0')(bn_up0)

		up1 = UpSampling2D(size=(2,2))(ac_up0)
		up1_sum = add([box[-2], up1])
		conv_up1 = Conv2D(304, (3, 3), padding="same", kernel_initializer="normal", name = "conv_up1")(up1_sum)
		bn_up1 = BatchNormalization(name = "bn_up1")(conv_up1)
		ac_up1 = Activation('relu', name='ac_up1')(bn_up1)

		up2 = UpSampling2D(size=(2,2))(ac_up1)
		up2_sum = add([box[-3], up2])
		conv_up2 = Conv2D(112, (3, 3), padding="same", kernel_initializer="normal", name = "conv_up2")(up2_sum)
		bn_up2 = BatchNormalization(name = "bn_up2")(conv_up2)
		ac_up2 = Activation('relu', name='ac_up2')(bn_up2)

		up3 = UpSampling2D(size=(2,2))(ac_up2)
		up3_sum = add([box[-4], up3])
		conv_up3 = Conv2D(96, (3, 3), padding="same", kernel_initializer="normal", name = "conv_up3")(up3_sum)
		bn_up3 = BatchNormalization(name = "bn_up3")(conv_up3)
		ac_up3 = Activation('relu', name='ac_up3')(bn_up3)

		up4 = UpSampling2D(size=(2, 2))(ac_up3)
		conv_up4 = Conv2D(64, (3, 3), padding="same", kernel_initializer="normal", name="conv_up4")(up4)
		conv_up4 = Dropout(rate=0.3)(conv_up4)
		bn_up4 = BatchNormalization(name="bn_up4")(conv_up4)
		ac_up4 = Activation('relu', name='ac_up4')(bn_up4)

		x = Conv2D(1, (1,1), padding="same", kernel_initializer="normal", name="dense167classifer")(ac_up4)

		model = Model(inputs, x, name='denseu161')
		model.compile(optimizer='adam', loss = dice_coef_loss, metrics = [dice_coef])
		self.model = model

	def conv_block(self,x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
		'''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
			# Arguments
				x: input tensor
				stage: index for dense block
				branch: layer index within each dense block
				nb_filter: number of filters
				dropout_rate: dropout rate
				weight_decay: weight decay factor
		'''
		eps = 1.1e-5
		conv_name_base = 'conv' + str(stage) + '_' + str(branch)
		relu_name_base = 'relu' + str(stage) + '_' + str(branch)

		# 1x1 Convolution (Bottleneck layer)
		inter_channel = nb_filter * 4
		x = BatchNormalization(epsilon=eps, axis=self.concat_axis, name=conv_name_base+'_x1_bn')(x)
		x = Scale(axis=self.concat_axis, name=conv_name_base+'_x1_scale')(x)
		x = Activation('relu', name=relu_name_base+'_x1')(x)
		x = Conv2D(inter_channel, (1, 1), name=conv_name_base+'_x1', use_bias=False)(x)

		if dropout_rate:
			x = Dropout(dropout_rate)(x)

		# 3x3 Convolution
		x = BatchNormalization(epsilon=eps, axis=self.concat_axis, name=conv_name_base+'_x2_bn')(x)
		x = Scale(axis=self.concat_axis, name=conv_name_base+'_x2_scale')(x)
		x = Activation('relu', name=relu_name_base+'_x2')(x)
		x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
		x = Conv2D(nb_filter, (3, 3), name=conv_name_base+'_x2', use_bias=False)(x)

		if dropout_rate:
			x = Dropout(dropout_rate)(x)

		return x


	def transition_block(self,x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
		''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
			# Arguments
				x: input tensor
				stage: index for dense block
				nb_filter: number of filters
				compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
				dropout_rate: dropout rate
				weight_decay: weight decay factor
		'''

		eps = 1.1e-5
		conv_name_base = 'conv' + str(stage) + '_blk'
		relu_name_base = 'relu' + str(stage) + '_blk'
		pool_name_base = 'pool' + str(stage)

		x = BatchNormalization(epsilon=eps, axis=self.concat_axis, name=conv_name_base+'_bn')(x)
		x = Scale(axis=self.concat_axis, name=conv_name_base+'_scale')(x)
		x = Activation('relu', name=relu_name_base)(x)
		x = Conv2D(int(nb_filter * compression), (1, 1), name=conv_name_base, use_bias=False)(x)

		if dropout_rate:
			x = Dropout(dropout_rate)(x)

		x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

		return x


	def dense_block(self,x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
		''' Build a dense_block where the output of each conv_block is fed to subsequent ones
			# Arguments
				x: input tensor
				stage: index for dense block
				nb_layers: the number of layers of conv_block to append to the model.
				nb_filter: number of filters
				growth_rate: growth rate
				dropout_rate: dropout rate
				weight_decay: weight decay factor
				grow_nb_filters: flag to decide to allow number of filters to grow
		'''

		concat_feat = x

		for i in range(nb_layers):
			branch = i+1
			x = self.conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
			concat_feat = concatenate([concat_feat, x], axis=self.concat_axis, name='concat_'+str(stage)+'_'+str(branch))

			if grow_nb_filters:
				nb_filter += growth_rate

		return concat_feat, nb_filter
