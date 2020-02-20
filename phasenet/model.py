import numpy as np
from distutils.version import LooseVersion

import keras
import keras.backend as K
from keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, TensorBoard

from csbdeep.utils import _raise, axes_check_and_normalize
from csbdeep.models import BaseConfig, BaseModel

from .psf import PsfGenerator3D
from .zernike import random_zernike_wavefront, ensure_dict
from .noise import add_random_noise
from .phantoms import PhantomGenerator3D
from .utils import cropper3D

from scipy.signal import convolve

class Data:

    def __init__(self,
                 amplitude_ranges, order='noll', normed=True,
                 batch_size=1,
                 psf_shape=(64,64,64), units=(0.1,0.1,0.1), na_detection=1.1, lam_detection=.5, n=1.33, n_threads=4,
                 noise_params = None, phantom_name=None, phantom_params=None, crop_params=None,
                 ):
        """
            Encapsulates data gerenation

            :param psf_shape: tuple, shape of psf, eg (32,32,32)
            :param units: tuple, units in microns, eg (0.1,0.1,0.1)
            :param lam_detection: scalar, wavelength in micrometer, eg 0.5
            :param n: scalar, refractive index, eg 1.33
            :param na_detection: scalar, numerical aperture of detection objective, eg 1.1
            :param n_threads: integer, for multiprocessing
            :param noise_params : dictionary, values for mean, sigma and snr
            :param phantom_name : string, phantom name
            :param phantom_params : dictionary, parameters according to the phantom chosen
            :param crop_params : dictionary, values for crop_shape, jitter, max_jitter
        """

        self.psfgen = PsfGenerator3D(psf_shape=psf_shape, units=units, lam_detection=lam_detection, n=n, na_detection=na_detection, n_threads=n_threads)
        self.order = order
        self.normed = normed
        self.amplitude_ranges = ensure_dict(amplitude_ranges, order)
        self.batch_size = batch_size
        
        if noise_params is not None:
            self.noise_flag = True
            self.noise_params = noise_params
        else:
            self.noise_flag=False

        if phantom_name is not None:
            self.phantom_flag = True
            isinstance(phantom_params,dict) or _raise(ValueError('phantom_params has to be a dictionary'))
            self.phantom_params = phantom_params
            self.phantom_shape = self.phantom_params.get('phantom_shape', psf_shape)
            self.phantomgen = PhantomGenerator3D(self.phantom_shape, units)
            self.phantom_name = phantom_name
            self.phantom_img = self.phantomgen.get_phantom_img(self.phantom_name, self.phantom_params)
        else:
            self.phantom_flag=False

        if crop_params is not None:
            self.crop_flag = True
            self.crop_params = crop_params
        else:
            self.crop_flag=False

    def _single_psf(self):
        phi = random_zernike_wavefront(self.amplitude_ranges, order=self.order)
        psf = self.psfgen.incoherent_psf(phi, normed=self.normed)
        psf = np.fft.fftshift(psf)
        if self.phantom_flag:
            psf = convolve(self.phantom_img, psf, mode='same')
        if self.noise_flag:
            psf = add_random_noise(psf, self.noise_params)
        if self.crop_flag:
            psf = cropper3D(psf, self.crop_params)
        return psf, phi.amplitudes_requested


    def generator(self):
        while True:
            psfs, amplitudes = zip(*(self._single_psf() for _ in range(self.batch_size)))
            X = np.expand_dims(np.stack(psfs, axis=0), -1)
            Y = np.stack(amplitudes, axis=0)
            yield X, Y



class Config(BaseConfig):
    """
        Configuration for phasenet models
        
        :param zernike_amplitude_ranges: dictionary or list, the values should either a scalar indicating the absolute magnitude 
                or a tuple with upper and lower bound, default is {'vertical coma': (-0.2,0.2)}
        :param zernike_order: string, zernike nomeclature used when the amplitude ranges are given as a list, default is 'noll'
        :param zernike_normed: booelan, whether the zzernikes are normalized according, default is True
        :param net_kernel_size: convoltuion kernel size, default is (3,3,3)
        :param net_pool_size: max pool kernel size, default is (1,2,2)
        :param net_activation: activation, default is 'tanh'
        :param net_padding: padding for convolution, default is 'same'
        
        :param psf_shape: tuple, shpae of the psf, default is (64,64,64)
        :param psf_units: tuple, voxel unit (z,y,x) in um, default is (0.1,0.1,0.1)
        :param psf_na_detection: scalar, numerical apperture default is 1.1
        :param psf_lam_detection: scalar, wavelength in um, default is 0.5
        :param psf_n: scalar, refractive index of immersion medium, default is 1.33
        :param noise_params: dictionary, noise  parameters, the value of each parameter can be a scalar for constant or a tuple 
                for upper and lower bound, if no noise is required set to None, default is {'mean':100, 'sigma':3.5, 'snr':(1.,5)}
        :param phantom_name: string, name of convolving object, set to None if not needed, default is 'sphere' 
        :param phantom_params: dictionary, parameters for the chosen phantom, e.g. {'radius':0.1} 
        :param crop_params: dictionary, parameters for cropping the psf, default is {'crop_shape':(32,32,32), 'jitter':True},
                max_jitter can also be passed as a key to the dictionary
        
        :param train_loss: string, training loss, default is 'mse'
        :param train_epochs: integer, number of epochs for trianing, default is 400
        :param train_steps_per_epoch: integer, number of steps per epoch, default is 5
        :param train_learning_rate: scalar, leaning rate, default is 0.0003
        :param train_batch_size: integer, batch size for training the network, default is 8
        :param train_n_val: integer, number of validation data, default is 128 
        :param train_tensorboard: boolean, create tensorboard, default is True

    """

    def __init__(self, axes='ZYX', n_channel_in=1, **kwargs):
        """See class docstring."""

        super().__init__(axes=axes, n_channel_in=n_channel_in, n_channel_out=1)

        # directly set by parameters
        # ...

        # default config (can be overwritten by kwargs below)
        self.zernike_amplitude_ranges  = {'vertical coma': (-0.2,0.2)}
        self.zernike_order             = 'noll'
        self.zernike_normed            = True

        self.net_kernel_size           = (3,3,3)
        self.net_pool_size             = (1,2,2)
        self.net_activation            = 'tanh'
        self.net_padding               = 'same'

        self.psf_shape                 = (64,64,64)
        self.psf_units                 = (0.1,0.1,0.1)
        self.psf_na_detection          = 1.1
        self.psf_lam_detection         = 0.5
        self.psf_n                     = 1.33
        self.noise_params              = {'mean':100, 'sigma':3.5, 'snr':(1.,5)}
        self.phantom_name              = 'sphere' 
        self.phantom_params            = {'radius':0.1} 
        self.crop_params               = {'crop_shape':(32,32,32), 'jitter':True}

        self.train_loss                = 'mse'
        self.train_epochs              = 400
        self.train_steps_per_epoch     = 5
        self.train_learning_rate       = 0.0003
        self.train_batch_size          = 8
        self.train_n_val               = 128
        self.train_tensorboard         = True
        # # the parameter 'min_delta' was called 'epsilon' for keras<=2.1.5
        # min_delta_key = 'epsilon' if LooseVersion(keras.__version__)<=LooseVersion('2.1.5') else 'min_delta'
        # self.train_reduce_lr           = {'factor': 0.5, 'patience': 40, min_delta_key: 0}

        # remove derived attributes that shouldn't be overwritten
        for k in ('n_dim', 'n_channel_out'):
            try: del kwargs[k]
            except KeyError: pass

        self.update_parameters(False, **kwargs)

        self.n_channel_out = len(random_zernike_wavefront(self.zernike_amplitude_ranges))

        if isinstance(self.crop_params,dict) and 'crop_shape' in self.crop_params:
            self._model_input_shape = self.crop_params.get('crop_shape')
        else:
            self._model_input_shape = self.psf_shape



class PhaseNet(BaseModel):
    """PhaseNet model.

    Parameters
    ----------
    config : :class:`Config` or None
        Will be saved to disk as JSON (``config.json``).
        If set to ``None``, will be loaded from disk (must exist).
    name : str or None
        Model name. Uses a timestamp if set to ``None`` (default).
    basedir : str
        Directory that contains (or will contain) a folder with the given model name.

    Raises
    ------
    FileNotFoundError
        If ``config=None`` and config cannot be loaded from disk.
    ValueError
        Illegal arguments, including invalid configuration.

    Attributes
    ----------
    config : :class:`Config`
        Configuration, as provided during instantiation.
    keras_model : `Keras model <https://keras.io/getting-started/functional-api-guide/>`_
        Keras neural network model.
    name : str
        Model name.
    logdir : :class:`pathlib.Path`
        Path to model folder (which stores configuration, weights, etc.)
    """

    @property
    def _config_class(self):
        return Config


    @property
    def _axes_out(self):
        return 'C'


    def _build(self):
        input_shape = tuple(self.config._model_input_shape) + (self.config.n_channel_in,)
        output_size = self.config.n_channel_out
        kernel_size = self.config.net_kernel_size
        pool_size = self.config.net_pool_size
        activation = self.config.net_activation
        padding = self.config.net_padding

        inp = Input(name='X', shape=input_shape)
        t = Conv3D(8, name='conv1', kernel_size=kernel_size, activation=activation, padding=padding)(inp)
        t = Conv3D(8, name='conv2', kernel_size=kernel_size, activation=activation, padding=padding)(t)
        t = MaxPooling3D(name='maxpool1', pool_size=pool_size)(t)
        t = Conv3D(16, name='conv3', kernel_size=kernel_size, activation=activation, padding=padding)(t)
        t = Conv3D(16, name='conv4', kernel_size=kernel_size, activation=activation, padding=padding)(t)
        t = MaxPooling3D(name='maxpool2', pool_size=pool_size)(t)
        t = Conv3D(32, name='conv5', kernel_size=kernel_size, activation=activation, padding=padding)(t)
        t = Conv3D(32, name='conv6', kernel_size=kernel_size, activation=activation, padding=padding)(t)
        t = MaxPooling3D(name='maxpool3', pool_size=pool_size)(t)
        t = Conv3D(64, name='conv7', kernel_size=kernel_size, activation=activation, padding=padding)(t)
        t = Conv3D(64, name='conv8', kernel_size=kernel_size, activation=activation, padding=padding)(t)
        t = MaxPooling3D(name='maxpool4', pool_size=pool_size)(t)
        t = Conv3D(128, name='conv9', kernel_size=kernel_size, activation=activation, padding=padding)(t)
        t = Conv3D(128, name='conv10', kernel_size=kernel_size, activation=activation, padding=padding)(t)

        if input_shape[0] == 1:
            t = MaxPooling3D(name='maxpool5', pool_size=(1, 2, 2))(t)
        else:
            t = MaxPooling3D(name='maxpool5', pool_size=(2, 2, 2))(t)
        t = Flatten(name='flat')(t)
        t = Dense(64, name='dense1', activation=activation)(t)
        t = Dense(64, name='dense2', activation=activation)(t)

        oup = Dense(output_size, name='Y', activation='linear')(t)
        return Model(inputs=inp, outputs=oup)


    def prepare_for_training(self, optimizer=None):
        """Prepare for neural network training.

        Compiles the model and creates
        `Keras Callbacks <https://keras.io/callbacks/>`_ to be used for training.

        Note that this method will be implicitly called once by :func:`train`
        (with default arguments) if not done so explicitly beforehand.

        Parameters
        ----------
        optimizer : obj or None
            Instance of a `Keras Optimizer <https://keras.io/optimizers/>`_ to be used for training.
            If ``None`` (default), uses ``Adam`` with the learning rate specified in ``config``.

        """
        if optimizer is None:
            optimizer = Adam(lr=self.config.train_learning_rate)

        self.keras_model.compile(optimizer, loss=self.config.train_loss)

        self.callbacks = []
        if self.basedir is not None:
            self.callbacks += self._checkpoint_callbacks()

            if self.config.train_tensorboard:
                self.callbacks.append(TensorBoard(log_dir=str(self.logdir), write_graph=False))

        # if self.config.train_reduce_lr is not None:
        #     rlrop_params = self.config.train_reduce_lr
        #     if 'verbose' not in rlrop_params:
        #         rlrop_params['verbose'] = True
        #     self.callbacks.append(ReduceLROnPlateau(**rlrop_params))

        self._model_prepared = True


    def train(self, seed=None, epochs=None, steps_per_epoch=None):
        """Train the neural network with the given data.

        Parameters
        ----------
        seed : int
            Convenience to set ``np.random.seed(seed)``. (To obtain reproducible validation patches, etc.)
        epochs : int
            Optional argument to use instead of the value from ``config``.
        steps_per_epoch : int
            Optional argument to use instead of the value from ``config``.

        Returns
        -------
        ``History`` object
            See `Keras training history <https://keras.io/models/model/#fit>`_.

        """
        if seed is not None:
            # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
            np.random.seed(seed)
        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        # patch_size = self.config.psf_shape
        # axes = self.config.axes.replace('C','')
        # b = self.config.train_completion_crop if self.config.train_shape_completion else 0
        # div_by = self._axes_div_by(axes)
        # [(p-2*b) % d == 0 or _raise(ValueError(
        #     "'train_patch_size' - 2*'train_completion_crop' must be divisible by {d} along axis '{a}'".format(a=a,d=d) if self.config.train_shape_completion else
        #     "'train_patch_size' must be divisible by {d} along axis '{a}'".format(a=a,d=d)
        #  )) for p,d,a in zip(patch_size,div_by,axes)]

        if not self._model_prepared:
            self.prepare_for_training()

        data_kwargs = dict (
            amplitude_ranges     = self.config.zernike_amplitude_ranges,
            order                = self.config.zernike_order,
            normed               = self.config.zernike_normed,
            psf_shape            = self.config.psf_shape,
            units                = self.config.psf_units,
            na_detection         = self.config.psf_na_detection,
            lam_detection        = self.config.psf_lam_detection,
            n                    = self.config.psf_n,
            noise_params         = self.config.noise_params,
            phantom_name         = self.config.phantom_name,
            phantom_params       = self.config.phantom_params,
            crop_params          = self.config.crop_params,
        )

        # generate validation data and store in numpy arrays
        data_val = Data(batch_size=self.config.train_n_val, **data_kwargs) # TODO: turn augmentation off here
        data_val = next(data_val.generator())

        data_train = Data(batch_size=self.config.train_batch_size, **data_kwargs)

        history = self.keras_model.fit_generator(generator=data_train.generator(), validation_data=data_val,
                                                 epochs=epochs, steps_per_epoch=steps_per_epoch,
                                                 callbacks=self.callbacks, verbose=1)
        self._training_finished()
        return history


    def predict(self, img, axes=None, normalizer=None, **predict_kwargs):
        """Predict.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Input image
        axes : str or None
            Axes of the input ``img``.
            ``None`` denotes that axes of img are the same as denoted in the config.
        normalizer : :class:`csbdeep.data.Normalizer` or None
            (Optional) normalization of input image before prediction.
            Note that the default (``None``) assumes ``img`` to be already normalized.
        predict_kwargs: dict
            Keyword arguments for ``predict`` function of Keras model.

        """
        if axes is None:
            axes = self.config.axes
            assert 'C' in axes
            if img.ndim == len(axes)-1 and self.config.n_channel_in == 1:
                # img has no dedicated channel axis, but 'C' always part of config axes
                axes = axes.replace('C','')

        axes     = axes_check_and_normalize(axes,img.ndim)
        axes_net = self.config.axes

        _permute_axes = self._make_permute_axes(axes, axes_net)
        x = _permute_axes(img) # x has axes_net semantics
        x.shape == tuple(self.config.psf_shape) + (self.config.n_channel_in,) or _raise(ValueError())

        normalizer = self._check_normalizer_resizer(normalizer, None)[0]
        x = normalizer.before(x, axes_net)

        return self.keras_model.predict(x[np.newaxis], **predict_kwargs)[0,0]
