import os
import sys
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import time
import math
import random

from . import networks_tensorflow as networks
from AnetLib.data.image_utils import GaussianBlurring
#from AnetLib.data.normalization import get_norm

class AnetModel():
    def name(self):
        return 'A-NET Model'

    def initialize(self, opt):
        # BaseModel.initialize(self, opt)
        self.opt = opt
        if 'dim_ordering' in opt:
            assert opt.dim_ordering == 'channels_last'
        else:
            opt.dim_ordering = 'channels_last'
        assert self.opt.which_direction == 'AtoB'
        self._current_report = {}
        self._current_visuals = {}
        self._current_config = {}

        self._current_epoch = 0
        self.global_step = 0
        self.uncertainty_blur_sigma = 5.0

        use_gpu = len(opt.gpu_ids) > 0
        if tf.__version__.split('.')[0] != "1":
            raise Exception("Tensorflow version 1 required")

        if opt.seed is None:
            opt.seed = random.randint(0, 2**31 - 1)
        tf.set_random_seed(opt.seed)
        # random.seed(opt.seed)
        # np.random.seed(opt.seed)

        if 'dropout_prob' in opt:
            self.dropout_prob = opt.dropout_prob
        else:
            self.dropout_prob = 0.2 if 'bayesian' in opt.model else 0.5
            vars(opt)['dropout_prob'] = self.dropout_prob

        self.__dropout_switch = not opt.no_dropout

        vars(opt).update({'summary_freq': opt.display_freq, 'progress_freq': opt.print_freq, 'trace_freq':0, 'display_freq': opt.display_freq})

        if self.opt.load_dir is not None and os.path.exists(os.path.join(self.opt.load_dir, '__model__')):
            self.opt.load_dir = os.path.join(self.opt.load_dir, '__model__')

        if opt.phase != 'train' or (opt.continue_train and self.opt.load_dir is not None):
            self.load_config('latest')

        tf.reset_default_graph()
        model = networks.build_network(opt.model.replace('_tensorflow', ''), opt.fineSize, opt.input_nc, opt.output_nc, opt.batchSize,
                                       use_resize_conv=opt.use_resize_conv, gan_weight=opt.lambda_G, l1_weight=opt.lambda_A,
                                       lr=opt.lr, beta1=opt.beta1, lambda_tv=opt.lambda_tv, ngf=opt.ngf, ndf=opt.ndf,
                                       control_nc=opt.control_nc, control_classes=opt.control_classes,
                                       lr_nc=opt.lr_nc, lr_scale=opt.lr_scale, squirrel_weight=opt.lambda_LR,
                                       norm_A=self.opt.norm_A, norm_B=self.opt.norm_B, norm_LR=self.opt.norm_LR,
                                       use_gaussd=opt.use_gaussd, lr_loss_mode=self.opt.lr_loss_mode, use_queue=not opt.no_queue)
        self.model, queue_funcs, self.display_fetches, losses, self.summary_merged = model
        enqueue_data, close_queue = queue_funcs
        self.loss_fetches, self.averaged_loss_fetches = losses
        self.enqueue_data = lambda *args: enqueue_data(self.sess, *args)
        self.close_queue = lambda : close_queue(self.sess)
        init_op = tf.global_variables_initializer()
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

        writer = tf.summary.FileWriter(logdir=self.opt.tb_dir, graph=tf.get_default_graph())
        writer.flush()

        # logdir = self.opt.workdir if (opt.trace_freq > 0 or opt.summary_freq > 0) else None
        # self.sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
        self.sess = tf.Session()
        self.__coord_stopped = True
        self.start_coord()
        print("parameter_count =", self.sess.run(parameter_count))

        self.sess.run(init_op)
        if opt.phase != 'train' or (opt.continue_train and self.opt.load_dir is not None):
            self.load('latest')

        # self.stop_coord()
        with open(os.path.join(self.opt.workdir, "options.json"), "w") as f:
            f.write(json.dumps(vars(opt), sort_keys=True, indent=4))
        self._current_config = vars(opt).copy()

        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def initialize_from_config(self, config_path, **kwargs):
        with open(config_path, 'r') as f:
            config_json = json.load(f)
            parser = argparse.ArgumentParser()
            opt = parser.parse_args([])
            config = vars(opt)
            config.update(config_json)
            config.update(kwargs)
            self.initialize(opt)

    def start_coord(self):
        if self.__coord_stopped:
            self.coord = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
            self.__coord_stopped = False

    def stop_coord(self):
        self.coord.request_stop()
        self.coord.join(self.threads, stop_grace_period_secs=3)
        self.__coord_stopped = True

    def close(self):
        self.sess.close()
        tf.reset_default_graph()

    def load_network(self, sess, network_label, epoch_label):
        assert epoch_label == 'latest'
        assert os.path.exists(self.opt.load_dir)
        print("loading model from checkpoint ", self.opt.load_dir)
        checkpoint = tf.train.latest_checkpoint(self.opt.load_dir)
        assert checkpoint, 'checkpoint not found.'
        if 'all' in network_label:
            all_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            saver = tf.train.Saver(all_trainables)
            saver.restore(sess, checkpoint)
        else:
            generator_filters = []
            if 'G' in network_label:
                generator_filters = generator_filters + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
            if 'D' in network_label:
                generator_filters = generator_filters + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
            if 'S' in network_label:
                generator_filters = generator_filters + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="squirrel_discriminator")

            if len(generator_filters) > 0:
                saver = tf.train.Saver(generator_filters)
                saver.restore(sess, checkpoint)
            else:
                raise Exception('invalid network label.')

    def save_network(self, sess, network_label, epoch_label, gpu_ids):
        assert epoch_label == 'latest'
        print("saving model to checkpoint")
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(sess, os.path.join(self.opt.checkpoints_dir, "model"), global_step=self.global_step)
        if self.model.squirrel_error_map is not None:
            self.save_squirrel_discriminator()

    def save_squirrel_discriminator(self, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.opt.checkpoints_dir, "squirrel_discriminator")
        generator_filters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="squirrel_discriminator")
        saver = tf.train.Saver(generator_filters, max_to_keep=1)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        saver.save(self.sess, os.path.join(save_path, "model"), global_step=self.global_step)

    def write_saved_model(self, saved_model_path, inputs, outputs):
        from tensorflow.python.saved_model import signature_constants
        saver = tf.train.Saver()

        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)

        tensor_info_outputs = {}
        for k in outputs:
          v = self.sess.graph.get_tensor_by_name(k+":0")
          tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(v)

        tensor_info_inputs = {}
        for k in inputs:
          v = self.sess.graph.get_tensor_by_name(k+":0")
          tensor_info_inputs[k] = tf.saved_model.utils.build_tensor_info(v)

        detection_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                  inputs=tensor_info_inputs,
                  outputs=tensor_info_outputs,
                  method_name=signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
              self.sess, [tf.saved_model.tag_constants.SERVING],
              signature_def_map={
                  signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                      detection_signature,
              },
          )
        builder.save()

    def set_input(self, input):
        if input is not None:
            if isinstance(input, (np.ndarray, np.generic)):
                input = {'A': input}
            self.input = input

            if input['A'].ndim == 2:
                input['A'] = input['A'][None, :, :, None]
            elif input['A'].ndim == 3:
                input['A'] = input['A'][None, :, :, :]
            assert input['A'].ndim == 4 and input['A'].shape[3] == self.opt.input_nc, 'input channel number should match the input_nc option'
            self.real_A = input['A']
            if 'B' not in input:
                self.real_B = np.zeros([self.real_A.shape[0], self.real_A.shape[1], self.real_A.shape[2], self.opt.output_nc])
            else:
                if input['B'].ndim == 2:
                    input['B'] = input['B'][None, :, :, None]
                elif input['B'].ndim == 3:
                    input['B'] = input['B'][None, :, :, :]
                assert input['B'].ndim == 4
                self.real_B = input['B']
            if 'path' not in input:
                self.paths = ['__']
            else:
                self.paths = input['path']
            curr_control = np.zeros([self.real_A.shape[0], 1, 1, self.opt.control_nc])
            if self.opt.control_nc > 0:
                if 'control' in input and input['control'] is not None:
                    for i in range(self.real_A.shape[0]):
                        curr_control[i, 0, 0, :] = np.array(input['control'])
            self.control = curr_control
            curr_channel_mask = np.ones([self.real_A.shape[0], 1, 1, self.real_A.shape[3]])
            if 'channel_mask' in input and input['channel_mask'] is not None:
                for i in range(self.real_A.shape[0]):
                    curr_channel_mask[i, 0, 0, :] = np.array(input['channel_mask'])
            self.channel_mask = curr_channel_mask
            # print('inputs:', self.real_A.shape, self.real_B.shape, self.paths)
        curr_input, curr_target, curr_path, curr_control, curr_channel_mask = self.real_A, self.real_B, np.array([self.paths]), self.control, self.channel_mask
        self.enqueue_data(curr_path, curr_input, curr_target, curr_control, curr_channel_mask)

    def forward(self):
        try:
            results = self.sess.run(self.display_fetches, feed_dict={'dropout_prob:0': self.get_dropout_prob()})
            return results
        except tf.errors.DeadlineExceededError:
            raise Exception('timeout error, please try again.')
        except Exception as e:
            raise

    def retrieve_results(self, results):
        self.real_A = results['inputs']
        self.real_B = results['targets']
        self.reco_B = results['outputs']
        self.paths = results['paths']
        self._current_visuals['real_A'] = results['inputs']
        self._current_visuals['real_B'] = results['targets']
        self._current_visuals['reco_B'] = results['outputs']

        if 'squirrel_error_map' in results:
            self._current_visuals['squirrel_error_map'] = results['squirrel_error_map']
            self.squirrel_error_map = results['squirrel_error_map']
        else:
            self.squirrel_error_map = None

        if 'lr_predict_fake' in results:
            self._current_visuals['lr_predict_fake'] = results['lr_predict_fake']
            self.lr_predict_fake = results['lr_predict_fake']
        else:
            self.lr_predict_fake = None

        if 'lr_predict_real' in results:
            self._current_visuals['lr_predict_real'] = results['lr_predict_real']
            self.lr_predict_real = results['lr_predict_real']
        else:
            self.lr_predict_real = None

        if 'lr_inputs' in results:
            self._current_visuals['lr_inputs'] = results['lr_inputs']
            self.lr_inputs = results['lr_inputs']
        else:
            self.lr_inputs = None

        if 'aleatoric_uncertainty' in results:
            self._current_visuals['aleatoric_uncertainty'] = results['aleatoric_uncertainty']
            self.aleatoric_uncertainty = results['aleatoric_uncertainty']
        else:
            self.aleatoric_uncertainty = None

        if 'epistemic_uncertainty' in results:
            self._current_visuals['epistemic_uncertainty'] = results['epistemic_uncertainty']
            self.epistemic_uncertainty = results['epistemic_uncertainty']
        else:
            self.epistemic_uncertainty = None

        if 'uncertainty_var' in results:
            self._current_visuals['uncertainty_var'] = results['uncertainty_var']
            self.uncertainty_var = results['uncertainty_var']
        else:
            self.uncertainty_var = None

    def get_dropout_prob(self):
        if self.__dropout_switch:
            return self.dropout_prob
        else:
            return 0

    def switch_dropout(self, on=True):
        self.__dropout_switch = on

    # no backprop gradients
    def test(self, dropout=0):
        self._current_visuals = {}
        self._current_report = {}
        if dropout <= 1:
            if dropout <=0:
                self.switch_dropout(False)
            else:
                self.switch_dropout(True)
            results = self.forward()
        else:
            self.switch_dropout(True)
            outputsList = []
            uncertaintyList = []
            lastInputs = None
            for i in range(repeat):
                if i>0:
                    self.set_input()
                results = self.forward()
                assert lastInputs is None or np.all(lastInputs == results['inputs']), 'inputs must be the same.'
                lastInputs = results['inputs']
                outputsList.append(results['outputs'])
                if 'aleatoric_uncertainty' in results:
                    uncertaintyList.append(results['aleatoric_uncertainty'])

            if repeat>1:
                outputss = np.stack(outputsList)
                vs = np.var(outputss, axis=0)
                results['outputs'] = outputss.mean(axis=0)
                blur = GaussianBlurring(sigma=self.uncertainty_blur_sigma)
                for outputs in outputsList:
                    for j in range(outputs.shape[0]):
                        outputs[j] = blur(outputs[j])
                outputssb = np.stack(outputsList)
                results['epistemic_uncertainty'] = outputssb.std(axis=0)
                if len(uncertaintyList)>0:
                    auncertainty = np.stack(uncertaintyList)
                    uncertainty = auncertainty.mean(axis=0)
                    results['aleatoric_uncertainty'] = uncertainty
                    results['uncertainty_var'] = vs + np.mean(np.square(auncertainty), axis=0)

        self.retrieve_results(results)
        # self.save_current_visuals(label)

    def backward(self):
        pass

    def optimize_parameters(self):
        fetches = {"train": self.model.train}
        fetches.update(self.averaged_loss_fetches)
        options = tf.RunOptions(timeout_in_ms=10000)
        results = self.sess.run(fetches, feed_dict={'dropout_prob:0': self.get_dropout_prob()}, options=options)
        self.global_step += 1
        for k, v in results.items():
            if k in self.averaged_loss_fetches:
                self._current_report[k] = v

    def train(self, data_source_train, data_source_test=None, epoch_callback=None, step_callback=None, transform=None, cached=True, max_epochs=None, max_steps=None, verbose=1):
        train_writer = tf.summary.FileWriter(self.opt.tb_dir, self.sess.graph)

        opt = self.opt
        queue_start, queue_stop = networks.setup_data_loader(data_source_train, self.enqueue_data, shuffle=True, control_nc=self.opt.control_nc, use_mixup=self.opt.use_mixup, seed=self.opt.seed)
        steps_per_epoch = int(math.ceil(len(data_source_train) / opt.batchSize))
        print("#samples = {}".format(len(data_source_train)))

        if max_epochs is not None:
            max_steps = max_epochs*steps_per_epoch
        if max_steps is None:
            max_steps = 2**32
        # start the data queue
        queue_start(self.sess, callback=self.stop_coord)

        # training
        start = time.time()

        start_step = self.global_step
        last_epoch = math.ceil(self.global_step / steps_per_epoch)
        for step in range(start_step, max_steps+start_step):
            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

            options = None
            run_metadata = None
            if should(opt.trace_freq):
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, timeout_in_ms=1000000)
                run_metadata = tf.RunMetadata()
            else:
                options = tf.RunOptions(timeout_in_ms=1000000)

            fetches = {
                "train": self.model.train,
            }

            if should(opt.progress_freq):
                fetches.update(self.averaged_loss_fetches)
            if should(opt.summary_freq):
                fetches["summary"] = self.summary_merged
            if should(opt.display_freq):
                fetches["display"] = self.display_fetches

            results = self.sess.run(fetches, feed_dict={'dropout_prob:0': self.get_dropout_prob()}, options=options, run_metadata=run_metadata)
            self.global_step = step
            self._current_epoch = math.ceil(self.global_step / steps_per_epoch)
            self._current_report['epoch'] = self._current_epoch
            self._current_report['global_step'] = self.global_step


            if 'display' in results:
                display = results["display"]
                self.retrieve_results(display)
                self.save_current_visuals(str(self._current_epoch))

            if 'summary' in results:
                print("recording summary")
                train_writer.add_summary(results["summary"], self.global_step)

            if should(opt.trace_freq):
                print("recording trace")
                train_writer.add_run_metadata(run_metadata, "step_%d" % self.global_step)

            if should(opt.progress_freq):
                # global_step will have the correct step count if we resume from a checkpoint
                train_step = (self.global_step - 1) % steps_per_epoch + 1
                rate = (step + 1) * opt.batchSize / (time.time() - start)
                remaining = (max_steps - step) * opt.batchSize / rate
                print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (self._current_epoch, train_step, rate, remaining / 60))
                for k, v in results.items():
                    if k in self.averaged_loss_fetches:
                        self._current_report[k] = v
                        print('{}={}'.format(k, v), end=', ')
                sys.stdout.flush()

            if should(opt.save_latest_freq):
                self.save('latest')

            if step_callback:
                try:
                    details = {}
                    for k, v in self._current_report.items():
                        if isinstance(v, np.generic):
                            details[k] = np.asscalar(v)
                        else:
                            details[k] = v

                    if 'display' in results:
                        details['display'] = self.get_current_visuals()

                    ret = step_callback(self, details)
                    if ret == 'stop':
                        break
                except Exception as e:
                    print('\nerror in step callback: ' + str(e))
            if self._current_epoch > last_epoch:
                last_epoch = self._current_epoch
                if epoch_callback:
                    try:
                        details = {'epoch': self._current_epoch, 'step': step}
                        ret = epoch_callback(self, details)
                        if ret == 'stop':
                            break
                    except Exception as e:
                        print('\nerror in epoch callback: ' + str(e))
        train_writer.close()
        queue_stop()

    def predict(self, data_source, dropout=0, cached=False, label=None, step_callback=None, repeat_callback=None, max_steps=None, verbose=1):
        repeat = dropout
        if dropout == 0:
            self.switch_dropout(False)
            repeat = 1
        else:
            self.switch_dropout(True)
        # data_loader_test = CreateDataLoader(data_source, self.opt, cached=cached, verbose=verbose)
        # dataset_test = data_loader_test.load_data()
        # dataset_test_size = len(data_loader_test)

        queue_start, queue_stop = networks.setup_data_loader(data_source, self.enqueue_data, shuffle=False, repeat=repeat, control_nc=self.opt.control_nc, seed=self.opt.seed)
        print("#samples = {}".format(len(data_source)))
        steps_per_epoch = int(math.ceil(len(data_source) / self.opt.batchSize))
        self._current_visuals = {}

        if max_steps is None:
            max_steps = 2**32
        # start the data queue
        queue_start(self.sess, callback=self.stop_coord)

        options = tf.RunOptions(timeout_in_ms=500000)
        max_steps = min(steps_per_epoch, max_steps)

        for step in range(max_steps):
            outputsList = []
            uncertaintyList = []
            lastInputs = None
            for i in range(repeat):
                fetches = {}
                fetches.update(self.loss_fetches)
                fetches.update(self.display_fetches)
                results = self.sess.run(fetches, feed_dict={'dropout_prob:0': self.get_dropout_prob()}, options=options)
                assert lastInputs is None or np.all(lastInputs == results['inputs']), 'inputs must be the same.'
                print('{}-{}'.format(step, results['paths'][0][0]))
                for k, v in results.items():
                    if k in self.loss_fetches:
                        self._current_report[k] = v
                        print('{}={}'.format(k, v), end=', ')
                print('')
                lastInputs = results['inputs']
                outputsList.append(results['outputs'])
                if 'aleatoric_uncertainty' in results:
                    uncertaintyList.append(results['aleatoric_uncertainty'])
                if repeat_callback:
                    try:
                        details = {'step': step, 'repeat': i, 'display': results}
                        repeat_callback(self, details)
                    except Exception as e:
                        print('\nerror in repeat callback: ' + str(e))
            if repeat>1:
                outputss = np.stack(outputsList)
                vs = np.var(outputss, axis=0)
                results['outputs'] = outputss.mean(axis=0)
                blur = GaussianBlurring(sigma=self.uncertainty_blur_sigma)
                for outputs in outputsList:
                    for j in range(outputs.shape[0]):
                        outputs[j] = blur(outputs[j])
                outputssb = np.stack(outputsList)
                results['epistemic_uncertainty'] = outputssb.std(axis=0)
                if len(uncertaintyList)>0:
                    auncertainty = np.stack(uncertaintyList)
                    uncertainty = auncertainty.mean(axis=0)
                    results['aleatoric_uncertainty'] = uncertainty
                    results['uncertainty_var'] = vs + np.mean(np.square(auncertainty), axis=0)
            self.retrieve_results(results)
            self.save_current_visuals(label)
            if step_callback:
                try:
                    details = {'step': step, 'display': self.get_current_visuals()}
                    step_callback(self, details)
                except Exception as e:
                    print('\nerror in step callback: ' + str(e))
        queue_stop()

    def save(self, label='latest'):
        self._current_config['global_step'] = self.global_step
        self._current_config['dropout_prob'] = self.dropout_prob
        self.save_config(label)
        self.save_network(self.sess, 'G', label, [])

    def load(self, label='latest'):
        # self.load_config(label)
        self.old_lr = self._current_config.get('_old_lr', self.opt.lr)
        self.global_step = self._current_config.get('global_step', self.global_step)
        self.dropout_prob = self._current_config.get('dropout_prob', self.dropout_prob)
        if self.opt.which_network is not None:
            self.load_network(self.sess, self.opt.which_network, label)
        else:
            if self.opt.phase != 'train':
                self.load_network(self.sess, 'G', label)
                if self.model.squirrel_error_map is not None:
                    self.load_network(self.sess, 'S', label)
            else:
                self.load_network(self.sess, 'all', label)

    def load_squirrel_discriminator(self, load_dir):
        generator_filters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="squirrel_discriminator")
        saver = tf.train.Saver(generator_filters)
        if os.path.exists(os.path.join(load_dir, "squirrel_discriminator")):
            load_dir = os.path.join(load_dir, "squirrel_discriminator")
        checkpoint = tf.train.latest_checkpoint(load_dir)
        saver.restore(self.sess, checkpoint)

    def load_config(self, label='latest'):
        load_filename = '%s_config.json' % (label)
        assert os.path.exists(self.opt.load_dir), "{} doesn't exist.".format(self.opt.load_dir)
        load_path = os.path.join(self.opt.load_dir, load_filename)
        if os.path.exists(load_path):
            print('loading config.json from ', load_path)
            with open(load_path, 'r') as f:
                config_json = json.load(f)
                # print('config({}) loaded:'.format(label))
                # print('-------')
                # for k, v in sorted(config_json.items()):
                #     if k not in self._current_config or self._current_config[k] != v:
                #         print('%s: %s' % (str(k), str(v)))
                # print('-------')
                for k in config_json.keys():
                    self._current_config[k] = config_json[k]

                    argparse_dict = vars(self.opt)
                    if k in ['model', 'norm_A', 'input_nc', 'output_nc', 'ngf', 'ndf',
                              'use_resize_conv', 'control_nc', 'control_classes',
                              'add_data_type_channel', 'lr_scale', 'lr_nc', 'use_gaussd']:
                        if k in config_json:
                            if k in argparse_dict and argparse_dict[k] != config_json[k]:
                                print("WARNING: {}={} is different in config.json ({}={})".format(k, argparse_dict[k], k, config_json[k]))
                                if not self.opt.no_config_override:
                                    print('override: {}={} --> {}={}'.format(k, argparse_dict[k], k, config_json[k]))
                                    argparse_dict.update({k: config_json[k]})

                        else:
                            print("WARNING: key {} doesn't exist in the config.json file".format(k))
        else:
            print('no config file found in: ', load_path)

        self._current_epoch = self._current_config.get('_current_epoch', self._current_epoch)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def get_current_visuals(self):
        self.input = {'A': self.real_A, 'B': self.real_B, 'path': [str(p[0],'utf-8') for p in self.paths]}
        return self._current_visuals

    def get_current_report(self):
        report = {}
        for k, v in self._current_report.items():
            report[k] = v[0]
        return report

    def train_image(self, img):
        self.set_input(img)
        self.optimize_parameters()

    def predict_image(self, img, dropout=0):
        self.set_input(img)
        self.test(dropout=dropout)
        # self.save_current_visuals()

    def save_current_visuals(self, label=None):
        images = self.get_current_visuals()
        if 'path' in self.input:
            output_paths = [os.path.join(self.opt.save_dir, os.path.split(p)[1]) for p in self.input['path']]
        else:
            output_paths = [self.opt.save_dir for i in range(len(images))]
        if not os.path.exists(self.opt.save_dir):
            os.makedirs(self.opt.save_dir)
        for k, v in images.items():
            for b in range(v.shape[0]):
                ima = v[b]
                channels = ima.shape[2]
                for i in range(channels):
                    im = Image.fromarray(ima[:, :, i])
                    d, n = os.path.split(output_paths[b])
                    n = '{}_{}_b{}_i{}.tif'.format(n, k, b, i)
                    if label:
                        n = '{}_'.format(label) + n
                    im.save(os.path.join(d, n))

    def save_config(self, label):
        self._current_config['_current_epoch'] = self._current_epoch
        opt = vars(self.opt)
        for k in opt:
            self._current_config[k] = opt[k]

        save_filename = '%s_config.json' % (label)
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        with open(save_path, 'w') as f:
            json.dump(self._current_config, f)
        print('config({})  saved to {}'.format(label, save_path))
