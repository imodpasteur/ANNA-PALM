import tensorflow  as tf

class ReversibleNet():

    def __init__(self, num_blocks, num_channels):
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.weights_per_layer = 12
        self.weight_list = self.def_rev_block_weights(self.num_channels)
        self.weight_ta = tf.TensorArray(dtype=tf.float32, size=len(self.weight_list), dynamic_size=False, clear_after_read=False,
                            infer_shape=False)
        for i in range(len(self.weight_list)):
            self.weight_ta = self.weight_ta.write(i, self.weight_list[i])

    def forward_pass(self, inputs):
        """
        Generates forward pass of the revnet
        Args:
            inputs: tuple of two same sized input tensors
        Returns:
            layer_rev_out: tuple of the two ouput tensors of the last network layer
        """
        print("revnet")
        print(inputs[0].shape)
        print(inputs[1].shape)

        def loop_body(layer_index, inputs, weights):
            layer_weights = []
            for i in range(self.weights_per_layer):
                layer_weights.append(weights.read(layer_index * self.weights_per_layer + i))

            in_1, in_2 = inputs
            out_1, out_2 = rev_block(in_1, in_2, layer_weights, reverse=False)
            out_1.set_shape(in_1.get_shape())
            out_2.set_shape(in_1.get_shape())
            output = (out_1, out_2)

            return [layer_index + 1, output, weights]

        _, layer_rev_out, ta = tf.while_loop(lambda i, _, __: i < self.num_blocks, loop_body,
                                             loop_vars=[tf.constant(0), inputs, self.weight_ta], parallel_iterations=1,
                                             back_prop=False)
        return layer_rev_out


    def backward_pass(self, inputs):
        """
        Generates backward pass of the revnet
        Args:
            inputs: tuple of two same sized input tensors going into the last layer
        Returns:
            layer_rev_out: tuple of the two ouput tensors of the first network layer
        """
        print("revnet_backward")
        print(inputs[0].shape)
        print(inputs[1].shape)

        def loop_body(layer_index, inputs, weights):
            layer_weights = []
            for i in range(self.weights_per_layer):
                layer_weights.append(weights.read(layer_index * self.weights_per_layer + i))

            in_1, in_2 = inputs
            out_1, out_2 = rev_block(in_1, in_2, layer_weights, reverse=True)
            out_1.set_shape(in_1.get_shape())
            out_2.set_shape(in_1.get_shape())
            output = (out_1, out_2)

            return [layer_index - 1, output, weights]

        _, layer_rev_out, ta = tf.while_loop(lambda i, _, __: i >= 0, loop_body,
                                             loop_vars=[tf.constant(self.num_blocks - 1), inputs, self.weight_ta], parallel_iterations=1,
                                             back_prop=False)
        return layer_rev_out


    def def_rev_block_weights(self, channels):
        """
        Defines weights for the revnet.
        Args:
            channels: channel number of the input tensors
        Returns:
            weight_list: list of all weights for the revnet, sorted alphabetically by variable-scope/name
        """
        weight_list = []
        for i in range(self.num_blocks):
            with tf.variable_scope("rev_core_%03d" % (i + 1)):
                with tf.variable_scope("rev_block"):
                    with tf.variable_scope("f"):
                        with tf.variable_scope("res_block"):
                            with tf.variable_scope("sub_1"):
                                with tf.variable_scope("batchnorm"):
                                    weight_list.append(tf.get_variable("offset", [channels], dtype=tf.float32,
                                                             initializer=tf.zeros_initializer()))
                                    weight_list.append(tf.get_variable("scale", [channels], dtype=tf.float32,
                                                            initializer=tf.random_normal_initializer(1.0, 0.02)))
                                with tf.variable_scope("conv3x3"):
                                    weight_list.append(tf.get_variable("filter", [3, 3, channels, channels], dtype=tf.float32,
                                                             initializer=tf.random_normal_initializer(0, 0.02)))
                            with tf.variable_scope("sub_2"):
                                with tf.variable_scope("batchnorm"):
                                    weight_list.append(tf.get_variable("offset", [channels], dtype=tf.float32,
                                                             initializer=tf.zeros_initializer()))
                                    weight_list.append(tf.get_variable("scale", [channels], dtype=tf.float32,
                                                            initializer=tf.random_normal_initializer(1.0, 0.02)))
                                with tf.variable_scope("conv3x3"):
                                    weight_list.append(tf.get_variable("filter", [3, 3, channels, channels], dtype=tf.float32,
                                                             initializer=tf.random_normal_initializer(0, 0.02)))
                    with tf.variable_scope("g"):
                        with tf.variable_scope("res_block"):
                            with tf.variable_scope("sub_1"):
                                with tf.variable_scope("batchnorm"):
                                    weight_list.append(tf.get_variable("offset", [channels], dtype=tf.float32,
                                                             initializer=tf.zeros_initializer()))
                                    weight_list.append(tf.get_variable("scale", [channels], dtype=tf.float32,
                                                            initializer=tf.random_normal_initializer(1.0, 0.02)))
                                with tf.variable_scope("conv3x3"):
                                    weight_list.append(tf.get_variable("filter", [3, 3, channels, channels], dtype=tf.float32,
                                                             initializer=tf.random_normal_initializer(0, 0.02)))
                            with tf.variable_scope("sub_2"):
                                with tf.variable_scope("batchnorm"):
                                    weight_list.append(tf.get_variable("offset", [channels], dtype=tf.float32,
                                                             initializer=tf.zeros_initializer()))
                                    weight_list.append(tf.get_variable("scale", [channels], dtype=tf.float32,
                                                            initializer=tf.random_normal_initializer(1.0, 0.02)))
                                with tf.variable_scope("conv3x3"):
                                    weight_list.append(tf.get_variable("filter", [3, 3, channels, channels], dtype=tf.float32,
                                                             initializer=tf.random_normal_initializer(0, 0.02)))
        return weight_list

    def compute_revnet_gradients_of_forward_pass(self, y1, y2, dy1, dy2):
        """
        Computes gradients.
        Args:
          y1: Output activation 1.
          y2: Output activation 2.
          dy1: Output gradient 1.
          dy2: Output gradient 2.
        Returns:
          dx1: Input gradient 1.
          dx2: Input gradient 2.
          grads_and_vars: List of tuples of gradient and variable.
        """
        with tf.name_scope("manual_gradients"):
            print("Manually building gradient graph.")
            tf.get_variable_scope().reuse_variables()

            grads_list = []

            weights = self.weight_ta
            weights_grads = tf.TensorArray(dtype=tf.float32, size=len(self.weight_list), dynamic_size=False,
                                           clear_after_read=False, infer_shape=False)

            outputs = (y1, y2)
            output_grads = (dy1, dy2)

            def loop_body(layer_index, outputs, output_grads, weights, weights_grads):
                layer_weights = []
                for i in range(self.weights_per_layer):
                    layer_weights.append(weights.read(layer_index * self.weights_per_layer + i))

                (inputs, input_grads, layer_weights_grads) = self.backprop_layer_forward_pass(outputs, output_grads,
                                                                                              layer_weights)

                for i in range(self.weights_per_layer):
                    weights_grads = weights_grads.write(layer_index * self.weights_per_layer + i,
                                                        tf.squeeze(layer_weights_grads[i]))

                inputs[1].set_shape(outputs[1].get_shape())
                inputs[0].set_shape(outputs[0].get_shape())
                input_grads[1].set_shape(output_grads[1].get_shape())
                input_grads[0].set_shape(output_grads[0].get_shape())
                return [layer_index - 1, inputs, input_grads, weights, weights_grads]

            _, inputs, input_grads, _, weights_grads = tf.while_loop(lambda i, *_: i >= 0, loop_body,
                                                                     [tf.constant(self.num_blocks - 1), outputs,
                                                                      output_grads, weights, weights_grads],
                                                                     parallel_iterations=1, back_prop=False)

            for i in range(self.num_blocks):
                grads_list.append(weights_grads.read(index=i))

            return input_grads, list(zip(grads_list, self.weight_list))


    def backprop_layer_forward_pass(self, outputs, output_grads, layer_weights):
        """
        Computes gradient for one layer.
        Args:
          outputs: Outputs of the layer
          output_grads: gradients for the layer outputs
          layer_weights: all weights for the rev_layer
        Returns:
          inputs: inputs of the layer
          input_grads: gradients for the layer inputs
          weight_grads: gradients for the layer weights
        """
        # First , reverse the layer to r e t r i e v e inputs
        y1, y2 = outputs[0], outputs[1]
        # F_weights , G_weights = tf.split(layer_weights, num_or_size_splits=2, axis=1)
        F_weights = layer_weights[0:6]
        G_weights = layer_weights[6:12]

        with tf.variable_scope("rev_block"):
            z1_stop = tf.stop_gradient(y1)
            with tf.variable_scope("g"):
                G_z1 = res_block(z1_stop, G_weights)
                x2 = y2 - G_z1
                x2_stop = tf.stop_gradient(x2)
            with tf.variable_scope("f"):
                F_x2 = res_block(x2_stop, F_weights)
                x1 = y1 - F_x2
                x1_stop = tf.stop_gradient(x1)

        y1_grad = output_grads[0]
        y2_grad = output_grads[1]
        z1 = x1_stop + F_x2
        y2 = x2_stop + G_z1
        y1 = z1

        z1_grad = tf.gradients(y2, z1_stop, y2_grad) + y1_grad
        x2_grad = tf.gradients(y1, x2_stop, z1_grad) + y2_grad
        x1_grad = z1_grad
        G_grads = tf.gradients(y2, G_weights, y2_grad)
        F_grads = tf.gradients(y1, F_weights, z1_grad)

        inputs = (x1_stop, x2_stop)
        input_grads = (tf.squeeze(x1_grad, axis=0), tf.squeeze(x2_grad, axis=0))
        weight_grads = F_grads + G_grads
        return inputs, input_grads, weight_grads


    def compute_revnet_gradients_of_backward_pass(self, x1, x2, dx1, dx2):
        """
        Computes gradients.
        Args:
          y1: Output activation 1.
          y2: Output activation 2.
          dy1: Output gradient 1.
          dy2: Output gradient 2.
        Returns:
          dx1: Input gradient 1.
          dx2: Input gradient 2.
          grads_and_vars: List of tuples of gradient and variable.
        """
        with tf.name_scope("manual_gradients"):
            print("Manually building gradient graph.")
            tf.get_variable_scope().reuse_variables()

            grads_list = []

            weights = self.weight_ta
            weights_grads = tf.TensorArray(dtype=tf.float32, size=len(self.weight_list), dynamic_size=False,
                                           clear_after_read=False, infer_shape=False)

            outputs = (x1, x2)
            output_grads = (dx1, dx2)

            def loop_body(layer_index, outputs, output_grads, weights, weights_grads):
                layer_weights = []
                for i in range(self.weights_per_layer):
                    layer_weights.append(weights.read(layer_index * self.weights_per_layer + i))

                (inputs, input_grads, layer_weights_grads) = self.backprop_layer_forward_pass(outputs, output_grads, layer_weights)

                for i in range(self.weights_per_layer):
                    weights_grads = weights_grads.write(layer_index * self.weights_per_layer + i, tf.squeeze(layer_weights_grads[i]))

                inputs[1].set_shape(outputs[1].get_shape())
                inputs[0].set_shape(outputs[0].get_shape())
                input_grads[1].set_shape(output_grads[1].get_shape())
                input_grads[0].set_shape(output_grads[0].get_shape())
                return [layer_index + 1, inputs, input_grads, weights, weights_grads]

            _, inputs, input_grads, _, weights_grads = tf.while_loop(lambda i, *_: i < self.num_blocks - 1, loop_body,
                                                                     [tf.constant(0), outputs,
                                                                      output_grads, weights, weights_grads],
                                                                     parallel_iterations=1, back_prop=False)

            for i in range(self.num_blocks):
                grads_list.append(weights_grads.read(index=i))

            return input_grads, list(zip(grads_list, self.weight_list))


    def backprop_layer_backward_pass(self, outputs, output_grads, layer_weights):
        """
        Computes gradient for one layer.
        Args:
          outputs: Outputs of the layer
          output_grads: gradients for the layer outputs
          layer_weights: all weights for the rev_layer
        Returns:
          inputs: inputs of the layer
          input_grads: gradients for the layer inputs
          weight_grads: gradients for the layer weights
        """
        # First , reverse the layer to r e t r i e v e inputs
        x1, x2 = outputs[0], outputs[1]
        # F_weights , G_weights = tf.split(layer_weights, num_or_size_splits=2, axis=1)
        F_weights = layer_weights[0:6]
        G_weights = layer_weights[6:12]

        with tf.variable_scope("rev_block"):
            x2_stop = tf.stop_gradient(x2)
            with tf.variable_scope("f"):
                F_x2 = res_block(x2_stop, F_weights)
                y1 = x1 + F_x2
                z1_stop = tf.stop_gradient(y1)
            with tf.variable_scope("g"):
                G_z1 = res_block(z1_stop, G_weights)
                y2 = x2 + G_z1
                y2_stop = tf.stop_gradient(y2)

        x1_grad = output_grads[0]
        x2_grad = output_grads[1]
        z1 = z1_stop
        x2 = y2_stop - G_z1
        x1 = z1 - F_x2

        y2_grad = tf.gradients([x1, x2], y2_stop, [x1_grad, x2_grad])
        y1_grad = tf.gradients([x1, x2], z1_stop, [x1_grad, x2_grad])
        z1_grad = y1_grad
        G_grads = tf.gradients([x1, x2], G_weights, [x1_grad, x2_grad])
        F_grads = tf.gradients([x1, x2], F_weights, [x1_grad, x2_grad])

        inputs = (z1_stop, y2_stop)
        input_grads = (tf.squeeze(y1_grad, axis=0), tf.squeeze(y2_grad, axis=0))
        weight_grads = F_grads + G_grads
        return inputs, input_grads, weight_grads


def rev_block(in_1, in_2, weights, reverse):
    # weights_f, weights_g = tf.split(weights, num_or_size_splits=2, axis=1)
    weights_f = weights[0:6]
    weights_g = weights[6:12]
    with tf.variable_scope("rev_block"):
        if reverse:
            # x2 = y2 - NN2(y1)
            with tf.variable_scope("g"):
                out_2 = in_2 - res_block(in_1, weights_g)

            # x1 = y1 - NN1(x2)
            with tf.variable_scope("f"):
                out_1 = in_1 - res_block(out_2, weights_f)
        else:
            # y1 = x1 - NN1(x2)
            with tf.variable_scope("f"):
                out_1 = in_1 - res_block(in_2, weights_f)

            # y2 = x2 - NN2(y1)
            with tf.variable_scope("g"):
                out_2 = in_2 - res_block(out_1, weights_g)

        return [out_1, out_2]

def res_block(in_1, weights):
    with tf.variable_scope("res_block"):
        # weights_1, weights_2 = tf.split(weights, num_or_size_splits=2, axis=1)
        weights_1 = weights[0:3]
        weights_2 = weights[3:6]
        with tf.variable_scope("sub_1"):
            out_1 = rev_conv3x3(in_1, tf.squeeze(weights_1[2]))
            out_1 = rev_batchnorm(out_1, weights_1[0:2])
            out_1 = rev_lrelu(out_1, 0.2)
        with tf.variable_scope("sub_2"):
            out_1 = rev_conv3x3(out_1, tf.squeeze(weights_2[2]))
            out_1 = rev_batchnorm(out_1, weights_2[0:2])
            out_1 = out_1 + in_1
            out_1 = rev_lrelu(out_1, 0.2)
        return out_1

def rev_conv3x3(batch_input, weight):
    with tf.variable_scope("conv3x3"):
        in_channels = batch_input.get_shape()[3]
        out_channels = in_channels
        filter = weight
        padded_in_1 = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        out_1 = tf.nn.conv2d(padded_in_1, filter, [1, 1, 1, 1], padding="VALID")
        return out_1

def rev_lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def rev_batchnorm(input, weights):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)
        channels = input.get_shape()[3]
        offset = weights[0]
        scale = weights[1]
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale,
                                               variance_epsilon=variance_epsilon)
        return normalized