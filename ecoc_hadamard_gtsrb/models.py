""" networks structures that used """
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Input, Concatenate, Conv2D, Activation, MaxPooling2D, Dropout
from keras.models import Model
import numpy as np

def get_1bit_ensemble_vgg16(model_weights_path=None, input_tensor=None):
    bottom = VGG16(include_top=False, weights='imagenet', input_shape=(64, 64, 3), input_tensor=input_tensor)
    flatten = Flatten()(bottom.output)
    fc1 = Dense(units=4096, activation='relu', name='fc1')(flatten)
    fc2 = Dense(units=4096, activation='relu', name='fc2')(fc1)
    output = Dense(units=1, kernel_initializer="he_normal", activation='tanh', name='output')(fc2)
    model = Model(inputs=bottom.input, outputs=output)
    if model_weights_path:
        model.load_weights(model_weights_path)
    return model

def get_32class_surrogate_model(model_weights_path=None, input_tensor=None):
    bottom = VGG16(include_top=False, weights='imagenet', input_shape=(64, 64, 3), input_tensor=input_tensor)
    flatten = Flatten()(bottom.output)
    fc1 = Dense(units=4096, activation='relu', name='fc1')(flatten)
    fc2 = Dense(units=4096, activation='relu', name='fc2')(fc1)
    output = Dense(units=32, kernel_initializer="he_normal", activation='softmax', name='output')(fc2)
    model = Model(inputs=bottom.input, outputs=output)
    if model_weights_path:
        model.load_weights(model_weights_path)
    return model

class ECOC_Hadamard_Model():

    def __init__(self, bit_model_weight_path_in_order, hadamard_matrix):
        print('models are prepared in order, i.e.')
        for idx, path in enumerate(bit_model_weight_path_in_order):
            print('Nuber {} model/bit is {}'.format(idx, path))
        self.__model = self.build_ge_model(bit_model_weight_path_in_order)
        self.__hadamard_matrix = hadamard_matrix # ndarray


    # properties
    @property
    def model(self):
        return self.__model
    @property
    def hadamard_matrix(self):
        return self.__hadamard_matrix
    @property
    def num_class(self):
        return self.model.output_shape

        # build model
    def build_chuan_model(self, bit_model_weight_path_in_order):
        # Chinese character 川 that all models works independently but sharing one same input image
        # base models been stored into a list, needs test
        input = Input((64, 64, 3), name='very_input')

        bit_models = []
        for idx, path in enumerate(bit_model_weight_path_in_order):
            base = get_1bit_ensemble_vgg16(path, input_tensor=input)
            model = Model(inputs=input, outputs=base.outputs)
            for layer in model.layers:
                layer.name = 'bit{}_'.format(idx+1) + layer.name
            bit_models.append(model)

        # base1 = get_1bit_ensemble_vgg16(weight_paths[0], input_tensor=input)
        # model1 = Model(inputs=input, outputs=base1.outputs)
        # for layer in model1.layers:
        #     layer.name = 'm1_' + layer.name
        #
        # base2 = get_1bit_ensemble_vgg16(weight_paths[1], input_tensor=input)
        # model2 = Model(inputs=input, outputs=base2.outputs)
        # for layer in model2.layers:
        #     layer.name = 'm2_' + layer.name
        #
        # base3 = get_1bit_ensemble_vgg16(weight_paths[2], input_tensor=input)
        # model3 = Model(inputs=input, outputs=base3.outputs)
        # for layer in model3.layers:
        #     layer.name = 'm3_' + layer.name
        #
        # base4 = get_1bit_ensemble_vgg16(weight_paths[3], input_tensor=input)
        # model4 = Model(inputs=input, outputs=base4.outputs)
        # for layer in model4.layers:
        #     layer.name = 'm4_' + layer.name
        #
        # base5 = get_1bit_ensemble_vgg16(weight_paths[4], input_tensor=input)
        # model5 = Model(inputs=input, outputs=base5.outputs)
        # for layer in model5.layers:
        #     layer.name = 'm5_' + layer.name
        #
        # base6 = get_1bit_ensemble_vgg16(weight_paths[5], input_tensor=input)
        # model6 = Model(inputs=input, outputs=base6.outputs)
        # for layer in model6.layers:
        #     layer.name = 'm6_' + layer.name
        #
        # base7 = get_1bit_ensemble_vgg16(weight_paths[6], input_tensor=input)
        # model7 = Model(inputs=input, outputs=base7.outputs)
        # for layer in model7.layers:
        #     layer.name = 'm7_' + layer.name
        #
        # base8 = get_1bit_ensemble_vgg16(weight_paths[7], input_tensor=input)
        # model8 = Model(inputs=input, outputs=base8.outputs)
        # for layer in model8.layers:
        #     layer.name = 'm8_' + layer.name
        #
        # base9 = get_1bit_ensemble_vgg16(weight_paths[8], input_tensor=input)
        # model9 = Model(inputs=input, outputs=base9.outputs)
        # for layer in model9.layers:
        #     layer.name = 'm9_' + layer.name
        #
        # base10 = get_1bit_ensemble_vgg16(weight_paths[9], input_tensor=input)
        # model10 = Model(inputs=input, outputs=base10.outputs)
        # for layer in model10.layers:
        #     layer.name = 'm10_' + layer.name
        #
        # base11 = get_1bit_ensemble_vgg16(weight_paths[10], input_tensor=input)
        # model11 = Model(inputs=input, outputs=base11.outputs)
        # for layer in model11.layers:
        #     layer.name = 'm11_' + layer.name
        #
        # base12 = get_1bit_ensemble_vgg16(weight_paths[11], input_tensor=input)
        # model12 = Model(inputs=input, outputs=base12.outputs)
        # for layer in model12.layers:
        #     layer.name = 'm12_' + layer.name
        #
        # base13 = get_1bit_ensemble_vgg16(weight_paths[12], input_tensor=input)
        # model13 = Model(inputs=input, outputs=base13.outputs)
        # for layer in model13.layers:
        #     layer.name = 'm13_' + layer.name
        #
        # base14 = get_1bit_ensemble_vgg16(weight_paths[13], input_tensor=input)
        # model14 = Model(inputs=input, outputs=base14.outputs)
        # for layer in model14.layers:
        #     layer.name = 'm14_' + layer.name
        #
        # base15 = get_1bit_ensemble_vgg16(weight_paths[14], input_tensor=input)
        # model15 = Model(inputs=input, outputs=base15.outputs)
        # for layer in model15.layers:
        #     layer.name = 'm15_' + layer.name
        #
        # base16 = get_1bit_ensemble_vgg16(weight_paths[15], input_tensor=input)
        # model16 = Model(inputs=input, outputs=base16.outputs)
        # for layer in model16.layers:
        #     layer.name = 'm16_' + layer.name
        #
        # base17 = get_1bit_ensemble_vgg16(weight_paths[16], input_tensor=input)
        # model17 = Model(inputs=input, outputs=base17.outputs)
        # for layer in model17.layers:
        #     layer.name = 'm17_' + layer.name
        #
        # base32 = get_1bit_ensemble_vgg16(weight_paths[31], input_tensor=input)
        # model32 = Model(inputs=input, outputs=base32.outputs)
        # for layer in model32.layers:
        #     layer.name = 'm32_' + layer.name

        cat_model = Model(input = input,
                          outputs = Concatenate()(
                              [model.layers[-1].output for model in bit_models]
                          ))

        # cat_model = Model(inputs=input,
        #                   outputs=Concatenate()([
        #                       model1.layers[-1].output, model2.layers[-1].output, model3.layers[-1].output,
        #                       model4.layers[-1].output, model5.layers[-1].output, model6.layers[-1].output,
        #                       model7.layers[-1].output, model8.layers[-1].output, model9.layers[-1].output,
        #                       model10.layers[-1].output, model11.layers[-1].output, model12.layers[-1].output,
        #                       model13.layers[-1].output, model14.layers[-1].output, model15.layers[-1].output,
        #                   ]))
        return cat_model

    def build_ge_model(self, bit_model_weight_path_in_order):
        # Chinese character 个 that all models share some bottom layer and input image, while work independently later at topper layers
        def get_ge_shared_and_unique_parts( num_bottom_layers, output_model_type, full_model_weight_path=None, input_tensor=None):
            """

            :param full_model_weight_path:
            :param input_tensor: must be passed if shared bottom is desired
            :param num_bottom_layers:  layers that should be abandon to get branch
            :return:
            """
            bottom = VGG16(include_top=False, weights='imagenet', input_shape=(64, 64, 3), input_tensor=input_tensor)
            flatten = Flatten()(bottom.output)
            fc1 = Dense(units=4096, activation='relu', name='fc1')(flatten)
            fc2 = Dense(units=4096, activation='relu', name='fc2')(fc1)
            output = Dense(units=1, kernel_initializer="he_normal", activation='tanh', name='output')(fc2)
            model = Model(inputs=bottom.input, outputs=output)
            if full_model_weight_path:
                model.load_weights(full_model_weight_path)
            else:
                model.load_weights(r'H:\LableCodingNetwork\trained-models\GTSRB\surrogate_VGG16_Dec23\output_replaced_with_dense_for_bit_model.hdf5')
            if output_model_type == 'bottom':
                shared_bottom = Model(inputs = model.inputs,
                               outputs = model.layers[num_bottom_layers].output) # todo considering add name to models
                # unique_branch = Model(inputs = model.layers[num_bottom_layers+1].input,
                #                       outputs = model.output)
                return shared_bottom
            elif output_model_type == 'branch':
                branch_input = Input(model.layers[num_bottom_layers+1].input_shape[1:])
                unique_branch = branch_input
                for layer in model.layers[num_bottom_layers+1:]:
                    unique_branch = layer(unique_branch)
                unique_branch = Model(inputs=branch_input, outputs=unique_branch)
                return unique_branch
            else:
                raise TypeError('output_model_type has to be either bottom or branch')


        num_freezed_layer = int(bit_model_weight_path_in_order[0].split('freeze')[1].split('_')[0])
        very_input = Input((64, 64, 3), name='very_input')
        # the bottom x layers that shared by all ensemble branches, if no weights specified then 32-class surrogate model weights is by default used
        shared_bottom = get_ge_shared_and_unique_parts(num_freezed_layer, output_model_type= 'bottom', input_tensor = very_input)

        bit_top_branches = []
        for idx, path in enumerate(bit_model_weight_path_in_order):
            # top y layers that is unique for every branches, well takes the same input, always need pass new weights
            branch = get_ge_shared_and_unique_parts(num_freezed_layer, full_model_weight_path=path,
                                                    output_model_type= 'branch', input_tensor=very_input)
            for layer in branch.layers:
                layer.name = 'bit{}_'.format(idx+1) + layer.name
            bit_top_branches.append(branch)

        # reconnecting everything
        reconnect_branch_ends = []
        for bit_branch in bit_top_branches:
            connected = shared_bottom.output
            for layer in bit_branch.layers:
                connected = layer(connected)
            reconnect_branch_ends.append(connected)

        reconnect_cat_model = Model(inputs=very_input,
                                    outputs=Concatenate()(
                                        [bit_branch_end(shared_bottom.output) for bit_branch_end in
                                         reconnect_branch_ends]
                                    ))
        # end reconnection

        cat_model = Model(inputs = very_input,
                          outputs = Concatenate()(
                              [bit_branch(shared_bottom.output) for bit_branch in bit_top_branches]
                          ))
        return cat_model

    def predict(self, img, output_type = 'probability'):
        """
        predict image as a concatnate model, so the output size should be 32 since it't tanh activation and hinge loss
        :param img:
        :return:
        """
        # outputs that been through activation (tanh, -1,1)
        outputs = self.model.predict(img)  # shape should be (,32)
        if output_type == 'activated':
            return outputs
        # calculate dot product with all hadamard codewords to get what I called dot product logits
        dp_logits = np.array([np.vdot(outputs, codeword) for codeword in self.hadamard_matrix])
        dp_logits2 = np.dot(self.hadamard_matrix, outputs)
        assert dp_logits == dp_logits2
        if output_type == 'dot_product_logits':
            return dp_logits2
        codeword = self.hadamard_matrix[np.argmax(dp_logits)]
        if output_type == 'codeword':
            return codeword
        # knid of softmax
        probabilities = dp_logits / np.sum(dp_logits)
        if output_type == 'probability':
            return probabilities

if __name__ == '__main__':
    # module test
    from glob import glob
    bit_model_weights_list = glob(r'H:\LableCodingNetwork\trained-models\GTSRB\ECOC\Hadamard32_surrogate_weights_freeze6_bit_*\final_trained_weights.hdf5')
    # sort by real number instead of str order
    bit_model_weights_list.sort(key=lambda x: int(x.split('bit_')[-1].split('\\')[0]))
    HADAMARD_MATRIX = np.load(r'H:\LableCodingNetwork\ecoc_hadamard_gtsrb\hadamard32.npy')
    TOP32GTSRB_CATEGORIES = np.load(r'H:\LableCodingNetwork\database\GTSRB\gysrb_top32category_label.npy')


    ecoc_model = ECOC_Hadamard_Model(bit_model_weights_list[:4], HADAMARD_MATRIX)
