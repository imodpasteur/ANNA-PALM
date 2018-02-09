import json
import argparse

def create_model_from_config(config_path, **kwargs):
    with open(config_path, 'r') as f:
        config_json = json.load(f)
        parser = argparse.ArgumentParser()
        opt = parser.parse_args([])
        config = vars(opt)
        config.update(config_json)
        config.update(kwargs)
        return create_model(opt)

def create_model(opt):
    model = None
    print('model:', opt.model)
    if 'tensorflow' in opt.model:
        if 'align_data' in opt:
            assert(opt.align_data == True)
        if 'dataset_mode' in opt:
            assert(opt.dataset_mode == 'aligned')
        from .anet_tensorflow_model import AnetModel
        model = AnetModel()
    elif opt.model == 'one_direction_test':
        from .one_direction_test_model import OneDirectionTestModel
        model = OneDirectionTestModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
