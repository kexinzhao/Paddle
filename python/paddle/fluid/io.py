#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import shutil

from paddle.fluid.evaluator import Evaluator
from paddle.fluid.framework import Program, Parameter, default_main_program, Variable
from . import core

__all__ = [
    'save_vars', 'save_params', 'save_persistables', 'load_vars', 'load_params',
    'load_persistables', 'save_inference_model', 'load_inference_model',
    'get_inference_program', 'save_checkpoint', 'load_checkpoint',
    'clean_checkpoint', 'load_persist_vars_without_grad',
    'save_persist_vars_without_grad', 'get_latest_checkpoint_serial'
]


def is_parameter(var):
    """Check whether the variable is a Parameter.

    This function checks whether the input variable is a Parameter.

    Args:
        var : The input variable.

    Returns:
        boolean result whether the variable is a Parameter.
    """
    return isinstance(var, Parameter)


def is_persistable(var):
    if var.desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
            var.desc.type() == core.VarDesc.VarType.FETCH_LIST:
        return False
    return var.persistable


def _clone_var_in_block_(block, var):
    assert isinstance(var, Variable)
    return block.create_var(
        name=var.name,
        shape=var.shape,
        dtype=var.dtype,
        type=var.type,
        lod_level=var.lod_level,
        persistable=True)


def save_vars(executor,
              dirname,
              main_program=None,
              vars=None,
              predicate=None,
              filename=None):
    """
    Save variables to directory by executor.

    :param executor: executor that save variable
    :param dirname: directory path
    :param main_program: program. If vars is None, then filter all variables in this
    program which fit `predicate`. Default default_main_program.
    :param predicate: The Predicate describes a callable that returns a variable
    as a bool. If it returns true, the corresponding input variable will be saved.
    :param vars: variables need to be saved. If vars is specified, program & predicate
    will be ignored
    :param filename: The name of a single file that all vars are saved to.
        If it is None, save variables to separate files.

    :return: None
    """
    if vars is None:
        if main_program is None:
            main_program = default_main_program()
        if not isinstance(main_program, Program):
            raise TypeError("program should be as Program type or None")

        save_vars(
            executor,
            dirname=dirname,
            vars=filter(predicate, main_program.list_vars()),
            filename=filename)
    else:
        save_program = Program()
        save_block = save_program.global_block()

        save_var_map = {}
        for each_var in vars:
            # NOTE: don't save the variable which type is RAW
            if each_var.type == core.VarDesc.VarType.RAW:
                continue
            new_var = _clone_var_in_block_(save_block, each_var)
            if filename is None:
                save_block.append_op(
                    type='save',
                    inputs={'X': [new_var]},
                    outputs={},
                    attrs={'file_path': os.path.join(dirname, new_var.name)})
            else:
                save_var_map[new_var.name] = new_var

        if filename is not None:
            save_var_list = []
            for name in sorted(save_var_map.keys()):
                save_var_list.append(save_var_map[name])

            save_block.append_op(
                type='save_combine',
                inputs={'X': save_var_list},
                outputs={},
                attrs={'file_path': os.path.join(dirname, filename)})

        executor.run(save_program)


def save_params(executor, dirname, main_program=None, filename=None):
    """
    Save all parameters to directory with executor.
    """
    save_vars(
        executor,
        dirname=dirname,
        main_program=main_program,
        vars=None,
        predicate=is_parameter,
        filename=filename)


def save_persistables(executor, dirname, main_program=None, filename=None):
    """
    Save all persistables to directory with executor.
    """
    save_vars(
        executor,
        dirname=dirname,
        main_program=main_program,
        vars=None,
        predicate=is_persistable,
        filename=filename)


def load_vars(executor,
              dirname,
              main_program=None,
              vars=None,
              predicate=None,
              filename=None):
    """
    Load variables from directory by executor.

    :param executor: executor that load variable
    :param dirname: directory path
    :param main_program: program. If vars is None, then filter all variables in this
    program which fit `predicate`. Default default_main_program().
    :param predicate: The Predicate describes a callable that returns a variable
    as a bool. If it returns true, the corresponding input variable will be loaded.
    :param vars: variables need to be loaded. If vars is specified, program &
    predicate will be ignored
    :param filename: The name of the single file that all vars are loaded from.
        If it is None, load variables from separate files.

    :return: None
    """
    if vars is None:
        if main_program is None:
            main_program = default_main_program()
        if not isinstance(main_program, Program):
            raise TypeError("program's type should be Program")

        load_vars(
            executor,
            dirname=dirname,
            vars=filter(predicate, main_program.list_vars()),
            filename=filename)
    else:
        load_prog = Program()
        load_block = load_prog.global_block()

        load_var_map = {}
        for each_var in vars:
            assert isinstance(each_var, Variable)
            if each_var.type == core.VarDesc.VarType.RAW:
                continue
            new_var = _clone_var_in_block_(load_block, each_var)
            if filename is None:
                load_block.append_op(
                    type='load',
                    inputs={},
                    outputs={'Out': [new_var]},
                    attrs={'file_path': os.path.join(dirname, new_var.name)})
            else:
                load_var_map[new_var.name] = new_var

        if filename is not None:
            load_var_list = []
            for name in sorted(load_var_map.keys()):
                load_var_list.append(load_var_map[name])

            load_block.append_op(
                type='load_combine',
                inputs={},
                outputs={"Out": load_var_list},
                attrs={'file_path': os.path.join(dirname, filename)})

        executor.run(load_prog)


def load_params(executor, dirname, main_program=None, filename=None):
    """
    load all parameters from directory by executor.
    """
    load_vars(
        executor,
        dirname=dirname,
        main_program=main_program,
        predicate=is_parameter,
        filename=filename)


def load_persistables(executor, dirname, main_program=None, filename=None):
    """
    load all persistables from directory by executor.
    """
    load_vars(
        executor,
        dirname=dirname,
        main_program=main_program,
        predicate=is_persistable,
        filename=filename)


def get_inference_program(target_vars, main_program=None):
    if main_program is None:
        main_program = default_main_program()
    if not isinstance(target_vars, list):
        target_vars = [target_vars]
    vars = []
    for var in target_vars:
        if isinstance(var, Evaluator):
            vars.extend(var.states)
            vars.extend(var.metrics)
        else:
            vars.append(var)
    pruned_program = main_program.prune(targets=vars)
    inference_program = pruned_program.inference_optimize()
    return inference_program


def prepend_feed_ops(inference_program,
                     feed_target_names,
                     feed_holder_name='feed'):
    if len(feed_target_names) == 0:
        return

    global_block = inference_program.global_block()
    feed_var = global_block.create_var(
        name=feed_holder_name,
        type=core.VarDesc.VarType.FEED_MINIBATCH,
        persistable=True)

    for i, name in enumerate(feed_target_names):
        out = global_block.var(name)
        global_block.prepend_op(
            type='feed',
            inputs={'X': [feed_var]},
            outputs={'Out': [out]},
            attrs={'col': i})


def append_fetch_ops(inference_program,
                     fetch_target_names,
                     fetch_holder_name='fetch'):
    global_block = inference_program.global_block()
    fetch_var = global_block.create_var(
        name=fetch_holder_name,
        type=core.VarDesc.VarType.FETCH_LIST,
        persistable=True)

    for i, name in enumerate(fetch_target_names):
        global_block.append_op(
            type='fetch',
            inputs={'X': [name]},
            outputs={'Out': [fetch_var]},
            attrs={'col': i})


def save_inference_model(dirname,
                         feeded_var_names,
                         target_vars,
                         executor,
                         main_program=None,
                         model_filename=None,
                         params_filename=None):
    """
    Build a model especially for inference,
    and save it to directory by the executor.

    :param dirname: directory path
    :param feeded_var_names: Names of variables that need to be feeded data during inference
    :param target_vars: Variables from which we can get inference results.
    :param executor: executor that save inference model
    :param main_program: original program, which will be pruned to build the inference model.
            Default default_main_program().
    :param model_filename: The name of file to save inference program.
        If not specified, default filename `__model__` will be used.
    :param params_filename: The name of file to save parameters.
        It is used for the case that all parameters are saved in a single binary file.
        If not specified, parameters are considered saved in separate files.

    :return: None
    """
    if isinstance(feeded_var_names, basestring):
        feeded_var_names = [feeded_var_names]
    else:
        if len(feeded_var_names) > 0:
            if not (bool(feeded_var_names) and all(
                    isinstance(name, basestring) for name in feeded_var_names)):
                raise ValueError("'feed_var_names' should be a list of str.")

    if isinstance(target_vars, Variable):
        target_vars = [target_vars]
    else:
        if not (bool(target_vars) and all(
                isinstance(var, Variable) for var in target_vars)):
            raise ValueError("'target_vars' should be a list of Variable.")

    if main_program is None:
        main_program = default_main_program()
    copy_program = main_program.clone()

    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    # Clear the is_target information and remove the existed feed and fetch op
    global_block = copy_program.global_block()
    for i, op in enumerate(global_block.ops):
        op.desc.set_is_target(False)
        if op.type == "feed" or op.type == "fetch":
            global_block.remove_op(i)
    copy_program.desc.flush()

    pruned_program = copy_program.prune(targets=target_vars)
    inference_program = pruned_program.inference_optimize()
    fetch_var_names = [v.name for v in target_vars]

    prepend_feed_ops(inference_program, feeded_var_names)
    append_fetch_ops(inference_program, fetch_var_names)

    if model_filename is not None:
        model_filename = os.path.basename(model_filename)
    else:
        model_filename = "__model__"
    model_filename = os.path.join(dirname, model_filename)

    if params_filename is not None:
        params_filename = os.path.basename(params_filename)

    with open(model_filename, "wb") as f:
        f.write(inference_program.desc.serialize_to_string())

    save_persistables(executor, dirname, inference_program, params_filename)


def load_inference_model(dirname,
                         executor,
                         model_filename=None,
                         params_filename=None):
    """
    Load inference model from a directory

    :param dirname: directory path
    :param executor: executor that load inference model
    :param model_filename: The name of file to load inference program.
        If not specified, default filename `__model__` will be used.
    :param params_filename: The name of file to load parameters.
        It is used for the case that all parameters are saved in a single binary file.
        If not specified, parameters are considered saved in separate files.

    :return: [program, feed_target_names, fetch_targets]
             program: program especially for inference.
             feed_target_names: Names of variables that need to feed data
             fetch_targets: Variables from which we can get inference results.
    """
    if not os.path.isdir(dirname):
        raise ValueError("There is no directory named '%s'", dirname)

    if model_filename is not None:
        model_filename = os.path.basename(model_filename)
    else:
        model_filename = "__model__"
    model_filename = os.path.join(dirname, model_filename)

    if params_filename is not None:
        params_filename = os.path.basename(params_filename)

    with open(model_filename, "rb") as f:
        program_desc_str = f.read()

    program = Program.parse_from_string(program_desc_str)
    load_persistables(executor, dirname, program, params_filename)

    feed_target_names = program.desc.get_feed_target_names()
    fetch_target_names = program.desc.get_fetch_target_names()
    fetch_targets = [
        program.global_block().var(name) for name in fetch_target_names
    ]

    return [program, feed_target_names, fetch_targets]


def get_parameter_value(para, executor):
    """
    Get the LoDTensor for the parameter

    :param executor: executor for retrieving the value
    :param para: the given parameter

    :return: the LoDTensor for the parameter
    """
    assert is_parameter(para)

    get_program = Program()
    block = get_program.global_block()
    new_var = _clone_var_in_block_(block, para)
    return executor.run(get_program, feed={}, fetch_list=[new_var])[0]


def get_parameter_value_by_name(name, executor, program=None):
    """
    Get the LoDTensor for paramter with the given name

    :param executor: executor for retrieving the value
    :param name: the name of the parameter
    :param program: the program where the variable is found
            Default default_main_program().

    :return: the LoDTensor for the variable
    """
    if program is None:
        program = default_main_program()
    var = program.global_block().var(name)
    return get_parameter_value(var, executor)


SUCCESS_MARK_FILENAME = "_SUCCESS"
CHECKPOINT_PREFIX = "checkpoint"
MODEL_DIR = "__model__"
TRAINER_PREFIX = "trainer"
CHECKPOINT_SEPARATOR = "_"


def save_checkpoint(executor,
                    checkpoint_dir,
                    trainer_id,
                    trainer_args=None,
                    main_program=None,
                    max_num_checkpoints=3):
    """
    Save Checkpoint will save persistable LodTensor variables from main_program in checkpoint directory,
    the directory named by serial number from 0 to (n -1), save_checkpoint use LRU strategy
    to keep numbers of checkpoint directory,  the numbers of checkpoint directory are max_num_checkpoints at most,
    The interval between two saved checkpoints must greater than save_interval_secs.

    :param executor executor for save the value
    :param checkpoint_dir the checkpoint directory 
    :param trainer_id currect trainer id, if id is equal to 0, the trainer is chief
    :param main_program   will save all variables in program 
    :param max_num_checkpoints will keep numbers of checkpoint serials not bigger than max_num_checkpoints
    """
    if checkpoint_dir is None:
        raise ValueError("'checkpoint_dir' should not be None")

    if trainer_args:
        assert isinstance(trainer_args, dict)

    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    serial = get_latest_checkpoint_serial(checkpoint_dir) + 1
    cur_dir = _get_serial_dir(checkpoint_dir, serial)

    save_trainer_args(cur_dir, trainer_id, trainer_args)

    if trainer_id == 0:
        save_persist_vars_without_grad(executor, cur_dir, main_program)

    _scroll_delete(checkpoint_dir, max_num_checkpoints)


def load_checkpoint(executor, checkpoint_dir, serial, main_program):
    """
    Load checkpoint from a directory by executor,
    it will find  the most recent saved checkpoint file and load it auto.

    :param executor executor for load the value
    :param checkpoint_dir  the checkpoint directory 
    :param serial the serial folder in checkpoint directory will be load
    :param main_program  will load all variables in program 
    """

    if checkpoint_dir is None:
        raise ValueError("'checkpoint_dir' should not be None")

    if serial is None or serial < 0:
        raise ValueError("'serial' should not be None or <0 ")

    if main_program is None:
        raise ValueError('main_program should not be None.')

    cur_dir = _get_serial_dir(checkpoint_dir, serial)
    load_persist_vars_without_grad(executor, cur_dir, main_program, True)


def clean_checkpoint(checkpoint_dir, delete_dir=False):
    """
    clean the checkpoint dir, when the train exits normally, the trainer will call clean_checkpoint to delete checkpoint directory saved before.
    delete_dir only works when the directory is empty, otherwise, OSError is raised.  

    :param checkpoint_dir
    :param delete_dir
    """

    if checkpoint_dir is None:
        raise ValueError("'checkpoint_dir' should not be None")
    _scroll_delete(checkpoint_dir, max_num_checkpoints=0)

    if delete_dir and not os.listdir(checkpoint_dir):
        os.rmdir(checkpoint_dir)


def load_persist_vars_without_grad(executor,
                                   dirname,
                                   program,
                                   has_model_dir=False):
    """
    load_persist_vars_without_grad will load variables from a directory by an executor,
    the variable named end with "@GRAD" will not be loaded.

    :param executor  executor for load the value
    :param dirname the checkpoint directory 
    :param program   will load all variables in program 
    :param has_model_dir if has_model_dir is True, will load variables from  sub directory named __model__
    """

    if has_model_dir:
        dirname = _get_model_dir(dirname)

    load_vars(
        executor,
        dirname=dirname,
        main_program=program,
        predicate=_is_checkpoint_var,
        filename=None)


def save_persist_vars_without_grad(executor, dirname, program):
    """
    save_persist_vars_without_grad  will save variables to a directory by an executor,
    the variable named end with "@GRAD" will not be saved.

    :param executor  executor for load the value
    :param dirname the checkpoint directory 
    :param program   will load all variables in program
    """
    cur_dir = _get_model_dir(dirname)
    save_vars(
        executor,
        dirname=cur_dir,
        main_program=program,
        vars=None,
        predicate=_is_checkpoint_var,
        filename=None)
    _write_success(cur_dir)


def save_trainer_args(dirname, trainer_id, trainer_args):
    assert isinstance(trainer_args, dict)

    cur_dir = _get_trainer_dir(dirname, trainer_id)

    for name, value in trainer_args.iteritems():
        args_file = os.path.join(cur_dir, name)
        with open(args_file, 'w') as f:
            f.write(str(value))
    _write_success(cur_dir)


def load_trainer_args(checkpoint_dir, serial, trainer_id, trainer_args):
    assert isinstance(trainer_args, list)

    cur_dir = _get_serial_dir(checkpoint_dir, serial)
    cur_dir = _get_trainer_dir(cur_dir, trainer_id)

    ret_values = []

    for arg in trainer_args:
        cur_file = os.path.join(cur_dir, arg)
        with open(cur_file, 'r') as f:
            contents = f.read()
            ret_values.append(contents.strip())
    return ret_values


def _is_checkpoint_var(var):
    """
    the checkpoint will not save or load all the variables.
    var type is FEED_MINIBATCH/FETCH_LIST/RAW or var name ends with @GRAD are discarded.

    :param var
    """
    if var.desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
            var.desc.type() == core.VarDesc.VarType.FETCH_LIST or \
            var.desc.type() == core.VarDesc.VarType.RAW:
        return False
    # @GRAD are named for gradient variables, checkpoint will not save it.
    if "@GRAD" in var.name:
        return False
    # .trainer_ are named for distribute train variables, checkpoint will not save it.
    if ".trainer_" in var.name:
        return False

    # .block is named for distribute train variables, checkpoint will not save it.
    if ".block" in var.name:
        return False

    return var.persistable


def _get_dir_serial(dirname):
    _, serial = dirname.split(CHECKPOINT_SEPARATOR)

    try:
        serial_num = int(serial)
    except ValueError:
        serial_num = -1
    return serial_num


def _get_serial_dir(dirname, serial):
    serial_folder = CHECKPOINT_PREFIX + CHECKPOINT_SEPARATOR + str(serial)
    serial_dir = os.path.join(dirname, serial_folder)

    if not os.path.isdir(serial_dir):
        os.makedirs(serial_dir)

    return serial_dir


def _get_model_dir(dirname):
    model_dir = os.path.join(dirname, MODEL_DIR)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    return model_dir


def _get_trainer_dir(dirname, trainer_id):
    trainer_folder = TRAINER_PREFIX + CHECKPOINT_SEPARATOR + str(trainer_id)
    trainer_dir = os.path.join(dirname, trainer_folder)

    if not os.path.isdir(trainer_dir):
        os.makedirs(trainer_dir)

    return trainer_dir


def _scroll_delete(dirname, max_num_checkpoints=3):
    dirs = os.listdir(dirname)
    serial_map = {}
    for serial in dirs:
        serial_num = _get_dir_serial(serial)
        serial_map[serial_num] = serial

    if len(serial_map.keys()) <= max_num_checkpoints:
        return

    serials = serial_map.keys()
    serials.sort(reverse=True)
    serials = serials[max_num_checkpoints:]
    for serial in serials:
        cur_dir = _get_serial_dir(dirname, serial)
        shutil.rmtree(cur_dir)


def _write_success(dirname):
    """
    write an empty file named "_SUCCESS" in checkpoint dir, indicate this checkpoint is correct.

    :param dirname
    """
    success_file = os.path.join(dirname, SUCCESS_MARK_FILENAME)
    with open(success_file, 'a') as f:
        now = time.ctime()
        f.write(now)


def get_latest_checkpoint_serial(checkpoint_dir):
    """
    get the latest file in checkpoint directory, the _SUCCESS file must exist in the directory

    :param checkpoint_dir
    """
    if not checkpoint_dir:
        return -1

    def has_success(checkpoint_dir, cur_dir):
        """
        is _SUCCESS in this dir
        """

        serial = _get_dir_serial(cur_dir)
        if serial == -1 or not os.path.isdir(
                os.path.join(checkpoint_dir, cur_dir)):
            return -1

        success_path = os.path.join(
            _get_serial_dir(checkpoint_dir, serial), MODEL_DIR,
            SUCCESS_MARK_FILENAME)
        if os.path.isfile(success_path):
            return serial

    if not os.path.isdir(checkpoint_dir):
        return -1

    current_dir = -1
    dirs = os.listdir(checkpoint_dir)
    for cur_dir in dirs:
        success_num = has_success(checkpoint_dir, cur_dir)
        if success_num > current_dir:
            current_dir = success_num
    return current_dir
