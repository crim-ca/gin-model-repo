import thelper
from google_drive_downloader import GoogleDriveDownloader as gdd
import os.path as osp
import sys
from six.moves.urllib.parse import urlparse
from six.moves.urllib.request import urlopen
from io import BytesIO
import six
import ssl

def update_class_mapping(class_mappings, model_task):
    # type: (List[Tuple[Union[str,int], Union[str,int]]], str, Optional[str]) -> None
    """Updates the model task using provided class mapping."""
 
    model_task = thelper.tasks.utils.create_task(model_task)
    if len(model_task.class_names) != len(class_mappings):
        raise ValueError(f"Task classes and class mapping size do not match "
                         f"({len(model_task.class_names)} != {len(class_mappings)}) :\n"
                         f"  {model_task.class_names}\n  {class_mappings} ")
    class_mapped = {}
    class_mappings = dict(class_mappings)
    for class_name in model_task.class_names:
        if class_name not in class_mappings:
            raise ValueError(f"Missing mapping for class '{class_name}'.")
        new_class = class_mappings[class_name]
        idx_class = model_task.class_indices[class_name]
        class_mapped[new_class] = idx_class
    setattr(model_task, "_class_indices", class_mapped)
    model_outputs_sorted_by_index = list(sorted(class_mapped.items(), key=lambda _map: _map[1]))
    setattr(model_task, "_class_names", [str(_map[0]) for _map in model_outputs_sorted_by_index])
    return model_task

def validate_model(model_data):
    # type: (CkptData) -> Tuple[bool, Optional[Exception]]
    """
    Accomplishes required model checkpoint validation to restrict unexpected behaviour during other function calls.

    All security checks or alternative behaviours allowed by native :mod:`thelper` library but that should be forbidden
    within this API for process execution should be done here.

    :param model_data: model checkpoint data with configuration parameters (typically loaded by :func:`load_model`)
    :return: tuple of (success, exception) accordingly
    :raises: None (nothrow)
    """
    model_task = model_data.get("task")
    if isinstance(model_task, dict):
        model_type = model_task.get("type")
        if not isinstance(model_type, six.string_types):
            print(f"Model task: [{model_type!s}]")
            print(
                f"Forbidden model checkpoint task defines unknown operation: [{model_type!s}]"
            )
            return False
        model_params = model_task.get("params")
        if not isinstance(model_params, dict):
            print(f"Model task: [{model_params!s}]")
            print(
                "Forbidden model checkpoint task missing JSON definition of parameter section."
            )
            return False
        model_classes = model_params.get("class_names")
        if not (isinstance(model_classes, list) and all([isinstance(c, (int, str)) for c in model_classes])):
            print(f"Model task: [{model_classes!s}]")
            print(
                "Forbidden model checkpoint task contains invalid JSON class names parameter section."
            )
            return False
    elif isinstance(model_task, thelper.tasks.Task):
        model_type = fully_qualified_name(model_task)
        if model_type not in MODEL_TASK_MAPPING:
            print(f"Model task: [{model_type!s}]")
            print(
                f"Forbidden model checkpoint task defines unknown operation: [{model_type!s}]"
            )
            return False
    else:
        # thelper security risk, refuse literal string definition of task loaded by eval() unless it can be validated
        print(f"Model task not defined as dictionary nor `thelper.task.Task` class: [{model_task!s}]")
        if not (isinstance(model_task, str) and model_task.startswith("thelper.task")):
            return False, print(
                "Forbidden model checkpoint task definition as string doesn't refer to a `thelper.task`."
            )
        model_task_cls = model_task.split("(")[0]
        print(f"Verifying model task as string: {model_task_cls!s}")
        model_task_cls = thelper.utils.import_class(model_task_cls)
        if not (isclass(model_task_cls) and issubclass(model_task_cls, thelper.tasks.Task)):
            print(
                "Forbidden model checkpoint task definition as string is not a known `thelper.task`."
            )
            return False
        if model_task.count("(") != 1 or model_task.count(")") != 1:
            print(
                "Forbidden model checkpoint task definition as string has unexpected syntax."
            )
            return False
        print("Model task defined as string allowed after basic validation.")
        try:
            fix_str_model_task(model_task)  # attempt update but don't actually apply it
        except ValueError:
            print(
                "Forbidden model checkpoint task defined as string doesn't respect expected syntax."
            )
            return False
        print("Model task as string validated with successful parameter conversion")
    return True, None

def fully_qualified_name(obj):
    # type: (Union[Any, Type[Any]]) -> AnyStr
    """Obtains the ``'<module>.<name>'`` full path definition of the object to allow finding and importing it."""
    cls = obj if isclass(obj) else type(obj)
    return '.'.join([obj.__module__, cls.__name__])

def isclass(obj):
    # type: (Any) -> bool
    """Evaluates ``obj`` for ``class`` type (ie: class definition, not an instance nor any other type)."""
    return isinstance(obj, six.class_types)

def fix_str_model_task(model_task):
    # type: (str) -> ParamsType
    """
    Attempts to convert the input model task definition as literal string to the equivalent dictionary of task
    input parameters.

    For example, a model with classification task is expected to have the following format::

        "thelper.tasks.classif.Classification(class_names=['cls1', 'cls2'], input_key='0', label_key='1', meta_keys=[])"

    And will be converted to::

        {'class_names': ['cls1', 'cls2'], 'input_key': '0', 'label_key': '1', 'meta_keys': []}

    :return: dictionary of task input parameters converted from the literal string definition
    :raises ValueError: if the literal string cannot be parsed as a task input parameters definition
    """
    try:
        if not isinstance(model_task, str):
            raise ValueError(f"Invalid input is not a literal string for model task parsing, got '{type(model_task)}'")
        params = model_task.split("(", 1)[-1].split(")", 1)[0]
        params = re.sub(r"(\w+)\s*=", r"'\1': ", params)
        return ast.literal_eval(f"{{{params}}}")
    except ValueError:
        raise   # failing ast converting raises ValueError
    except Exception as exc:
        raise ValueError(f"Failed literal string parsing for model task, exception: [{exc!s}]")

def load_model(model_file):
    # type: (Union[Any, AnyStr]) -> Tuple[bool, CkptData, Optional[BytesIO], Optional[Exception]]
    """
    Tries to load a model checkpoint file from the file-like object, file path or URL.

    :return: tuple of (success, data, buffer, exception) accordingly.
    :raises: None (nothrow)
    """
    try:
        model_buffer = model_file
        if isinstance(model_file, six.string_types):
            if urlparse(model_file).scheme:
                no_ssl = ssl.create_default_context()
                no_ssl.check_hostname = False
                no_ssl.verify_mode = ssl.CERT_NONE
                url_buffer = urlopen(model_file, context=no_ssl)
                model_buffer = BytesIO(url_buffer.read())
            else:
                with open(model_file, 'rb') as f:
                    model_buffer = BytesIO(f.read())
        thelper.utils.bypass_queries = True     # avoid blocking ui query
        model_checkpoint_info = thelper.utils.load_checkpoint(model_buffer)
    except Exception as ex:
        return False, {}, None, ex
    if model_checkpoint_info:
        return True, model_checkpoint_info, model_buffer, None
    return False, {}, None, None


def maybe_download_and_extract(file_id, dest_path ):
    filename = dest_path.split('/')[-1]
    file_path = dest_path
    download_dir= osp.dirname(osp.abspath(dest_path))
    if not osp.isfile(dest_path):
      gdd.download_file_from_google_drive(file_id= file_id, dest_path= file_path)
      print("Download finished. Extracting files.")

      if file_path.endswith(".zip"):
          # Unpack the zip-file.
          zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
      elif file_path.endswith((".tar.gz", ".tgz")):
          # Unpack the tar-ball.
          tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)
      print("Done.")
    else:
        print("Data has apparently already been downloaded and unpacked.")

# see bottom for mapping definition
MAPPING_TASK = "task"


MODEL_TASK_MAPPING = {
    fully_qualified_name(thelper.tasks.classif.Classification): {
        MAPPING_TASK:   fully_qualified_name(thelper.tasks.classif.Classification),
    },
    fully_qualified_name(thelper.tasks.segm.Segmentation): {
        MAPPING_TASK:   fully_qualified_name(thelper.tasks.segm.Segmentation),
    },
}