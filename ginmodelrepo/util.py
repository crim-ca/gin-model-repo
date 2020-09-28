import thelper
from google_drive_downloader import GoogleDriveDownloader as gdd
import os.path as osp
from osgeo import gdal
import cv2
import sys
from six.moves.urllib.parse import urlparse
from six.moves.urllib.request import urlopen
from io import BytesIO
import six
import ssl
from collections import Counter
from copy import deepcopy
import os
import cv2
import numpy as np

# keys used across methods to find matching configs, must be unique and non-conflicting with other sample keys
IMAGE_DATA_KEY = "data"     # key used to store temporarily the loaded image data
IMAGE_LABEL_KEY = "label"   # key used to store the class label used by the model
IMAGE_MASK_KEY = "mask"   # key used to store the mask used by the model
TEST_DATASET_KEY = "dataset"

DATASET_FILES_KEY = "files"             # list of all files in the dataset batch
DATASET_DATA_KEY = "data"               # dict of data below
DATASET_DATA_TAXO_KEY = "taxonomy"
DATASET_DATA_MAPPING_KEY = "taxonomy_model_map"     # taxonomy ID -> model labels
DATASET_DATA_ORDERING_KEY = "model_class_order"     # model output classes (same indices)
DATASET_DATA_MODEL_MAPPING = "model_output_mapping"    # model output classes (same indices)
DATASET_DATA_PATCH_KEY = "patches"
DATASET_DATA_PATCH_CLASS_KEY = "class"       # class id associated to the patch
DATASET_DATA_PATCH_SPLIT_KEY = "split"       # group train/test of the patch
DATASET_DATA_PATCH_CROPS_KEY = "crops"       # extra data such as coordinates
DATASET_DATA_PATCH_IMAGE_KEY = "image"       # original image path that was used to generate the patch
DATASET_DATA_PATCH_PATH_KEY = "path"         # crop image path of the generated patch
DATASET_DATA_PATCH_MASK_KEY = "mask"         # mask image path of the generated patch
DATASET_DATA_PATCH_MASK_PATH_KEY = 'mask'    # original mask path that was used to generate the patch
DATASET_DATA_PATCH_INDEX_KEY = "index"       # data loader getter index reference
DATASET_DATA_PATCH_FEATURE_KEY = "feature"   # annotation reference id
DATASET_BACKGROUND_ID = 999                  # background class id
DATASET_DATA_PATCH_DONTCARE = 255            # dontcare value in the test set
DATASET_DATA_CHANNELS = "channels"           # channels information
# see bottom for mapping definition
MAPPING_TASK = "task"
MAPPING_LOADER = "loader"
MAPPING_RESULT = "result"
MAPPING_TESTER = "tester"

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

class ImageFolderSegDataset(thelper.data.SegmentationDataset):
    """Image folder dataset specialization interface for segmentation tasks.

    This specialization is used to parse simple image subfolders, and it essentially replaces the very
    basic ``torchvision.datasets.ImageFolder`` interface with similar functionalities. It it used to provide
    a proper task interface as well as path metadata in each loaded packet for metrics/logging output.

    .. seealso::
        | :class:`thelper.data.parsers.ImageDataset`
        | :class:`thelper.data.parsers.SegmentationDataset`
    """

    def __init__(self, root, transforms=None, channels= None, image_key="image", label_key="label", mask_key="mask", mask_path_key="mask_path", path_key="path", idx_key="idx"):
        """Image folder dataset parser constructor."""
        self.root = root
        if self.root is None or not os.path.isdir(self.root):
            raise AssertionError("invalid input data root '%s'" % self.root)
        class_map = {}
        for child in os.listdir(self.root):
            if os.path.isdir(os.path.join(self.root, child)):
                class_map[child] = []
        if not class_map:
            raise AssertionError("could not find any image folders at '%s'" % self.root)
        image_exts = [".jpg", ".jpeg", ".bmp", ".png", ".ppm", ".pgm", ".tif"]
        self.image_key = image_key
        self.path_key = path_key
        self.idx_key = idx_key
        self.label_key = label_key
        self.mask_key = mask_key
        self.mask_path_key = mask_path_key
        self.channels = channels if channels else [1, 2, 3]
        samples = []
        for class_name in class_map:
            class_folder = os.path.join(self.root, class_name)
            for folder, subfolder, files in os.walk(class_folder):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in image_exts:
                        class_map[class_name].append(len(samples))
                        samples.append({
                            self.path_key: os.path.join(folder, file),
                            self.label_key: class_name
                        })
        old_unsorted_class_names = list(class_map.keys())
        class_map = {k: class_map[k] for k in sorted(class_map.keys()) if len(class_map[k]) > 0}
        if old_unsorted_class_names != list(class_map.keys()):
            # new as of v0.4.4; this may only be an issue for old models trained on windows and ported to linux
            # (this is caused by the way os.walk returns folders in an arbitrary order on some platforms)
            logger.warning("class name ordering changed due to folder name sorting; this may impact the "
                           "behavior of previously-trained models as task class indices may be swapped!")
        if not class_map:
            raise AssertionError("could not locate any subdir in '%s' with images to load" % self.root)
        meta_keys = [self.path_key, self.idx_key]
        super(ImageFolderSegDataset, self).__init__(class_names=list(class_map.keys()), input_key=self.image_key,
                                                 label_key=self.label_key, meta_keys=meta_keys, transforms=transforms)
        self.samples = samples

    def __getitem__(self, idx):
        """Returns the data sample (a dictionary) for a specific (0-based) index."""
        if isinstance(idx, slice):
            return self._getitems(idx)
        if idx >= len(self.samples):
            raise AssertionError("sample index is out-of-range")
        if idx < 0:
            idx = len(self.samples) + idx
        sample = self.samples[idx]
        image_path = sample[self.path_key]
        rasterfile = gdal.Open(image_path, gdal.GA_ReadOnly)
        # image = cv2.imread(image_path)
        image = []
        for raster_band_idx in self.channels:
            curr_band = rasterfile.GetRasterBand(raster_band_idx)  # offset, starts at 1
            band_array = curr_band.ReadAsArray()
            band_nodataval = curr_band.GetNoDataValue()
            # band_ma = np.ma.array(band_array.astype(np.float32))
            image.append(band_array)
        image = np.dstack(image)
        rasterfile = None  # close input fd
        mask_path = sample[self.mask_path_key] if hasattr(self, 'mask_path_key') else None
        mask = None
        def convert(img, target_type_min, target_type_max, target_type):
            imin = img.min()
            imax = img.max()

            a = (target_type_max - target_type_min) / (imax - imin)
            b = target_type_max - a * imax
            new_img = (a * img + b).astype(target_type)
            return new_img
        if mask_path is not None:
            mask = cv2.imread(mask_path)
            not_zero=np.count_nonzero(mask)
            #assert not_zero > 0
            mask = mask if mask.ndim == 2 else mask[:, :, 0] # masks saved with PIL have three bands
            mask = (mask > 0) *255
            if False:
                import matplotlib
                #matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 2, figsize=(10, 20))
                plt.tight_layout()
                axes[0].imshow(convert(image,0,255, np.uint8))
                axes[0].get_xaxis().set_visible(False)
                axes[0].get_yaxis().set_visible(False)
                axes[1].imshow(mask)
                axes[1].get_xaxis().set_visible(False)
                axes[1].get_yaxis().set_visible(False)
                plt.show()
                plt.savefig('/home/sfoucher/DEV/eval.png')
                plt.show()
                plt.close()
            
        if image is None:
            raise AssertionError("invalid image at '%s'" % image_path)
        sample = {
            self.image_key: np.array(image.data, copy=True, dtype='float32'),
            self.mask_key: mask,
            self.label_key: sample[self.label_key],
            self.idx_key: idx,
            # **sample
        }
        # FIXME: not clear how to handle transformations on the image as well as on the mask
        #  in particular for geometric transformations
        if self.transforms:
            sample = self.transforms(sample)
        return sample

class BatchTestPatchesBaseSegDatasetLoader(ImageFolderSegDataset):
    """
    Batch dataset parser that loads only patches from 'test' split and matching
    class IDs (or their parents) known by the model as defined in its ``task``.

    .. note::

        Uses :class:`thelper.data.SegmentationDataset` ``__getitem__`` implementation to load image
        from a folder, but overrides the ``__init__`` to adapt the configuration to batch format.
    """

    # noinspection PyMissingConstructor
    def __init__(self, dataset=None, transforms=None):
        if not (isinstance(dataset, dict) and len(dataset)):
            raise ValueError("Expected dataset parameters as configuration input.")
        thelper.data.Dataset.__init__(self, transforms=transforms, deepcopy=False)
        self.root = dataset["path"]
        # keys matching dataset config for easy loading and referencing to same fields
        self.image_key = IMAGE_DATA_KEY     # key employed by loader to extract image data (pixel values)
        self.label_key = IMAGE_LABEL_KEY    # class id from API mapped to match model task
        self.path_key = DATASET_DATA_PATCH_PATH_KEY  # actual file path of the patch
        self.idx_key = DATASET_DATA_PATCH_INDEX_KEY  # increment for __getitem__
        self.mask_key = DATASET_DATA_PATCH_MASK_KEY  # actual mask path of the patch
        self.mask_path_key = DATASET_DATA_PATCH_MASK_PATH_KEY  # actual mask path of the patch
        self.meta_keys = [self.path_key, self.idx_key, self.mask_key, DATASET_DATA_PATCH_CROPS_KEY,
                          DATASET_DATA_PATCH_IMAGE_KEY, DATASET_DATA_PATCH_FEATURE_KEY]
        model_class_map = dataset[DATASET_DATA_KEY][DATASET_DATA_MODEL_MAPPING]
        sample_class_ids = set()
        samples = []
        channels = dataset.get(DATASET_DATA_CHANNELS, None)  # FIXME: the user needs to specified the channels used by the model
        self.channels = channels if channels else [1, 2, 3]  # by default we take the first 3 channels
        for patch_path, patch_info in zip(dataset[DATASET_FILES_KEY],
                                          dataset[DATASET_DATA_KEY][DATASET_DATA_PATCH_KEY]):
            if patch_info[DATASET_DATA_PATCH_SPLIT_KEY] == "test":
                # convert the dataset class ID into the model class ID using mapping, drop sample if not found
                class_name = model_class_map.get(patch_info[DATASET_DATA_PATCH_CLASS_KEY])
                if class_name is not None:
                    sample_class_ids.add(class_name)
                    samples.append(deepcopy(patch_info))
                    samples[-1][self.path_key] = os.path.join(self.root, patch_path)
                    samples[-1][self.label_key] = class_name
                    mask_name = patch_info.get(DATASET_DATA_PATCH_CROPS_KEY)[0].get(DATASET_DATA_PATCH_MASK_PATH_KEY, None)
                    if mask_name is not None:
                        samples[-1][self.mask_path_key] = os.path.join(self.root, mask_name)
        if not len(sample_class_ids):
            raise ValueError("No patch/class could be retrieved from batch loading for specific model task.")
        self.samples = samples
        self.sample_class_ids = sample_class_ids




def get_dataset_classes(dataset):
    # type: (Dataset) -> Tuple[Dict, Dict, List]
    """
    Generates the list of classes with files.

    :param model_task: original task defined by the model training which specifies known classes.
    :return: class_mapping, class ids, class names
    """
    
    samples_all = dataset[DATASET_DATA_KEY][DATASET_DATA_PATCH_KEY]  # type: JSON
    all_classes_with_files = Counter([s["class"] for s in samples_all])
    all_child_classes = set()           # only taxonomy child classes IDs
    all_classes_mapping = dict()        # child->parent taxonomy class ID mapping
    all_child_names = dict()
    def find_class_mapping(taxonomy_class, parent=None):
        """Finds existing mappings defined by taxonomy."""
        children = taxonomy_class.get("children")
        class_id = taxonomy_class.get("id")
        name = taxonomy_class.get("name_en")
        if children:
            for child in children:
                find_class_mapping(child, taxonomy_class)
        elif class_id in all_classes_with_files:
            all_child_classes.add(class_id)
            all_child_names[class_id] = name
        all_classes_mapping[class_id] = None if not parent else parent.get("id")
    for taxo in dataset[DATASET_DATA_KEY][DATASET_DATA_TAXO_KEY]:
        find_class_mapping(taxo)
    # print("Taxonomy class mapping:  {}".format(all_classes_mapping))
    all_classes_names = [all_child_names[c] for c in all_classes_with_files]

    print("Dataset class id and occurences: {}".format(all_classes_with_files))
    print("Dataset class names: {}".format(all_classes_names))
    # print("Dataset class parents: {}".format(all_class_parents))
    class_mapping = dict(zip(all_classes_names, all_classes_with_files.keys()))
    return class_mapping, all_classes_with_files, all_classes_names

def adapt_dataset_for_model_task(model_task, dataset):
    # type: (AnyTask, Dataset) -> JSON
    """
    Generates dataset parameter definition for loading from checkpoint configuration with ``thelper``.

    Retrieves available classes from the loaded dataset parameters and preserves only matching classes with the task
    defined by the original model task. Furthermore, parent/child class IDs are resolved recursively in a bottom-top
    manner to adapt specific classes into corresponding `categories` in the case the model uses them as more generic
    classes.

    .. seealso::
        - :class:`BatchTestPatchesBaseDatasetLoader` for dataset parameters used for loading filtered patches.

    :param model_task: original task defined by the model training which specifies known classes.
    :param dataset: batch of patches from which to extract matching classes known to the model.
    :return: configuration that allows ``thelper`` to generate a data loader for testing the model on desired patches.
    """
    try:
        dataset_params = dataset #.json()     # json required because thelper dumps config during logs
        all_classes_mapping = dict()        # child->parent taxonomy class ID mapping
        all_model_ordering = list()         # class ID order as defined by the model
        all_model_mapping = dict()          # taxonomy->model class ID mapping
        all_child_classes = set()           # only taxonomy child classes IDs
        all_test_patch_files = list()       # list of the test patch files

        def find_class_mapping(taxonomy_class, parent=None):
            """Finds existing mappings defined by taxonomy."""
            children = taxonomy_class.get("children")
            class_id = taxonomy_class.get("id")
            if children:
                for child in children:
                    find_class_mapping(child, taxonomy_class)
            else:
                all_child_classes.add(class_id)
            all_classes_mapping[class_id] = None if not parent else parent.get("id")

        # Some models will use a generic background class so we add it systematically in case the model needs it
        for taxo in dataset_params[DATASET_DATA_KEY][DATASET_DATA_TAXO_KEY]:
            taxo.get("children").insert(0, {"id": DATASET_BACKGROUND_ID,
                                            "name_fr": "Classe autre",
                                            "taxonomy_id": taxo.get("taxonomy_id"),
                                            "code": "BACK",
                                            "name_en": "Background",
                                            "children": []})

        for taxo in dataset_params[DATASET_DATA_KEY][DATASET_DATA_TAXO_KEY]:
            find_class_mapping(taxo)
        print("Taxonomy class mapping:  {}".format(all_classes_mapping))
        print("Taxonomy class children: {}".format(all_child_classes))

        # find model mapping using taxonomy hierarchy
        def get_children_class_ids(parent_id):
            children_ids = set()
            filtered_ids = set([c for c, p in all_classes_mapping.items() if p == parent_id])
            for c in filtered_ids:
                if c not in all_child_classes:
                    children_ids = children_ids | get_children_class_ids(c)
            return children_ids | filtered_ids

        for model_class_id in model_task.class_names:
            # attempt str->int conversion of model string, they should match taxonomy class IDs
            try:
                model_class_id = int(model_class_id)
            except ValueError:
                raise ValueError("Unknown class ID '{}' cannot be matched with taxonomy classes".format(model_class_id))
            if model_class_id not in all_classes_mapping:
                raise ValueError("Unknown class ID '{}' cannot be found in taxonomy".format(model_class_id))
            # while looking for parent/child mapping, also convert IDs as thelper requires string labels
            if model_class_id in all_child_classes:
                print("Class {0}: found direct child ID ({0}->{0})".format(model_class_id))
                all_model_mapping[model_class_id] = str(model_class_id)
            else:
                categorized_classes = get_children_class_ids(model_class_id)
                for cat_id in categorized_classes:
                    all_model_mapping[cat_id] = str(model_class_id)
                print("Class {0}: found category class IDs ({0}->AnyOf{1})"
                             .format(model_class_id, list(categorized_classes)))
            all_model_ordering.append(model_class_id)
        all_model_mapping = {c: all_model_mapping[c] for c in sorted(all_model_mapping)}
        print("Model class mapping (only supported classes): {}".format(all_model_mapping))
        print("Model class ordering (indexed class outputs): {}".format(all_model_ordering))

        # add missing classes mapping
        all_model_mapping.update({c: None for c in sorted(set(all_classes_mapping) - set(all_model_mapping))})
        print("Model class mapping (added missing classes): {}".format(all_model_mapping))

        # update obtained mapping with dataset parameters for loader
        dataset_params[DATASET_DATA_KEY][DATASET_DATA_MAPPING_KEY] = all_model_mapping
        dataset_params[DATASET_DATA_KEY][DATASET_DATA_ORDERING_KEY] = all_model_ordering
        dataset_params[DATASET_DATA_KEY][DATASET_DATA_MODEL_MAPPING] = model_task.class_indices
        dataset_params[DATASET_FILES_KEY] = all_test_patch_files

        # update patch info for classes of interest
        # this is necessary for BatchTestPatchesClassificationDatasetLoader
        class_mapped = [c for c, m in all_model_mapping.items() if m is not None]
        samples_all = dataset_params[DATASET_DATA_KEY][DATASET_DATA_PATCH_KEY]  # type: JSON
        all_classes_with_files = sorted(set([s["class"] for s in samples_all]))
        all_model_classes = set(class_mapped + all_model_ordering)
        samples_mapped = [s for s in samples_all if s[DATASET_DATA_PATCH_CLASS_KEY] in all_model_classes]
        # retain class Ids with test patches
        classes_with_files = sorted(set([s["class"] for s in samples_mapped if s["split"] == "test"]))
        if len(classes_with_files) == 0:
            raise ValueError("No test patches for the classes of interest!")
        all_test_patch_files = [s[DATASET_DATA_PATCH_CROPS_KEY][0]["path"] for s in samples_mapped]
        dataset_params[DATASET_FILES_KEY] = all_test_patch_files

        # test_samples = [
        #    {"class_id": s[DATASET_DATA_PATCH_CLASS_KEY],
        #     "sample_id": s[DATASET_DATA_PATCH_FEATURE_KEY]} for s in samples_all
        # ]

        model_task_name = fully_qualified_name(model_task)
        return {
            "type": MODEL_TASK_MAPPING[model_task_name][MAPPING_LOADER],
            "params": {TEST_DATASET_KEY: dataset_params},
            "task": model_task
        }
    except Exception as exc:
        raise RuntimeError("Failed dataset adaptation to model task classes for evaluation. [{!r}]".format(exc))


# FIXME: add definitions/implementations to support other task types (ex: object-detection)
MODEL_TASK_MAPPING = {
    fully_qualified_name(thelper.tasks.classif.Classification): {
        MAPPING_TASK:   fully_qualified_name(thelper.tasks.classif.Classification),
        MAPPING_LOADER: fully_qualified_name(BatchTestPatchesBaseSegDatasetLoader),
        MAPPING_TESTER: fully_qualified_name(thelper.train.classif.ImageClassifTrainer),
    },
    fully_qualified_name(thelper.tasks.segm.Segmentation): {
        MAPPING_TASK:   fully_qualified_name(thelper.tasks.segm.Segmentation),
        MAPPING_LOADER: fully_qualified_name(BatchTestPatchesBaseSegDatasetLoader),
        MAPPING_TESTER: fully_qualified_name(thelper.train.segm.ImageSegmTrainer),
    },
}