
import thelper

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