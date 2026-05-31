from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
)


_dummy_objects = {}
_import_structure = {}

try:
    if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.27.0")):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # This points to dummy objects so the library doesn't crash if transformers isn't installed
    from ...utils.dummy_torch_and_transformers_objects import (
        AudioLDMPipeline as TangoPipeline, # Fallback to a similar dummy
    )

    _dummy_objects.update({"TangoPipeline": TangoPipeline})
else:
    # This tells Diffusers where to find your code
    _import_structure["pipeline_tango"] = ["TangoPipeline"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.27.0")):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import (
            AudioLDMPipeline as TangoPipeline,
        )

    else:
        # Static import for IDEs/Type Checking
        from .pipeline_tango import TangoPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)