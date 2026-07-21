import inspect

from fastapi.params import Form
from pydantic_core import PydanticUndefined

from key_matching.production_api import register_key


def test_registration_metadata_fields_are_required_forms():
    signature = inspect.signature(register_key)
    names = (
        "key_type",
        "manufacturer",
        "lock_brand",
        "key_code",
        "key_bitting",
        "num_pins",
        "brand",
        "code",
        "x",
    )

    for name in names:
        field = signature.parameters[name].default
        assert isinstance(field, Form)
        assert field.default is PydanticUndefined
