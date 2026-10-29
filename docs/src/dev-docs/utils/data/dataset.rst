Dataset
#######

.. automodule:: metatrain.utils.data.dataset
    :members:
    :undoc-members:
    :show-inheritance:

Batch identity
==============

``DiskDataset.__getitem__`` returns source-numbered sample labels and the
``mtt::aux::system_index`` extra field. Collation assigns each record occurrence a
batch-local ``system`` sample label, across targets, extra data, and explicit
``system`` columns in gradient samples. For a batch drawn at indices ``[1, 0, 1]``,
these labels are ``[0, 1, 2]``; atomic rows carry the corresponding occurrence id.
Gradient ``sample`` references are offset by the join to address the concatenated
parent rows.

Auxiliary **values** remain ``[1, 0, 1]``. They identify dataset-local source
indices, which can overlap between datasets and do not uniquely identify batch
occurrences. Transforms that align targets to the systems list must use the
auxiliary block's ``system`` sample column. Direct transform callers can use
source-numbered labels when auxiliary labels and target/extra labels agree.
