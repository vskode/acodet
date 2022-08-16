# Humpback-specific model Python code

This is not part of the multispecies\_whale\_detection package proper but rather
related prior work being made available for reference to users who want to make
fine-grained modifications using the weights from the SavedModel released at

https://tfhub.dev/google/humpback\_whale/1

The best way to learn details is to read the comments in the Python source
files. Very basic usage:

```
import humpback_model

model = humpback_model.Model.load_from_tf_hub()
