import os

import onnxruntime

print("ort", onnxruntime.__version__)
capi = os.path.join(os.path.dirname(onnxruntime.__file__), "capi")
libs = sorted(f for f in os.listdir(capi) if f.startswith("libonnxruntime"))
print("capi:", capi)
print("libs:", libs)