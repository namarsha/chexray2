```python
## Required files:

## an Input file containing China and Montgomery Xrays, diagnoses, and masks (provided by Dr. A. through Seton Hall University Google Drive)

## The CheXpert-V1.0-small.zip file, provided by the Stanford ML group.
```

## If on AWS, set this variable to True

## If not running on AWS, ensure that input file is in same directory as this notebook. If you don't have the input file, you will need to download it.

## Note that the full Stanford ML file must also be downloaded separately. the dataset (CheXpert-V1.0-small.zip) from the Stanford ML group, and unzip it. A script for unzipping the files is provided below. 



```python

RUNNING_ON_AWS = True
```

# Unzipping very large files: https://stackoverflow.com/questions/339053/how-do-you-unzip-very-large-files-in-python


```python


# import errno
# import os
# import shutil
# import zipfile

# TARGETDIR = os.path.join('.')

# src = os.path.join('.', 'CheXpert-V1.0-small.zip')

# with open(src, "rb") as zipsrc:
#     zfile = zipfile.ZipFile(zipsrc)
#     for member in zfile.infolist():
#         print("Now serving member: {}".format(member))
#         target_path = os.path.join(TARGETDIR, member.filename)
#         if target_path.endswith('/'):  # folder entry, create
#             try:
#                 os.makedirs(target_path)
#             except (OSError, IOError) as err:
#                # Windows may complain if the folders already exist
#                 if err.errno != errno.EEXIST:
#                     raise
#             continue
#         with open(target_path, 'wb') as outfile, zfile.open(member) as infile:
#            shutil.copyfileobj(infile, outfile)
```


```python

import torch
import torchvision
import os
import matplotlib.pyplot as plt
import glob
import pandas as pd

```

## TODO: Add Montgomery file S3 bucket info here


```python
import s3fs
fs = s3fs.S3FileSystem()
china_Files = fs.ls('s3://chexrayproject-allchinafiles')
print(china_Files)
```

    ['chexrayproject-allchinafiles/CHNCXR_0001_0.png', 'chexrayproject-allchinafiles/CHNCXR_0001_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0002_0.png', 'chexrayproject-allchinafiles/CHNCXR_0002_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0003_0.png', 'chexrayproject-allchinafiles/CHNCXR_0003_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0004_0.png', 'chexrayproject-allchinafiles/CHNCXR_0004_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0005_0.png', 'chexrayproject-allchinafiles/CHNCXR_0005_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0006_0.png', 'chexrayproject-allchinafiles/CHNCXR_0006_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0007_0.png', 'chexrayproject-allchinafiles/CHNCXR_0007_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0008_0.png', 'chexrayproject-allchinafiles/CHNCXR_0008_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0009_0.png', 'chexrayproject-allchinafiles/CHNCXR_0009_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0010_0.png', 'chexrayproject-allchinafiles/CHNCXR_0010_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0011_0.png', 'chexrayproject-allchinafiles/CHNCXR_0011_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0012_0.png', 'chexrayproject-allchinafiles/CHNCXR_0012_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0013_0.png', 'chexrayproject-allchinafiles/CHNCXR_0013_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0014_0.png', 'chexrayproject-allchinafiles/CHNCXR_0014_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0015_0.png', 'chexrayproject-allchinafiles/CHNCXR_0015_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0016_0.png', 'chexrayproject-allchinafiles/CHNCXR_0016_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0017_0.png', 'chexrayproject-allchinafiles/CHNCXR_0017_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0018_0.png', 'chexrayproject-allchinafiles/CHNCXR_0018_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0019_0.png', 'chexrayproject-allchinafiles/CHNCXR_0019_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0020_0.png', 'chexrayproject-allchinafiles/CHNCXR_0020_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0021_0.png', 'chexrayproject-allchinafiles/CHNCXR_0021_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0022_0.png', 'chexrayproject-allchinafiles/CHNCXR_0022_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0023_0.png', 'chexrayproject-allchinafiles/CHNCXR_0023_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0024_0.png', 'chexrayproject-allchinafiles/CHNCXR_0024_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0025_0.png', 'chexrayproject-allchinafiles/CHNCXR_0025_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0026_0.png', 'chexrayproject-allchinafiles/CHNCXR_0026_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0027_0.png', 'chexrayproject-allchinafiles/CHNCXR_0027_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0028_0.png', 'chexrayproject-allchinafiles/CHNCXR_0028_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0029_0.png', 'chexrayproject-allchinafiles/CHNCXR_0029_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0030_0.png', 'chexrayproject-allchinafiles/CHNCXR_0030_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0031_0.png', 'chexrayproject-allchinafiles/CHNCXR_0031_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0032_0.png', 'chexrayproject-allchinafiles/CHNCXR_0032_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0033_0.png', 'chexrayproject-allchinafiles/CHNCXR_0033_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0034_0.png', 'chexrayproject-allchinafiles/CHNCXR_0034_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0035_0.png', 'chexrayproject-allchinafiles/CHNCXR_0035_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0036_0.png', 'chexrayproject-allchinafiles/CHNCXR_0036_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0037_0.png', 'chexrayproject-allchinafiles/CHNCXR_0037_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0038_0.png', 'chexrayproject-allchinafiles/CHNCXR_0038_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0039_0.png', 'chexrayproject-allchinafiles/CHNCXR_0039_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0040_0.png', 'chexrayproject-allchinafiles/CHNCXR_0040_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0041_0.png', 'chexrayproject-allchinafiles/CHNCXR_0041_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0042_0.png', 'chexrayproject-allchinafiles/CHNCXR_0042_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0043_0.png', 'chexrayproject-allchinafiles/CHNCXR_0043_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0044_0.png', 'chexrayproject-allchinafiles/CHNCXR_0044_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0045_0.png', 'chexrayproject-allchinafiles/CHNCXR_0045_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0046_0.png', 'chexrayproject-allchinafiles/CHNCXR_0046_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0047_0.png', 'chexrayproject-allchinafiles/CHNCXR_0047_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0048_0.png', 'chexrayproject-allchinafiles/CHNCXR_0048_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0049_0.png', 'chexrayproject-allchinafiles/CHNCXR_0049_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0050_0.png', 'chexrayproject-allchinafiles/CHNCXR_0050_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0051_0.png', 'chexrayproject-allchinafiles/CHNCXR_0051_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0052_0.png', 'chexrayproject-allchinafiles/CHNCXR_0052_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0053_0.png', 'chexrayproject-allchinafiles/CHNCXR_0053_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0054_0.png', 'chexrayproject-allchinafiles/CHNCXR_0054_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0055_0.png', 'chexrayproject-allchinafiles/CHNCXR_0055_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0056_0.png', 'chexrayproject-allchinafiles/CHNCXR_0056_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0057_0.png', 'chexrayproject-allchinafiles/CHNCXR_0057_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0058_0.png', 'chexrayproject-allchinafiles/CHNCXR_0058_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0059_0.png', 'chexrayproject-allchinafiles/CHNCXR_0059_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0060_0.png', 'chexrayproject-allchinafiles/CHNCXR_0060_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0061_0.png', 'chexrayproject-allchinafiles/CHNCXR_0061_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0062_0.png', 'chexrayproject-allchinafiles/CHNCXR_0062_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0063_0.png', 'chexrayproject-allchinafiles/CHNCXR_0063_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0064_0.png', 'chexrayproject-allchinafiles/CHNCXR_0064_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0065_0.png', 'chexrayproject-allchinafiles/CHNCXR_0065_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0066_0.png', 'chexrayproject-allchinafiles/CHNCXR_0066_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0067_0.png', 'chexrayproject-allchinafiles/CHNCXR_0067_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0068_0.png', 'chexrayproject-allchinafiles/CHNCXR_0068_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0069_0.png', 'chexrayproject-allchinafiles/CHNCXR_0069_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0070_0.png', 'chexrayproject-allchinafiles/CHNCXR_0070_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0071_0.png', 'chexrayproject-allchinafiles/CHNCXR_0071_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0072_0.png', 'chexrayproject-allchinafiles/CHNCXR_0072_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0073_0.png', 'chexrayproject-allchinafiles/CHNCXR_0073_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0074_0.png', 'chexrayproject-allchinafiles/CHNCXR_0074_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0075_0.png', 'chexrayproject-allchinafiles/CHNCXR_0075_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0076_0.png', 'chexrayproject-allchinafiles/CHNCXR_0076_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0077_0.png', 'chexrayproject-allchinafiles/CHNCXR_0077_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0078_0.png', 'chexrayproject-allchinafiles/CHNCXR_0078_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0079_0.png', 'chexrayproject-allchinafiles/CHNCXR_0079_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0080_0.png', 'chexrayproject-allchinafiles/CHNCXR_0080_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0081_0.png', 'chexrayproject-allchinafiles/CHNCXR_0081_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0082_0.png', 'chexrayproject-allchinafiles/CHNCXR_0082_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0083_0.png', 'chexrayproject-allchinafiles/CHNCXR_0083_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0084_0.png', 'chexrayproject-allchinafiles/CHNCXR_0084_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0085_0.png', 'chexrayproject-allchinafiles/CHNCXR_0085_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0086_0.png', 'chexrayproject-allchinafiles/CHNCXR_0086_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0087_0.png', 'chexrayproject-allchinafiles/CHNCXR_0087_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0088_0.png', 'chexrayproject-allchinafiles/CHNCXR_0088_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0089_0.png', 'chexrayproject-allchinafiles/CHNCXR_0089_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0090_0.png', 'chexrayproject-allchinafiles/CHNCXR_0090_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0091_0.png', 'chexrayproject-allchinafiles/CHNCXR_0091_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0092_0.png', 'chexrayproject-allchinafiles/CHNCXR_0092_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0093_0.png', 'chexrayproject-allchinafiles/CHNCXR_0093_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0094_0.png', 'chexrayproject-allchinafiles/CHNCXR_0094_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0095_0.png', 'chexrayproject-allchinafiles/CHNCXR_0095_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0096_0.png', 'chexrayproject-allchinafiles/CHNCXR_0096_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0097_0.png', 'chexrayproject-allchinafiles/CHNCXR_0097_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0098_0.png', 'chexrayproject-allchinafiles/CHNCXR_0098_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0099_0.png', 'chexrayproject-allchinafiles/CHNCXR_0099_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0100_0.png', 'chexrayproject-allchinafiles/CHNCXR_0100_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0101_0.png', 'chexrayproject-allchinafiles/CHNCXR_0101_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0102_0.png', 'chexrayproject-allchinafiles/CHNCXR_0102_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0103_0.png', 'chexrayproject-allchinafiles/CHNCXR_0103_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0104_0.png', 'chexrayproject-allchinafiles/CHNCXR_0104_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0105_0.png', 'chexrayproject-allchinafiles/CHNCXR_0105_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0106_0.png', 'chexrayproject-allchinafiles/CHNCXR_0106_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0107_0.png', 'chexrayproject-allchinafiles/CHNCXR_0107_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0108_0.png', 'chexrayproject-allchinafiles/CHNCXR_0108_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0109_0.png', 'chexrayproject-allchinafiles/CHNCXR_0109_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0110_0.png', 'chexrayproject-allchinafiles/CHNCXR_0110_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0111_0.png', 'chexrayproject-allchinafiles/CHNCXR_0111_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0112_0.png', 'chexrayproject-allchinafiles/CHNCXR_0112_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0113_0.png', 'chexrayproject-allchinafiles/CHNCXR_0113_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0114_0.png', 'chexrayproject-allchinafiles/CHNCXR_0114_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0115_0.png', 'chexrayproject-allchinafiles/CHNCXR_0115_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0116_0.png', 'chexrayproject-allchinafiles/CHNCXR_0116_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0117_0.png', 'chexrayproject-allchinafiles/CHNCXR_0117_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0118_0.png', 'chexrayproject-allchinafiles/CHNCXR_0118_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0119_0.png', 'chexrayproject-allchinafiles/CHNCXR_0119_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0120_0.png', 'chexrayproject-allchinafiles/CHNCXR_0120_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0121_0.png', 'chexrayproject-allchinafiles/CHNCXR_0121_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0122_0.png', 'chexrayproject-allchinafiles/CHNCXR_0122_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0123_0.png', 'chexrayproject-allchinafiles/CHNCXR_0123_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0124_0.png', 'chexrayproject-allchinafiles/CHNCXR_0124_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0125_0.png', 'chexrayproject-allchinafiles/CHNCXR_0125_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0126_0.png', 'chexrayproject-allchinafiles/CHNCXR_0126_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0127_0.png', 'chexrayproject-allchinafiles/CHNCXR_0127_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0128_0.png', 'chexrayproject-allchinafiles/CHNCXR_0128_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0129_0.png', 'chexrayproject-allchinafiles/CHNCXR_0129_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0130_0.png', 'chexrayproject-allchinafiles/CHNCXR_0130_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0131_0.png', 'chexrayproject-allchinafiles/CHNCXR_0131_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0132_0.png', 'chexrayproject-allchinafiles/CHNCXR_0132_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0133_0.png', 'chexrayproject-allchinafiles/CHNCXR_0133_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0134_0.png', 'chexrayproject-allchinafiles/CHNCXR_0134_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0135_0.png', 'chexrayproject-allchinafiles/CHNCXR_0135_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0136_0.png', 'chexrayproject-allchinafiles/CHNCXR_0136_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0137_0.png', 'chexrayproject-allchinafiles/CHNCXR_0137_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0138_0.png', 'chexrayproject-allchinafiles/CHNCXR_0138_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0139_0.png', 'chexrayproject-allchinafiles/CHNCXR_0139_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0140_0.png', 'chexrayproject-allchinafiles/CHNCXR_0140_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0141_0.png', 'chexrayproject-allchinafiles/CHNCXR_0141_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0142_0.png', 'chexrayproject-allchinafiles/CHNCXR_0142_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0143_0.png', 'chexrayproject-allchinafiles/CHNCXR_0143_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0144_0.png', 'chexrayproject-allchinafiles/CHNCXR_0144_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0145_0.png', 'chexrayproject-allchinafiles/CHNCXR_0145_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0146_0.png', 'chexrayproject-allchinafiles/CHNCXR_0146_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0147_0.png', 'chexrayproject-allchinafiles/CHNCXR_0147_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0148_0.png', 'chexrayproject-allchinafiles/CHNCXR_0148_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0149_0.png', 'chexrayproject-allchinafiles/CHNCXR_0149_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0150_0.png', 'chexrayproject-allchinafiles/CHNCXR_0150_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0151_0.png', 'chexrayproject-allchinafiles/CHNCXR_0151_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0152_0.png', 'chexrayproject-allchinafiles/CHNCXR_0152_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0153_0.png', 'chexrayproject-allchinafiles/CHNCXR_0153_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0154_0.png', 'chexrayproject-allchinafiles/CHNCXR_0154_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0155_0.png', 'chexrayproject-allchinafiles/CHNCXR_0155_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0156_0.png', 'chexrayproject-allchinafiles/CHNCXR_0156_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0157_0.png', 'chexrayproject-allchinafiles/CHNCXR_0157_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0158_0.png', 'chexrayproject-allchinafiles/CHNCXR_0158_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0159_0.png', 'chexrayproject-allchinafiles/CHNCXR_0159_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0160_0.png', 'chexrayproject-allchinafiles/CHNCXR_0160_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0161_0.png', 'chexrayproject-allchinafiles/CHNCXR_0161_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0162_0.png', 'chexrayproject-allchinafiles/CHNCXR_0162_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0163_0.png', 'chexrayproject-allchinafiles/CHNCXR_0163_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0164_0.png', 'chexrayproject-allchinafiles/CHNCXR_0164_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0165_0.png', 'chexrayproject-allchinafiles/CHNCXR_0165_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0166_0.png', 'chexrayproject-allchinafiles/CHNCXR_0166_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0167_0.png', 'chexrayproject-allchinafiles/CHNCXR_0167_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0168_0.png', 'chexrayproject-allchinafiles/CHNCXR_0168_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0169_0.png', 'chexrayproject-allchinafiles/CHNCXR_0169_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0170_0.png', 'chexrayproject-allchinafiles/CHNCXR_0170_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0171_0.png', 'chexrayproject-allchinafiles/CHNCXR_0171_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0172_0.png', 'chexrayproject-allchinafiles/CHNCXR_0172_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0173_0.png', 'chexrayproject-allchinafiles/CHNCXR_0173_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0174_0.png', 'chexrayproject-allchinafiles/CHNCXR_0174_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0175_0.png', 'chexrayproject-allchinafiles/CHNCXR_0175_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0176_0.png', 'chexrayproject-allchinafiles/CHNCXR_0176_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0177_0.png', 'chexrayproject-allchinafiles/CHNCXR_0177_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0178_0.png', 'chexrayproject-allchinafiles/CHNCXR_0178_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0179_0.png', 'chexrayproject-allchinafiles/CHNCXR_0179_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0180_0.png', 'chexrayproject-allchinafiles/CHNCXR_0180_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0181_0.png', 'chexrayproject-allchinafiles/CHNCXR_0181_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0182_0.png', 'chexrayproject-allchinafiles/CHNCXR_0182_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0183_0.png', 'chexrayproject-allchinafiles/CHNCXR_0183_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0184_0.png', 'chexrayproject-allchinafiles/CHNCXR_0184_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0185_0.png', 'chexrayproject-allchinafiles/CHNCXR_0185_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0186_0.png', 'chexrayproject-allchinafiles/CHNCXR_0186_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0187_0.png', 'chexrayproject-allchinafiles/CHNCXR_0187_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0188_0.png', 'chexrayproject-allchinafiles/CHNCXR_0188_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0189_0.png', 'chexrayproject-allchinafiles/CHNCXR_0189_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0190_0.png', 'chexrayproject-allchinafiles/CHNCXR_0190_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0191_0.png', 'chexrayproject-allchinafiles/CHNCXR_0191_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0192_0.png', 'chexrayproject-allchinafiles/CHNCXR_0192_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0193_0.png', 'chexrayproject-allchinafiles/CHNCXR_0193_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0194_0.png', 'chexrayproject-allchinafiles/CHNCXR_0194_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0195_0.png', 'chexrayproject-allchinafiles/CHNCXR_0195_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0196_0.png', 'chexrayproject-allchinafiles/CHNCXR_0196_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0197_0.png', 'chexrayproject-allchinafiles/CHNCXR_0197_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0198_0.png', 'chexrayproject-allchinafiles/CHNCXR_0198_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0199_0.png', 'chexrayproject-allchinafiles/CHNCXR_0199_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0200_0.png', 'chexrayproject-allchinafiles/CHNCXR_0200_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0201_0.png', 'chexrayproject-allchinafiles/CHNCXR_0201_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0202_0.png', 'chexrayproject-allchinafiles/CHNCXR_0202_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0203_0.png', 'chexrayproject-allchinafiles/CHNCXR_0203_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0204_0.png', 'chexrayproject-allchinafiles/CHNCXR_0204_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0205_0.png', 'chexrayproject-allchinafiles/CHNCXR_0205_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0206_0.png', 'chexrayproject-allchinafiles/CHNCXR_0206_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0207_0.png', 'chexrayproject-allchinafiles/CHNCXR_0207_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0208_0.png', 'chexrayproject-allchinafiles/CHNCXR_0208_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0209_0.png', 'chexrayproject-allchinafiles/CHNCXR_0209_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0210_0.png', 'chexrayproject-allchinafiles/CHNCXR_0210_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0211_0.png', 'chexrayproject-allchinafiles/CHNCXR_0211_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0212_0.png', 'chexrayproject-allchinafiles/CHNCXR_0212_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0213_0.png', 'chexrayproject-allchinafiles/CHNCXR_0213_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0214_0.png', 'chexrayproject-allchinafiles/CHNCXR_0214_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0215_0.png', 'chexrayproject-allchinafiles/CHNCXR_0215_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0216_0.png', 'chexrayproject-allchinafiles/CHNCXR_0216_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0217_0.png', 'chexrayproject-allchinafiles/CHNCXR_0217_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0218_0.png', 'chexrayproject-allchinafiles/CHNCXR_0218_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0219_0.png', 'chexrayproject-allchinafiles/CHNCXR_0219_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0220_0.png', 'chexrayproject-allchinafiles/CHNCXR_0220_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0221_0.png', 'chexrayproject-allchinafiles/CHNCXR_0221_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0222_0.png', 'chexrayproject-allchinafiles/CHNCXR_0222_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0223_0.png', 'chexrayproject-allchinafiles/CHNCXR_0223_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0224_0.png', 'chexrayproject-allchinafiles/CHNCXR_0224_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0225_0.png', 'chexrayproject-allchinafiles/CHNCXR_0225_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0226_0.png', 'chexrayproject-allchinafiles/CHNCXR_0226_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0227_0.png', 'chexrayproject-allchinafiles/CHNCXR_0227_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0228_0.png', 'chexrayproject-allchinafiles/CHNCXR_0228_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0229_0.png', 'chexrayproject-allchinafiles/CHNCXR_0229_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0230_0.png', 'chexrayproject-allchinafiles/CHNCXR_0230_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0231_0.png', 'chexrayproject-allchinafiles/CHNCXR_0231_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0232_0.png', 'chexrayproject-allchinafiles/CHNCXR_0232_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0233_0.png', 'chexrayproject-allchinafiles/CHNCXR_0233_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0234_0.png', 'chexrayproject-allchinafiles/CHNCXR_0234_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0235_0.png', 'chexrayproject-allchinafiles/CHNCXR_0235_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0236_0.png', 'chexrayproject-allchinafiles/CHNCXR_0236_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0237_0.png', 'chexrayproject-allchinafiles/CHNCXR_0237_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0238_0.png', 'chexrayproject-allchinafiles/CHNCXR_0238_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0239_0.png', 'chexrayproject-allchinafiles/CHNCXR_0239_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0240_0.png', 'chexrayproject-allchinafiles/CHNCXR_0240_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0241_0.png', 'chexrayproject-allchinafiles/CHNCXR_0241_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0242_0.png', 'chexrayproject-allchinafiles/CHNCXR_0242_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0243_0.png', 'chexrayproject-allchinafiles/CHNCXR_0243_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0244_0.png', 'chexrayproject-allchinafiles/CHNCXR_0244_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0245_0.png', 'chexrayproject-allchinafiles/CHNCXR_0245_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0246_0.png', 'chexrayproject-allchinafiles/CHNCXR_0246_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0247_0.png', 'chexrayproject-allchinafiles/CHNCXR_0247_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0248_0.png', 'chexrayproject-allchinafiles/CHNCXR_0248_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0249_0.png', 'chexrayproject-allchinafiles/CHNCXR_0249_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0250_0.png', 'chexrayproject-allchinafiles/CHNCXR_0250_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0251_0.png', 'chexrayproject-allchinafiles/CHNCXR_0251_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0252_0.png', 'chexrayproject-allchinafiles/CHNCXR_0252_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0253_0.png', 'chexrayproject-allchinafiles/CHNCXR_0253_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0254_0.png', 'chexrayproject-allchinafiles/CHNCXR_0254_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0255_0.png', 'chexrayproject-allchinafiles/CHNCXR_0255_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0256_0.png', 'chexrayproject-allchinafiles/CHNCXR_0256_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0257_0.png', 'chexrayproject-allchinafiles/CHNCXR_0257_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0258_0.png', 'chexrayproject-allchinafiles/CHNCXR_0258_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0259_0.png', 'chexrayproject-allchinafiles/CHNCXR_0259_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0260_0.png', 'chexrayproject-allchinafiles/CHNCXR_0260_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0261_0.png', 'chexrayproject-allchinafiles/CHNCXR_0261_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0262_0.png', 'chexrayproject-allchinafiles/CHNCXR_0262_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0263_0.png', 'chexrayproject-allchinafiles/CHNCXR_0263_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0264_0.png', 'chexrayproject-allchinafiles/CHNCXR_0264_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0265_0.png', 'chexrayproject-allchinafiles/CHNCXR_0265_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0266_0.png', 'chexrayproject-allchinafiles/CHNCXR_0266_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0267_0.png', 'chexrayproject-allchinafiles/CHNCXR_0267_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0268_0.png', 'chexrayproject-allchinafiles/CHNCXR_0268_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0269_0.png', 'chexrayproject-allchinafiles/CHNCXR_0269_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0270_0.png', 'chexrayproject-allchinafiles/CHNCXR_0270_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0271_0.png', 'chexrayproject-allchinafiles/CHNCXR_0271_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0272_0.png', 'chexrayproject-allchinafiles/CHNCXR_0272_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0273_0.png', 'chexrayproject-allchinafiles/CHNCXR_0273_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0274_0.png', 'chexrayproject-allchinafiles/CHNCXR_0274_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0275_0.png', 'chexrayproject-allchinafiles/CHNCXR_0275_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0276_0.png', 'chexrayproject-allchinafiles/CHNCXR_0276_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0277_0.png', 'chexrayproject-allchinafiles/CHNCXR_0277_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0278_0.png', 'chexrayproject-allchinafiles/CHNCXR_0278_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0279_0.png', 'chexrayproject-allchinafiles/CHNCXR_0279_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0280_0.png', 'chexrayproject-allchinafiles/CHNCXR_0280_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0281_0.png', 'chexrayproject-allchinafiles/CHNCXR_0281_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0282_0.png', 'chexrayproject-allchinafiles/CHNCXR_0282_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0283_0.png', 'chexrayproject-allchinafiles/CHNCXR_0283_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0284_0.png', 'chexrayproject-allchinafiles/CHNCXR_0284_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0285_0.png', 'chexrayproject-allchinafiles/CHNCXR_0285_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0286_0.png', 'chexrayproject-allchinafiles/CHNCXR_0286_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0287_0.png', 'chexrayproject-allchinafiles/CHNCXR_0287_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0288_0.png', 'chexrayproject-allchinafiles/CHNCXR_0288_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0289_0.png', 'chexrayproject-allchinafiles/CHNCXR_0289_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0290_0.png', 'chexrayproject-allchinafiles/CHNCXR_0290_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0291_0.png', 'chexrayproject-allchinafiles/CHNCXR_0291_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0292_0.png', 'chexrayproject-allchinafiles/CHNCXR_0292_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0293_0.png', 'chexrayproject-allchinafiles/CHNCXR_0293_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0294_0.png', 'chexrayproject-allchinafiles/CHNCXR_0294_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0295_0.png', 'chexrayproject-allchinafiles/CHNCXR_0295_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0296_0.png', 'chexrayproject-allchinafiles/CHNCXR_0296_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0297_0.png', 'chexrayproject-allchinafiles/CHNCXR_0297_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0298_0.png', 'chexrayproject-allchinafiles/CHNCXR_0298_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0299_0.png', 'chexrayproject-allchinafiles/CHNCXR_0299_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0300_0.png', 'chexrayproject-allchinafiles/CHNCXR_0300_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0301_0.png', 'chexrayproject-allchinafiles/CHNCXR_0301_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0302_0.png', 'chexrayproject-allchinafiles/CHNCXR_0302_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0303_0.png', 'chexrayproject-allchinafiles/CHNCXR_0303_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0304_0.png', 'chexrayproject-allchinafiles/CHNCXR_0304_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0305_0.png', 'chexrayproject-allchinafiles/CHNCXR_0305_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0306_0.png', 'chexrayproject-allchinafiles/CHNCXR_0306_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0307_0.png', 'chexrayproject-allchinafiles/CHNCXR_0307_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0308_0.png', 'chexrayproject-allchinafiles/CHNCXR_0308_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0309_0.png', 'chexrayproject-allchinafiles/CHNCXR_0309_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0310_0.png', 'chexrayproject-allchinafiles/CHNCXR_0310_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0311_0.png', 'chexrayproject-allchinafiles/CHNCXR_0311_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0312_0.png', 'chexrayproject-allchinafiles/CHNCXR_0312_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0313_0.png', 'chexrayproject-allchinafiles/CHNCXR_0313_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0314_0.png', 'chexrayproject-allchinafiles/CHNCXR_0314_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0315_0.png', 'chexrayproject-allchinafiles/CHNCXR_0315_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0316_0.png', 'chexrayproject-allchinafiles/CHNCXR_0316_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0317_0.png', 'chexrayproject-allchinafiles/CHNCXR_0317_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0318_0.png', 'chexrayproject-allchinafiles/CHNCXR_0318_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0319_0.png', 'chexrayproject-allchinafiles/CHNCXR_0319_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0320_0.png', 'chexrayproject-allchinafiles/CHNCXR_0320_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0321_0.png', 'chexrayproject-allchinafiles/CHNCXR_0321_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0322_0.png', 'chexrayproject-allchinafiles/CHNCXR_0322_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0323_0.png', 'chexrayproject-allchinafiles/CHNCXR_0323_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0324_0.png', 'chexrayproject-allchinafiles/CHNCXR_0324_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0325_0.png', 'chexrayproject-allchinafiles/CHNCXR_0325_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0326_0.png', 'chexrayproject-allchinafiles/CHNCXR_0326_0.txt', 'chexrayproject-allchinafiles/CHNCXR_0327_1.png', 'chexrayproject-allchinafiles/CHNCXR_0327_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0328_1.png', 'chexrayproject-allchinafiles/CHNCXR_0328_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0329_1.png', 'chexrayproject-allchinafiles/CHNCXR_0329_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0330_1.png', 'chexrayproject-allchinafiles/CHNCXR_0330_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0331_1.png', 'chexrayproject-allchinafiles/CHNCXR_0331_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0332_1.png', 'chexrayproject-allchinafiles/CHNCXR_0332_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0333_1.png', 'chexrayproject-allchinafiles/CHNCXR_0333_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0334_1.png', 'chexrayproject-allchinafiles/CHNCXR_0334_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0335_1.png', 'chexrayproject-allchinafiles/CHNCXR_0335_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0336_1.png', 'chexrayproject-allchinafiles/CHNCXR_0336_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0337_1.png', 'chexrayproject-allchinafiles/CHNCXR_0337_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0338_1.png', 'chexrayproject-allchinafiles/CHNCXR_0338_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0339_1.png', 'chexrayproject-allchinafiles/CHNCXR_0339_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0340_1.png', 'chexrayproject-allchinafiles/CHNCXR_0340_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0341_1.png', 'chexrayproject-allchinafiles/CHNCXR_0341_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0342_1.png', 'chexrayproject-allchinafiles/CHNCXR_0342_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0343_1.png', 'chexrayproject-allchinafiles/CHNCXR_0343_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0344_1.png', 'chexrayproject-allchinafiles/CHNCXR_0344_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0345_1.png', 'chexrayproject-allchinafiles/CHNCXR_0345_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0346_1.png', 'chexrayproject-allchinafiles/CHNCXR_0346_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0347_1.png', 'chexrayproject-allchinafiles/CHNCXR_0347_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0348_1.png', 'chexrayproject-allchinafiles/CHNCXR_0348_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0349_1.png', 'chexrayproject-allchinafiles/CHNCXR_0349_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0350_1.png', 'chexrayproject-allchinafiles/CHNCXR_0350_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0351_1.png', 'chexrayproject-allchinafiles/CHNCXR_0351_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0352_1.png', 'chexrayproject-allchinafiles/CHNCXR_0352_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0353_1.png', 'chexrayproject-allchinafiles/CHNCXR_0353_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0354_1.png', 'chexrayproject-allchinafiles/CHNCXR_0354_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0355_1.png', 'chexrayproject-allchinafiles/CHNCXR_0355_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0356_1.png', 'chexrayproject-allchinafiles/CHNCXR_0356_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0357_1.png', 'chexrayproject-allchinafiles/CHNCXR_0357_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0358_1.png', 'chexrayproject-allchinafiles/CHNCXR_0358_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0359_1.png', 'chexrayproject-allchinafiles/CHNCXR_0359_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0360_1.png', 'chexrayproject-allchinafiles/CHNCXR_0360_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0361_1.png', 'chexrayproject-allchinafiles/CHNCXR_0361_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0362_1.png', 'chexrayproject-allchinafiles/CHNCXR_0362_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0363_1.png', 'chexrayproject-allchinafiles/CHNCXR_0363_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0364_1.png', 'chexrayproject-allchinafiles/CHNCXR_0364_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0365_1.png', 'chexrayproject-allchinafiles/CHNCXR_0365_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0366_1.png', 'chexrayproject-allchinafiles/CHNCXR_0366_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0367_1.png', 'chexrayproject-allchinafiles/CHNCXR_0367_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0368_1.png', 'chexrayproject-allchinafiles/CHNCXR_0368_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0369_1.png', 'chexrayproject-allchinafiles/CHNCXR_0369_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0370_1.png', 'chexrayproject-allchinafiles/CHNCXR_0370_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0371_1.png', 'chexrayproject-allchinafiles/CHNCXR_0371_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0372_1.png', 'chexrayproject-allchinafiles/CHNCXR_0372_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0373_1.png', 'chexrayproject-allchinafiles/CHNCXR_0373_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0374_1.png', 'chexrayproject-allchinafiles/CHNCXR_0374_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0375_1.png', 'chexrayproject-allchinafiles/CHNCXR_0375_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0376_1.png', 'chexrayproject-allchinafiles/CHNCXR_0376_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0377_1.png', 'chexrayproject-allchinafiles/CHNCXR_0377_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0378_1.png', 'chexrayproject-allchinafiles/CHNCXR_0378_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0379_1.png', 'chexrayproject-allchinafiles/CHNCXR_0379_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0380_1.png', 'chexrayproject-allchinafiles/CHNCXR_0380_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0381_1.png', 'chexrayproject-allchinafiles/CHNCXR_0381_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0382_1.png', 'chexrayproject-allchinafiles/CHNCXR_0382_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0383_1.png', 'chexrayproject-allchinafiles/CHNCXR_0383_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0384_1.png', 'chexrayproject-allchinafiles/CHNCXR_0384_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0385_1.png', 'chexrayproject-allchinafiles/CHNCXR_0385_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0386_1.png', 'chexrayproject-allchinafiles/CHNCXR_0386_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0387_1.png', 'chexrayproject-allchinafiles/CHNCXR_0387_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0388_1.png', 'chexrayproject-allchinafiles/CHNCXR_0388_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0389_1.png', 'chexrayproject-allchinafiles/CHNCXR_0389_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0390_1.png', 'chexrayproject-allchinafiles/CHNCXR_0390_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0391_1.png', 'chexrayproject-allchinafiles/CHNCXR_0391_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0392_1.png', 'chexrayproject-allchinafiles/CHNCXR_0392_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0393_1.png', 'chexrayproject-allchinafiles/CHNCXR_0393_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0394_1.png', 'chexrayproject-allchinafiles/CHNCXR_0394_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0395_1.png', 'chexrayproject-allchinafiles/CHNCXR_0395_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0396_1.png', 'chexrayproject-allchinafiles/CHNCXR_0396_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0397_1.png', 'chexrayproject-allchinafiles/CHNCXR_0397_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0398_1.png', 'chexrayproject-allchinafiles/CHNCXR_0398_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0399_1.png', 'chexrayproject-allchinafiles/CHNCXR_0399_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0400_1.png', 'chexrayproject-allchinafiles/CHNCXR_0400_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0401_1.png', 'chexrayproject-allchinafiles/CHNCXR_0401_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0402_1.png', 'chexrayproject-allchinafiles/CHNCXR_0402_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0403_1.png', 'chexrayproject-allchinafiles/CHNCXR_0403_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0404_1.png', 'chexrayproject-allchinafiles/CHNCXR_0404_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0405_1.png', 'chexrayproject-allchinafiles/CHNCXR_0405_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0406_1.png', 'chexrayproject-allchinafiles/CHNCXR_0406_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0407_1.png', 'chexrayproject-allchinafiles/CHNCXR_0407_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0408_1.png', 'chexrayproject-allchinafiles/CHNCXR_0408_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0409_1.png', 'chexrayproject-allchinafiles/CHNCXR_0409_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0410_1.png', 'chexrayproject-allchinafiles/CHNCXR_0410_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0411_1.png', 'chexrayproject-allchinafiles/CHNCXR_0411_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0412_1.png', 'chexrayproject-allchinafiles/CHNCXR_0412_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0413_1.png', 'chexrayproject-allchinafiles/CHNCXR_0413_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0414_1.png', 'chexrayproject-allchinafiles/CHNCXR_0414_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0415_1.png', 'chexrayproject-allchinafiles/CHNCXR_0415_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0416_1.png', 'chexrayproject-allchinafiles/CHNCXR_0416_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0417_1.png', 'chexrayproject-allchinafiles/CHNCXR_0417_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0418_1.png', 'chexrayproject-allchinafiles/CHNCXR_0418_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0419_1.png', 'chexrayproject-allchinafiles/CHNCXR_0419_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0420_1.png', 'chexrayproject-allchinafiles/CHNCXR_0420_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0421_1.png', 'chexrayproject-allchinafiles/CHNCXR_0421_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0422_1.png', 'chexrayproject-allchinafiles/CHNCXR_0422_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0423_1.png', 'chexrayproject-allchinafiles/CHNCXR_0423_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0424_1.png', 'chexrayproject-allchinafiles/CHNCXR_0424_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0425_1.png', 'chexrayproject-allchinafiles/CHNCXR_0425_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0426_1.png', 'chexrayproject-allchinafiles/CHNCXR_0426_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0427_1.png', 'chexrayproject-allchinafiles/CHNCXR_0427_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0428_1.png', 'chexrayproject-allchinafiles/CHNCXR_0428_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0429_1.png', 'chexrayproject-allchinafiles/CHNCXR_0429_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0430_1.png', 'chexrayproject-allchinafiles/CHNCXR_0430_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0431_1.png', 'chexrayproject-allchinafiles/CHNCXR_0431_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0432_1.png', 'chexrayproject-allchinafiles/CHNCXR_0432_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0433_1.png', 'chexrayproject-allchinafiles/CHNCXR_0433_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0434_1.png', 'chexrayproject-allchinafiles/CHNCXR_0434_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0435_1.png', 'chexrayproject-allchinafiles/CHNCXR_0435_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0436_1.png', 'chexrayproject-allchinafiles/CHNCXR_0436_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0437_1.png', 'chexrayproject-allchinafiles/CHNCXR_0437_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0438_1.png', 'chexrayproject-allchinafiles/CHNCXR_0438_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0439_1.png', 'chexrayproject-allchinafiles/CHNCXR_0439_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0440_1.png', 'chexrayproject-allchinafiles/CHNCXR_0440_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0441_1.png', 'chexrayproject-allchinafiles/CHNCXR_0441_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0442_1.png', 'chexrayproject-allchinafiles/CHNCXR_0442_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0443_1.png', 'chexrayproject-allchinafiles/CHNCXR_0443_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0444_1.png', 'chexrayproject-allchinafiles/CHNCXR_0444_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0445_1.png', 'chexrayproject-allchinafiles/CHNCXR_0445_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0446_1.png', 'chexrayproject-allchinafiles/CHNCXR_0446_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0447_1.png', 'chexrayproject-allchinafiles/CHNCXR_0447_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0448_1.png', 'chexrayproject-allchinafiles/CHNCXR_0448_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0449_1.png', 'chexrayproject-allchinafiles/CHNCXR_0449_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0450_1.png', 'chexrayproject-allchinafiles/CHNCXR_0450_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0451_1.png', 'chexrayproject-allchinafiles/CHNCXR_0451_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0452_1.png', 'chexrayproject-allchinafiles/CHNCXR_0452_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0453_1.png', 'chexrayproject-allchinafiles/CHNCXR_0453_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0454_1.png', 'chexrayproject-allchinafiles/CHNCXR_0454_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0455_1.png', 'chexrayproject-allchinafiles/CHNCXR_0455_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0456_1.png', 'chexrayproject-allchinafiles/CHNCXR_0456_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0457_1.png', 'chexrayproject-allchinafiles/CHNCXR_0457_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0458_1.png', 'chexrayproject-allchinafiles/CHNCXR_0458_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0459_1.png', 'chexrayproject-allchinafiles/CHNCXR_0459_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0460_1.png', 'chexrayproject-allchinafiles/CHNCXR_0460_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0461_1.png', 'chexrayproject-allchinafiles/CHNCXR_0461_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0462_1.png', 'chexrayproject-allchinafiles/CHNCXR_0462_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0463_1.png', 'chexrayproject-allchinafiles/CHNCXR_0463_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0464_1.png', 'chexrayproject-allchinafiles/CHNCXR_0464_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0465_1.png', 'chexrayproject-allchinafiles/CHNCXR_0465_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0466_1.png', 'chexrayproject-allchinafiles/CHNCXR_0466_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0467_1.png', 'chexrayproject-allchinafiles/CHNCXR_0467_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0468_1.png', 'chexrayproject-allchinafiles/CHNCXR_0468_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0469_1.png', 'chexrayproject-allchinafiles/CHNCXR_0469_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0470_1.png', 'chexrayproject-allchinafiles/CHNCXR_0470_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0471_1.png', 'chexrayproject-allchinafiles/CHNCXR_0471_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0472_1.png', 'chexrayproject-allchinafiles/CHNCXR_0472_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0473_1.png', 'chexrayproject-allchinafiles/CHNCXR_0473_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0474_1.png', 'chexrayproject-allchinafiles/CHNCXR_0474_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0475_1.png', 'chexrayproject-allchinafiles/CHNCXR_0475_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0476_1.png', 'chexrayproject-allchinafiles/CHNCXR_0476_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0477_1.png', 'chexrayproject-allchinafiles/CHNCXR_0477_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0478_1.png', 'chexrayproject-allchinafiles/CHNCXR_0478_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0479_1.png', 'chexrayproject-allchinafiles/CHNCXR_0479_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0480_1.png', 'chexrayproject-allchinafiles/CHNCXR_0480_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0481_1.png', 'chexrayproject-allchinafiles/CHNCXR_0481_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0482_1.png', 'chexrayproject-allchinafiles/CHNCXR_0482_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0483_1.png', 'chexrayproject-allchinafiles/CHNCXR_0483_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0484_1.png', 'chexrayproject-allchinafiles/CHNCXR_0484_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0485_1.png', 'chexrayproject-allchinafiles/CHNCXR_0485_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0486_1.png', 'chexrayproject-allchinafiles/CHNCXR_0486_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0487_1.png', 'chexrayproject-allchinafiles/CHNCXR_0487_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0488_1.png', 'chexrayproject-allchinafiles/CHNCXR_0488_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0489_1.png', 'chexrayproject-allchinafiles/CHNCXR_0489_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0490_1.png', 'chexrayproject-allchinafiles/CHNCXR_0490_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0491_1.png', 'chexrayproject-allchinafiles/CHNCXR_0491_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0492_1.png', 'chexrayproject-allchinafiles/CHNCXR_0492_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0493_1.png', 'chexrayproject-allchinafiles/CHNCXR_0493_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0494_1.png', 'chexrayproject-allchinafiles/CHNCXR_0494_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0495_1.png', 'chexrayproject-allchinafiles/CHNCXR_0495_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0496_1.png', 'chexrayproject-allchinafiles/CHNCXR_0496_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0497_1.png', 'chexrayproject-allchinafiles/CHNCXR_0497_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0498_1.png', 'chexrayproject-allchinafiles/CHNCXR_0498_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0499_1.png', 'chexrayproject-allchinafiles/CHNCXR_0499_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0500_1.png', 'chexrayproject-allchinafiles/CHNCXR_0500_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0501_1.png', 'chexrayproject-allchinafiles/CHNCXR_0501_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0502_1.png', 'chexrayproject-allchinafiles/CHNCXR_0502_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0503_1.png', 'chexrayproject-allchinafiles/CHNCXR_0503_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0504_1.png', 'chexrayproject-allchinafiles/CHNCXR_0504_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0505_1.png', 'chexrayproject-allchinafiles/CHNCXR_0505_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0506_1.png', 'chexrayproject-allchinafiles/CHNCXR_0506_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0507_1.png', 'chexrayproject-allchinafiles/CHNCXR_0507_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0508_1.png', 'chexrayproject-allchinafiles/CHNCXR_0508_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0509_1.png', 'chexrayproject-allchinafiles/CHNCXR_0509_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0510_1.png', 'chexrayproject-allchinafiles/CHNCXR_0510_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0511_1.png', 'chexrayproject-allchinafiles/CHNCXR_0511_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0512_1.png', 'chexrayproject-allchinafiles/CHNCXR_0512_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0513_1.png', 'chexrayproject-allchinafiles/CHNCXR_0513_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0514_1.png', 'chexrayproject-allchinafiles/CHNCXR_0514_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0515_1.png', 'chexrayproject-allchinafiles/CHNCXR_0515_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0516_1.png', 'chexrayproject-allchinafiles/CHNCXR_0516_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0517_1.png', 'chexrayproject-allchinafiles/CHNCXR_0517_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0518_1.png', 'chexrayproject-allchinafiles/CHNCXR_0518_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0519_1.png', 'chexrayproject-allchinafiles/CHNCXR_0519_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0520_1.png', 'chexrayproject-allchinafiles/CHNCXR_0520_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0521_1.png', 'chexrayproject-allchinafiles/CHNCXR_0521_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0522_1.png', 'chexrayproject-allchinafiles/CHNCXR_0522_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0523_1.png', 'chexrayproject-allchinafiles/CHNCXR_0523_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0524_1.png', 'chexrayproject-allchinafiles/CHNCXR_0524_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0525_1.png', 'chexrayproject-allchinafiles/CHNCXR_0525_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0526_1.png', 'chexrayproject-allchinafiles/CHNCXR_0526_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0527_1.png', 'chexrayproject-allchinafiles/CHNCXR_0527_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0528_1.png', 'chexrayproject-allchinafiles/CHNCXR_0528_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0529_1.png', 'chexrayproject-allchinafiles/CHNCXR_0529_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0530_1.png', 'chexrayproject-allchinafiles/CHNCXR_0530_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0531_1.png', 'chexrayproject-allchinafiles/CHNCXR_0531_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0532_1.png', 'chexrayproject-allchinafiles/CHNCXR_0532_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0533_1.png', 'chexrayproject-allchinafiles/CHNCXR_0533_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0534_1.png', 'chexrayproject-allchinafiles/CHNCXR_0534_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0535_1.png', 'chexrayproject-allchinafiles/CHNCXR_0535_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0536_1.png', 'chexrayproject-allchinafiles/CHNCXR_0536_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0537_1.png', 'chexrayproject-allchinafiles/CHNCXR_0537_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0538_1.png', 'chexrayproject-allchinafiles/CHNCXR_0538_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0539_1.png', 'chexrayproject-allchinafiles/CHNCXR_0539_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0540_1.png', 'chexrayproject-allchinafiles/CHNCXR_0540_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0541_1.png', 'chexrayproject-allchinafiles/CHNCXR_0541_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0542_1.png', 'chexrayproject-allchinafiles/CHNCXR_0542_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0543_1.png', 'chexrayproject-allchinafiles/CHNCXR_0543_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0544_1.png', 'chexrayproject-allchinafiles/CHNCXR_0544_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0545_1.png', 'chexrayproject-allchinafiles/CHNCXR_0545_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0546_1.png', 'chexrayproject-allchinafiles/CHNCXR_0546_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0547_1.png', 'chexrayproject-allchinafiles/CHNCXR_0547_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0548_1.png', 'chexrayproject-allchinafiles/CHNCXR_0548_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0549_1.png', 'chexrayproject-allchinafiles/CHNCXR_0549_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0550_1.png', 'chexrayproject-allchinafiles/CHNCXR_0550_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0551_1.png', 'chexrayproject-allchinafiles/CHNCXR_0551_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0552_1.png', 'chexrayproject-allchinafiles/CHNCXR_0552_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0553_1.png', 'chexrayproject-allchinafiles/CHNCXR_0553_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0554_1.png', 'chexrayproject-allchinafiles/CHNCXR_0554_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0555_1.png', 'chexrayproject-allchinafiles/CHNCXR_0555_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0556_1.png', 'chexrayproject-allchinafiles/CHNCXR_0556_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0557_1.png', 'chexrayproject-allchinafiles/CHNCXR_0557_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0558_1.png', 'chexrayproject-allchinafiles/CHNCXR_0558_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0559_1.png', 'chexrayproject-allchinafiles/CHNCXR_0559_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0560_1.png', 'chexrayproject-allchinafiles/CHNCXR_0560_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0561_1.png', 'chexrayproject-allchinafiles/CHNCXR_0561_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0562_1.png', 'chexrayproject-allchinafiles/CHNCXR_0562_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0563_1.png', 'chexrayproject-allchinafiles/CHNCXR_0563_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0564_1.png', 'chexrayproject-allchinafiles/CHNCXR_0564_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0565_1.png', 'chexrayproject-allchinafiles/CHNCXR_0565_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0566_1.png', 'chexrayproject-allchinafiles/CHNCXR_0566_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0567_1.png', 'chexrayproject-allchinafiles/CHNCXR_0567_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0568_1.png', 'chexrayproject-allchinafiles/CHNCXR_0568_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0569_1.png', 'chexrayproject-allchinafiles/CHNCXR_0569_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0570_1.png', 'chexrayproject-allchinafiles/CHNCXR_0570_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0571_1.png', 'chexrayproject-allchinafiles/CHNCXR_0571_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0572_1.png', 'chexrayproject-allchinafiles/CHNCXR_0572_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0573_1.png', 'chexrayproject-allchinafiles/CHNCXR_0573_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0574_1.png', 'chexrayproject-allchinafiles/CHNCXR_0574_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0575_1.png', 'chexrayproject-allchinafiles/CHNCXR_0575_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0576_1.png', 'chexrayproject-allchinafiles/CHNCXR_0576_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0577_1.png', 'chexrayproject-allchinafiles/CHNCXR_0577_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0578_1.png', 'chexrayproject-allchinafiles/CHNCXR_0578_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0579_1.png', 'chexrayproject-allchinafiles/CHNCXR_0579_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0580_1.png', 'chexrayproject-allchinafiles/CHNCXR_0580_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0581_1.png', 'chexrayproject-allchinafiles/CHNCXR_0581_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0582_1.png', 'chexrayproject-allchinafiles/CHNCXR_0582_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0583_1.png', 'chexrayproject-allchinafiles/CHNCXR_0583_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0584_1.png', 'chexrayproject-allchinafiles/CHNCXR_0584_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0585_1.png', 'chexrayproject-allchinafiles/CHNCXR_0585_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0586_1.png', 'chexrayproject-allchinafiles/CHNCXR_0586_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0587_1.png', 'chexrayproject-allchinafiles/CHNCXR_0587_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0588_1.png', 'chexrayproject-allchinafiles/CHNCXR_0588_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0589_1.png', 'chexrayproject-allchinafiles/CHNCXR_0589_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0590_1.png', 'chexrayproject-allchinafiles/CHNCXR_0590_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0591_1.png', 'chexrayproject-allchinafiles/CHNCXR_0591_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0592_1.png', 'chexrayproject-allchinafiles/CHNCXR_0592_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0593_1.png', 'chexrayproject-allchinafiles/CHNCXR_0593_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0594_1.png', 'chexrayproject-allchinafiles/CHNCXR_0594_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0595_1.png', 'chexrayproject-allchinafiles/CHNCXR_0595_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0596_1.png', 'chexrayproject-allchinafiles/CHNCXR_0596_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0597_1.png', 'chexrayproject-allchinafiles/CHNCXR_0597_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0598_1.png', 'chexrayproject-allchinafiles/CHNCXR_0598_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0599_1.png', 'chexrayproject-allchinafiles/CHNCXR_0599_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0600_1.png', 'chexrayproject-allchinafiles/CHNCXR_0600_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0601_1.png', 'chexrayproject-allchinafiles/CHNCXR_0601_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0602_1.png', 'chexrayproject-allchinafiles/CHNCXR_0602_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0603_1.png', 'chexrayproject-allchinafiles/CHNCXR_0603_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0604_1.png', 'chexrayproject-allchinafiles/CHNCXR_0604_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0605_1.png', 'chexrayproject-allchinafiles/CHNCXR_0605_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0606_1.png', 'chexrayproject-allchinafiles/CHNCXR_0606_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0607_1.png', 'chexrayproject-allchinafiles/CHNCXR_0607_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0608_1.png', 'chexrayproject-allchinafiles/CHNCXR_0608_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0609_1.png', 'chexrayproject-allchinafiles/CHNCXR_0609_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0610_1.png', 'chexrayproject-allchinafiles/CHNCXR_0610_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0611_1.png', 'chexrayproject-allchinafiles/CHNCXR_0611_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0612_1.png', 'chexrayproject-allchinafiles/CHNCXR_0612_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0613_1.png', 'chexrayproject-allchinafiles/CHNCXR_0613_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0614_1.png', 'chexrayproject-allchinafiles/CHNCXR_0614_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0615_1.png', 'chexrayproject-allchinafiles/CHNCXR_0615_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0616_1.png', 'chexrayproject-allchinafiles/CHNCXR_0616_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0617_1.png', 'chexrayproject-allchinafiles/CHNCXR_0617_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0618_1.png', 'chexrayproject-allchinafiles/CHNCXR_0618_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0619_1.png', 'chexrayproject-allchinafiles/CHNCXR_0619_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0620_1.png', 'chexrayproject-allchinafiles/CHNCXR_0620_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0621_1.png', 'chexrayproject-allchinafiles/CHNCXR_0621_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0622_1.png', 'chexrayproject-allchinafiles/CHNCXR_0622_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0623_1.png', 'chexrayproject-allchinafiles/CHNCXR_0623_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0624_1.png', 'chexrayproject-allchinafiles/CHNCXR_0624_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0625_1.png', 'chexrayproject-allchinafiles/CHNCXR_0625_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0626_1.png', 'chexrayproject-allchinafiles/CHNCXR_0626_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0627_1.png', 'chexrayproject-allchinafiles/CHNCXR_0627_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0628_1.png', 'chexrayproject-allchinafiles/CHNCXR_0628_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0629_1.png', 'chexrayproject-allchinafiles/CHNCXR_0629_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0630_1.png', 'chexrayproject-allchinafiles/CHNCXR_0630_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0631_1.png', 'chexrayproject-allchinafiles/CHNCXR_0631_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0632_1.png', 'chexrayproject-allchinafiles/CHNCXR_0632_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0633_1.png', 'chexrayproject-allchinafiles/CHNCXR_0633_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0634_1.png', 'chexrayproject-allchinafiles/CHNCXR_0634_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0635_1.png', 'chexrayproject-allchinafiles/CHNCXR_0635_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0636_1.png', 'chexrayproject-allchinafiles/CHNCXR_0636_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0637_1.png', 'chexrayproject-allchinafiles/CHNCXR_0637_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0638_1.png', 'chexrayproject-allchinafiles/CHNCXR_0638_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0639_1.png', 'chexrayproject-allchinafiles/CHNCXR_0639_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0640_1.png', 'chexrayproject-allchinafiles/CHNCXR_0640_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0641_1.png', 'chexrayproject-allchinafiles/CHNCXR_0641_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0642_1.png', 'chexrayproject-allchinafiles/CHNCXR_0642_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0643_1.png', 'chexrayproject-allchinafiles/CHNCXR_0643_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0644_1.png', 'chexrayproject-allchinafiles/CHNCXR_0644_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0645_1.png', 'chexrayproject-allchinafiles/CHNCXR_0645_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0646_1.png', 'chexrayproject-allchinafiles/CHNCXR_0646_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0647_1.png', 'chexrayproject-allchinafiles/CHNCXR_0647_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0648_1.png', 'chexrayproject-allchinafiles/CHNCXR_0648_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0649_1.png', 'chexrayproject-allchinafiles/CHNCXR_0649_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0650_1.png', 'chexrayproject-allchinafiles/CHNCXR_0650_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0651_1.png', 'chexrayproject-allchinafiles/CHNCXR_0651_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0652_1.png', 'chexrayproject-allchinafiles/CHNCXR_0652_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0653_1.png', 'chexrayproject-allchinafiles/CHNCXR_0653_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0654_1.png', 'chexrayproject-allchinafiles/CHNCXR_0654_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0655_1.png', 'chexrayproject-allchinafiles/CHNCXR_0655_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0656_1.png', 'chexrayproject-allchinafiles/CHNCXR_0656_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0657_1.png', 'chexrayproject-allchinafiles/CHNCXR_0657_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0658_1.png', 'chexrayproject-allchinafiles/CHNCXR_0658_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0659_1.png', 'chexrayproject-allchinafiles/CHNCXR_0659_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0660_1.png', 'chexrayproject-allchinafiles/CHNCXR_0660_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0661_1.png', 'chexrayproject-allchinafiles/CHNCXR_0661_1.txt', 'chexrayproject-allchinafiles/CHNCXR_0662_1.png', 'chexrayproject-allchinafiles/CHNCXR_0662_1.txt', 'chexrayproject-allchinafiles/Thumbs.db']



```python
#Get all China set files:

if RUNNING_ON_AWS == True:
    import s3fs
    fs = s3fs.S3FileSystem()
    china_Files = fs.ls('s3://chexrayproject-allchinafiles')
    shen_paths = []
    #for filename in glob.iglob("china_Files" + "**/*", recursive=False):
        #shen_paths.append(filename)
    for filename in china_Files:
        shen_paths.append(filename)
else:
    shen_dir = ".\\input\\ChinaSet_AllFiles\\ChinaSet_AllFiles\\"
    shen_paths = []
    for filename in glob.iglob(shen_dir + "**/*", recursive=False):
        shen_paths.append(filename)


#get all the Montgomery set files

if RUNNING_ON_AWS == True:
    import s3fs
    fs = s3fs.S3FileSystem()
    montgomery_Files = fs.ls('s3://chexrayproject-allmontgomeryfiles')
    mont_paths = []
    #for filename in glob.iglob("montgomery_Files" + "**/*", recursive=False):
        #mont_paths.append(filename)
    for filename in montgomery_Files:
        mont_paths.append(filename)
    
else:
    mont_dir = ".\\input\\Montgomery\\MontgomerySet\\"
    mont_paths = []
    for filename in glob.iglob(mont_dir + "**/*", recursive=False):
         mont_paths.append(filename)
```


```python
print('Montgomery Files', len(mont_paths))
print('Shenzhen Files', len(shen_paths))
```

    Montgomery Files 277
    Shenzhen Files 1325



```python
#### Now combine al the files into a dataframe: all_paths_df
```


```python
all_paths_df = pd.DataFrame(dict(path = mont_paths + shen_paths))
all_paths_df
all_paths_df['path'][79]
```




    'chexrayproject-allmontgomeryfiles/MCUCXR_0054_0.txt'




```python
all_paths_df = pd.DataFrame(dict(path = mont_paths + shen_paths))
all_paths_df['source'] = all_paths_df['path'].map(lambda x: x.split('/')[3])
all_paths_df['file_id'] = all_paths_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
all_paths_df['patient_group']  = all_paths_df['file_id'].map(lambda x: x.split('_')[0])

all_paths_df['file_ext'] = all_paths_df['path'].map(lambda x: os.path.splitext(x)[1][1:])
all_paths_df = all_paths_df[all_paths_df.file_ext.isin(['png', 'txt'])]
all_paths_df['pulm_state']  = all_paths_df['file_id'].map(lambda x: int(x.split('_')[-1]))
all_paths_df.sample(5)
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-20-95bfb6b12fb1> in <module>
          1 all_paths_df = pd.DataFrame(dict(path = mont_paths + shen_paths))
    ----> 2 all_paths_df['source'] = all_paths_df['path'].map(lambda x: x.split('/')[3])
          3 all_paths_df['file_id'] = all_paths_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
          4 all_paths_df['patient_group']  = all_paths_df['file_id'].map(lambda x: x.split('_')[0])
          5 


    /opt/conda/lib/python3.6/site-packages/pandas/core/series.py in map(self, arg, na_action)
       3981         dtype: object
       3982         """
    -> 3983         new_values = super()._map_values(arg, na_action=na_action)
       3984         return self._constructor(new_values, index=self.index).__finalize__(
       3985             self, method="map"


    /opt/conda/lib/python3.6/site-packages/pandas/core/base.py in _map_values(self, mapper, na_action)
       1158 
       1159         # mapper is a function
    -> 1160         new_values = map_f(values, mapper)
       1161 
       1162         return new_values


    pandas/_libs/lib.pyx in pandas._libs.lib.map_infer()


    <ipython-input-20-95bfb6b12fb1> in <lambda>(x)
          1 all_paths_df = pd.DataFrame(dict(path = mont_paths + shen_paths))
    ----> 2 all_paths_df['source'] = all_paths_df['path'].map(lambda x: x.split('/')[3])
          3 all_paths_df['file_id'] = all_paths_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
          4 all_paths_df['patient_group']  = all_paths_df['file_id'].map(lambda x: x.split('_')[0])
          5 


    IndexError: list index out of range


## Create Report DF


```python
clean_patients_df = all_paths_df.pivot_table(index = ['patient_group', 'pulm_state', 'file_id'], 
                                             columns=['file_ext'], 
                                             values = 'path', aggfunc='first').reset_index()
clean_patients_df.sample(5)
from warnings import warn
def report_to_dict(in_path):
    with open(in_path, 'r') as f:
        all_lines = [x.strip() for x in f.read().split('\n')]
    info_dict = {}
    try:
        if "Patient's Sex" in all_lines[0]:
            info_dict['age'] = all_lines[1].split(':')[-1].strip().replace('Y', '')
            info_dict['gender'] = all_lines[0].split(':')[-1].strip()
            info_dict['report'] = ' '.join(all_lines[2:]).strip()
        else:
            info_dict['age'] = all_lines[0].split(' ')[-1].replace('yrs', '').replace('yr', '')
            info_dict['gender'] = all_lines[0].split(' ')[0].strip()
            info_dict['report'] = ' '.join(all_lines[1:]).strip()
        
        info_dict['gender'] = info_dict['gender'].upper().replace('FEMALE', 'F').replace('MALE', 'M').replace('FEMAL', 'F')[0:1]
        if 'month' in info_dict.get('age', ''):
            info_dict.pop('age') # invalid
        if 'day' in info_dict.get('age', ''):
            info_dict.pop('age') # invalid
        elif len(info_dict.get('age',''))>0:
            info_dict['age'] = float(info_dict['age'])
        else:
            info_dict.pop('age')
        return info_dict
    except Exception as e:
        print(all_lines)
        warn(str(e), RuntimeWarning)
        return {}
report_df = pd.DataFrame([dict(**report_to_dict(c_row.pop('txt')), **c_row) 
              for  _, c_row in clean_patients_df.iterrows()])
report_df.sample(5)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-16-b1a9cf3aed60> in <module>
          1 clean_patients_df = all_paths_df.pivot_table(index = ['patient_group', 'pulm_state', 'file_id'], 
          2                                              columns=['file_ext'],
    ----> 3                                              values = 'path', aggfunc='first').reset_index()
          4 clean_patients_df.sample(5)
          5 from warnings import warn


    /opt/conda/lib/python3.7/site-packages/pandas/core/frame.py in pivot_table(self, values, index, columns, aggfunc, fill_value, margins, dropna, margins_name, observed)
       6078             dropna=dropna,
       6079             margins_name=margins_name,
    -> 6080             observed=observed,
       6081         )
       6082 


    /opt/conda/lib/python3.7/site-packages/pandas/core/reshape/pivot.py in pivot_table(data, values, index, columns, aggfunc, fill_value, margins, dropna, margins_name, observed)
        180     # GH 15193 Make sure empty columns are removed if dropna=True
        181     if isinstance(table, ABCDataFrame) and dropna:
    --> 182         table = table.dropna(how="all", axis=1)
        183 
        184     return table


    /opt/conda/lib/python3.7/site-packages/pandas/core/frame.py in dropna(self, axis, how, thresh, subset, inplace)
       4749             agg_obj = self.take(indices, axis=agg_axis)
       4750 
    -> 4751         count = agg_obj.count(axis=agg_axis)
       4752 
       4753         if thresh is not None:


    /opt/conda/lib/python3.7/site-packages/pandas/core/frame.py in count(self, axis, level, numeric_only)
       7793         # GH #423
       7794         if len(frame._get_axis(axis)) == 0:
    -> 7795             result = Series(0, index=frame._get_agg_axis(axis))
       7796         else:
       7797             if frame._is_mixed_type or frame._data.any_extension_types:


    /opt/conda/lib/python3.7/site-packages/pandas/core/series.py in __init__(self, data, index, dtype, name, copy, fastpath)
        303                     data = data.copy()
        304             else:
    --> 305                 data = sanitize_array(data, index, dtype, copy, raise_cast_failure=True)
        306 
        307                 data = SingleBlockManager(data, index, fastpath=True)


    /opt/conda/lib/python3.7/site-packages/pandas/core/construction.py in sanitize_array(data, index, dtype, copy, raise_cast_failure)
        463                 value = maybe_cast_to_datetime(value, dtype)
        464 
    --> 465             subarr = construct_1d_arraylike_from_scalar(value, len(index), dtype)
        466 
        467         else:


    /opt/conda/lib/python3.7/site-packages/pandas/core/dtypes/cast.py in construct_1d_arraylike_from_scalar(value, length, dtype)
       1459                 value = ensure_str(value)
       1460 
    -> 1461         subarr = np.empty(length, dtype=dtype)
       1462         subarr.fill(value)
       1463 


    TypeError: Cannot interpret '<attribute 'dtype' of 'numpy.generic' objects>' as a data type



```python
## Segmentation
```


```python
#get all the mask files
#mask_path = os.path.join("D:\\", "Documents", "Medical", "TB","Lung Segmentation","masks")
mask_path = os.path.join('.', 'input', 'masks')
#mask directory
masks = os.listdir(mask_path)


#clean it up to align with images names: Remove the .png and the _mask(from China masks)


mask_ids_temp = [fName.split(".png")[0] for fName in masks]

mask_ids = [fName.split("_mask")[0] for fName in mask_ids_temp]

#The total # of masks
mask_file_names = [os.path.join(mask_path, mask) for mask in masks]

#masks

#Total number of modified masks - China masks
check = [i for i in masks if "mask" in i]
print("Total masks that have modified names:",len(check))

## ??? There seems to be 704 masks before modification
```

    Total mask that has modified name: 566



```python
#get all the image files
image_path = os.path.join('.', 'input',"CXR_png")

#image directory
images = os.listdir(image_path)

#clean it up to align with images names: Remove the .png and the _mask(from China masks)
image_ids = [fName.split(".png")[0] for fName in images]
#mask_file_names = [fName.split("_mask")[0] for fName in mask_id]

image_file_names = [os.path.join(image_path, image) for image in images]

#The total # of images
print('Total X-ray images: ', len(image_file_names))
```

    Total X-ray images:  800



```python
#Put all the names into a dataframe for convenience
images_df = pd.DataFrame()
images_df['xrays'] = image_file_names
images_df['file_id'] = image_ids
images_df['has_mask'] = images_df['file_id'].isin(mask_ids)

images_with_masks_df = images_df[images_df['file_id'].isin(mask_ids)]

images_with_masks_df['masks'] = mask_file_names

print("There are {} x-rays with masks".format(len(images_with_masks_df)))
images_df
print("True indicates the x-ray has a mask:")
images_df['has_mask'].value_counts()
```

    There are 704 x-rays with masks
    True indicates the x-ray has a mask:


    <ipython-input-14-075844446854>:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      images_with_masks_df['masks'] = mask_file_names





    True     704
    False     96
    Name: has_mask, dtype: int64




```python
from sklearn.model_selection import train_test_split
#Do a train-test split
#??? So, here we are splitting the xrays from the masks, in segmentation we are trying to predict the mask.
# We use 90% of the data for the training set.
train_x,test_x,train_y,test_y = train_test_split(images_with_masks_df['xrays'],
                                                   images_with_masks_df['masks'],test_size    = 0.1,
                                                   random_state = 42)

#size of the training set should be 90% of 704
#len(train_x)
```


```python
#We are also going to make a validation set.
trainx,validationx,trainy,validationy = train_test_split(train_x,train_y,test_size = 0.1,random_state = 42)

#len(trainx)
```


```python
#Put all these data sets into data frames
train_df = pd.DataFrame(index=trainx.index)
train_df['xrays'] = trainx
train_df['masks'] = trainy

test_df = pd.DataFrame(index=test_x.index)
test_df['xrays'] = test_x
test_df['masks'] = test_y

validation_df = pd.DataFrame(index=validationx.index)
validation_df['xrays'] = validationx
validation_df['masks'] = validationy
```


```python
## Now that we have a dataframe of training and test examples, can we mask them?
```

## Need a train info dataframe


```python

```


```python
train_info_loc = os.path.join(".", "CheXpert-v1.0-small") #Need the file path to the CheXpert-V1.0-small file (this must be downloaded independently through Stanford ML)
train_file_name = "train.csv"
train_info = pd.read_csv(os.path.join(train_info_loc, train_file_name))

```


```python
train_info.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Path</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Frontal/Lateral</th>
      <th>AP/PA</th>
      <th>No Finding</th>
      <th>Enlarged Cardiomediastinum</th>
      <th>Cardiomegaly</th>
      <th>Lung Opacity</th>
      <th>Lung Lesion</th>
      <th>Edema</th>
      <th>Consolidation</th>
      <th>Pneumonia</th>
      <th>Atelectasis</th>
      <th>Pneumothorax</th>
      <th>Pleural Effusion</th>
      <th>Pleural Other</th>
      <th>Fracture</th>
      <th>Support Devices</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CheXpert-v1.0-small/train/patient00001/study1/...</td>
      <td>Female</td>
      <td>68</td>
      <td>Frontal</td>
      <td>AP</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CheXpert-v1.0-small/train/patient00002/study2/...</td>
      <td>Female</td>
      <td>87</td>
      <td>Frontal</td>
      <td>AP</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>NaN</td>
      <td>-1.0</td>
      <td>NaN</td>
      <td>-1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CheXpert-v1.0-small/train/patient00002/study1/...</td>
      <td>Female</td>
      <td>83</td>
      <td>Frontal</td>
      <td>AP</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CheXpert-v1.0-small/train/patient00002/study1/...</td>
      <td>Female</td>
      <td>83</td>
      <td>Lateral</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CheXpert-v1.0-small/train/patient00003/study1/...</td>
      <td>Male</td>
      <td>41</td>
      <td>Frontal</td>
      <td>AP</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(train_info)
```




    223414




```python
train_info.fillna(0, inplace=True)
train_info.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Path</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Frontal/Lateral</th>
      <th>AP/PA</th>
      <th>No Finding</th>
      <th>Enlarged Cardiomediastinum</th>
      <th>Cardiomegaly</th>
      <th>Lung Opacity</th>
      <th>Lung Lesion</th>
      <th>Edema</th>
      <th>Consolidation</th>
      <th>Pneumonia</th>
      <th>Atelectasis</th>
      <th>Pneumothorax</th>
      <th>Pleural Effusion</th>
      <th>Pleural Other</th>
      <th>Fracture</th>
      <th>Support Devices</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CheXpert-v1.0-small/train/patient00001/study1/...</td>
      <td>Female</td>
      <td>68</td>
      <td>Frontal</td>
      <td>AP</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CheXpert-v1.0-small/train/patient00002/study2/...</td>
      <td>Female</td>
      <td>87</td>
      <td>Frontal</td>
      <td>AP</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CheXpert-v1.0-small/train/patient00002/study1/...</td>
      <td>Female</td>
      <td>83</td>
      <td>Frontal</td>
      <td>AP</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CheXpert-v1.0-small/train/patient00002/study1/...</td>
      <td>Female</td>
      <td>83</td>
      <td>Lateral</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CheXpert-v1.0-small/train/patient00003/study1/...</td>
      <td>Male</td>
      <td>41</td>
      <td>Frontal</td>
      <td>AP</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## create a new dataframe with a column for complete path and diagnostic columns of interest:



```python
train_df = pd.DataFrame(index=train_info.index)
data_df = train_info.iloc[:, 5:].copy()
data_df['xrays'] = [os.path.join('.', x) for x in train_info['Path'].values]
```


```python
data_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No Finding</th>
      <th>Enlarged Cardiomediastinum</th>
      <th>Cardiomegaly</th>
      <th>Lung Opacity</th>
      <th>Lung Lesion</th>
      <th>Edema</th>
      <th>Consolidation</th>
      <th>Pneumonia</th>
      <th>Atelectasis</th>
      <th>Pneumothorax</th>
      <th>Pleural Effusion</th>
      <th>Pleural Other</th>
      <th>Fracture</th>
      <th>Support Devices</th>
      <th>xrays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>.\CheXpert-v1.0-small/train/patient00001/study...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>.\CheXpert-v1.0-small/train/patient00002/study...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>.\CheXpert-v1.0-small/train/patient00002/study...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>.\CheXpert-v1.0-small/train/patient00002/study...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>.\CheXpert-v1.0-small/train/patient00003/study...</td>
    </tr>
  </tbody>
</table>
</div>




```python
import pickle

#save huge dataframe to pickle
train_df.to_pickle("train_df.pkl")
```

## get and view file from the data_df (checking understanding of file formats)


```python

```


```python
rnd_xray = data_df['xrays'][354]
```


```python
from PIL import Image

img = Image.open(rnd_xray) #Note, these .jpg files are PIL objects...

img
```




![png](output_36_0.png)




```python
## Let's convert it to a tensor

from torchvision import transforms

convert_tensor = transforms.ToTensor()

img_t = convert_tensor(img)


print(img_t.shape)

shifted = img_t.permute(1, 2, 0)

print(shifted.shape)
```

    torch.Size([1, 320, 390])
    torch.Size([320, 390, 1])



```python
#Create test train split

from sklearn.model_selection import GroupShuffleSplit

# Initialize the GroupShuffleSplit.
gss = GroupShuffleSplit(n_splits=1, test_size=0.01)

# Get the indexers for the split.
idx1, idx2 = next(gss.split(data_df, groups=data_df.index))

# Get the split DataFrames.
df1, df2 = data_df.iloc[idx1], data_df.iloc[idx2]



#Just use a slice of the images for now:

train_temp_df = df1.sample(100000)
test_temp_df = df2.copy()
```


```python
train_temp_df['xrays'][432]
```




    '.\\CheXpert-v1.0-small/train/patient00119/study5/view1_frontal.jpg'




```python
len(train_temp_df)
```




    100000



## MODEL DEVELOPMENT

## 
1) Load 1 Resnet pretrained model
2) Apply this model to data (What's the input and what's the output?)


```python
import torch.nn as nn
## Try implementing a Resnet from scratch  (tutorial here: https://www.youtube.com/watch?v=DkNIBBBvcPs)


class resblock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample = None, stride = 1):
        super(resblock, self).__init__()
        self.expansion = 4 # "number of channels after a block is 4x what it was when it entered"
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride =1, padding = 0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, padding =1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels *self.expansion, kernel_size =1, stride=1, padding = 0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
            
            print('SHAPES:')
            print(x.shape)
            print(identity.shape)
        
        x += identity
        x = self.relu(x)
        return x
    

class ResNet(nn.Module): # note, the layers argument corresponds to the number of resnet blocks
    def __init__(self, resblock, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        #ResNet layers
        
        self.layer1 = self._make_layer(resblock, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(resblock, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(resblock, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(resblock, layers[3], out_channels=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        x = self.bn1(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.maxpool(x)
        print(x.shape)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
        
        
    def _make_layer(self, resblock, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels *4, kernel_size = 1,
                                                         stride = stride),
                                               nn.BatchNorm2d(out_channels*4))
        
        layers.append(resblock(self.in_channels, out_channels, identity_downsample, stride)) #changes the number of channels
        self.in_channels = out_channels * 4
        
        for i in range(num_residual_blocks - 1):
            layers.append(resblock(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)
        
        
```


```python
#initialize resnet 50 with our parameters: 1 channel for grayscale images, 14 classes.

def ResNet50(img_channels=1, num_classes=14):
    return ResNet(resblock, [3, 4, 6, 3], img_channels, num_classes)
```


```python

```


```python
def test():
    net = ResNet50()
    x = torch.randn(2, 1, 224, 224)
    y = net(x).to('cuda')
    print(y.shape)
    
test()
```

## MAYBE WE DON'T NEED TO DEVELOP OUR OWN MODELS RIGHT NOW. TRY LOADING DR. A's PreTrained Model

## TO RUN THIS CODE MAKE SURE THE h5 MODEL IS IN THE SAME DIRECTORY AS THE NOTEBOOK, OR IF ON AWS, MAKE SURE THE h5 MODEL IS IN A BUCKET.

## TODO: THIS NEEDS TO BE TRANSLATED TO PYTORCH, 


```python
loaded_model = load_model(os.path.join('.', 'CheXpert_ResNet20v1_224_multiclass_model_epochs5.h5'))
```


```python
#masked_test_gen.reset()
pred=model.predict_generator(masked_test_gen, steps=len(test_temp_df), verbose=1)
```


```python
pred_bool = []
pred_bool = [(pred[i] > 0.5) for i in range(len(pred))]
```


```python
predictions = [item.astype(int) for item in pred_bool]
predictions[10]
```


```python
columns = diag_cols
#columns should be the same order of y_col
results=pd.DataFrame(index = test_temp_df.index)
for i in range(len(columns)):
    results[columns[i]] = predictions[i]
#ordered_cols = ["Filenames"] + columns
#results=results[ordered_cols] #To get the same column order
#results.to_csv("results.csv",index=False)
results.tail()
```


```python
results.describe()
```

## TRANSFER LEARNING WORK (THIS MUST BE TRANSLATED TO PYTORCH)


```python
from tensorflow.keras.applications.inception_v3 import preprocess_input
```


```python
# initialize the input image shape (224x224 pixels) along with
# the pre-processing function (this might need to be changed
# based on which model we use to classify our image)
inputShape = (224, 224)
preprocess = preprocess_input
# if we are using the InceptionV3 or Xception networks, then we
# need to set the input shape to (299x299) [rather than (224x224)]
# and use a different image processing function
#if args["model"] in ("inception", "xception"):
#	inputShape = (299, 299)
#	preprocess = preprocess_input
```


```python
# load the input image using the Keras helper utility while ensuring
# the image is resized to `inputShape`, the required input dimensions
# for the ImageNet pre-trained network
print("[INFO] loading and pre-processing image...")

# (Nick: trying to understand what this code is doing.)
# Keras has a load_image function that loads the image and reduces its size to 224 x 224
# Torchvision will do this with a resize transformation

import torchvision.transforms as transforms

#from earlier image loading code above:
rnd_xray = data_df['xrays'][354]
from PIL import Image
img_norm = Image.open(rnd_xray) #Note, these .jpg files are PIL objects...

img_resized = transforms.Resize(size = (224, 224))(img_norm) #resized image must be an image, not a string.

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
plt.imshow(img_norm)
ax.set_title('Normal')
ax = fig.add_subplot(1,2,2)
ax.set_title('Resized')
plt.imshow(img_resized)

```

    [INFO] loading and pre-processing image...





    <matplotlib.image.AxesImage at 0x1f1a86b7f10>




![png](output_59_2.png)



```python
## We may want to string together a number of transformations:

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channel = 1),
    transforms.ToTensor()
])

img_transformed = preprocess(img_norm) #resized image must be an image, not a string.

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
plt.imshow(img_norm)
ax.set_title('Normal')
ax = fig.add_subplot(1,2,2)
ax.set_title('Transformed')
plt.imshow(img_transformed.permute(1,2,0))
```




    <matplotlib.image.AxesImage at 0x1f1a9c23520>




![png](output_60_1.png)



```python

```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-70-719f4b8b8838> in <module>
    ----> 1 img_transformed[1]
    

    IndexError: index 1 is out of bounds for dimension 0 with size 1


## Why is this image appearing in the blue, green, and yellow spectrum? When I check the shape, the first channel is 1, which indicates it's b+w, or does it?


```python
image = load_img(data_df['xrays'].values[1], target_size = (224, 224))

image = img_to_array(image)
# our input image is now represented as a NumPy array of shape
# (inputShape[0], inputShape[1], 3) however we need to expand the
# dimension by making the shape (1, inputShape[0], inputShape[1], 3)
# so we can pass it through the network
image = np.expand_dims(image, axis=0)
# pre-process the image using the appropriate function based on the
# model that has been loaded (i.e., mean subtraction, scaling, etc.)
image = preprocess(image)
```

## Load ResNet50 in Pytorch and override last fully connected layer


```python
from torchvision import models
import torch.nn as nn

rn50 = models.resnet50(pretrained=True)
rn50.fc = nn.Linear(in_features=2048, out_features=14, bias=True) #This overrides the final fc layer
```


```python

```


```python
print(rn50)
```

    ResNet(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (layer2): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (3): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (layer3): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (3): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (4): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (5): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (layer4): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      (fc): Linear(in_features=2048, out_features=14, bias=True)
    )


# Dr. A's code calls for making several layers untrainable. This is not that easy in pytorch. Some insight might be gleaned here: https://github.com/mortezamg63/Accessing-and-modifying-different-layers-of-a-pretrained-model-in-pytorch

#### The key insight seems to be that torch models use 'children' to hierarchically organize their internal modules



```python
child_counter = 0
for child in rn50.children():
    print("child ", str(child_counter), "is: ")
    print(child)
    child_counter +=1
```

    child  0 is: 
    Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    child  1 is: 
    BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    child  2 is: 
    ReLU(inplace=True)
    child  3 is: 
    MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    child  4 is: 
    Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (downsample): Sequential(
          (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    child  5 is: 
    Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (3): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    child  6 is: 
    Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (downsample): Sequential(
          (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (3): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (4): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (5): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    child  7 is: 
    Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (downsample): Sequential(
          (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    child  8 is: 
    AdaptiveAvgPool2d(output_size=(1, 1))
    child  9 is: 
    Linear(in_features=2048, out_features=14, bias=True)


#### freeze all layers


```python
child_counter = 0
for child in rn50.children():
    print("child ", str(child_counter), "was frozen.")
    for param in child.parameters():
        param.requires_grad = False
    child_counter +=1
    
    

```

    child  0 was frozen.
    child  1 was frozen.
    child  2 was frozen.
    child  3 was frozen.
    child  4 was frozen.
    child  5 was frozen.
    child  6 was frozen.
    child  7 was frozen.
    child  8 was frozen.
    child  9 was frozen.


#### If certain layers are to be frozen (in Keras this is setting the trainable attribute to False)


```python
layers_to_be_frozen []
child_counter = 0
for child in rn50.children():
    if child in layers_to_be_frozen:
        print("child ", str(child_counter), "was frozen.")
        for param in child.parameters():
            param.requires_grad = False
    else:
        print('child ', str(child_counter), "not frozen.")
    child_counter +=1
```


```python

```


```python
#If we don't want to train the weights:
#mark loaded layers as not trainable
for layer in model.layers:
    layer.trainable = False
    
#Also if don't want to train particular layers:
#mark some layers as not trainable
#model.get_layer('block1_conv1').trainable = False
#model.get_layer('block1_conv2').trainable = False
#model.get_layer('block2_conv1').trainable = False
#model.get_layer('block2_conv2').trainable = False
```


```python
baseModel = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(299, 299, 3)))
# construct the head of the model that will be placed on top of the
# the base model

#mark loaded layers as not trainable
#for layer in baseModel.layers:
#    layer.trainable = False


headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
y = Dense(1024, activation="relu")(headModel)
#headModel = Dropout(0.5)(headModel)
output1 = Dense(1, activation = 'sigmoid',kernel_initializer='he_normal')(y)
output2 = Dense(1, activation = 'sigmoid',kernel_initializer='he_normal')(y)
output3 = Dense(1, activation = 'sigmoid',kernel_initializer='he_normal')(y)
output4 = Dense(1, activation = 'sigmoid',kernel_initializer='he_normal')(y)
output5 = Dense(1, activation = 'sigmoid',kernel_initializer='he_normal')(y)
output6 = Dense(1, activation = 'sigmoid',kernel_initializer='he_normal')(y)
output7 = Dense(1, activation = 'sigmoid',kernel_initializer='he_normal')(y)
output8 = Dense(1, activation = 'sigmoid',kernel_initializer='he_normal')(y)
output9 = Dense(1, activation = 'sigmoid',kernel_initializer='he_normal')(y)
output10 = Dense(1, activation = 'sigmoid',kernel_initializer='he_normal')(y)
output11 = Dense(1, activation = 'sigmoid',kernel_initializer='he_normal')(y)
output12 = Dense(1, activation = 'sigmoid',kernel_initializer='he_normal')(y)
output13 = Dense(1, activation = 'sigmoid',kernel_initializer='he_normal')(y)
output14 = Dense(1, activation = 'sigmoid',kernel_initializer='he_normal')(y)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
IV3model = Model(inputs=baseModel.input, outputs=[output1,output2,output3,output4,output5,output6, 
                                               output7,output8,output9,output10, output11,output12,output13,output14])
#model = resnet_v2(input_shape=input_shape, depth=depth)
IV3model.compile(loss = ["binary_crossentropy","binary_crossentropy", "binary_crossentropy","binary_crossentropy", 
                      "binary_crossentropy","binary_crossentropy","binary_crossentropy", "binary_crossentropy",
                      "binary_crossentropy",      "binary_crossentropy", "binary_crossentropy","binary_crossentropy", 
                      "binary_crossentropy","binary_crossentropy"],
              optimizer='adam', metrics=['accuracy'])


```


```python
IV3results = IV3model.fit_generator(masked_train_gen, steps_per_epoch= 10000, epochs=10,
                              validation_data=masked_val_gen, validation_steps=100,
                              callbacks=callbacks, verbose=2)
```
