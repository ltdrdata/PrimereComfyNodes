import os as os
import hashlib

def model_hash(filename):
    try:
        with open(filename, "rb") as file:
            m = hashlib.sha256()
            blksize = 1024 * 1024
            for chunk in iter(lambda: file.read(blksize), b""):
                m.update(chunk)
            nameoffile = os.path.basename(filename)
            thehash = m.hexdigest()[0:10]
            print(thehash, "|", nameoffile)
    except FileNotFoundError:
        return 'NOFILE'

print("  Hash   | Model Name")
print("----------------------")

filenamex = 'h:/Resources/Stable-diffusion/BestAll/dvarchExterior.safetensors'
model_hash(filenamex)

'''
1638fa9a88
BestOf_Hardblend

import hashlib

def model_hash(filename):
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()[0:10]

filenamex = 'h:/Resources/Stable-diffusion/BestAll/dvarchExterior.safetensors'
print(model_hash(filenamex))
'''