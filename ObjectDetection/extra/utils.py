
import os
import io
import time
import tarfile
import pickle

def fetch(url):
  import requests, os, hashlib, tempfile
  fp = os.path.join(tempfile.gettempdir(), hashlib.md5(url.encode('utf-8')).hexdigest())
  if os.path.isfile(fp) and os.stat(fp).st_size > 0:
    with open(fp, "rb") as f:
      dat = f.read()
  else:
    print("fetching %s" % url)
    dat = requests.get(url).content
    with open(fp + ".tmp", "wb") as f:
      f.write(dat)
    os.rename(fp + ".tmp", fp)
  return dat

def load_cifar():
  tt = tarfile.open(fileobj=io.BytesIO(fetch('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')), mode='r:gz')
  db = pickle.load(tt.extractfile('cifar-10-batches-py/data_batch_1'), encoding="bytes")
  X = db[b'data'].reshape((-1, 3, 32, 32))
  Y = np.array(db[b'labels'])
  return X, Y