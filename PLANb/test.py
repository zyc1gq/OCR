import os

files=os.listdir('image/')
for file in files:
    path='image/'+file
    os.remove(path)
    print("del "+path)

files=os.listdir('image_test/')
for file in files:
    path='image_test/'+file
    os.remove(path)
    print("del "+path)

files=os.listdir('stat/')
for file in files:
    path='stat/'+file
    os.remove(path)
    print("del "+path)