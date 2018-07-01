import mmap, os, struct

mfd = os.open("mmapped.bin", os.O_RDONLY)
mfile = mmap.mmap(mfd, 0, prot=mmap.PROT_READ)
msg = mfile.read(8000)

print len(msg)
l = []
for i in range(1000):
    res = struct.unpack('i', msg[4*i:4*(i+1)])
    l.append(res[0])
print l[:5], l[-5:]

mfile.close()

# with open("mmapped.bin", "rb") as f:
#     # memory-map the file, size 0 means whole file
#     mm = mmap.mmap(f.fileno(), 0)
#     print len(mm.read())