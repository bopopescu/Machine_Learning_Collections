fid = open('../data/Training_halos.csv', 'r')
data = fid.read()
fid.close()

lines = data.splitlines()[1:]

fout = open('Training_halos', 'w')
for line in lines:
    fields = line.split(',')
    fout.write(' '.join(fields[1:]) + '\n')
fout.close()
