fid = open('../data/Training_halos.csv', 'r')
data = fid.read()
fid.close()

lines = data.splitlines()[1:]

floc = open('locations', 'w')
fref = open('ref_points', 'w')
for line in lines:
    fields = line.split(',')
    if fields[1] == '1':
        floc.write(fields[4] + ' ' + fields[5] + '\n')
        fref.write(fields[2] + ' ' + fields[3] + '\n')
floc.close()
fref.close()
