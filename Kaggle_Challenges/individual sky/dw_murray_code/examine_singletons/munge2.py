for ii in range(1,100+1):
    fid = open('../data/Train_Skies/Training_Sky%d.csv' % ii, 'r')
    data = fid.read()
    fid.close()
    lines = data.splitlines()[1:]
    fsky = open('sky/%d' % ii, 'w')
    for line in lines:
        fields = line.split(',')[1:]
        fsky.write(' '.join(fields) + '\n')
    fsky.close()
