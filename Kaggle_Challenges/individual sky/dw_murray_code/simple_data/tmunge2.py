for ii in range(1,120+1):
    fid = open('../data/Test_Skies/Test_Sky%d.csv' % ii, 'r')
    data = fid.read()
    fid.close()
    lines = data.splitlines()[1:]
    fsky = open('Test_Skies/%d' % ii, 'w')
    for line in lines:
        fields = line.split(',')[1:]
        fsky.write(' '.join(fields) + '\n')
    fsky.close()
