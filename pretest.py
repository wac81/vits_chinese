import os

path = 'test/woman_csmsc.txt'
with open(path, 'r') as f:
    texts = f.readlines()
    
    


with open(path+'.2', 'w') as f:
    temp = []
    for text in texts:
        line = text.split('\t')
        temp.append( '/data/audio/data/csmsc/' + line[0] + '.wav'+'|' + line[1].replace('#',''))  #  test/data/0.wav
        
    texts = f.writelines(temp)